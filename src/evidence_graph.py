"""Temporal evidence graph: dependency-aware obligation tracking with deterministic, evidence-referenced state derivation.

Design rules (task specification):
* completion of an upstream obligation NEVER completes a downstream one; each obligation is completed only by ITS OWN completion
  evidence (COMPLETED_BY edge from a verified completion event of its step type);
* missing documentation is not a confirmed failure: after a deadline an obligation is OVERDUE only with positive evidence of non-completion
  (tracker snapshot 'open' / documented no-show / cancellation); otherwise it is CENSORED (inside the grace window) or UNVERIFIED;
* event time and ingestion time are stored separately; state is derived from the SET of ingested evidence (ordered by event time), so duplicate
  deliveries are no-ops and delayed / out-of-order evidence reconciles automatically;
* dependencies must stay acyclic (cycles are rejected), and an obligation whose parent has not arrived yet keeps a pending dependency link.

Features for the risk model are computed here (`features`) with a strict event-time cutoff, so the same code serves offline training
(features at the dataset's assessment times) and live scoring (features at the current time).
"""
from __future__ import annotations

import bisect
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

import networkx as nx

from .ontology import (COMPLETION_EVENT, CONFLICT_EVENT, EVENT_NODE_TYPE, MISS_EVENTS, OPEN_EVENTS, REVIEW_STEPS, SNAPSHOT_EVENT, State)

_EPOCH_CACHE: dict[str, float] = {}


def to_epoch(s: str) -> float:
    v = _EPOCH_CACHE.get(s)
    if v is None:
        v = datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
        if len(_EPOCH_CACHE) < 2_000_000:
            _EPOCH_CACHE[s] = v
    return v


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


H = 3600.0


@dataclass
class Ev:
    eid: str
    etype: str
    t_event: float
    t_ingest: float
    verification: str
    provenance: str
    obligation_id: str | None
    payload: dict = field(default_factory=dict)


@dataclass
class Ob:
    oid: str
    patient_id: str
    encounter_id: str | None
    category: str
    step_type: str
    role: str
    created: float
    deadline: float
    depends_on: list[str]
    chain_id: str
    chain_position: int
    chain_length: int
    extraction_confidence: float
    source_dataset: str = ""


@dataclass
class StateDecision:
    state: State
    rule: str
    evidence_ids: list[str]
    late: bool = False
    detail: dict = field(default_factory=dict)

    def explain(self) -> str:
        return f"{self.state.value} [{self.rule}] evidence={self.evidence_ids} {self.detail if self.detail else ''}".strip()


@dataclass
class IngestResult:
    duplicate: bool = False
    affected: set = field(default_factory=set)
    kind: str = ""
    rejected: str | None = None
    notes: list = field(default_factory=list)


class TemporalEvidenceGraph:
    def __init__(self, grace_hours: float = 12.0, build_nx: bool = True):
        self.grace = grace_hours * H
        self.build_nx = build_nx
        self.G = nx.MultiDiGraph()               # full typed graph (task-specification node/edge types)
        self.dep = nx.DiGraph()                  # obligation -> parent obligation, kept acyclic
        self.obs: dict[str, Ob] = {}
        self.ob_events: dict[str, list[Ev]] = defaultdict(list)   # sorted by (t_event, eid)
        self._keys: dict[str, list[tuple[float, str]]] = defaultdict(list)
        self.dependents: dict[str, list[str]] = defaultdict(list)
        self.pending_deps: dict[str, list[str]] = defaultdict(list)   # parent id -> children waiting for it
        self.orphans: dict[str, list[Ev]] = defaultdict(list)         # events that arrived before their obligation
        self.patient_obs: dict[str, list[str]] = defaultdict(list)
        self.patient_encs: dict[str, list[tuple[float, str, str, str | None, str | None]]] = defaultdict(list)  # (t, enc_id, dept_or_type, type, dept)
        self.enc_info: dict[str, tuple[str | None, str | None, float]] = {}
        self.patient_noshow: dict[str, list[float]] = defaultdict(list)
        self.seen: set[str] = set()
        self.rejected_cycles: list[tuple[str, str]] = []
        self.n_duplicates = 0
        self._anc_cache: dict[str, list[str]] = {}

    # ------------------------------------------------------------------------------------------ construction
    def _node(self, nid: str, ntype: str, **attrs):
        if self.build_nx and nid not in self.G:
            self.G.add_node(nid, ntype=ntype, **attrs)

    def _edge(self, u: str, v: str, etype: str, **attrs):
        if self.build_nx:
            self.G.add_edge(u, v, key=etype + "|" + str(attrs.get("eid", "")), etype=etype, **attrs)

    def add_obligation(self, oid: str, pid: str, enc: str | None, payload: dict, t_event: float, t_ingest: float) -> IngestResult:
        res = IngestResult(kind="obligation_created", affected={oid})
        if oid in self.obs:
            self.n_duplicates += 1
            res.duplicate = True
            return res
        ob = Ob(oid, pid, enc, payload["category"], payload["step_type"], payload["responsible_role"], to_epoch(payload.get("created_time_utc") or iso(t_event)),
                to_epoch(payload["due_time_utc"]), list(payload["depends_on"]), payload["chain_id"], int(payload["chain_position"]), int(payload["chain_length"]),
                float(payload["extraction_confidence"]), payload.get("source_dataset", ""))
        self.obs[oid] = ob
        self.patient_obs[pid].append(oid)
        self._node(pid, "Patient")
        self._node(oid, "Obligation", step=ob.step_type, category=ob.category, deadline=ob.deadline, t_ingest=t_ingest)
        self._node(f"role:{ob.role}", "ResponsibleRole")
        self._edge(oid, pid, "BELONGS_TO")
        self._edge(oid, f"role:{ob.role}", "ASSIGNED_TO")
        if enc:
            self._node(enc, "Encounter")
            self._edge(oid, enc, "EXTRACTED_FROM")
        self.dep.add_node(oid)
        for parent in ob.depends_on:
            if parent in self.obs:
                self._link(oid, parent, res)
            else:
                self.pending_deps[parent].append(oid)       # parent not delivered yet: unresolved dependency
        for child in self.pending_deps.pop(oid, []):
            self._link(child, oid, res)
        for e in self.orphans.pop(oid, []):
            self._attach(e, res)
        self._anc_cache.clear()
        return res

    def _link(self, child: str, parent: str, res: IngestResult):
        if child == parent or nx.has_path(self.dep, parent, child):       # would create a cycle
            self.rejected_cycles.append((child, parent))
            res.rejected = f"cycle:{child}->{parent}"
            res.notes.append(f"dependency {child}->{parent} rejected (cycle)")
            return
        self.dep.add_edge(child, parent)
        self.dependents[parent].append(child)
        self._edge(child, parent, "DEPENDS_ON")
        res.affected.update({child, parent})

    # ------------------------------------------------------------------------------------------ events
    def ingest(self, e: dict) -> IngestResult:
        """Apply one delivery. Returns the affected obligation ids. Idempotent per event_id."""
        eid, et = e["event_id"], e["event_type"]
        if eid in self.seen:
            self.n_duplicates += 1
            return IngestResult(duplicate=True, kind=et)
        self.seen.add(eid)
        t_event, t_ing = to_epoch(e["event_time_utc"]), to_epoch(e["ingest_time_utc"])
        pl = e.get("payload") or {}
        pid = e["patient_id"]
        if et == "encounter_start":
            cid = e["encounter_id"]
            dept = pl.get("department")
            typ = pl.get("encounter_type")
            self.enc_info[cid] = (typ, dept, t_event)
            bisect.insort(self.patient_encs[pid], (t_event, cid, str(dept) if dept is not None else str(typ), typ, dept))
            self._node(cid, "Encounter", t_event=t_event)
            self._node(pid, "Patient")
            self._edge(cid, pid, "BELONGS_TO")
            return IngestResult(kind=et)
        if et == "obligation_created":
            return self.add_obligation(e["obligation_id"], pid, e.get("encounter_id"), pl, t_event, t_ing)
        if et == "deadline_reached":
            return IngestResult(kind=et, affected={e["obligation_id"]})
        ev = Ev(eid, et, t_event, t_ing, pl.get("verification", "verified"), e.get("provenance", ""), e.get("obligation_id"), pl)
        if et == "visit_no_show":
            bisect.insort(self.patient_noshow[pid], t_event)
        res = IngestResult(kind=et)
        if ev.obligation_id is None:
            return res
        if ev.obligation_id not in self.obs:
            self.orphans[ev.obligation_id].append(ev)           # out-of-order: obligation not delivered yet
            res.notes.append("orphan_buffered")
            return res
        self._attach(ev, res)
        return res

    def _attach(self, ev: Ev, res: IngestResult):
        oid = ev.obligation_id
        ob = self.obs[oid]
        key = (ev.t_event, ev.eid)
        i = bisect.bisect(self._keys[oid], key)
        self._keys[oid].insert(i, key)
        self.ob_events[oid].insert(i, ev)
        res.affected.add(oid)
        for child in self.dependents.get(oid, []):
            res.affected.add(child)                              # descendants' dependency features changed
        if self.build_nx:
            nt = EVENT_NODE_TYPE.get(ev.etype, "EvidenceEvent")
            self._node(ev.eid, nt, etype=ev.etype, t_event=ev.t_event, t_ingest=ev.t_ingest, provenance=ev.provenance, verification=ev.verification)
            self._edge(ev.eid, ob.patient_id, "BELONGS_TO", eid=ev.eid)
            self._edge(oid, ev.eid, "EVIDENCED_BY", eid=ev.eid, t_event=ev.t_event)
            if ev.etype == COMPLETION_EVENT[ob.step_type] and ev.verification == "verified":
                self._edge(oid, ev.eid, "COMPLETED_BY", eid=ev.eid, t_event=ev.t_event)
            if ev.etype == CONFLICT_EVENT:
                comp = next((x for x in self.ob_events[oid] if x.etype == COMPLETION_EVENT[ob.step_type]), None)
                if comp:
                    self._edge(ev.eid, comp.eid, "CONTRADICTS", eid=ev.eid, t_event=ev.t_event)
                self._edge(oid, f"role:{ob.role}", "REQUIRES_REVIEW", eid=ev.eid, reason="conflicting_evidence", t_event=ev.t_event)
            if ev.etype in (SNAPSHOT_EVENT, *MISS_EVENTS):
                self._edge(oid, f"role:{ob.role}", "REQUIRES_REVIEW", eid=ev.eid, reason="open_past_deadline" if ev.etype == SNAPSHOT_EVENT else ev.etype, t_event=ev.t_event)

    # ------------------------------------------------------------------------------------------ queries
    def ancestors(self, oid: str) -> list[str]:
        if oid in self._anc_cache:
            return self._anc_cache[oid]
        out: list[str] = []
        for p in self.dep.successors(oid) if oid in self.dep else []:
            out.append(p)
            out.extend(self.ancestors(p))
        self._anc_cache[oid] = list(dict.fromkeys(out))
        return self._anc_cache[oid]

    def completion_time(self, oid: str, as_of: float, live: bool = False) -> float | None:
        ob = self.obs.get(oid)
        if ob is None:
            return None
        ct = COMPLETION_EVENT[ob.step_type]
        for e in self.ob_events.get(oid, ()):
            if e.t_event > as_of:
                break
            if e.etype == ct and (not live or e.t_ingest <= as_of):
                return e.t_event
        return None

    def derive_state(self, oid: str, now: float, live: bool = False) -> StateDecision:
        """live=True: only evidence already INGESTED by `now` counts (what the running service could know); live=False: all evidence with event_time <= now."""
        ob = self.obs[oid]
        evs = [e for e in self.ob_events.get(oid, ()) if e.t_event <= now and (not live or e.t_ingest <= now)]
        ct = COMPLETION_EVENT[ob.step_type]
        comp = next((e for e in evs if e.etype == ct), None)
        conflict = next((e for e in evs if e.etype == CONFLICT_EVENT), None)
        anc_open = [a for a in self.ancestors(oid) if self.completion_time(a, now, live) is None]
        detail = {"upstream_incomplete": anc_open[:5]} if anc_open else {}
        if conflict is not None:
            return StateDecision(State.CONTRADICTORY, "R2_conflicting_secondary_record", [x.eid for x in (comp, conflict) if x], detail=detail)
        if comp is not None:
            if comp.verification != "verified":
                return StateDecision(State.UNVERIFIED, "R3_completion_evidence_not_verified", [comp.eid], detail=detail)
            parents_missing = [p for p in ob.depends_on if p in self.obs and self.completion_time(p, now, live) is None]
            d = dict(detail)
            if parents_missing:
                d["dependency_evidence_gap"] = parents_missing
            return StateDecision(State.COMPLETED, "R1_own_verified_completion_evidence", [comp.eid], late=comp.t_event > ob.deadline, detail=d)
        if now <= ob.deadline:
            return StateDecision(State.PENDING, "R4_before_deadline_no_completion", [], detail=detail)
        miss = [e for e in evs if (e.etype == SNAPSHOT_EVENT and e.payload.get("details", {}).get("status", "open") == "open") or e.etype in MISS_EVENTS]
        if miss:
            return StateDecision(State.OVERDUE, "R5_deadline_passed_open_confirmed", [miss[-1].eid], detail=detail)
        if now < ob.deadline + self.grace:
            return StateDecision(State.CENSORED, "R6_deadline_passed_awaiting_late_evidence", [], detail=detail)
        return StateDecision(State.UNVERIFIED, "R7_deadline_passed_no_evidence_either_way", [], detail=detail)

    # ------------------------------------------------------------------------------------------ features (dataset-compatible)
    def features(self, oid: str, now: float, live: bool = False) -> dict:
        ob = self.obs[oid]
        A, D = now, ob.deadline
        anc = self.ancestors(oid)
        chain = [oid] + anc
        evs = sorted((e for o in chain for e in self.ob_events.get(o, ()) if e.t_event <= A and (not live or e.t_ingest <= A)), key=lambda e: e.t_event)
        own = [e for e in self.ob_events.get(oid, ()) if e.t_event <= A and (not live or e.t_ingest <= A)]
        comp = lambda o: self.completion_time(o, A, live)  # noqa: E731
        direct = [p for p in ob.depends_on if p in self.obs]
        inc_dep = sum(1 for p in ob.depends_on if not (p in self.obs and comp(p) is not None))
        inc_anc = sum(1 for p in anc if comp(p) is None)
        if not ob.depends_on:
            pstate, hs_parent = "none", float("nan")
        else:
            ptimes = [comp(p) for p in direct if comp(p) is not None]
            full = len(ptimes) == len(ob.depends_on)
            pstate = "completed" if full else "pending"
            hs_parent = (A - max(ptimes)) / H if full else float("nan")
        appt = [e for e in evs if e.etype in ("appointment_scheduled", "appointment_rescheduled")]
        appt_h = float("nan")
        if appt:
            at = appt[-1].payload.get("details", {}).get("appointment_time")
            if at:
                appt_h = (to_epoch(at) - D) / H
        if ob.step_type in REVIEW_STEPS:
            ack = "viewed_unacknowledged" if any(e.etype in OPEN_EVENTS for e in own) else ("not_viewed" if pstate == "completed" else "awaiting_result")
        else:
            ack = "not_applicable"
        nv = sum(e.verification == "verified" for e in evs)
        nu = len(evs) - nv
        # same-patient context (other chains only)
        due_prior = ont = conc = conc_role = 0
        for o in self.patient_obs[ob.patient_id]:
            ot = self.obs[o]
            if ot.chain_id == ob.chain_id or ot.created > A:
                continue
            oc = comp(o)
            done = oc is not None
            if ot.deadline < A:
                due_prior += 1
                ont += int(done and oc <= ot.deadline)
            elif not done:
                conc += 1
                conc_role += int(ot.role == ob.role)
        ns = bisect.bisect_right(self.patient_noshow[ob.patient_id], A)
        n_enc = n_dep = 0
        otype, odept = "none", "none"
        if ob.encounter_id and ob.encounter_id in self.enc_info:
            typ, dept, t0 = self.enc_info[ob.encounter_id]
            otype = str(typ)
            odept = str(dept) if dept is not None else str(typ)
            for (t, _cid, cur, _typ, _dept) in self.patient_encs[ob.patient_id]:
                if t0 < t <= A:
                    n_enc += 1
                    n_dep += int(cur != odept)
        roles = [self.obs[o].role for o in sorted(chain, key=lambda o: self.obs[o].chain_position)]
        handoffs = sum(1 for a, b in zip(roles, roles[1:]) if a != b)
        last = evs[-1].t_event if evs else None
        last_own = own[-1].t_event if own else None
        dt = datetime.fromtimestamp(D, timezone.utc)
        win = (D - ob.created) / H
        return {
            "category": ob.category, "step_type": ob.step_type, "responsible_role": ob.role, "origin_encounter_type": otype, "origin_department": odept,
            "chain_position": ob.chain_position, "chain_length": ob.chain_length, "window_hours": win, "hours_to_deadline": (D - A) / H,
            "elapsed_fraction": ((A - ob.created) / H) / max(win, 1e-6), "deadline_dow": dt.weekday(), "deadline_is_weekend": int(dt.weekday() >= 5),
            "n_dependencies": len(ob.depends_on), "n_incomplete_dependencies": inc_dep, "n_incomplete_ancestors": inc_anc, "parent_state": pstate,
            "hours_since_parent_completed": hs_parent, "n_events_so_far": len(evs),
            "hours_since_last_event": (A - last) / H if last is not None else (A - ob.created) / H, "no_events_yet": int(not evs),
            "n_reminders": sum(e.etype == "reminder_sent" for e in evs), "n_contact_attempts": sum(e.etype == "contact_attempt" for e in evs),
            "has_scheduled_appointment": int(bool(appt)), "appointment_minus_deadline_hours": appt_h,
            "n_reschedules": sum(e.etype == "appointment_rescheduled" for e in evs), "has_cancellation": int(any(e.etype == "appointment_cancelled" for e in evs)),
            "ack_state": ack, "n_verified_events": nv, "n_unverified_events": nu, "has_conflicting_record": int(any(e.etype == CONFLICT_EVENT for e in evs)),
            "evidence_uncertainty": (nu / len(evs)) if evs else 0.0, "has_documented_source_evidence": int(any(e.provenance == "documented_source" for e in evs)),
            "extraction_confidence": ob.extraction_confidence, "prior_n_due": due_prior, "prior_n_completed_ontime": ont,
            "prior_completion_rate": (ont + 1) / (due_prior + 2), "prior_noshow_count": ns, "n_concurrent_open": conc, "n_concurrent_open_same_role": conc_role,
            "role_handoffs": handoffs, "n_encounters_since_origin": n_enc, "n_department_transitions_since_origin": n_dep,
            # independent-task (own-event) view used by the baseline that does not know about dependencies
            "own_n_events": len(own), "own_hours_since_last_event": (A - last_own) / H if last_own is not None else (A - ob.created) / H, "own_no_events": int(not own),
            "own_n_reminders": sum(e.etype == "reminder_sent" for e in own), "own_n_contact_attempts": sum(e.etype == "contact_attempt" for e in own),
            "own_viewed": int(any(e.etype in OPEN_EVENTS for e in own)), "own_has_cancellation": int(any(e.etype == "appointment_cancelled" for e in own)),
        }

    # ------------------------------------------------------------------------------------------ views
    def snapshot(self, t: float) -> nx.MultiDiGraph:
        """Subgraph knowable at event time t (nodes/edges stamped with t_event <= t; structural nodes always included)."""
        keep = [n for n, d in self.G.nodes(data=True) if d.get("t_event", -1) <= t and d.get("ntype") not in ("Obligation",) or
                (d.get("ntype") == "Obligation" and self.obs[n].created <= t)]
        sub = self.G.subgraph(keep).copy()
        drop = [(u, v, k) for u, v, k, d in sub.edges(keys=True, data=True) if d.get("t_event", -1) > t]
        sub.remove_edges_from(drop)
        return sub

    def stats(self) -> dict:
        return {"obligations": len(self.obs), "events_ingested": len(self.seen), "duplicates_ignored": self.n_duplicates,
                "nx_nodes": self.G.number_of_nodes(), "nx_edges": self.G.number_of_edges(), "rejected_cycles": len(self.rejected_cycles),
                "orphan_events_waiting": sum(len(v) for v in self.orphans.values()), "unresolved_dependencies": sum(len(v) for v in self.pending_deps.values())}
