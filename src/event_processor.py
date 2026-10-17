"""Real-time event processor: idempotent, transaction-safe replay engine around the temporal evidence graph.

Per delivery: (1) validate schema/provenance, (2) identify patient/obligation, (3) update the graph, (4) recompute affected workflow states,
(5) compute current-time features, (6) score risk with the live models, (7) apply deterministic alert rules, (8) build evidence-grounded
explanations, (9) append to the audit log. SQLite (WAL) is the source of truth: the in-memory graph is a cache rebuilt from the event table
by `recover`. Duplicate deliveries are ignored (event_id primary key); late/out-of-order evidence reconciles state and is logged with both
event time and ingestion time. Time is simulated from the stream's ingest timestamps (`speed` only throttles wall-clock for demos).
"""
from __future__ import annotations

import heapq
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .alert_engine import AlertEngine, PriorityScorer
from .evidence_graph import StateDecision, TemporalEvidenceGraph, iso, to_epoch
from .ontology import COMPLETION_EVENT, State

H = 3600.0
ALLOWED_PROVENANCE = {"documented_source", "extracted_rule", "extracted_genai", "simulated", "inferred"}
KNOWN_EVENTS = set(COMPLETION_EVENT.values()) | {"encounter_start", "obligation_created", "deadline_reached", "result_opened", "medication_opened", "reminder_sent",
                                                  "contact_attempt", "appointment_rescheduled", "visit_no_show", "appointment_cancelled", "task_status_snapshot",
                                                  "conflicting_record", "imaging_order_documented"}
REQUIRED = ("event_id", "patient_id", "event_time_utc", "ingest_time_utc", "event_type", "provenance")
SCHEMA = """
CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE, patient_id TEXT, obligation_id TEXT, encounter_id TEXT, event_type TEXT,
  event_time TEXT, ingest_time TEXT, provenance TEXT, payload TEXT, deliveries INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS obligations(oid TEXT PRIMARY KEY, patient_id TEXT, state TEXT, rule TEXT, late INTEGER, state_time TEXT, last_risk REAL, data TEXT);
CREATE TABLE IF NOT EXISTS transitions(id INTEGER PRIMARY KEY AUTOINCREMENT, oid TEXT, t_ingest TEXT, trigger_event_id TEXT, old_state TEXT, new_state TEXT,
  rule TEXT, evidence TEXT, explanation TEXT);
CREATE TABLE IF NOT EXISTS alerts(alert_id TEXT PRIMARY KEY, oid TEXT, rule TEXT, severity TEXT, t_ingest TEXT, status TEXT, resolved_at TEXT, resolution TEXT, risk REAL,
  priority REAL, payload TEXT);
CREATE TABLE IF NOT EXISTS risk_log(oid TEXT, t_clock REAL, trigger TEXT, scorer TEXT, risk REAL, hours_to_deadline REAL);
CREATE TABLE IF NOT EXISTS rejected(delivery_id TEXT, reason TEXT, raw TEXT);
CREATE INDEX IF NOT EXISTS ix_events_oid ON events(obligation_id);
CREATE INDEX IF NOT EXISTS ix_trans_oid ON transitions(oid);
"""


def deadline_only_score(rows: list[dict]) -> list[float]:
    """System A: deadline-driven reminder urgency (fraction of the window elapsed); knows nothing about dependencies or evidence."""
    return [float(min(max(r["elapsed_fraction"], 0.0), 1.0)) for r in rows]


class EventProcessor:
    def __init__(self, cfg: dict, db_path: str | Path = ":memory:", scorers: dict | None = None, primary: str = "C_graph_ml", alert_engine: AlertEngine | None = None,
                 commit_every: int | None = None, rescore_batch: int = 1, log_risk: bool = True, build_nx: bool = True, track_latency: bool = False,
                 checkpoints: tuple[float, ...] = (0.25, 0.5, 0.75), checkpoint_hours: tuple[float, ...] = (24.0, 6.0), enable_alerts: bool = True):
        self.cfg = cfg
        self.graph = TemporalEvidenceGraph(cfg["engine"]["grace_hours"], build_nx=build_nx)
        self.db_path = str(db_path)
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL" if self.db_path != ":memory:" else "PRAGMA journal_mode=MEMORY")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self.scorers = scorers or {}
        self.primary = primary
        self.alerts = alert_engine or AlertEngine(cfg)
        self.enable_alerts = enable_alerts
        self.commit_every = commit_every or cfg["engine"]["commit_every"]
        self.rescore_batch = rescore_batch
        self.log_risk = log_risk
        self.track_latency = track_latency
        self.checkpoints, self.checkpoint_hours = checkpoints, checkpoint_hours
        self.clock = 0.0
        self.states: dict[str, tuple[State, str, bool]] = {}
        self.last_risk: dict[str, dict] = {}
        self.timers: list[tuple[float, int, str, str]] = []
        self._seq = 0
        self._dirty: dict[str, tuple[str | None, float]] = {}
        self._buf = {"events": [], "dup": [], "trans": [], "alerts": [], "alert_upd": [], "risk": [], "rej": [], "obl": {}}
        self.stats = Counter()
        self.latencies: list[float] = []
        self.transitions: list[dict] = []          # in-memory audit mirror (also persisted)
        self.alert_log: list = []
        self._n_since_commit = 0

    # ----------------------------------------------------------------------------------------- validation
    def _validate(self, ev: dict) -> str | None:
        for k in REQUIRED:
            if ev.get(k) in (None, ""):
                return f"missing_field:{k}"
        if ev["provenance"] not in ALLOWED_PROVENANCE:
            return "unknown_provenance"
        if ev["event_type"] not in KNOWN_EVENTS:
            return "unknown_event_type"
        try:
            te, ti = to_epoch(ev["event_time_utc"]), to_epoch(ev["ingest_time_utc"])
        except Exception:  # noqa: BLE001
            return "bad_timestamp"
        if ti + 5 < te:
            return "ingest_before_event"
        if ev["event_type"] not in ("encounter_start",) and ev.get("obligation_id") in (None, ""):
            return "missing_obligation_id"
        if ev["event_type"] == "obligation_created":
            pl = ev.get("payload") or {}
            for k in ("category", "step_type", "responsible_role", "due_time_utc", "depends_on", "chain_id", "chain_position", "chain_length", "extraction_confidence"):
                if k not in pl:
                    return f"obligation_payload_missing:{k}"
        return None

    # ----------------------------------------------------------------------------------------- timers
    def _schedule(self, oid: str):
        ob = self.graph.obs[oid]
        win = ob.deadline - ob.created
        ts: list[tuple[float, str]] = [(ob.deadline + 1.0, "deadline"), (ob.deadline + self.graph.grace + 1.0, "grace_end")]
        for f in self.checkpoints:
            ts.append((ob.created + f * win, f"checkpoint_{f}"))
        for h in self.checkpoint_hours:
            if ob.deadline - h * H > ob.created:
                ts.append((ob.deadline - h * H, f"checkpoint_T-{h:g}h"))
        for t, kind in ts:
            self._seq += 1
            heapq.heappush(self.timers, (t, self._seq, oid, kind))

    def _advance(self, upto: float):
        if self._dirty and self.timers and self.timers[0][0] <= upto:       # keep per-obligation transitions chronological in micro-batch mode
            self._refresh(self._dirty, self.clock)
            self._dirty = {}
        due = []
        while self.timers and self.timers[0][0] <= upto:
            t, _, oid, kind = heapq.heappop(self.timers)
            if oid in self.graph.obs:
                self.stats["timer_fired"] += 1
                due.append((oid, f"timer:{kind}", t))
        if due:
            self._refresh(due)

    # ----------------------------------------------------------------------------------------- main entry
    def process(self, ev: dict) -> dict:
        t0 = time.perf_counter() if self.track_latency else 0.0
        out = {"event_id": ev.get("event_id"), "status": "ok"}
        self.stats["deliveries"] += 1
        bad = self._validate(ev)
        if bad:
            self._buf["rej"].append((ev.get("delivery_id") or ev.get("event_id"), bad, json.dumps(ev, default=str)[:2000]))
            self.stats["rejected"] += 1
            out["status"] = f"rejected:{bad}"
            return out
        t_ing = to_epoch(ev["ingest_time_utc"])
        self._advance(t_ing)
        self.clock = max(self.clock, t_ing)
        res = self.graph.ingest(ev)
        if res.duplicate:
            self.stats["duplicates"] += 1
            self._buf["dup"].append(ev["event_id"])
            out["status"] = "duplicate"
        else:
            self._buf["events"].append((ev["event_id"], ev["patient_id"], ev.get("obligation_id"), ev.get("encounter_id"), ev["event_type"], ev["event_time_utc"], ev["ingest_time_utc"], ev["provenance"],
                                        json.dumps(ev.get("payload") or {})))
            if res.kind == "obligation_created" and not res.rejected:
                self._schedule(ev["obligation_id"])
            if res.rejected:
                self.stats["rejected_dependency_cycles"] += 1
            if t_ing - to_epoch(ev["event_time_utc"]) > 3600 and res.kind not in ("deadline_reached",):
                self.stats["late_events"] += 1
            for oid in res.affected:
                self._dirty[oid] = (ev["event_id"], t_ing)
            out["affected"] = len(res.affected)
            if len(self._dirty) >= self.rescore_batch:
                self._refresh(self._dirty, t_ing)
                self._dirty = {}
        self._n_since_commit += 1
        if self._n_since_commit >= self.commit_every:
            self.commit()
        if self.track_latency:
            self.latencies.append(time.perf_counter() - t0)
        return out

    def flush(self):
        if self._dirty:
            self._refresh(self._dirty, self.clock)
            self._dirty = {}

    # ----------------------------------------------------------------------------------------- state/risk/alerts
    def _refresh(self, work, now_default: float | None = None):
        """work: dict oid -> (trigger, t) or list of (oid, trigger, t). Each item is evaluated at ITS OWN time t using only evidence ingested by t,
        so batching never changes results; model scoring is batched across all items."""
        g = self.graph
        if isinstance(work, dict):
            work = [(oid, trig, t) for oid, (trig, t) in work.items()]
        items = sorted([(oid, trig, t) for oid, trig, t in work if oid in g.obs], key=lambda x: x[2])
        decs: dict[int, StateDecision] = {}
        feats: dict[int, dict] = {}
        for k, (oid, trig, t) in enumerate(items):
            decs[k] = g.derive_state(oid, t, live=True)
            if decs[k].state == State.PENDING and (self.scorers or self.enable_alerts):
                feats[k] = g.features(oid, t, live=True)
        risks: dict[int, dict[str, float]] = {k: {} for k in feats}
        if feats and self.scorers:
            ks = list(feats)
            rows = [feats[k] for k in ks]
            for name, sc in self.scorers.items():
                preds = sc(rows) if callable(sc) and not hasattr(sc, "predict_dicts") else sc.predict_dicts(rows)
                for k, p in zip(ks, preds):
                    risks[k][name] = float(p)
        for k, (oid, trig, t) in enumerate(items):
            dec = decs[k]
            prev = self.states.get(oid)
            changed = prev is None or prev[0] != dec.state or prev[2] != dec.late
            if changed:
                self._transition(oid, prev[0] if prev else None, dec, trig, t)
                self.states[oid] = (dec.state, dec.rule, dec.late)
                self._buf["obl"][oid] = None
            rk = risks.get(k)
            if rk:
                self.last_risk[oid] = {"t": t, **rk}
                if self.log_risk:
                    htd = feats[k]["hours_to_deadline"]
                    for name, v in rk.items():
                        self._buf["risk"].append((oid, t, trig, name, v, htd))
            if self.enable_alerts:
                prim = rk.get(self.primary) if rk else None
                new, resolved = self.alerts.evaluate(g, oid, t, dec, prev[0] if prev else None, prim, feats.get(k), trig)
                for a in new:
                    pr = a.priority
                    self._buf["alerts"].append(a)
                    self.alert_log.append(a)
                    self.stats[f"alert:{a.rule}"] += 1
                for a in resolved:
                    self._buf["alert_upd"].append(a)
                    self.stats["alerts_resolved"] += 1

    def _transition(self, oid, old, dec: StateDecision, trig, t):
        ob = self.graph.obs[oid]
        txt = f"{ob.step_type}: {old.value if old else 'new'} -> {dec.state.value}{' (late)' if dec.late else ''} by {dec.rule}; evidence={dec.evidence_ids}"
        rec = {"oid": oid, "t_ingest": iso(t), "trigger_event_id": trig, "old_state": old.value if old else None, "new_state": dec.state.value, "late": dec.late,
               "rule": dec.rule, "evidence": dec.evidence_ids, "explanation": txt}
        self.transitions.append(rec)
        self._buf["trans"].append((oid, rec["t_ingest"], trig, rec["old_state"], rec["new_state"], dec.rule, json.dumps(dec.evidence_ids), txt))
        self.stats["transitions"] += 1

    # ----------------------------------------------------------------------------------------- persistence
    def commit(self):
        b = self._buf
        try:
            with self.db:                                                        # one atomic transaction per batch
                if b["events"]:
                    self.db.executemany("INSERT OR IGNORE INTO events(event_id,patient_id,obligation_id,encounter_id,event_type,event_time,ingest_time,provenance,payload) VALUES (?,?,?,?,?,?,?,?,?)", b["events"])
                if b["dup"]:
                    self.db.executemany("UPDATE events SET deliveries = deliveries + 1 WHERE event_id = ?", [(e,) for e in b["dup"]])
                if b["trans"]:
                    self.db.executemany("INSERT INTO transitions(oid,t_ingest,trigger_event_id,old_state,new_state,rule,evidence,explanation) VALUES (?,?,?,?,?,?,?,?)", b["trans"])
                for oid in b["obl"]:
                    st = self.states[oid]
                    ob = self.graph.obs[oid]
                    self.db.execute("INSERT OR REPLACE INTO obligations(oid,patient_id,state,rule,late,state_time,last_risk,data) VALUES (?,?,?,?,?,?,?,?)",
                                    (oid, ob.patient_id, st[0].value, st[1], int(st[2]), iso(self.clock), (self.last_risk.get(oid) or {}).get(self.primary),
                                     json.dumps({"step": ob.step_type, "category": ob.category, "role": ob.role, "deadline": iso(ob.deadline), "chain": ob.chain_id})))
                if b["alerts"]:
                    self.db.executemany("INSERT OR REPLACE INTO alerts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                        [(a.alert_id, a.oid, a.rule, a.severity, iso(a.t_ingest), a.status, None, None, a.risk, a.priority, json.dumps(a.to_dict(), default=str)) for a in b["alerts"]])
                if b["alert_upd"]:
                    self.db.executemany("UPDATE alerts SET status='resolved', resolved_at=?, resolution=? WHERE alert_id=?",
                                        [(iso(a.resolved_at), a.resolution, a.alert_id) for a in b["alert_upd"]])
                if b["risk"]:
                    self.db.executemany("INSERT INTO risk_log VALUES (?,?,?,?,?,?)", b["risk"])
                if b["rej"]:
                    self.db.executemany("INSERT INTO rejected VALUES (?,?,?)", b["rej"])
        except Exception:
            self.db.rollback()
            raise
        self._buf = {"events": [], "dup": [], "trans": [], "alerts": [], "alert_upd": [], "risk": [], "rej": [], "obl": {}}
        self._n_since_commit = 0

    def finalize(self, until: float | None = None):
        """Fire remaining timers (deadline / grace expiry) up to `until` (default: after the last timer) and commit."""
        self.flush()
        self._advance(until if until is not None else (max((t for t, *_ in self.timers), default=self.clock) + 1))
        self.commit()

    def close(self):
        self.commit()
        self.db.close()

    # ----------------------------------------------------------------------------------------- recovery
    @classmethod
    def recover(cls, cfg: dict, db_path: str | Path, **kw) -> "EventProcessor":
        """Rebuild the in-memory graph from the persisted event log (SQLite is the source of truth) and restore stored states."""
        p = cls(cfg, db_path, **kw)
        rows = p.db.execute("SELECT event_id, patient_id, obligation_id, encounter_id, event_type, event_time, ingest_time, provenance, payload FROM events ORDER BY seq").fetchall()
        for eid, pid, oid, enc, et, tev, ti, prov, pl in rows:
            p.graph.ingest({"event_id": eid, "patient_id": pid, "obligation_id": oid, "event_type": et, "event_time_utc": tev, "ingest_time_utc": ti,
                            "provenance": prov, "payload": json.loads(pl), "encounter_id": enc})
            p.clock = max(p.clock, to_epoch(ti))
        for oid, st, rule, late in p.db.execute("SELECT oid, state, rule, late FROM obligations"):
            p.states[oid] = (State(st), rule, bool(late))
        for oid in p.graph.obs:
            p._schedule(oid)
        p.timers = [t for t in p.timers if t[0] > p.clock]
        heapq.heapify(p.timers)
        return p

    # ----------------------------------------------------------------------------------------- streaming driver
    def run(self, events, speed: float | None = None, on_event=None, limit: int | None = None) -> dict:
        """Process an iterable of deliveries. speed=None: as fast as possible. speed=k: 1 simulated hour = 3600/k wall seconds (demo throttling)."""
        n, last_sim = 0, None
        t0 = time.perf_counter()
        for ev in events:
            if speed:
                ts = to_epoch(ev["ingest_time_utc"])
                if last_sim is not None and ts > last_sim:
                    time.sleep(min((ts - last_sim) / speed, 2.0))
                last_sim = ts
            r = self.process(ev)
            if on_event:
                on_event(ev, r, self)
            n += 1
            if limit and n >= limit:
                break
        self.flush()
        self.commit()
        return {"events": n, "seconds": time.perf_counter() - t0, "events_per_s": n / max(time.perf_counter() - t0, 1e-9), **dict(self.stats)}

    # ----------------------------------------------------------------------------------------- reporting
    def state_counts(self) -> dict:
        return dict(Counter(s[0].value for s in self.states.values()))

    def risk_log_df(self) -> pd.DataFrame:
        self.commit()
        return pd.read_sql("SELECT * FROM risk_log", self.db)

    def alerts_df(self) -> pd.DataFrame:
        self.commit()
        return pd.read_sql("SELECT alert_id, oid, rule, severity, t_ingest, status, resolved_at, resolution, risk, priority FROM alerts", self.db)

    def audit_df(self, oid: str | None = None) -> pd.DataFrame:
        self.commit()
        q = "SELECT * FROM transitions" + (" WHERE oid = ?" if oid else "") + " ORDER BY id"
        return pd.read_sql(q, self.db, params=(oid,) if oid else ())

    def integrity_check(self) -> dict:
        self.commit()
        c = self.db
        return {"sqlite_integrity": c.execute("PRAGMA integrity_check").fetchone()[0], "events": c.execute("SELECT COUNT(*) FROM events").fetchone()[0],
                "event_ids_unique": c.execute("SELECT COUNT(*) = COUNT(DISTINCT event_id) FROM events").fetchone()[0] == 1,
                "orphan_transitions": c.execute("SELECT COUNT(*) FROM transitions WHERE oid NOT IN (SELECT oid FROM obligations)").fetchone()[0],
                "alerts": c.execute("SELECT COUNT(*) FROM alerts").fetchone()[0], "rejected": c.execute("SELECT COUNT(*) FROM rejected").fetchone()[0]}
