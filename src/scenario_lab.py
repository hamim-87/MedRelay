"""Synthetic discharge episodes with explicit, seeded event timelines (novel templates reserved for evaluation).

Used for (i) the end-to-end GenAI-extraction ablation (instruction text -> extracted obligations -> monitoring -> alerts),
(ii) evidence-uncertainty injections and (iii) the live demonstration. NOTHING here is real patient data. Outcomes are derived from the emitted
events with the same deadline/evidence rules as the dataset pipeline (completion by deadline = completed; open with tracker confirmation = overdue;
no evidence either way = unverified), never drawn directly.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from .clinical_extractor import ExtractedObligation, interval_hours, step_of
from .evidence_graph import iso, to_epoch
from .ontology import COMPLETION_EVENT, STEP_SPEC

H = 3600.0
DEFAULT_WINDOW_H = {"laboratory_test": 168, "imaging": 336, "appointment_referral": 720, "followup_review": 720, "procedure": 720, "medication_action": 336,
                    "patient_communication": 72, "other_action": 336}   # simulated workflow definitions used ONLY when no deadline is stated
GAP_AFTER_PARENT_H = 72                                                   # review / notification due this long after the parent's deadline
T0 = to_epoch("2024-03-04T09:00:00Z")


def engine_step(task_type: str, clause: str, has_parent: bool) -> str:
    st = step_of(task_type, clause)
    if st == "review":
        return "result_reviewed"
    if st == "notify":
        return "patient_notified" if has_parent else "patient_contacted"
    if st == "appointment":
        return "appointment_attended"
    if task_type == "followup_review":
        return "followup_visit_completed"
    return {"laboratory_test": "lab_collected", "imaging": "imaging_performed", "procedure": "procedure_performed", "medication_action": "medication_dispensed"}.get(task_type, "action_completed")


def impute_deadlines(items: list[dict], t0: float) -> list[tuple[float, bool]]:
    """items: {task_type, due_hours|None, deps: [indices]} in order -> (deadline_epoch, imputed)."""
    out: list[tuple[float, bool]] = []
    for it in items:
        if it["due_hours"] is not None:
            out.append((t0 + it["due_hours"] * H, False))
        elif it["deps"]:
            out.append((max(out[d][0] for d in it["deps"] if d < len(out)) + GAP_AFTER_PARENT_H * H, True))
        else:
            out.append((t0 + DEFAULT_WINDOW_H[it["task_type"]] * H, True))
    return out


def true_obligations(doc: dict, t0: float) -> list[dict]:
    items = [{"task_type": o["task_type"], "due_hours": interval_hours(o["due_interval"]), "deps": o["depends_on"]} for o in doc["obligations"]]
    dl = impute_deadlines(items, t0)
    out = []
    for j, (o, (d, imp)) in enumerate(zip(doc["obligations"], dl)):
        step = engine_step(o["task_type"], o["span"], bool(o["depends_on"]))
        out.append({"idx": j, "id": f"{doc['doc_id']}:t{j}", "task_type": o["task_type"], "step": step, "deadline": d, "imputed": imp, "deps": o["depends_on"], "start": o["start"], "end": o["end"],
                    "span": o["span"]})
    return out


# ------------------------------------------------------------------------------------------------ explicit event simulation
BEHAVIOURS = ["all_good", "ack_missing", "late_ack", "no_show", "missed_test", "delayed_documentation", "contradiction", "slow_but_ok"]


def simulate_events(tobs: list[dict], patient: str, doc_id: str, behaviour: str, rng: np.random.Generator, t0: float) -> tuple[list[dict], dict]:
    """Emit stream-format events for the TRUE obligations of one episode and derive the outcome of each from those events."""
    evs: list[dict] = []
    done: dict[int, float | None] = {}
    seq = [0]
    delay_doc = behaviour == "delayed_documentation"

    def emit(oid, etype, t, ver="verified", details=None, ingest_extra=0.0):
        seq[0] += 1
        ing = t + float(np.exp(np.log(20) + 0.6 * rng.standard_normal())) + ingest_extra
        evs.append({"event_id": f"{doc_id}:e{seq[0]}", "patient_id": patient, "encounter_id": None, "obligation_id": oid, "event_time_utc": iso(t), "ingest_time_utc": iso(max(ing, t)),
                    "event_type": etype, "payload": {"status": "ok", "evidence_source": "scenario_lab", "verification": ver, "details": details or {}, "source_event_id": None},
                    "provenance": "simulated", "is_simulated": True, "split": "test", "source_dataset": "scenario_lab", "delivery_attempt": 1})
        return evs[-1]

    for o in tobs:
        parents_done = [done.get(d) for d in o["deps"]]
        if any(p is None for p in parents_done):
            done[o["idx"]] = None
            continue
        start = max([t0] + [p for p in parents_done if p is not None])
        win = max(o["deadline"] - start, 6 * H)
        step, typ = o["step"], o["task_type"]
        ok, t_done = True, None
        if behaviour == "missed_test" and step in ("lab_collected", "imaging_performed", "procedure_performed"):
            ok = False
        if behaviour == "ack_missing" and step == "result_reviewed":
            ok = False
        if behaviour == "no_show" and step in ("appointment_attended", "followup_visit_completed"):
            ok = False
        if not ok:
            done[o["idx"]] = None
            if step in ("appointment_attended", "followup_visit_completed"):
                emit(o["id"], "visit_no_show", min(o["deadline"] - 2 * H, start + 0.6 * win))
            if step == "result_reviewed":                                    # the result was posted/opened but never acknowledged
                emit(o["id"], "result_opened", start + float(rng.uniform(6, 30)) * H)
            continue
        frac = float(rng.uniform(0.3, 0.8)) if behaviour != "slow_but_ok" else float(rng.uniform(0.85, 0.97))
        t_done = start + frac * win
        if behaviour == "late_ack" and step == "result_reviewed":
            t_done = o["deadline"] + float(rng.uniform(6, 60)) * H
        if step == "result_reviewed":
            emit(o["id"], "result_opened", min(t_done - 2 * H, start + float(rng.uniform(4, 24)) * H))
        if step in ("lab_collected", "imaging_performed"):
            emit(o["id"], "result_posted" if step == "lab_collected" else "report_posted", t_done + float(rng.uniform(6, 30)) * H)
        if step in ("lab_collected", "patient_notified", "patient_contacted") and rng.random() < 0.4:
            emit(o["id"], "reminder_sent", start + 0.3 * win)
        emit(o["id"], COMPLETION_EVENT[step], t_done, ingest_extra=float(rng.uniform(24, 96)) * H if delay_doc else 0.0)
        done[o["idx"]] = t_done
        if behaviour == "contradiction" and o["idx"] == 0:
            emit(o["id"], "conflicting_record", t_done + float(rng.uniform(1, 30)) * H, ver="unverified")
    outcome = {}
    for o in tobs:
        mine = [e for e in evs if e["obligation_id"] == o["id"]]
        comp = [e for e in mine if e["event_type"] == COMPLETION_EVENT[o["step"]]]
        conflict = any(e["event_type"] == "conflicting_record" for e in mine)
        miss = any(e["event_type"] in ("visit_no_show",) for e in mine)
        late_ts = None
        if conflict and comp:
            outcome[o["id"]] = "unverified"
        elif comp and to_epoch(comp[0]["event_time_utc"]) <= o["deadline"]:
            outcome[o["id"]] = "completed"
        elif comp:
            outcome[o["id"]] = "overdue"
        else:
            snap = rng.random() < 0.9 and not delay_doc
            if snap or miss:
                emit(o["id"], "task_status_snapshot", o["deadline"] + H, details={"status": "open"})
                outcome[o["id"]] = "overdue"
            else:
                outcome[o["id"]] = "unverified"
    evs.sort(key=lambda e: (e["ingest_time_utc"], e["event_id"]))
    return evs, outcome


def created_event(oid: str, patient: str, step: str, deadline: float, deps: list[str], chain: str, pos: int, n: int, conf: float, t0: float, provenance: str = "extracted_rule") -> dict:
    return {"event_id": f"{oid}:created", "patient_id": patient, "encounter_id": None, "obligation_id": oid, "event_time_utc": iso(t0), "ingest_time_utc": iso(t0 + 5),
            "event_type": "obligation_created", "provenance": provenance, "is_simulated": True, "split": "test", "source_dataset": "scenario_lab", "delivery_attempt": 1,
            "payload": {"category": STEP_SPEC[step][0], "step_type": step, "description": step, "responsible_role": STEP_SPEC[step][1], "due_time_utc": iso(deadline), "depends_on": deps,
                        "extraction_confidence": conf, "label_tier": "silver", "chain_id": chain, "chain_position": pos, "chain_length": n, "source_dataset": "scenario_lab"}}


def match_extracted(extracted: list[ExtractedObligation], tobs: list[dict], thr: float = 0.4) -> dict[int, int]:
    """true index -> extracted index (greedy by span IoU, same type not required: the engine monitors what was extracted)."""
    cands = []
    for i, e in enumerate(extracted):
        for t in tobs:
            a0, a1, b0, b1 = e.evidence_span.start, e.evidence_span.end, t["start"], t["end"]
            inter = max(0, min(a1, b1) - max(a0, b0))
            iou = inter / max(a1 - a0 + b1 - b0 - inter, 1) if a0 >= 0 else 0.0
            if iou >= thr:
                cands.append((iou, i, t["idx"]))
    used_i, used_t, out = set(), set(), {}
    for iou, i, j in sorted(cands, reverse=True):
        if i not in used_i and j not in used_t:
            used_i.add(i)
            used_t.add(j)
            out[j] = i
    return out


def build_monitoring_stream(doc: dict, extracted: list[ExtractedObligation], tobs: list[dict], events: list[dict], patient: str, t0: float) -> tuple[list[dict], dict]:
    """Obligations the system monitors = the EXTRACTED ones. True events are relabelled onto matched extracted obligations; events of true
    obligations that were not extracted are dropped (nothing monitors them); spurious extracted obligations simply get no events."""
    items = [{"task_type": e.task_type, "due_hours": interval_hours(e.due_interval), "deps": [int(d.rsplit("ob", 1)[1]) for d in e.dependencies if "ob" in d]} for e in extracted]
    dl = impute_deadlines(items, t0)
    mp = match_extracted(extracted, tobs)
    inv = {i: j for j, i in mp.items()}
    out = []
    for i, e in enumerate(extracted):
        step = engine_step(e.task_type, e.evidence_span.text, bool(items[i]["deps"]))
        deps = [f"{doc['doc_id']}:x{d}" for d in items[i]["deps"] if d < len(extracted)]
        out.append(created_event(f"{doc['doc_id']}:x{i}", patient, step, dl[i][0], deps, doc["doc_id"], i, len(extracted), e.confidence, t0))
    rename = {tobs[j]["id"]: f"{doc['doc_id']}:x{i}" for i, j in inv.items()}
    for ev in events:
        if ev["obligation_id"] in rename:
            out.append({**ev, "obligation_id": rename[ev["obligation_id"]]})
    out.sort(key=lambda e: (e["ingest_time_utc"], e["event_id"]))
    return out, {"matched": mp, "n_extracted": len(extracted), "n_true": len(tobs)}


# ------------------------------------------------------------------------------------------------ the live demonstration scenario
DEMO_TEXT = ("Patient discharged after pneumonia admission. Repeat the CBC within 48 hours. The treating physician must review the result. "
             "Contact the patient if the result is abnormal. Arrange a follow-up appointment in 2 weeks.")


def demo_obligations(patient: str = "demo:patient-001") -> list[dict]:
    t0 = T0
    L = lambda i: f"demo:ob{i}"  # noqa: E731
    return [
        created_event(L(0), patient, "lab_collected", t0 + 48 * H, [], "demo:chain", 0, 4, 0.9, t0),
        created_event(L(1), patient, "result_reviewed", t0 + (48 + 72) * H, [L(0)], "demo:chain", 1, 4, 0.8, t0),
        created_event(L(2), patient, "patient_notified", t0 + (48 + 144) * H, [L(1)], "demo:chain", 2, 4, 0.7, t0),
        created_event(L(3), patient, "followup_visit_completed", t0 + 14 * 24 * H, [L(1)], "demo:chain", 3, 4, 0.85, t0)]


def demo_events(variant: str = "delayed_ack", patient: str = "demo:patient-001") -> list[dict]:
    """Hand-scripted episode. variant: delayed_ack (ack happened on time but documented late) | missing_ack (never documented) | happy_path."""
    t0 = T0
    L = lambda i: f"demo:ob{i}"  # noqa: E731
    ev: list[dict] = list(demo_obligations(patient))
    n = [0]

    def add(oid, etype, t_event, t_ingest=None, ver="verified", details=None):
        n[0] += 1
        ev.append({"event_id": f"demo:e{n[0]:02d}", "patient_id": patient, "encounter_id": None, "obligation_id": oid, "event_time_utc": iso(t_event), "ingest_time_utc": iso(t_ingest or t_event + 30),
                   "event_type": etype, "payload": {"status": "ok", "evidence_source": "demo_script", "verification": ver, "details": details or {}, "source_event_id": None},
                   "provenance": "simulated", "is_simulated": True, "split": "demo", "source_dataset": "scenario_lab", "delivery_attempt": 1})
    add(L(0), "reminder_sent", t0 + 24 * H)
    add(L(0), "lab_collected", t0 + 30 * H)                                  # test completes (on time)
    add(L(0), "result_posted", t0 + 36 * H)                                  # result available
    add(L(1), "result_opened", t0 + 42 * H)                                  # clinician opened the result ...  but no acknowledgment is documented yet
    if variant == "happy_path":
        add(L(1), "result_acknowledged", t0 + 50 * H)
    elif variant == "delayed_ack":
        add(L(1), "result_acknowledged", t0 + 100 * H, t_ingest=t0 + 150 * H)   # acknowledged BEFORE the deadline, documented 50 h later
    add(L(1), "task_status_snapshot", t0 + 120 * H + H, details={"status": "open"}) if variant != "happy_path" else None
    ev = [e for e in ev if e is not None]
    if variant in ("delayed_ack", "happy_path"):
        add(L(2), "patient_notified", t0 + (160 if variant == "delayed_ack" else 70) * H)
    add(L(3), "appointment_scheduled", t0 + 180 * H)
    add(L(3), "visit_attended", t0 + 300 * H) if variant != "missing_ack" else add(L(3), "visit_no_show", t0 + 300 * H)
    ev.sort(key=lambda e: (e["ingest_time_utc"], e["event_id"]))
    return ev
