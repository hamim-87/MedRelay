"""Hybrid GenAI + ML + temporal-evidence-graph pipeline (integration layer).

  clinical text --extractor (rules / validated GenAI / hybrid)--> validated obligation records --> obligation_created events
  --> EventProcessor (graph, deterministic states, ML risk, alerts, audit) <-- streaming clinical events

The extractor only PROPOSES obligation records that already passed schema/grounding validation; workflow state is derived exclusively by the
deterministic evidence-graph rules from documented events. The LLM never sets a state."""
from __future__ import annotations

import pandas as pd

from .clinical_extractor import ExtractionResult
from .event_processor import EventProcessor
from .evidence_graph import iso, to_epoch
from .scenario_lab import T0, build_monitoring_stream, created_event, engine_step, impute_deadlines
from .clinical_extractor import interval_hours

H = 3600.0


class HybridPipeline:
    def __init__(self, cfg: dict, extractor, scorers: dict, db: str = ":memory:", primary: str = "C_graph_ml", **proc_kw):
        self.cfg, self.extractor = cfg, extractor
        self.proc = EventProcessor(cfg, db, scorers=scorers, primary=primary, rescore_batch=1, **proc_kw)
        self.extractions: dict[str, ExtractionResult] = {}

    def ingest_note(self, text: str, note_id: str, patient_id: str, t0: float = T0) -> tuple[ExtractionResult, list[dict]]:
        res = self.extractor.extract(text, note_id)
        self.extractions[note_id] = res
        doc = {"doc_id": note_id}
        created, _ = build_monitoring_stream(doc, res.obligations, [], [], patient_id, t0)
        prov = "extracted_rule" if res.extractor.startswith("A_") else "extracted_genai"
        for ev in created:
            ev["provenance"] = prov
            ev["payload"]["description"] = next((o.description for i, o in enumerate(res.obligations) if ev["obligation_id"].endswith(f":x{i}")), ev["payload"]["description"])
            self.proc.process(ev)
        self.proc.flush()
        return res, created

    def feed(self, events: list[dict]):
        for ev in events:
            self.proc.process(ev)
        self.proc.flush()

    def report(self, now: float | None = None) -> pd.DataFrame:
        g = self.proc.graph
        now = now or self.proc.clock
        rows = []
        for oid, ob in g.obs.items():
            dec = g.derive_state(oid, now, live=True)
            rk = (self.proc.last_risk.get(oid) or {}).get(self.proc.primary)
            rows.append({"obligation": oid, "step": ob.step_type, "category": ob.category, "deadline": iso(ob.deadline), "depends_on": ob.depends_on, "state": dec.state.value, "late": dec.late, "rule": dec.rule,
                         "risk": rk, "evidence": dec.evidence_ids})
        return pd.DataFrame(rows)


def script_events_for_created(created: list[dict], variant: str = "delayed_ack", t0: float = T0) -> list[dict]:
    """Narrative events for ANY extracted chain, keyed by workflow step: tests complete, results are posted/opened, the acknowledgment is
    (variant) documented late / never / on time, notification and visit follow. Same storyline as scenario_lab.demo_events."""
    n, out = [0], []

    def add(oid, patient, etype, t_event, t_ingest=None, details=None):
        n[0] += 1
        out.append({"event_id": f"script:e{n[0]:02d}", "patient_id": patient, "encounter_id": None, "obligation_id": oid, "event_time_utc": iso(t_event), "ingest_time_utc": iso(t_ingest or t_event + 30),
                    "event_type": etype, "payload": {"status": "ok", "evidence_source": "scripted_demo", "verification": "verified", "details": details or {}, "source_event_id": None},
                    "provenance": "simulated", "is_simulated": True, "split": "demo", "source_dataset": "scenario_lab", "delivery_attempt": 1})
    for c in created:
        step, oid, pid = c["payload"]["step_type"], c["obligation_id"], c["patient_id"]
        dl = to_epoch(c["payload"]["due_time_utc"])
        if step in ("lab_collected", "imaging_performed", "procedure_performed"):
            add(oid, pid, "reminder_sent", t0 + 24 * H)
            add(oid, pid, {"lab_collected": "lab_collected", "imaging_performed": "imaging_performed", "procedure_performed": "procedure_performed"}[step], t0 + 30 * H)
            add(oid, pid, "result_posted", t0 + 36 * H)
        elif step == "result_reviewed":
            add(oid, pid, "result_opened", t0 + 42 * H)
            if variant == "happy_path":
                add(oid, pid, "result_acknowledged", t0 + 50 * H)
            elif variant == "delayed_ack":
                add(oid, pid, "result_acknowledged", dl - 20 * H, t_ingest=dl + 30 * H)
            if variant != "happy_path":
                add(oid, pid, "task_status_snapshot", dl + H, details={"status": "open"})
        elif step in ("patient_notified", "patient_contacted") and variant != "missing_ack":
            add(oid, pid, "patient_notified", t0 + (70 if variant == "happy_path" else 160) * H)
        elif step in ("appointment_attended", "followup_visit_completed"):
            add(oid, pid, "visit_attended" if variant != "missing_ack" else "visit_no_show", t0 + 300 * H)
    out.sort(key=lambda e: (e["ingest_time_utc"], e["event_id"]))
    return out
