"""Deterministic alert rules, evidence-grounded explanations and intervention-priority scoring.

Separation of concerns (task specification):
  * workflow STATE comes only from the evidence graph (deterministic rules R1-R7);
  * RISK is a calibrated model probability that a still-open obligation will be unresolved at its deadline (a forecast);
  * ALERTS combine the two with fixed, auditable rules. An obligation that is already overdue is detected by deadline logic, not by the model.
Every alert carries the triggering event, state, evidence ids, missing information, responsible role, a suggested VERIFICATION action and an
uncertainty statement. Recommendations are for qualified human review only; nothing here changes orders, prescriptions or diagnoses.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .evidence_graph import StateDecision, TemporalEvidenceGraph, iso
from .ontology import REVIEW_STEPS, State

STEP_LABEL = {
    "lab_collected": "laboratory sample collection", "result_available": "laboratory result availability", "result_reviewed": "result review",
    "patient_notified": "patient notification", "imaging_scheduled": "imaging scheduling", "imaging_performed": "imaging study",
    "report_available": "imaging report availability", "report_reviewed": "imaging report review", "appointment_scheduled": "appointment scheduling",
    "appointment_attended": "appointment attendance", "consult_report_received": "consult report receipt", "followup_scheduled": "follow-up scheduling",
    "followup_visit_completed": "follow-up visit", "procedure_scheduled": "procedure scheduling", "procedure_performed": "procedure",
    "post_procedure_review": "post-procedure review", "medication_dispensed": "medication dispensing", "medication_reviewed": "medication review",
    "patient_contacted": "patient contact", "action_completed": "action completion"}
ROLE_LABEL = {"ordering_clinician": "ordering clinician", "nurse": "nurse", "laboratory": "laboratory", "radiology": "radiology", "scheduler": "scheduler",
              "pharmacy": "pharmacy", "specialist": "specialist", "patient": "patient"}
# Simulated workflow-definition importance weights (NOT clinical severity; not derived from diagnoses).
IMPORTANCE = {"result_reviewed": 1.0, "report_reviewed": 1.0, "post_procedure_review": 1.0, "patient_notified": 0.9, "medication_reviewed": 0.9, "appointment_attended": 0.8,
              "followup_visit_completed": 0.8, "procedure_performed": 0.8, "imaging_performed": 0.7, "lab_collected": 0.7}
VERIFY = {
    "result_reviewed": "Ask the {role} to confirm whether the result was reviewed and to document the acknowledgment.",
    "report_reviewed": "Ask the {role} to confirm review of the imaging report and document it.",
    "post_procedure_review": "Ask the {role} to confirm post-procedure review and document it.",
    "medication_reviewed": "Ask the {role} to confirm the medication review and document acknowledgment.",
    "patient_notified": "Confirm with the {role} whether the patient was contacted; document the contact (or the failed attempts).",
    "lab_collected": "Check with the patient/laboratory whether the specimen was collected; contact the patient if not.",
    "imaging_performed": "Check with radiology and the patient whether the study was performed.",
    "appointment_attended": "Verify with scheduling/registration whether the visit occurred; contact the patient if it did not.",
    "followup_visit_completed": "Verify whether the follow-up visit occurred; arrange a new visit if it did not.",
}


@dataclass
class Alert:
    alert_id: str
    oid: str
    patient_id: str
    rule: str
    severity: str
    t_ingest: float
    state: str
    risk: float | None
    priority: float | None
    trigger_event_id: str | None
    evidence_ids: list
    missing_information: list
    responsible_role: str | None
    suggested_verification: str
    uncertainty: str
    explanation: str
    status: str = "open"
    resolved_at: float | None = None
    resolution: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["t_ingest_utc"] = iso(self.t_ingest)
        return d


class PriorityScorer:
    """Intervention-priority score. Methods are compared experimentally (see experiments.run_exp_priority); weights for 'weighted' are tuned on validation."""

    FEATURES = ["risk", "urgency", "unresolved_dependencies", "evidence_incomplete", "importance"]

    def __init__(self, method: str = "weighted", weights: dict | None = None, urgency_scale_h: float = 48.0):
        self.method = method
        self.weights = weights or {"risk": 0.55, "urgency": 0.2, "unresolved_dependencies": 0.1, "evidence_incomplete": 0.1, "importance": 0.05}
        self.scale = urgency_scale_h
        self.model = None

    def components(self, risk: float, feats: dict) -> dict:
        h = max(feats["hours_to_deadline"], 0.0)
        return {"risk": float(risk), "urgency": 1.0 / (1.0 + h / self.scale),
                "unresolved_dependencies": min(feats["n_incomplete_ancestors"], 3) / 3.0,
                "evidence_incomplete": min(1.0, 0.5 * (feats["own_no_events"]) + 0.5 * (feats["n_verified_events"] == 0) + 0.3 * feats["has_conflicting_record"]),
                "importance": IMPORTANCE.get(feats["step_type"], 0.6)}

    def score(self, risk: float, feats: dict) -> float:
        c = self.components(risk, feats)
        if self.method == "risk_only":
            return c["risk"]
        if self.method == "deadline_only":
            return c["urgency"]
        if self.method == "risk_x_urgency":
            return c["risk"] * c["urgency"] ** 0.5
        if self.method == "learned" and self.model is not None:
            return float(self.model.predict_proba(np.array([[c[k] for k in self.FEATURES]]))[0, 1])
        return float(sum(self.weights[k] * c[k] for k in self.FEATURES))

    def fit_learned(self, comps: np.ndarray, y: np.ndarray):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(C=1.0, max_iter=1000).fit(comps, y)
        self.method = "learned"
        return self


class AlertEngine:
    def __init__(self, cfg: dict, scorer: PriorityScorer | None = None):
        a = cfg["alerts"]
        self.risk_thr = a["risk_threshold"]
        self.risk_window_h = a["hours_to_deadline_for_risk_alert"]
        self.unack_h = a["unacknowledged_hours"]
        self.scorer = scorer or PriorityScorer()
        self.open: dict[tuple[str, str], Alert] = {}
        self.n = 0

    # -- helpers
    def _evidence_lines(self, g: TemporalEvidenceGraph, oid: str, ids: list[str], now: float) -> list[str]:
        out = []
        for e in g.ob_events.get(oid, ()):
            if e.eid in ids:
                out.append(f"{e.etype} @ {iso(e.t_event)} (id {e.eid}, {e.verification}, {e.provenance})")
        return out

    def _new(self, g, oid, rule, severity, now, dec: StateDecision, risk, feats, trigger, msg, missing, verify_key=None, uncertainty="") -> Alert:
        ob = g.obs[oid]
        self.n += 1
        role = ROLE_LABEL.get(ob.role, ob.role)
        verify = VERIFY.get(verify_key or ob.step_type, "Verify completion evidence with the responsible role and document the outcome.").format(role=role)
        pr = self.scorer.score(risk, feats) if (risk is not None and feats is not None) else None
        ev_lines = self._evidence_lines(g, oid, dec.evidence_ids, now)
        text = (f"{msg} Task: {STEP_LABEL.get(ob.step_type, ob.step_type)} (obligation {oid}, due {iso(ob.deadline)}). State: {dec.state.value}. "
                f"Evidence used: {'; '.join(ev_lines) if ev_lines else 'none documented'}. Missing: {', '.join(missing) if missing else 'nothing identified'}. "
                f"Responsible role: {role}. Suggested verification (for qualified human review): {verify} "
                f"{uncertainty}").strip()
        return Alert(f"alert-{self.n:07d}", oid, ob.patient_id, rule, severity, now, dec.state.value, risk, pr, trigger, dec.evidence_ids, missing, ob.role, verify, uncertainty, text)

    def _resolve(self, key, now, why):
        a = self.open.pop(key, None)
        if a:
            a.status, a.resolved_at, a.resolution = "resolved", now, why
            return a
        return None

    # -- main entry: evaluate rules for one obligation after its state/risk were refreshed
    def evaluate(self, g: TemporalEvidenceGraph, oid: str, now: float, dec: StateDecision, prev_state: State | None, risk: float | None, feats: dict | None,
                 trigger: str | None) -> tuple[list[Alert], list[Alert]]:
        """Returns (new_alerts, resolved_alerts)."""
        ob = g.obs[oid]
        new, resolved = [], []
        st = dec.state
        if st == State.COMPLETED:
            for rule in list(self.open):
                if rule[0] == oid:
                    r = self._resolve(rule, now, "completion evidence received" + (" (late)" if dec.late else ""))
                    if r:
                        resolved.append(r)
            return new, resolved
        if st in (State.OVERDUE, State.UNVERIFIED, State.CONTRADICTORY):
            rule = {State.OVERDUE: "DEADLINE_MISSED_CONFIRMED", State.UNVERIFIED: "COMPLETION_UNVERIFIED", State.CONTRADICTORY: "CONFLICTING_EVIDENCE"}[st]
            for other in [k for k in self.open if k[0] == oid and k[1] != rule and k[1] in ("DEADLINE_MISSED_CONFIRMED", "COMPLETION_UNVERIFIED", "CONFLICTING_EVIDENCE", "HIGH_RISK", "RESULT_NOT_ACKNOWLEDGED", "BLOCKED_BY_UPSTREAM")]:
                r = self._resolve(other, now, f"superseded by {rule}")
                if r:
                    resolved.append(r)
            if (oid, rule) not in self.open:
                msgs = {"DEADLINE_MISSED_CONFIRMED": "Deadline passed and the task tracker/registration confirms the task is still open.",
                        "COMPLETION_UNVERIFIED": "Deadline passed with no completion evidence and no status evidence either way; this is NOT confirmed non-completion.",
                        "CONFLICTING_EVIDENCE": "A completion record and a secondary record disagree."}
                missing = {"DEADLINE_MISSED_CONFIRMED": ["verified completion evidence"], "COMPLETION_UNVERIFIED": ["completion evidence", "status confirmation"],
                           "CONFLICTING_EVIDENCE": ["reconciliation of conflicting records"]}[rule]
                unc = {"COMPLETION_UNVERIFIED": "Uncertainty: documentation may simply be missing or delayed.",
                       "CONFLICTING_EVIDENCE": "Uncertainty: which record is correct is unknown.", "DEADLINE_MISSED_CONFIRMED": "Uncertainty: the finding is evidence-based, but late documentation could still arrive."}[rule]
                a = self._new(g, oid, rule, "high" if st == State.OVERDUE else "medium", now, dec, risk, feats, trigger, msgs[rule], missing, uncertainty=unc)
                self.open[(oid, rule)] = a
                new.append(a)
            return new, resolved
        # PENDING / CENSORED -> forward-looking rules
        if feats is None:
            return new, resolved
        # R-UNACK: a review obligation whose prerequisite result exists but nobody has acknowledged it
        if ob.step_type in REVIEW_STEPS and feats["ack_state"] in ("not_viewed", "viewed_unacknowledged") and feats["hours_since_parent_completed"] == feats["hours_since_parent_completed"] \
                and feats["hours_since_parent_completed"] >= self.unack_h and st == State.PENDING:
            if (oid, "RESULT_NOT_ACKNOWLEDGED") not in self.open:
                msg = ("Result received" if feats["ack_state"] == "not_viewed" else "Result opened but not acknowledged") + \
                      f" {feats['hours_since_parent_completed']:.0f} h ago. No documented acknowledgment found. This review obligation requires verification before the dependent task can be considered completed."
                a = self._new(g, oid, "RESULT_NOT_ACKNOWLEDGED", "medium", now, dec, risk, feats, trigger, msg, ["acknowledgment of the result"],
                              uncertainty="Uncertainty: acknowledgment may have occurred without documentation.")
                self.open[(oid, "RESULT_NOT_ACKNOWLEDGED")] = a
                new.append(a)
        else:
            r = self._resolve((oid, "RESULT_NOT_ACKNOWLEDGED"), now, "acknowledgment evidence arrived or condition cleared")
            if r:
                resolved.append(r)
        # R-BLOCKED: pending task whose upstream obligations are incomplete close to its deadline
        if st == State.PENDING and feats["n_incomplete_ancestors"] > 0 and feats["hours_to_deadline"] <= self.risk_window_h:
            if (oid, "BLOCKED_BY_UPSTREAM") not in self.open:
                up = dec.detail.get("upstream_incomplete", [])
                a = self._new(g, oid, "BLOCKED_BY_UPSTREAM", "medium", now, dec, risk, feats, trigger,
                              f"{feats['n_incomplete_ancestors']} upstream obligation(s) have no completion evidence yet, so this task cannot be completed in sequence.",
                              [f"completion evidence for {u}" for u in up[:3]] or ["upstream completion evidence"], uncertainty="Uncertainty: upstream steps may be undocumented rather than missing.")
                self.open[(oid, "BLOCKED_BY_UPSTREAM")] = a
                new.append(a)
        else:
            r = self._resolve((oid, "BLOCKED_BY_UPSTREAM"), now, "upstream evidence arrived or deadline passed")
            if r:
                resolved.append(r)
        # R-RISK: model forecast above threshold inside the warning window
        if st == State.PENDING and risk is not None and risk >= self.risk_thr and feats["hours_to_deadline"] <= self.risk_window_h:
            if (oid, "HIGH_RISK") not in self.open:
                a = self._new(g, oid, "HIGH_RISK", "medium" if risk < 0.9 else "high", now, dec, risk, feats, trigger,
                              f"Predicted probability of being unresolved at the deadline is {risk:.0%} (calibrated model forecast, not an observed failure).",
                              ["completion evidence"] + (["upstream completion"] if feats["n_incomplete_ancestors"] else []),
                              uncertainty=f"Uncertainty: model forecast (calibrated probability {risk:.2f}); the forecast can be wrong.")
                self.open[(oid, "HIGH_RISK")] = a
                new.append(a)
        elif (oid, "HIGH_RISK") in self.open and (risk is None or risk < self.risk_thr * 0.8):
            r = self._resolve((oid, "HIGH_RISK"), now, "risk fell below threshold")
            if r:
                resolved.append(r)
        return new, resolved


def verbalize_alert(backend, alert: Alert, max_tokens: int = 140) -> dict:
    """OPTIONAL GenAI paraphrase of an alert. The structured, deterministic explanation stays the record of truth: the paraphrase is accepted only if it
    (a) mentions the obligation id and (b) introduces no id/time/number that is absent from the structured explanation; otherwise the deterministic text is used."""
    import re
    msgs = [{"role": "system", "content": "Rewrite the clinical follow-up alert in at most two plain sentences for a clinician. Use ONLY facts present in the text. Keep every id, date and number exactly as written. "
                                          "Do not add medical advice."}, {"role": "user", "content": alert.explanation}]
    r = backend.generate_text(msgs, max_tokens=max_tokens)
    txt = r.text.strip()
    tokens = set(re.findall(r"[A-Za-z0-9:_\-]*\d[A-Za-z0-9:_\-.]*", txt))
    foreign = sorted(t for t in tokens if t.strip(".,") not in alert.explanation)
    grounded = alert.oid in txt and not foreign
    return {"text": txt if grounded else alert.explanation, "llm_text": txt, "grounded": grounded, "foreign_tokens": foreign, "latency_s": r.latency_s, "tokens": r.completion_tokens}
