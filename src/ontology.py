"""Shared vocabulary: task steps, completion-event semantics, workflow states, graph node/edge types and feature groups.

Everything that more than one component needs to agree on lives here, so extraction, graph, risk model and alert rules
cannot silently diverge.
"""
from __future__ import annotations

from enum import Enum

CATEGORIES = ["appointment_referral", "laboratory_test", "imaging", "procedure", "medication_action",
              "patient_communication", "followup_review", "other_action"]
ALL_LABELS = CATEGORIES + ["no_action"]
ROLES = ["patient", "scheduler", "laboratory", "radiology", "ordering_clinician", "nurse", "specialist", "pharmacy"]

# step_type -> (category, responsible_role)
STEP_SPEC = {
    "lab_collected": ("laboratory_test", "patient"), "result_available": ("laboratory_test", "laboratory"),
    "result_reviewed": ("followup_review", "ordering_clinician"), "patient_notified": ("patient_communication", "nurse"),
    "imaging_scheduled": ("appointment_referral", "scheduler"), "imaging_performed": ("imaging", "radiology"),
    "report_available": ("imaging", "radiology"), "report_reviewed": ("followup_review", "ordering_clinician"),
    "appointment_scheduled": ("appointment_referral", "scheduler"), "appointment_attended": ("appointment_referral", "patient"),
    "consult_report_received": ("followup_review", "specialist"),
    "followup_scheduled": ("appointment_referral", "scheduler"), "followup_visit_completed": ("followup_review", "patient"),
    "procedure_scheduled": ("appointment_referral", "scheduler"), "procedure_performed": ("procedure", "specialist"),
    "post_procedure_review": ("followup_review", "ordering_clinician"),
    "medication_dispensed": ("medication_action", "pharmacy"), "medication_reviewed": ("medication_action", "ordering_clinician"),
    "patient_contacted": ("patient_communication", "nurse"), "action_completed": ("other_action", "ordering_clinician"),
}
# The single event type whose (verified) presence completes a given step.
COMPLETION_EVENT = {
    "lab_collected": "lab_collected", "result_available": "result_posted", "result_reviewed": "result_acknowledged",
    "patient_notified": "patient_notified", "imaging_scheduled": "appointment_scheduled", "imaging_performed": "imaging_performed",
    "report_available": "report_posted", "report_reviewed": "result_acknowledged", "appointment_scheduled": "appointment_scheduled",
    "appointment_attended": "visit_attended", "consult_report_received": "consult_report_received",
    "followup_scheduled": "appointment_scheduled", "followup_visit_completed": "visit_attended",
    "procedure_scheduled": "appointment_scheduled", "procedure_performed": "procedure_performed",
    "post_procedure_review": "result_acknowledged", "medication_dispensed": "medication_dispensed",
    "medication_reviewed": "medication_acknowledged", "patient_contacted": "patient_notified", "action_completed": "action_documented",
}
REVIEW_STEPS = {"result_reviewed", "report_reviewed", "post_procedure_review", "medication_reviewed"}
OPEN_EVENTS = {"result_opened", "medication_opened"}
# Events that positively evidence NON-completion (confirm an open/missed task).
MISS_EVENTS = {"visit_no_show", "appointment_cancelled"}
SNAPSHOT_EVENT = "task_status_snapshot"
CONFLICT_EVENT = "conflicting_record"


class State(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed_with_evidence"
    OVERDUE = "overdue_without_verified_completion"
    UNVERIFIED = "unverified_insufficient_evidence"
    CONTRADICTORY = "contradictory_evidence"
    CENSORED = "censored_not_yet_assessable"


# Graph vocabulary (task specification). Event nodes get the artefact type their event_type implies, else EvidenceEvent.
NODE_TYPES = ["Patient", "Encounter", "ClinicalNote", "Obligation", "Task", "Result", "Review", "Acknowledgment", "Notification",
              "EvidenceEvent", "ResponsibleRole"]
EDGE_TYPES = ["EXTRACTED_FROM", "DEPENDS_ON", "ASSIGNED_TO", "EVIDENCED_BY", "COMPLETED_BY", "REQUIRES_REVIEW", "CONTRADICTS", "BELONGS_TO"]
EVENT_NODE_TYPE = {
    **{e: "Task" for e in ("lab_collected", "imaging_performed", "procedure_performed", "medication_dispensed", "visit_attended",
                           "action_documented", "appointment_scheduled", "appointment_rescheduled")},
    **{e: "Result" for e in ("result_posted", "report_posted", "consult_report_received")},
    **{e: "Review" for e in ("result_opened", "medication_opened")},
    **{e: "Acknowledgment" for e in ("result_acknowledged", "medication_acknowledged")},
    **{e: "Notification" for e in ("patient_notified", "contact_attempt", "reminder_sent")},
}

# Feature groups (see followup_risk/feature_spec.json of the dataset pipeline for the base definitions).
FEATURE_GROUPS = {
    "task": ["category", "step_type", "responsible_role", "origin_encounter_type", "origin_department", "window_hours", "hours_to_deadline",
             "elapsed_fraction", "deadline_dow", "deadline_is_weekend", "extraction_confidence"],
    "own_events": ["own_n_events", "own_hours_since_last_event", "own_no_events", "own_n_reminders", "own_n_contact_attempts", "own_viewed",
                   "own_has_cancellation"],
    "workload_history": ["prior_n_due", "prior_n_completed_ontime", "prior_completion_rate", "prior_noshow_count", "n_concurrent_open",
                         "n_concurrent_open_same_role", "n_encounters_since_origin", "n_department_transitions_since_origin"],
    "graph_dependency": ["chain_position", "chain_length", "n_dependencies", "n_incomplete_dependencies", "n_incomplete_ancestors", "parent_state",
                         "hours_since_parent_completed", "role_handoffs"],
    "graph_temporal": ["n_events_so_far", "hours_since_last_event", "no_events_yet", "n_reminders", "n_contact_attempts", "has_scheduled_appointment",
                       "appointment_minus_deadline_hours", "n_reschedules", "has_cancellation"],
    "evidence": ["ack_state", "n_verified_events", "n_unverified_events", "has_conflicting_record", "evidence_uncertainty",
                 "has_documented_source_evidence"],
}
CATEGORICAL_FEATURES = ["category", "step_type", "responsible_role", "origin_encounter_type", "origin_department", "parent_state", "ack_state"]
# Features dropped by the "temporal" ablation: everything about recency/timing of events and of the deadline window.
TEMPORAL_FEATURES = ["hours_to_deadline", "elapsed_fraction", "own_hours_since_last_event", "hours_since_last_event",
                     "hours_since_parent_completed", "appointment_minus_deadline_hours", "window_hours"]
SYSTEM_FEATURES = {  # experiment 3 systems
    "B_independent_ml": ["task", "own_events", "workload_history"],
    "C_graph_ml": ["task", "own_events", "workload_history", "graph_dependency", "graph_temporal", "evidence"],
}


def features_of(groups: list[str], drop: list[str] | None = None) -> list[str]:
    cols = [c for g in groups for c in FEATURE_GROUPS[g]]
    return [c for c in cols if c not in set(drop or [])]
