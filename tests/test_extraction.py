"""Deadline parsing, GenAI JSON validation, unsupported-claim rejection, evidence grounding, dependency ordering."""
import json

import pytest
from pydantic import ValidationError

from src.clinical_extractor import (ExtractedObligation, EvidenceSpan, RuleExtractor, ValidatedExtractor, LLMExtractor, dependencies_valid, find_span, interval_hours, parse_deadline,
                                    unsupported_fields)
from src.llm_backend import LLMResult


@pytest.mark.parametrize("text,iso,hours", [("within 48 hours", "PT48H", 48), ("in two days", "P2D", 48), ("in 1-2 weeks", "P2W", 336), ("in a couple of days", "P2D", 48),
                                            ("next week", "P1W", 168), ("tomorrow", "P1D", 24), ("in 3 months", "P3M", 2160), ("immediately", "PT0H", 0)])
def test_valid_deadline_parsing(text, iso, hours):
    d = parse_deadline(f"Repeat the test {text}.")
    assert d.iso == iso and d.hours == hours


@pytest.mark.parametrize("text", ["Continue metformin for ten days.", "Take twice daily.", "Follow up at the next visit.", "Titrate every 3 days until glucose is below 130."])
def test_durations_and_frequencies_are_not_deadlines(text):
    assert parse_deadline(text) is None


def test_iso_validation_rejects_bad_interval():
    with pytest.raises(ValidationError):
        ExtractedObligation(obligation_id="a", task_type="laboratory_test", description="x test", due_interval="two_days", evidence_span=EvidenceSpan(start=0, end=1, text="x"), source_note_id="n", confidence=.5)
    with pytest.raises(ValidationError):
        ExtractedObligation(obligation_id="a", task_type="telepathy", description="x test", evidence_span=EvidenceSpan(start=0, end=1, text="x"), source_note_id="n", confidence=.5)
    with pytest.raises(ValidationError):
        ExtractedObligation(obligation_id="a", task_type="imaging", description="x test", responsible_role="wizard", evidence_span=EvidenceSpan(start=0, end=1, text="x"), source_note_id="n", confidence=.5)


class FakeBackend:
    name, size_bytes, n_threads, load_s = "fake", 0, 1, 0.0

    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), 0

    def generate_json(self, messages, schema, max_tokens=500, temperature=0.0):
        txt = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return LLMResult(txt if isinstance(txt, str) else json.dumps(txt), 10, 20, 0.01)


TEXT = "Repeat the CBC in 48 hours. The nurse should call the patient if hemoglobin is low."


def _item(**kw):
    base = {"task_type": "laboratory_test", "description": "Repeat the CBC", "due_interval": None, "responsible_role": None, "depends_on": [], "conditional": None, "evidence_span": "Repeat the CBC in 48 hours"}
    base.update(kw)
    return base


def test_unsupported_deadline_role_and_conditional_are_removed_by_validation():
    out = {"obligations": [_item(due_interval="P7D", responsible_role="pharmacy", conditional="if the moon is full")]}
    res = ValidatedExtractor(FakeBackend([out]), max_retries=0).extract(TEXT, "n")
    ob = res.obligations[0]
    assert ob.due_interval == "PT48H"                                  # the unsupported P7D is replaced by the deadline actually stated in the evidence span
    assert ob.responsible_role is None and ob.conditional_requirements is None
    assert {r["field"] for r in res.repairs} >= {"due_interval", "responsible_role", "conditional_requirements"}
    assert next(r for r in res.repairs if r["field"] == "due_interval")["action"] == "replaced"
    raw = LLMExtractor(FakeBackend([out])).extract(TEXT, "n").obligations[0]
    assert set(unsupported_fields(raw, TEXT)) == {"due_interval", "responsible_role", "conditional_requirements"}      # raw GenAI output carries the unsupported claims


def test_fabricated_evidence_span_is_rejected_not_repaired():
    res = ValidatedExtractor(FakeBackend([{"obligations": [_item(evidence_span="Order a colonoscopy within one year")]}]), max_retries=0).extract(TEXT, "n")
    assert res.obligations == [] and res.rejected[0]["reason"] == "evidence_span_not_in_source"


def test_near_verbatim_span_is_snapped_to_source():
    res = ValidatedExtractor(FakeBackend([{"obligations": [_item(evidence_span="repeat the cbc in 48 hrs")]}]), max_retries=0).extract(TEXT, "n")
    assert res.obligations and TEXT[res.obligations[0].evidence_span.start:res.obligations[0].evidence_span.end].startswith("Repeat the CBC")


def test_invalid_json_is_survived_and_flagged():
    res = ValidatedExtractor(FakeBackend(["not json at all", "still not json"]), max_retries=1).extract(TEXT, "n")
    assert res.schema_valid is False and res.obligations == [] and res.n_retries >= 1


def test_grounded_deadline_is_kept():
    res = ValidatedExtractor(FakeBackend([{"obligations": [_item(due_interval="PT48H")]}]), max_retries=0).extract(TEXT, "n")
    assert res.obligations[0].due_interval == "PT48H" and not res.repairs


def test_invalid_dependencies_are_dropped_and_detected():
    out = {"obligations": [_item(depends_on=[0]), _item(evidence_span="The nurse should call the patient if hemoglobin is low", task_type="patient_communication", depends_on=[1, 5])]}
    res = ValidatedExtractor(FakeBackend([out]), max_retries=0).extract(TEXT, "n")
    assert dependencies_valid(res.obligations) == []                    # self / forward / unknown references removed
    raw = LLMExtractor(FakeBackend([out])).extract(TEXT, "n").obligations
    assert dependencies_valid(raw)                                       # raw output has invalid references


def test_rule_extractor_orders_review_after_test_and_notification_after_review():
    r = RuleExtractor().extract("Repeat the blood test within 48 hours, review the results, and contact the patient if further action is necessary.", "d")
    types = [o.task_type for o in r.obligations]
    assert types == ["laboratory_test", "followup_review", "patient_communication"]
    assert r.obligations[1].dependencies == ["d:ob0"] and r.obligations[2].dependencies == ["d:ob1"]
    assert r.obligations[0].due_interval == "PT48H" and r.obligations[1].due_interval is None            # deadline NOT propagated to unstated steps
    assert dependencies_valid(r.obligations) == []


def test_find_span_exact_and_fuzzy():
    assert find_span(TEXT, "the cbc", fuzzy=1.0)[2] is True
    assert find_span(TEXT, "Repeat CBC in 48 hrs", fuzzy=0.8) is not None
    assert find_span(TEXT, "completely unrelated sentence about surgery", fuzzy=0.85) is None
