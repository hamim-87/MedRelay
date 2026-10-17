"""Temporal evidence graph semantics: dependency ordering, cycles, duplicates, delayed evidence, completion, missing acknowledgment, contradictions, feature cutoff."""
import pytest

from src.evidence_graph import TemporalEvidenceGraph, to_epoch
from src.ontology import State
from tests.conftest import mk_ev, mk_ob

H = 3600.0


def build(T0, with_ack=False):
    g = TemporalEvidenceGraph(grace_hours=12, build_nx=True)
    g.ingest(mk_ob("o0", "lab_collected", T0, T0 + 48 * H, [], pos=0, n=3))
    g.ingest(mk_ob("o1", "result_reviewed", T0, T0 + 120 * H, ["o0"], pos=1, n=3))
    g.ingest(mk_ob("o2", "patient_notified", T0, T0 + 192 * H, ["o1"], pos=2, n=3))
    return g


def test_completed_test_does_not_complete_the_chain(T0):
    g = build(T0)
    g.ingest(mk_ev("e1", "o0", "lab_collected", T0 + 10 * H))
    now = T0 + 20 * H
    assert g.derive_state("o0", now).state == State.COMPLETED
    assert g.derive_state("o1", now).state == State.PENDING and g.derive_state("o2", now).state == State.PENDING


def test_result_available_without_acknowledgment_is_not_review_completion(T0):
    g = build(T0)
    g.ingest(mk_ev("e1", "o0", "lab_collected", T0 + 10 * H))
    g.ingest(mk_ev("e2", "o0", "result_posted", T0 + 14 * H))
    g.ingest(mk_ev("e3", "o1", "result_opened", T0 + 20 * H))
    assert g.derive_state("o1", T0 + 30 * H).state == State.PENDING
    f = g.features("o1", T0 + 30 * H)
    assert f["ack_state"] == "viewed_unacknowledged" and f["parent_state"] == "completed"


def test_missing_documentation_is_not_confirmed_failure(T0):
    g = build(T0)                                            # no evidence of anything
    d = T0 + 48 * H
    assert g.derive_state("o0", d + 1 * H).state == State.CENSORED                       # inside the grace window
    assert g.derive_state("o0", d + 13 * H).state == State.UNVERIFIED                    # grace over, still no evidence either way (NOT 'overdue')
    g.ingest(mk_ev("s1", "o0", "task_status_snapshot", d + H, details={"status": "open"}))
    dec = g.derive_state("o0", d + 2 * H)
    assert dec.state == State.OVERDUE and dec.evidence_ids == ["s1"]                      # positive evidence of non-completion => confirmed overdue


def test_duplicates_are_ignored(T0):
    g = build(T0)
    r1 = g.ingest(mk_ev("e1", "o0", "lab_collected", T0 + 10 * H))
    r2 = g.ingest(mk_ev("e1", "o0", "lab_collected", T0 + 10 * H, t_ingest=T0 + 11 * H))
    assert not r1.duplicate and r2.duplicate and g.n_duplicates == 1 and len(g.ob_events["o0"]) == 1


def test_delayed_evidence_reconciles_state(T0):
    g = build(T0)
    d = T0 + 48 * H
    g.ingest(mk_ev("s1", "o0", "task_status_snapshot", d + H, details={"status": "open"}))
    assert g.derive_state("o0", d + 20 * H, live=True).state == State.OVERDUE
    g.ingest(mk_ev("late", "o0", "lab_collected", T0 + 40 * H, t_ingest=T0 + 90 * H))     # happened before the deadline, documented 42 h later
    assert g.derive_state("o0", d + 20 * H, live=True).state == State.OVERDUE            # at 68h the late record has not arrived (ingest 90h)
    assert g.derive_state("o0", d + 50 * H, live=True).state == State.COMPLETED           # after arrival: reconciled
    dec = g.derive_state("o0", d + 50 * H, live=True)
    assert not dec.late                                                                  # completed on time per EVENT time


def test_late_completion_is_flagged_late(T0):
    g = build(T0)
    g.ingest(mk_ev("e1", "o0", "lab_collected", T0 + 60 * H))
    dec = g.derive_state("o0", T0 + 70 * H)
    assert dec.state == State.COMPLETED and dec.late


def test_contradictory_evidence_state(T0):
    g = build(T0)
    g.ingest(mk_ev("e1", "o0", "lab_collected", T0 + 10 * H))
    g.ingest(mk_ev("c1", "o0", "conflicting_record", T0 + 12 * H, ver="unverified"))
    assert g.derive_state("o0", T0 + 13 * H).state == State.CONTRADICTORY
    assert any(d["etype"] == "CONTRADICTS" for _, _, d in g.G.edges(data=True))


def test_unverified_completion_evidence_is_not_completion(T0):
    g = build(T0)
    g.ingest(mk_ev("e1", "o0", "lab_collected", T0 + 10 * H, ver="unverified"))
    assert g.derive_state("o0", T0 + 20 * H).state == State.UNVERIFIED


def test_cycles_are_rejected(T0):
    g = TemporalEvidenceGraph()
    g.ingest(mk_ob("a", "lab_collected", T0, T0 + H, ["b"]))        # parent not yet known
    r = g.ingest(mk_ob("b", "result_reviewed", T0, T0 + 2 * H, ["a"]))
    assert r.rejected and g.rejected_cycles
    import networkx as nx
    assert nx.is_directed_acyclic_graph(g.dep)


def test_out_of_order_obligation_and_events_are_buffered(T0):
    g = TemporalEvidenceGraph()
    r = g.ingest(mk_ev("e1", "o9", "lab_collected", T0 + H))                         # event before its obligation
    assert "orphan_buffered" in r.notes and g.stats()["orphan_events_waiting"] == 1
    g.ingest(mk_ob("o9", "lab_collected", T0, T0 + 48 * H, [], n=1))
    assert g.stats()["orphan_events_waiting"] == 0 and g.derive_state("o9", T0 + 2 * H).state == State.COMPLETED
    g.ingest(mk_ob("child", "result_reviewed", T0, T0 + 100 * H, ["parent"], n=2))     # unresolved dependency until the parent arrives
    assert g.stats()["unresolved_dependencies"] == 1
    g.ingest(mk_ob("parent", "lab_collected", T0, T0 + 50 * H, [], n=2))
    assert g.stats()["unresolved_dependencies"] == 0 and "parent" in g.ancestors("child")


def test_event_time_and_ingest_time_are_kept_separately(T0):
    g = build(T0)
    g.ingest(mk_ev("e1", "o0", "lab_collected", T0 + 10 * H, t_ingest=T0 + 30 * H))
    e = g.ob_events["o0"][0]
    assert e.t_event == T0 + 10 * H and e.t_ingest == T0 + 30 * H


def test_features_use_only_events_up_to_the_cutoff(T0):
    g = build(T0)
    g.ingest(mk_ev("e1", "o0", "lab_collected", T0 + 10 * H))
    before = g.features("o1", T0 + 9 * H)
    g.ingest(mk_ev("e2", "o0", "result_posted", T0 + 30 * H))            # future event relative to the cutoff
    g.ingest(mk_ev("e3", "o1", "result_acknowledged", T0 + 50 * H))
    after = g.features("o1", T0 + 9 * H)
    import pandas as pd
    assert pd.Series(before).equals(pd.Series(after))                    # adding later events cannot change features at an earlier cutoff (NaN-safe comparison)
    assert before["n_events_so_far"] == 0 and before["parent_state"] == "pending"


def test_live_mode_excludes_not_yet_ingested_evidence(T0):
    g = build(T0)
    g.ingest(mk_ev("e1", "o0", "lab_collected", T0 + 10 * H, t_ingest=T0 + 40 * H))
    assert g.features("o0", T0 + 20 * H, live=False)["n_events_so_far"] == 1
    assert g.features("o0", T0 + 20 * H, live=True)["n_events_so_far"] == 0


def test_dependency_gap_is_flagged_when_parent_has_no_evidence(T0):
    g = build(T0)
    g.ingest(mk_ev("e3", "o1", "result_acknowledged", T0 + 50 * H))      # review documented but the test itself has no evidence
    dec = g.derive_state("o1", T0 + 60 * H)
    assert dec.state == State.COMPLETED and dec.detail.get("dependency_evidence_gap") == ["o0"]
