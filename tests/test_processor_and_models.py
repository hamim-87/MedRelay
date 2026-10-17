"""Event processor (idempotency, persistence, recovery, replay reproducibility, DB integrity), risk-model cutoff & serialization, alerts."""
import numpy as np
import pandas as pd
import pytest

from src.alert_engine import AlertEngine, PriorityScorer
from src.event_processor import EventProcessor, deadline_only_score
from src.evidence_graph import to_epoch
from src.ontology import FEATURE_GROUPS, State
from src.scenario_lab import demo_events
from tests.conftest import mk_ev, mk_ob

H = 3600.0


def run(cfg, events, db=":memory:", **kw):
    p = EventProcessor(cfg, db, scorers=kw.pop("scorers", {}), **kw)
    p.run(events)
    p.finalize()
    return p


def test_replay_is_reproducible_and_idempotent(cfg):
    ev = demo_events("delayed_ack")
    a, b = run(cfg, ev), run(cfg, ev + ev[:10])                      # second run also receives 10 duplicate deliveries
    assert a.state_counts() == b.state_counts()
    assert [t["new_state"] for t in a.transitions] == [t["new_state"] for t in b.transitions]
    assert b.stats["duplicates"] == 10


def test_delayed_acknowledgment_reconciles_and_audit_trail_is_kept(cfg):
    p = run(cfg, demo_events("delayed_ack"))
    tr = [t for t in p.transitions if t["oid"] == "demo:ob1"]
    states = [t["new_state"] for t in tr]
    assert states == ["pending", "censored_not_yet_assessable", "overdue_without_verified_completion", "completed_with_evidence"]
    assert tr[-1]["old_state"] == "overdue_without_verified_completion" and tr[-1]["evidence"]                  # transition references its evidence
    assert p.states["demo:ob1"][2] is False                                                                      # not late: acknowledged before the deadline per event time


def test_missing_acknowledgment_stays_unconfirmed_downstream(cfg):
    p = run(cfg, demo_events("missing_ack"))
    assert p.states["demo:ob0"][0] == State.COMPLETED                      # the test completed ...
    assert p.states["demo:ob1"][0] == State.OVERDUE                        # ... but its review has no evidence (tracker confirms open)
    assert p.states["demo:ob2"][0] == State.UNVERIFIED                     # dependent notification: no evidence either way
    assert p.states["demo:ob3"][0] == State.OVERDUE                        # documented no-show


def test_invalid_events_are_rejected_and_recorded(cfg):
    bad = [{"event_id": "x1", "patient_id": "p", "event_time_utc": "garbage", "ingest_time_utc": "2024-01-01T00:00:00Z", "event_type": "lab_collected", "provenance": "simulated", "obligation_id": "o"},
           {"event_id": "x2", "patient_id": "p", "event_time_utc": "2024-01-01T00:00:00Z", "ingest_time_utc": "2024-01-01T00:00:01Z", "event_type": "made_up", "provenance": "simulated", "obligation_id": "o"},
           {"event_id": "x3", "patient_id": "p", "event_time_utc": "2024-01-01T00:00:00Z", "ingest_time_utc": "2024-01-01T00:00:01Z", "event_type": "lab_collected", "provenance": "magic", "obligation_id": "o"},
           {"event_id": "x4", "patient_id": "p", "event_time_utc": "2024-01-01T01:00:00Z", "ingest_time_utc": "2024-01-01T00:00:01Z", "event_type": "lab_collected", "provenance": "simulated", "obligation_id": "o"}]
    p = run(cfg, bad)
    assert p.stats["rejected"] == 4
    assert p.integrity_check()["rejected"] == 4 and p.graph.stats()["events_ingested"] == 0


def test_sqlite_integrity_persistence_and_recovery(cfg, tmp_path):
    db = tmp_path / "state.sqlite"
    p = run(cfg, demo_events("delayed_ack"), db=db)
    chk = p.integrity_check()
    assert chk["sqlite_integrity"] == "ok" and chk["event_ids_unique"] and chk["orphan_transitions"] == 0
    states, counts = {o: s[0] for o, s in p.states.items()}, p.state_counts()
    p.close()
    q = EventProcessor.recover(cfg, db)                                   # cold start from the persisted event log
    assert {o: s[0] for o, s in q.states.items()} == states
    assert q.graph.stats()["obligations"] == 4 and q.graph.completion_time("demo:ob1", 1e12) is not None
    q.close()


def test_duplicate_delivery_updates_counter_not_state_in_db(cfg, tmp_path):
    ev = demo_events("happy_path")
    p = run(cfg, ev + ev[5:8], db=tmp_path / "d.sqlite")
    n = p.db.execute("SELECT SUM(deliveries) - COUNT(*) FROM events").fetchone()[0]
    assert n == 3


def test_transitions_are_chronological_per_obligation_in_batch_mode(cfg):
    ev = demo_events("delayed_ack")
    p = run(cfg, ev, rescore_batch=64)
    for oid in {t["oid"] for t in p.transitions}:
        ts = [to_epoch(t["t_ingest"]) for t in p.transitions if t["oid"] == oid]
        assert ts == sorted(ts)


def test_batching_does_not_change_results(cfg):
    ev = demo_events("delayed_ack")
    a, b = run(cfg, ev, rescore_batch=1), run(cfg, ev, rescore_batch=64)
    assert [(t["oid"], t["new_state"]) for t in a.transitions] == [(t["oid"], t["new_state"]) for t in b.transitions]


def test_alert_has_required_explanation_fields(cfg):
    p = run(cfg, demo_events("missing_ack"))
    a = next(x for x in p.alert_log if x.rule == "DEADLINE_MISSED_CONFIRMED")
    for part in ("Task:", "State:", "Evidence used:", "Missing:", "Responsible role:", "Suggested verification", "Uncertainty"):
        assert part in a.explanation
    assert a.evidence_ids and a.responsible_role and a.suggested_verification


def test_alert_rules_split_forecast_from_deadline_logic(cfg):
    class Const:
        def predict_dicts(self, rows):
            return [0.95] * len(rows)
    p = run(cfg, demo_events("missing_ack"), scorers={"C_graph_ml": Const()})
    rules = {a.rule for a in p.alert_log}
    assert "HIGH_RISK" in rules                                           # forecast alerts exist while obligations are pending
    over = [a for a in p.alert_log if a.rule == "DEADLINE_MISSED_CONFIRMED"]
    assert over and all(a.state == State.OVERDUE.value for a in over)    # the overdue alert is state-derived
    assert all(a.rule != "HIGH_RISK" or a.state == State.PENDING.value for a in p.alert_log)


def test_priority_score_methods(cfg):
    f = {"hours_to_deadline": 10.0, "n_incomplete_ancestors": 2, "own_no_events": 1, "n_verified_events": 0, "has_conflicting_record": 0, "step_type": "result_reviewed"}
    s = {m: PriorityScorer(m).score(0.8, f) for m in ("risk_only", "deadline_only", "risk_x_urgency", "weighted")}
    assert s["risk_only"] == 0.8 and 0 < s["deadline_only"] <= 1 and 0 < s["weighted"] <= 1
    assert PriorityScorer("weighted").score(0.9, f) > PriorityScorer("weighted").score(0.3, f)
    assert PriorityScorer("weighted").score(0.8, {**f, "hours_to_deadline": 1.0}) > PriorityScorer("weighted").score(0.8, {**f, "hours_to_deadline": 300.0})


def test_deadline_only_scorer_ignores_dependencies():
    assert deadline_only_score([{"elapsed_fraction": 0.4, "n_incomplete_ancestors": 5}, {"elapsed_fraction": 1.7}]) == [0.4, 1.0]


@pytest.fixture(scope="module")
def tiny_models(cfg):
    from src.risk_predictor import RiskModel, binary, risk_frames
    from src.data_loader import subset_patients, read_source
    pats = read_source(cfg, "interim/patients.csv.gz")
    keep = subset_patients(cfg, pats)
    F = risk_frames(cfg, keep)
    tr, ytr = binary(F["train"]); va, yva = binary(F["validation"])
    from src.ontology import features_of, SYSTEM_FEATURES
    cols = features_of(SYSTEM_FEATURES["C_graph_ml"])
    m = RiskModel("hgb", cols, {"max_depth": 3, "max_iter": 60}, 0).fit(tr, ytr).calibrate(va, yva)
    return m, va, cols


def test_model_features_exclude_forbidden_columns(tiny_models):
    m, _, cols = tiny_models
    forbidden = {"outcome_label", "label_binary", "scenario_profile", "latent_completed", "latent_completion_time", "doc_lost", "blocked_by_upstream", "observation_end", "resolved_late",
                 "completion_event_time", "documented_root", "deadline_utc", "assessment_time_utc", "obligation_id", "patient_id", "chain_id", "split", "source_dataset"}
    assert not (forbidden & set(m.features))


def test_model_serialization_roundtrip_and_calibration_bounds(tiny_models, tmp_path):
    from src.risk_predictor import RiskModel
    m, va, _ = tiny_models
    m.save(tmp_path / "m.joblib")
    m2 = RiskModel.load(tmp_path / "m.joblib")
    p1, p2 = m.predict(va), m2.predict(va)
    assert np.allclose(p1, p2) and ((p1 >= 0) & (p1 <= 1)).all()
    assert np.allclose(m.predict_dicts(va[m.features].head(5).to_dict("records")), p1[:5])


def test_risk_input_cutoff_future_events_do_not_change_the_score(cfg, tiny_models):
    m, _, _ = tiny_models
    from src.evidence_graph import TemporalEvidenceGraph
    T0 = to_epoch("2024-03-04T09:00:00Z")
    g = TemporalEvidenceGraph()
    g.ingest(mk_ob("o0", "lab_collected", T0, T0 + 48 * H, [], n=2))
    g.ingest(mk_ob("o1", "result_reviewed", T0, T0 + 120 * H, ["o0"], pos=1, n=2))
    A = T0 + 20 * H
    s1 = m.predict_dicts([g.features("o1", A)])[0]
    for i, (et, o, t) in enumerate((("lab_collected", "o0", 25), ("result_posted", "o0", 30), ("result_acknowledged", "o1", 60))):
        g.ingest(mk_ev(f"f{i}", o, et, T0 + t * H))
    assert m.predict_dicts([g.features("o1", A)])[0] == s1
