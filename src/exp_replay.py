"""Replay-based experiments: system comparison (Exp 3), evidence-uncertainty injection (Exp 4), ablations and priority scoring (Exp 5 / Section 11).

All three systems are evaluated on the SAME replayed stream of held-out (test) patients:
  A  deadline-only reminders   : score = fraction of the window elapsed (no dependencies, no evidence, no ML)
  B  independent tasks + ML    : calibrated model on task/own-event/workload features only
  C  dependency-aware graph+ML : calibrated model on all features computed by the temporal evidence graph (+ deterministic evidence states)
Operating thresholds are chosen on VALIDATION patients (target recall) and applied unchanged to TEST patients.
Labels are simulated follow-up outcomes; results describe simulated processes only.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd

from .alert_engine import AlertEngine, PriorityScorer
from .data_loader import iter_stream, load_stream_truth
from .evaluation import grouped_bootstrap_ci, paired_bootstrap_diff
from .event_processor import EventProcessor, deadline_only_score
from .evidence_graph import TemporalEvidenceGraph, to_epoch
from .experiments import HAS_XGB, Lab
from .ontology import FEATURE_GROUPS, State, TEMPORAL_FEATURES, features_of, SYSTEM_FEATURES
from .risk_predictor import RiskModel, binary

H = 3600.0
SYSTEMS = ["A_deadline_only", "B_independent_ml", "C_graph_ml"]
TRUE_CLASS = {"completed": "completed", "overdue_unresolved": "overdue", "unverified": "not_confirmed", "censored_or_unknown": "not_confirmed"}


# ------------------------------------------------------------------------------------------------ scorers
def load_scorers(lab: Lab, ablations: bool = True) -> dict:
    """A, B, C (+ ablation variants of C retrained with the SAME family/hyper-parameters minus a feature group; train/val only)."""
    mB = RiskModel.load(lab.models / "risk_B_independent.joblib")
    mC = RiskModel.load(lab.models / "risk_C_graph.joblib")
    sc = {"A_deadline_only": deadline_only_score, "A_reminder_24h": lambda rows: [1.0 if r["hours_to_deadline"] <= 24.5 else 0.0 for r in rows], "B_independent_ml": mB, "C_graph_ml": mC}
    if ablations:
        F = lab.dense()
        tr, ytr = binary(F["train"]); va, yva = binary(F["validation"])
        full = mC.features
        drops = {"C_no_graph_dependency": FEATURE_GROUPS["graph_dependency"], "C_no_temporal": TEMPORAL_FEATURES, "C_no_evidence": FEATURE_GROUPS["evidence"],
                 "C_no_graph_temporal": FEATURE_GROUPS["graph_temporal"]}
        for name, d in drops.items():
            cols = [c for c in full if c not in set(d)]
            m = RiskModel(mC.kind, cols, mC.params, lab.seed, name).fit(tr, ytr).calibrate(va, yva)
            m.save(lab.models / f"ablation_{name}.joblib")
            sc[name] = m
    return sc


def replay(lab: Lab, patients: set[str] | None, scorers: dict, events: list[dict] | None = None, enable_alerts: bool = True, rescore_batch: int = 1,
           db=":memory:", log_risk: bool = True, build_nx: bool = False, track_latency: bool = False) -> EventProcessor:
    ev = events if events is not None else list(iter_stream(lab.cfg, patients=patients))
    p = EventProcessor(lab.cfg, db, scorers=scorers, primary="C_graph_ml", enable_alerts=enable_alerts, rescore_batch=rescore_batch, log_risk=log_risk, build_nx=build_nx,
                       track_latency=track_latency)
    p.run(ev)
    p.finalize()
    return p


def split_patients(lab: Lab, split: str) -> set[str] | None:
    df = lab.patients_df
    ids = df[df.split == split].patient_id.tolist()
    n = (lab.p["replay_patients_per_split"] or {}).get(split)
    if n:
        from .data_loader import stable_hash
        ids.sort(key=lambda i: stable_hash(i, "replay_subset"))
        ids = ids[:n]
    return set(ids)


def truth_frame(lab: Lab, proc: EventProcessor) -> pd.DataFrame:
    t = load_stream_truth(lab.cfg)
    t = t[t.obligation_id.isin(proc.graph.obs)].copy()
    t = t[~t.documented_root.astype(bool)]
    t["y"] = t.outcome_label.map({"completed": 0, "overdue_unresolved": 1})
    t["deadline_t"] = t.deadline.map(to_epoch)
    t["true_class"] = t.outcome_label.map(TRUE_CLASS)
    return t.set_index("obligation_id")


# ------------------------------------------------------------------------------------------------ early warning evaluation
def first_alert_times(rl: pd.DataFrame, scorer: str, theta: float, min_lead_h: float = 0.0) -> pd.Series:
    """First time the score reaches theta while at least `min_lead_h` hours remain before the deadline (early-warning requirement)."""
    d = rl[(rl.scorer == scorer) & (rl.hours_to_deadline > 0) & (rl.hours_to_deadline >= min_lead_h) & (rl.risk >= theta)]
    return d.groupby("oid").t_clock.min()


def ew_metrics(truth: pd.DataFrame, first: pd.Series) -> dict:
    pos = truth.index[truth.y == 1]
    neg = truth.index[truth.y == 0]
    oth = truth.index[truth.y.isna()]
    al = first.index
    tp, fp = len(pos.intersection(al)), len(neg.intersection(al))
    lead = (truth.loc[pos.intersection(al), "deadline_t"] - first.loc[pos.intersection(al)]) / H
    return {"alerts": int(len(al)), "recall": tp / max(len(pos), 1), "precision": tp / max(tp + fp, 1), "false_alert_rate": fp / max(len(neg), 1),
            "alerts_on_unverified_or_censored": int(len(oth.intersection(al))), "median_lead_h": float(lead.median()) if len(lead) else float("nan"),
            "mean_lead_h": float(lead.mean()) if len(lead) else float("nan"), "n_pos": int(len(pos)), "n_neg": int(len(neg))}


def threshold_for_recall(truth_v: pd.DataFrame, rl_v: pd.DataFrame, scorer: str, target: float, min_lead_h: float = 0.0) -> float:
    """Largest threshold whose obligation-level recall (on VALIDATION, alerts at least min_lead_h before the deadline) is >= target."""
    d = rl_v[(rl_v.scorer == scorer) & (rl_v.hours_to_deadline > 0) & (rl_v.hours_to_deadline >= min_lead_h) & rl_v.oid.isin(truth_v.index[truth_v.y == 1])]
    mx = d.groupby("oid").risk.max()
    npos = int((truth_v.y == 1).sum())
    vals = np.sort(mx.values)[::-1]
    k = int(np.ceil(target * npos))
    return float(vals[k - 1]) if 0 < k <= len(vals) else float(vals[-1]) if len(vals) else 1.0


def max_score_table(rl: pd.DataFrame, truth: pd.DataFrame, scorer: str, horizon_h: float | None = None) -> pd.DataFrame:
    d = rl[(rl.scorer == scorer) & (rl.hours_to_deadline > (horizon_h if horizon_h else 0.0))]
    if horizon_h:                                               # score as known at T - horizon: use entries logged at least `horizon` before the deadline
        d = d.sort_values("t_clock").groupby("oid").tail(1)
        d = d.assign(risk=d.risk)
    s = d.groupby("oid").risk.max() if not horizon_h else d.set_index("oid").risk
    t = truth[truth.y.notna()].copy()
    t["score"] = s.reindex(t.index).fillna(0.0)
    return t


def pr_auc_of(t: pd.DataFrame) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(t.y.astype(int), t.score)) if t.y.sum() else float("nan")


def eval_states(proc: EventProcessor, truth: pd.DataFrame, grace_h: float) -> dict:
    tr = pd.DataFrame(proc.transitions)
    tr["t"] = tr.t_ingest.map(to_epoch)
    per = {o: g.sort_values("t") for o, g in tr.groupby("oid")}
    pred, naive = {}, {}
    for oid, row in truth.iterrows():
        tstar = row.deadline_t + (grace_h + 1) * H
        g = per.get(oid)
        c = "none"
        if g is not None:
            gg = g[g.t <= tstar]
            if len(gg):
                last = gg.iloc[-1]
                st, late = last.new_state, bool(last["late"])
                c = {"completed_with_evidence": "overdue" if late else "completed", "overdue_without_verified_completion": "overdue"}.get(st, "not_confirmed")
        pred[oid] = c
        naive[oid] = naive_class(proc.graph, oid, tstar)
    fin = {}
    for oid in truth.index:
        st = proc.states.get(oid)
        fin[oid] = "none" if st is None else {"completed_with_evidence": "overdue" if st[2] else "completed", "overdue_without_verified_completion": "overdue"}.get(st[0].value, "not_confirmed")
    out = {}
    for name, d in (("engine_evidence_aware", pred), ("engine_final_after_reconciliation", fin), ("naive_absence_is_failure", naive)):
        pc = pd.Series(d).reindex(truth.index)
        acc = float((pc == truth.true_class).mean())
        conf = pd.crosstab(truth.true_class, pc)
        rec = {c: float(((pc == c) & (truth.true_class == c)).sum() / max((truth.true_class == c).sum(), 1)) for c in ("completed", "overdue", "not_confirmed")}
        out[name] = {"accuracy": acc, "per_class_recall": rec, "confusion": conf}
    return out


def naive_class(g: TemporalEvidenceGraph, oid: str, tstar: float) -> str:
    """Independent-task deadline logic: an obligation is 'completed' if ANY completion-type event was ingested by t*, else 'overdue' once its deadline passed
    (absence of evidence treated as failure; conflicts, evidence quality and dependencies ignored)."""
    ob = g.obs[oid]
    from .ontology import COMPLETION_EVENT
    ct = COMPLETION_EVENT[ob.step_type]
    comp = [e for e in g.ob_events.get(oid, ()) if e.etype == ct and e.t_ingest <= tstar and e.t_event <= tstar]
    if comp:
        return "overdue" if min(e.t_event for e in comp) > ob.deadline else "completed"
    return "overdue" if tstar > ob.deadline else "pending"


# ------------------------------------------------------------------------------------------------ Experiment 3
def run_exp3(lab: Lab, scorers: dict | None = None, rescore_batch: int = 64) -> dict:
    t0 = time.time()
    sc = scorers or load_scorers(lab, ablations=True)
    pv, pt = split_patients(lab, "validation"), split_patients(lab, "test")
    prv = replay(lab, pv, sc, rescore_batch=rescore_batch)
    prt = replay(lab, pt, sc, rescore_batch=rescore_batch, track_latency=True)
    tv, tt = truth_frame(lab, prv), truth_frame(lab, prt)
    rlv, rlt = prv.risk_log_df(), prt.risk_log_df()
    rows = []
    for name in sc:
        for lead in (0.0, 24.0, 72.0):
            for target in (0.5, 0.7, 0.9):
                theta = threshold_for_recall(tv, rlv, name, target, lead)
                m = ew_metrics(tt, first_alert_times(rlt, name, theta, lead))
                rows.append({"system": name, "min_lead_h": lead, "target_recall(val)": target, "threshold(val)": theta, **m})
    ew = pd.DataFrame(rows)
    # threshold-free summaries with patient-grouped CIs
    summ = []
    for name in sc:
        t_ = max_score_table(rlt, tt, name)
        ci = grouped_bootstrap_ci(t_.reset_index().assign(patient_id=t_.patient_id.values), "patient_id", pr_auc_of, n_boot=lab.p["bootstrap"], seed=11) if "patient_id" in t_ else {}
        h24 = max_score_table(rlt, tt, name, horizon_h=24.0)
        h72 = max_score_table(rlt, tt, name, horizon_h=72.0)
        from sklearn.metrics import roc_auc_score
        summ.append({"system": name, "pr_auc_max_score": pr_auc_of(t_), "pr_auc_lo": ci.get("lo"), "pr_auc_hi": ci.get("hi"), "pr_auc_at_T-24h": pr_auc_of(h24), "pr_auc_at_T-72h": pr_auc_of(h72),
                     "roc_auc_max_score": float(roc_auc_score(t_.y.astype(int), t_.score))})
    summ = pd.DataFrame(summ)
    # paired differences C vs A and C vs B (same patients)
    pd_t = max_score_table(rlt, tt, "C_graph_ml").rename(columns={"score": "sC"})
    for nm, col in (("A_deadline_only", "sA"), ("B_independent_ml", "sB")):
        pd_t[col] = max_score_table(rlt, tt, nm).score
    pd_t = pd_t.reset_index()
    from sklearn.metrics import average_precision_score
    paired = {k: paired_bootstrap_diff(pd_t, "patient_id", lambda s: average_precision_score(s.y.astype(int), s.sC), lambda s, c=c: average_precision_score(s.y.astype(int), s[c]),
                                       n_boot=lab.p["bootstrap"], seed=12) for k, c in (("C_minus_A", "sA"), ("C_minus_B", "sB"))}
    states = eval_states(prt, tt, lab.cfg["engine"]["grace_hours"])
    # engine alerts (system C default configuration)
    al = prt.alerts_df()
    al = al.join(tt[["y", "deadline_t", "true_class"]], on="oid")
    al["t"] = al.t_ingest.map(to_epoch)
    alert_tab = []
    for rule, g in al.groupby("rule"):
        pos = g[g.y == 1]; neg = g[g.y == 0]
        lead = (pos.deadline_t - pos.t) / H
        alert_tab.append({"rule": rule, "alerts": len(g), "precision_vs_overdue": float(len(pos) / max(len(pos) + len(neg), 1)), "on_completed": int(len(neg)), "on_unverified_or_censored": int(g.y.isna().sum()),
                          "median_lead_h(before deadline)": float(lead.median()) if len(lead) else float("nan")})
    alert_tab = pd.DataFrame(alert_tab)
    lat = np.array(prt.latencies) * 1000
    out = {"early_warning": ew, "summary": summ, "paired": paired, "state_eval": states, "alert_table": alert_tab, "n": {"val_obligations": int(len(tv)), "test_obligations": int(len(tt)),
           "test_overdue": int((tt.y == 1).sum()), "test_completed": int((tt.y == 0).sum()), "test_unverified_or_censored": int(tt.y.isna().sum()), "test_patients": len(pt) if pt else None},
           "replay_stats": dict(prt.stats), "replay_latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)), "p99": float(np.percentile(lat, 99))},
           "graph_stats": prt.graph.stats(), "integrity": prt.integrity_check(), "seconds": time.time() - t0, "processor_test": prt, "scorers": sc}
    lab.save_csv("exp3_early_warning", ew); lab.save_csv("exp3_summary", summ); lab.save_csv("exp3_alerts_by_rule", alert_tab)
    lab.save_json("exp3_summary", {"paired": paired, "n": out["n"], "state_eval": {k: {"accuracy": v["accuracy"], "per_class_recall": v["per_class_recall"]} for k, v in states.items()},
                                   "replay_stats": out["replay_stats"], "replay_latency_ms": out["replay_latency_ms"], "integrity": out["integrity"]})
    lab.log.log("exp3_system_comparison", {"systems": list(sc), "thresholds": "chosen on validation replay"}, {"paired": paired, "state_accuracy": {k: v["accuracy"] for k, v in states.items()}}, lab.seed,
                model="A/B/C", duration_s=out["seconds"], warnings=["simulated follow-up outcomes"])
    return out


# ------------------------------------------------------------------------------------------------ Experiment 4: evidence uncertainty
def _clean_stream(events: list[dict]) -> list[dict]:
    pr = {"encounter_start": 0, "obligation_created": 1, "deadline_reached": 3}
    ev = [dict(e, ingest_time_utc=e["event_time_utc"]) for e in events if e.get("delivery_attempt", 1) == 1]
    ev.sort(key=lambda e: (e["event_time_utc"], pr.get(e["event_type"], 2), e["event_id"]))
    return ev


def _inject(events: list[dict], truth: pd.DataFrame, kind: str, rate: float, seed: int) -> tuple[list[dict], set[str]]:
    from .ontology import COMPLETION_EVENT
    rng = np.random.default_rng(seed)
    comp_ev = {}
    for e in events:
        if e["obligation_id"] in truth.index and e["event_type"] == COMPLETION_EVENT.get(truth.at[e["obligation_id"], "step_type"]) and e.get("delivery_attempt", 1) == 1:
            comp_ev[e["obligation_id"]] = e
    completed = [o for o in comp_ev if truth.at[o, "outcome_label"] == "completed"]
    chosen = set(rng.choice(completed, size=int(rate * len(completed)), replace=False)) if completed else set()
    out = []
    if kind == "missing_completion":
        out = [e for e in events if not (e["obligation_id"] in chosen and e["event_id"] == comp_ev[e["obligation_id"]]["event_id"])]
    elif kind == "delayed_evidence":
        for e in events:
            if e["obligation_id"] in chosen and e["event_id"] == comp_ev[e["obligation_id"]]["event_id"]:
                dly = float(rng.uniform(40, 120)) * H
                e = dict(e, ingest_time_utc=pd.Timestamp(to_epoch(e["event_time_utc"]) + dly, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"))
            out.append(e)
        out.sort(key=lambda e: (e["ingest_time_utc"], e["event_id"]))
    elif kind == "contradictory":
        out = list(events)
        for o in chosen:
            c = comp_ev[o]
            t = to_epoch(c["event_time_utc"]) + float(rng.uniform(1, 30)) * H
            ts = pd.Timestamp(t, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
            out.append({**c, "event_id": c["event_id"] + ":conflict", "event_type": "conflicting_record", "event_time_utc": ts, "ingest_time_utc": ts,
                        "payload": {**c["payload"], "verification": "unverified"}})
        out.sort(key=lambda e: (e["ingest_time_utc"], e["event_id"]))
    elif kind == "duplicates":
        out = list(events)
        for e in events:
            if rng.random() < rate:
                out.append({**e, "delivery_attempt": e.get("delivery_attempt", 1) + 1, "ingest_time_utc": pd.Timestamp(to_epoch(e["ingest_time_utc"]) + float(rng.uniform(1, 3600)), unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")})
        out.sort(key=lambda e: (e["ingest_time_utc"], e["event_id"]))
        chosen = set()
    elif kind == "unacknowledged_results":
        rev = [o for o in completed if truth.at[o, "step_type"] in ("result_reviewed", "report_reviewed", "post_procedure_review", "medication_reviewed")]
        chosen = set(rng.choice(rev, size=int(rate * len(rev)), replace=False)) if rev else set()
        out = [e for e in events if not (e["obligation_id"] in chosen and e["event_type"] in ("result_acknowledged", "medication_acknowledged"))]
    else:
        raise ValueError(kind)
    return out, chosen


def run_exp4(lab: Lab, rate: float = 0.3) -> dict:
    """Does the system distinguish UNKNOWN completion status from CONFIRMED overdue? Injected faults vs the naive 'absence = failure' deadline logic."""
    pt = split_patients(lab, "test")
    base = list(iter_stream(lab.cfg, patients=pt))
    p0 = replay(lab, pt, {}, events=base, enable_alerts=True, log_risk=False)
    truth = truth_frame(lab, p0)
    grace = lab.cfg["engine"]["grace_hours"]
    base_states = {o: s[0].value for o, s in p0.states.items()}
    rows = []
    for kind in ("missing_completion", "delayed_evidence", "contradictory", "unacknowledged_results", "duplicates"):
        ev, chosen = _inject(base, truth, kind, rate if kind != "duplicates" else 0.2, lab.seed + hash(kind) % 1000)
        p = replay(lab, pt, {}, events=ev, enable_alerts=True, log_risk=False)
        if kind == "duplicates":
            same = np.mean([p.states.get(o, (None,))[0] == s for o, s in ((o, State(v)) for o, v in base_states.items())])
            rows.append({"injection": kind, "affected": int(sum(1 for e in ev if e.get("delivery_attempt", 1) > 1 and e["event_id"] not in {x["event_id"] for x in base if x.get("delivery_attempt", 1) > 1})),
                         "engine_state_identical_to_clean_run": float(same), "duplicates_ignored_by_graph": int(p.graph.n_duplicates), "naive_would_double_count": "n/a",
                         "engine_metric": "idempotency (final states unchanged)", "naive_metric": "-"})
            continue
        idx = [o for o in chosen if o in truth.index]
        tstars = {o: truth.at[o, "deadline_t"] + (grace + 1) * H for o in idx}
        eng_state = {o: (lambda s: s)(None) for o in idx}
        trans = pd.DataFrame(p.transitions)
        trans["t"] = trans.t_ingest.map(to_epoch)
        per = {o: g.sort_values("t") for o, g in trans.groupby("oid")}

        def state_at(o, t):
            g = per.get(o)
            if g is None:
                return None
            g = g[g.t <= t]
            return (g.iloc[-1].new_state, bool(g.iloc[-1]["late"])) if len(g) else None
        eng = {o: state_at(o, tstars[o]) for o in idx}
        final = {o: (p.states[o][0].value, p.states[o][2]) for o in idx if o in p.states}
        naive = {o: naive_class(p.graph, o, tstars[o]) for o in idx}
        n = max(len(idx), 1)
        if kind == "missing_completion":
            rows.append({"injection": kind, "affected": len(idx), "engine_declared_confirmed_overdue": float(np.mean([(eng[o] or ("",))[0] == State.OVERDUE.value for o in idx])),
                         "engine_flagged_unverified_or_not_assessable": float(np.mean([(eng[o] or ("",))[0] in (State.UNVERIFIED.value, State.CENSORED.value) for o in idx])),
                         "naive_declared_overdue": float(np.mean([naive[o] == "overdue" for o in idx])), "engine_metric": "should NOT be 'confirmed overdue' (documentation missing, task actually done)", "naive_metric": "absence=failure"})
        elif kind == "delayed_evidence":
            rows.append({"injection": kind, "affected": len(idx), "engine_declared_confirmed_overdue": float(np.mean([(eng[o] or ("",))[0] == State.OVERDUE.value for o in idx])),
                         "engine_flagged_unverified_or_not_assessable": float(np.mean([(eng[o] or ("",))[0] in (State.UNVERIFIED.value, State.CENSORED.value) for o in idx])),
                         "naive_declared_overdue": float(np.mean([naive[o] == "overdue" for o in idx])),
                         "engine_final_completed_after_reconciliation": float(np.mean([final.get(o, ("",))[0] == State.COMPLETED.value for o in idx])),
                         "engine_metric": "interim: not confirmed overdue; final: reconciled to completed", "naive_metric": "interim false overdue"})
        elif kind == "contradictory":
            rows.append({"injection": kind, "affected": len(idx), "engine_flagged_contradictory": float(np.mean([final.get(o, ("",))[0] == State.CONTRADICTORY.value for o in idx])),
                         "naive_declared_completed": float(np.mean([naive[o] == "completed" for o in idx])),
                         "engine_metric": "contradictory state", "naive_metric": "treated as completed"})
        elif kind == "unacknowledged_results":
            al = p.alerts_df()
            flagged = set(al[al.rule.isin(["RESULT_NOT_ACKNOWLEDGED", "COMPLETION_UNVERIFIED", "DEADLINE_MISSED_CONFIRMED"])].oid)
            first = {o: al[(al.oid == o) & al.rule.isin(["RESULT_NOT_ACKNOWLEDGED"])].t_ingest.map(to_epoch).min() for o in idx}
            lead = [(truth.at[o, "deadline_t"] - t) / H for o, t in first.items() if t == t]
            rows.append({"injection": kind, "affected": len(idx), "engine_alerted_any_rule": float(np.mean([o in flagged for o in idx])),
                         "engine_early_alert_RESULT_NOT_ACK": float(np.mean([first[o] == first[o] for o in idx])), "median_early_alert_lead_h": float(np.median(lead)) if lead else float("nan"),
                         "naive_declared_overdue_at_t*": float(np.mean([naive[o] == "overdue" for o in idx])), "engine_metric": "alerts before the deadline (reactive rules only fire after)", "naive_metric": "flags only after the deadline"})
    res = pd.DataFrame(rows)
    lab.save_csv("exp4_evidence_uncertainty", res)
    lab.log.log("exp4_evidence_uncertainty", {"rate": rate}, {"rows": res.to_dict("records")}, lab.seed, model="evidence-graph states vs naive", warnings=["injected faults on simulated stream"])
    return {"table": res, "base_state_counts": p0.state_counts(), "rate": rate}


# ------------------------------------------------------------------------------------------------ Section 11: intervention-priority approaches
def components_df(df: pd.DataFrame, risk: np.ndarray, scale_h: float = 48.0) -> pd.DataFrame:
    from .alert_engine import IMPORTANCE
    h = df.hours_to_deadline.clip(lower=0)
    return pd.DataFrame({"risk": risk, "urgency": 1.0 / (1.0 + h / scale_h), "unresolved_dependencies": df.n_incomplete_ancestors.clip(upper=3) / 3.0,
                         "evidence_incomplete": np.minimum(1.0, 0.5 * df.own_no_events + 0.5 * (df.n_verified_events == 0) + 0.3 * df.has_conflicting_record),
                         "importance": df.step_type.map(IMPORTANCE).fillna(0.6)}, index=df.index)


def run_priority(lab: Lab, scorers: dict | None = None) -> dict:
    """Compare ways of combining risk, deadline proximity, dependencies, evidence completeness and (simulated-workflow) importance into one ranking.
    'weighted_tuned' weights are searched on VALIDATION only (random Dirichlet search, fixed budget); 'learned' is a logistic regression fitted on validation."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    sc = scorers or load_scorers(lab, ablations=False)
    D = lab.dense()
    va, yva = binary(D["validation"]); te, yte = binary(D["test"])
    mC = sc["C_graph_ml"]
    cv, ct = components_df(va, mC.predict(va)), components_df(te, mC.predict(te))
    keys = list(cv.columns)
    rng = np.random.default_rng(lab.seed)
    budget = 300
    best_w, best_ap = None, -1
    for _ in range(budget):
        w = rng.dirichlet(np.ones(len(keys)))
        ap = average_precision_score(yva, cv.values @ w)
        if ap > best_ap:
            best_w, best_ap = w, ap
    learned = LogisticRegression(C=1.0, max_iter=1000).fit(cv.values, yva)
    default_w = np.array([0.55, 0.2, 0.1, 0.1, 0.05])
    methods = {"deadline_only (urgency)": ct.urgency.values, "risk_only (calibrated model)": ct.risk.values, "risk x sqrt(urgency)": (ct.risk * np.sqrt(ct.urgency)).values,
               "weighted (fixed default weights)": ct.values @ default_w, "weighted (tuned on validation)": ct.values @ best_w, "learned logistic combination": learned.predict_proba(ct.values)[:, 1]}
    rows = []
    d = te.assign(_y=yte)
    for name, s in methods.items():
        d["_s"] = s
        ap = average_precision_score(yte, s)
        ci = grouped_bootstrap_ci(d, "patient_id", lambda x: average_precision_score(x._y, x._s), n_boot=lab.p["bootstrap"], seed=21)
        order = np.argsort(-s)
        prec = {f"precision@top{int(f*100)}%": float(yte[order[: max(1, int(f * len(s)))]].mean()) for f in (0.1, 0.2, 0.3)}
        rows.append({"method": name, "pr_auc": ap, "pr_auc_lo": ci["lo"], "pr_auc_hi": ci["hi"], **prec})
    tab = pd.DataFrame(rows)
    paired = {}
    d["_base"] = methods["risk_only (calibrated model)"]
    for name in ("weighted (tuned on validation)", "learned logistic combination", "deadline_only (urgency)"):
        d["_s"] = methods[name]
        paired[name + " - risk_only"] = paired_bootstrap_diff(d, "patient_id", lambda x: average_precision_score(x._y, x._s), lambda x: average_precision_score(x._y, x._base), n_boot=lab.p["bootstrap"], seed=22)
    out = {"table": tab, "tuned_weights": dict(zip(keys, map(float, best_w))), "learned_coefficients": dict(zip(keys, map(float, learned.coef_[0]))), "paired": paired, "search_budget": budget,
           "note": "importance = simulated workflow-definition weights (not clinical severity); weights/coefficients fitted on validation only"}
    lab.save_csv("priority_methods", tab)
    lab.save_json("priority_summary", {k: v for k, v in out.items() if k != "table"})
    lab.log.log("section11_priority", {"budget": budget}, {"table": tab.to_dict("records"), "tuned_weights": out["tuned_weights"]}, lab.seed, model="priority scorer")
    return out


# ------------------------------------------------------------------------------------------------ error analysis (Section 15)
def error_analysis(lab: Lab, e3: dict, k: int = 3, horizon_h: float = 24.0, scorer: str = "C_graph_ml") -> dict:
    """Concrete false-positive / false-negative cases for the graph+ML system at T-horizon, plus error rates by (hidden) generator scenario profile.
    The hidden profile is used ONLY for this post-hoc analysis, never as a feature."""
    prt = e3["processor_test"]
    tt = truth_frame(lab, prt)
    rl = prt.risk_log_df()
    t = max_score_table(rl, tt, scorer, horizon_h=horizon_h)
    thr = float(e3["early_warning"].query("system == @scorer and min_lead_h == 24.0 and `target_recall(val)` == 0.7")["threshold(val)"].iloc[0])
    t["flag"] = t.score >= thr
    cases = []
    g = prt.graph
    for label, sel in (("false positive (flagged, completed)", t[(t.y == 0) & t.flag].sort_values("score", ascending=False)), ("false negative (not flagged, overdue)", t[(t.y == 1) & ~t.flag].sort_values("score"))):
        for oid, row in sel.head(k).iterrows():
            ob = g.obs[oid]
            tcut = ob.deadline - horizon_h * H
            f = g.features(oid, tcut, live=True)
            evs = [(e.etype, iso_(e.t_event), iso_(e.t_ingest)) for e in g.ob_events.get(oid, ())]
            cases.append({"case": label, "obligation": oid, "step": ob.step_type, "score@T-24h": round(float(row.score), 3), "threshold": round(thr, 3), "true_outcome": row.outcome_label,
                          "hidden_scenario_profile": row.scenario_profile, "outcome_reason": row.outcome_reason, "n_incomplete_ancestors": f["n_incomplete_ancestors"], "parent_state": f["parent_state"],
                          "ack_state": f["ack_state"], "events (type, event time, ingest time)": evs[:6]})
    t["profile"] = t.scenario_profile
    prof = t[t.y.notna()].groupby("profile").apply(lambda d: pd.Series({"n": len(d), "overdue_share": float((d.y == 1).mean()), "flag_rate": float(d.flag.mean()),
                                                                          "recall": float(d[d.y == 1].flag.mean()) if (d.y == 1).any() else float("nan"),
                                                                          "false_alert_rate": float(d[d.y == 0].flag.mean()) if (d.y == 0).any() else float("nan")})).reset_index()
    lab.save_csv("error_by_profile", prof)
    return {"cases": pd.DataFrame(cases), "by_profile": prof, "threshold": thr, "horizon_h": horizon_h}


def iso_(t):
    from .evidence_graph import iso
    return iso(t)


# ------------------------------------------------------------------------------------------------ verification metrics (Section 13)
def verification_metrics(lab: Lab, proc: EventProcessor, truth: pd.DataFrame, events: list[dict]) -> dict:
    """Evidence matching, dependency verification, unresolved-evidence share and out-of-order reconciliation, against the simulated ground truth."""
    from .ontology import COMPLETION_EVENT
    g = proc.graph
    m_ok = eng_comp = tp = 0
    for oid, row in truth.iterrows():
        ct = g.completion_time(oid, 1e12)
        trt = row.completion_event_time
        has_t = isinstance(trt, str)
        if ct is not None:
            eng_comp += 1
        if ct is not None and has_t and abs(ct - to_epoch(trt)) < 2:
            tp += 1
        m_ok += int((ct is None and not has_t) or (ct is not None and has_t and abs(ct - to_epoch(trt)) < 2))
    # dependency verification: a completed child whose parent has NO documented completion must be flagged (dependency_evidence_gap)
    tmap = truth.completion_event_time.map(lambda x: isinstance(x, str)).to_dict()
    dv_ok = dv_n = 0
    for oid in truth.index:
        ob = g.obs[oid]
        if not ob.depends_on or not tmap.get(oid):
            continue
        dec = g.derive_state(oid, 1e12)
        flagged = bool(dec.detail.get("dependency_evidence_gap"))
        truth_gap = any((p in truth.index and not tmap.get(p, False)) for p in ob.depends_on)
        dv_n += 1
        dv_ok += int(flagged == truth_gap)
    st = pd.Series({o: s[0].value for o, s in proc.states.items()})
    unresolved = float(st.isin(["unverified_insufficient_evidence", "censored_not_yet_assessable", "contradictory_evidence"]).mean())
    pc = replay(lab, None, {}, events=_clean_stream(events), enable_alerts=False, log_risk=False)
    st2 = pd.Series({o: s[0].value for o, s in pc.states.items()})
    common = st.index.intersection(st2.index)
    return {"evidence_matching_accuracy": m_ok / max(len(truth), 1), "evidence_matching_precision": tp / max(eng_comp, 1), "completion_evidence_recall": tp / max(int(sum(tmap.values())), 1),
            "dependency_verification_accuracy": dv_ok / max(dv_n, 1), "dependency_pairs_checked": dv_n, "fraction_tasks_with_unresolved_evidence": unresolved,
            "out_of_order_final_state_agreement": float((st[common] == st2[common]).mean()), "n_compared": int(len(common)),
            "out_of_order_noisy_stream": {"late_events": int(proc.stats["late_events"]), "duplicates": int(proc.stats["duplicates"])}}
