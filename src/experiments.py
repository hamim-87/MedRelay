"""Experiment runners orchestrated by the master notebook. Every runner reads only train/validation data for fitting and tuning, touches the
test split once for reporting, writes tables/figures to results/ and figures/, and appends a record to results/experiments.jsonl."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import visualization as viz
from .action_classifier import ActionClassifier, evaluate, evaluate_on_spans, labels_matrix, tune as tune_action
from .data_loader import ROOT, SPLITS, load_action, load_config, load_noshow, path_of, read_source, subset_patients
from .eval_sets import challenge_docs
from .evaluation import (ExperimentLog, binary_metrics, calibration_curve_data, grouped_bootstrap_ci, paired_bootstrap_diff, precision_recall_at)
from .ontology import ALL_LABELS, FEATURE_GROUPS, SYSTEM_FEATURES, TEMPORAL_FEATURES, features_of
from .risk_predictor import ALL_FEATURES, RiskModel, binary, build_graph_from_stream, dense_frames, risk_frames, tune_and_fit

try:
    import xgboost  # noqa: F401
    HAS_XGB = True
except Exception:  # noqa: BLE001
    HAS_XGB = False


class Lab:
    """Shared experiment context (config, mode parameters, logger, cached heavy objects)."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.mode = self.cfg["mode"]
        self.p = self.cfg["experiments"][self.mode]
        self.seed = self.cfg["seed"]
        self.res = path_of(self.cfg, "results")
        self.models = path_of(self.cfg, "models")
        self.figs = path_of(self.cfg, "figures")
        self.log = ExperimentLog(self.res, ROOT)
        pats = read_source(self.cfg, "interim/patients.csv.gz")
        self.patients_df = pats
        self.patients = subset_patients(self.cfg, pats)
        self._frames = None
        self._graph = None
        self._dense = None

    def graph(self):
        if self._graph is None:
            self._graph = build_graph_from_stream(self.cfg, self.patients)
        return self._graph

    def frames(self) -> dict[str, pd.DataFrame]:
        if self._frames is None:
            self._frames = risk_frames(self.cfg, self.patients, self.graph())
        return self._frames

    def dense(self) -> dict[str, pd.DataFrame]:
        if self._dense is None:
            self._dense = dense_frames(self.cfg, self.patients, self.graph())
        return self._dense

    def save_json(self, name: str, obj):
        (self.res / f"{name}.json").write_text(json.dumps(obj, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else (o.tolist() if hasattr(o, "tolist") else str(o))))

    def save_csv(self, name: str, df: pd.DataFrame):
        df.to_csv(self.res / f"{name}.csv", index=False)


# ================================================================================================ Model A
def model_a_action_classifier(lab: Lab) -> dict:
    cfg = lab.cfg
    A = load_action(cfg)
    tr, va, te, aug = A["train"], A["validation"], A["test"], A["synthetic_train_augmentation"]
    docs = challenge_docs()
    rows, best, best_f, tables = [], {}, -1, {}
    for kind, budget in (("logreg", lab.p["action_budget"]), ("svm", lab.p["action_budget"]), ("mlp", max(2, lab.p["action_budget"] // 2))):
        t = time.time()
        m, trials = tune_action(tr, va, kind, budget, seed=lab.seed)
        rv, rt, sp = evaluate(m, va), evaluate(m, te), evaluate_on_spans(m, docs)
        rows.append({"model": kind, "params": json.dumps(m.params), "val_macro_f1": rv["macro"]["f1"], "val_micro_f1": rv["micro"]["f1"], "test_macro_f1": rt["macro"]["f1"],
                     "test_micro_f1": rt["micro"]["f1"], "test_micro_precision": rt["micro"]["precision"], "test_micro_recall": rt["micro"]["recall"],
                     "challenge_span_accuracy": sp["accuracy"], "challenge_span_macro_f1": sp["macro_f1"], "train_s": m.train_seconds, "search_s": time.time() - t})
        tables[kind] = {"test": rt, "span": sp, "trials": trials}
        lab.log.log(f"model_a_{kind}", {"params": m.params, "search_budget": budget}, {"val_macro_f1": rv["macro"]["f1"], "test_macro_f1": rt["macro"]["f1"], "challenge_span_acc": sp["accuracy"]},
                    lab.seed, model=f"tfidf+{kind}", duration_s=time.time() - t)
        if rv["macro"]["f1"] > best_f:
            best, best_f = m, rv["macro"]["f1"]
    # synthetic-template augmentation ablation for the winning family (train-only augmentation; val/test untouched)
    kind = best.kind
    m_aug = ActionClassifier(kind, seed=lab.seed, C=best.C, word_ngrams=best.word_ngrams, char=best.char, class_weight=best.class_weight).fit(
        pd.concat([tr.text, aug.text]), np.vstack([labels_matrix(tr), labels_matrix(aug)]))
    ra, rb = evaluate(m_aug, te), tables[kind]["test"]
    spa = evaluate_on_spans(m_aug, docs)
    comp = pd.DataFrame(rows)
    lab.save_csv("model_a_comparison", comp)
    best.save(lab.models / "action_classifier.joblib")
    per = pd.DataFrame(tables[kind]["test"]["per_class"]).T.reset_index().rename(columns={"index": "class"})
    lab.save_csv("model_a_per_class_test", per)
    viz.plot_confusion(tables[kind]["test"]["confusion_single_label"], f"Action classifier ({kind}) - silver test, single-label rows", lab.cfg, "model_a_confusion_silver")
    viz.plot_confusion(tables[kind]["span"]["confusion"], f"Action classifier ({kind}) - developer-annotated spans", lab.cfg, "model_a_confusion_spans")
    out = {"comparison": comp, "best_kind": kind, "best_params": best.params, "per_class_test": per,
           "synthetic_augmentation": {"test_macro_f1_without": rb["macro"]["f1"], "test_macro_f1_with": ra["macro"]["f1"], "challenge_span_acc_with": spa["accuracy"],
                                      "challenge_span_acc_without": tables[kind]["span"]["accuracy"], "n_synthetic": int(len(aug))},
           "label_provenance": "train/val/test labels are SILVER (rule-proposed); 'challenge_span_*' uses developer-annotated spans (not independently reviewed); 0 human-reviewed gold",
           "classifier": best}
    lab.save_json("model_a_summary", {k: v for k, v in out.items() if k not in ("classifier", "comparison", "per_class_test")} | {"comparison": comp.to_dict("records")})
    return out


# ================================================================================================ Model B (follow-up risk)
def _metrics_row(name, kind, fset, y, p_cal, p_raw, df, val_pr, boot):
    m = binary_metrics(y, p_cal)
    raw = binary_metrics(y, p_raw)
    d = df.assign(_y=y, _p=p_cal)
    from sklearn.metrics import average_precision_score
    ci = grouped_bootstrap_ci(d, "patient_id", lambda s: average_precision_score(s._y, s._p), n_boot=boot, seed=1)
    return {"model": name, "family": kind, "features": fset, "val_pr_auc": val_pr, "test_pr_auc": m["pr_auc"], "test_pr_auc_ci_lo": ci["lo"], "test_pr_auc_ci_hi": ci["hi"],
            "test_roc_auc": m["roc_auc"], "test_brier": m["brier"], "test_ece": m["ece"], "test_brier_uncalibrated": raw["brier"], "test_ece_uncalibrated": raw["ece"],
            "test_precision@0.5": m["precision@0.5"], "test_recall@0.5": m["recall@0.5"], "test_f1@0.5": m["f1@0.5"]}


def model_b_followup_risk(lab: Lab) -> dict:
    """Experiment 2: logistic regression vs tree models, task-only vs independent-task vs graph-feature-enhanced feature sets (all trained on SIMULATED outcomes)."""
    D = lab.dense()                                   # dense snapshots across each obligation's window (train/validation/test)
    tr, ytr = binary(D["train"]); va, yva = binary(D["validation"]); te, yte = binary(D["test"])
    og, yog = binary(lab.frames()["test"])                # the dataset pipeline's ORIGINAL single-snapshot protocol (20-80% of window), test split
    fsets = {"task_only": features_of(["task"]), "B_independent": features_of(SYSTEM_FEATURES["B_independent_ml"]), "C_graph": features_of(SYSTEM_FEATURES["C_graph_ml"])}
    fams = [f for f in ("logreg", "hgb", "rf", "xgboost") if f != "xgboost" or HAS_XGB]
    rows, best, curves, trial_tabs = [], {}, {}, []
    for fs_name, cols in fsets.items():
        for kind in fams:
            t = time.time()
            m, trials = tune_and_fit(kind, cols, tr, ytr, va, yva, lab.p["tuning_budget"], seed=lab.seed, name=f"{kind}/{fs_name}")
            pc, pr_ = m.predict(te), m.predict_raw(te)
            row = _metrics_row(m.name, kind, fs_name, yte, pc, pr_, te, float(trials.val_pr_auc.max()), lab.p["bootstrap"])
            mo = binary_metrics(yog, m.predict(og))
            row.update({"orig_protocol_test_pr_auc": mo["pr_auc"], "orig_protocol_test_roc_auc": mo["roc_auc"], "orig_protocol_test_brier": mo["brier"]})
            row["fit_search_s"] = time.time() - t
            rows.append(row)
            trial_tabs.append(trials.assign(features=fs_name))
            curves[(kind, fs_name)] = calibration_curve_data(yte, pc)
            if fs_name not in best or row["val_pr_auc"] > best[fs_name][1]:
                best[fs_name] = (m, row["val_pr_auc"], kind)
            lab.log.log("model_b_risk", {"family": kind, "features": fs_name, "n_features": len(cols), "budget": lab.p["tuning_budget"], "best_params": m.params},
                        {k: row[k] for k in ("val_pr_auc", "test_pr_auc", "test_roc_auc", "test_brier", "test_ece")}, lab.seed, model=m.name, duration_s=row["fit_search_s"],
                        data_version="followup_risk simulated outcomes", warnings=["simulation-generated labels: performance on simulated follow-up processes only"])
    res = pd.DataFrame(rows)
    # genuine baselines
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    tcols = ["hours_to_deadline", "window_hours", "elapsed_fraction"]
    dl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[tcols], ytr)
    pdl = dl.predict_proba(te[tcols])[:, 1]
    base_rows = [{"model": "majority(train prior)", **binary_metrics(yte, np.full(len(yte), ytr.mean()))}, {"model": "deadline_only_logreg", **binary_metrics(yte, pdl)}]
    base = pd.DataFrame(base_rows)
    lab.save_csv("model_b_comparison", res); lab.save_csv("model_b_baselines", base)
    lab.save_csv("model_b_search_trials", pd.concat(trial_tabs))
    for fs, (m, _, kind) in best.items():
        m.save(lab.models / f"risk_{fs}.joblib")
    cm, fm = best["C_graph"][0], best["task_only"][0]
    viz.plot_calibration({"C graph (calibrated)": calibration_curve_data(yte, cm.predict(te)), "C graph (raw)": calibration_curve_data(yte, cm.predict_raw(te)),
                          "B independent (calibrated)": calibration_curve_data(yte, best["B_independent"][0].predict(te))}, "Follow-up risk - test (simulated outcomes)", lab.cfg, "model_b_calibration")
    # paired comparison of graph-enhanced vs independent model on the same patients
    from sklearn.metrics import average_precision_score
    d = te.assign(_y=yte, pC=cm.predict(te), pB=best["B_independent"][0].predict(te), pT=fm.predict(te))
    paired = {"C_minus_B_pr_auc": paired_bootstrap_diff(d, "patient_id", lambda s: average_precision_score(s._y, s.pC), lambda s: average_precision_score(s._y, s.pB), n_boot=lab.p["bootstrap"], seed=2),
              "C_minus_task_only_pr_auc": paired_bootstrap_diff(d, "patient_id", lambda s: average_precision_score(s._y, s.pC), lambda s: average_precision_score(s._y, s.pT), n_boot=lab.p["bootstrap"], seed=3)}
    thr = precision_recall_at(yte, cm.predict(te))
    summ = {"best_per_featureset": {k: {"model": v[0].name, "val_pr_auc": v[1]} for k, v in best.items()}, "paired_bootstrap": paired, "precision_recall_at_thresholds_C": thr.to_dict("records"),
            "positive_rate_test": float(yte.mean()), "n": {"train": len(ytr), "validation": len(yva), "test": len(yte), "test_original_protocol": len(yog)}, "xgboost_available": HAS_XGB,
            "protocol": "primary rows = dense snapshots (8 points across each obligation window, open obligations only); 'orig_protocol_*' = dataset pipeline single snapshot at 20-80% of window",
            "note": "labels are simulated follow-up outcomes; 'unverified' and 'censored' rows are excluded from binary training/evaluation, never treated as failures"}
    lab.save_json("model_b_summary", summ)
    return {"comparison": res, "baselines": base, "best": best, "summary": summ, "thresholds": thr}


# ================================================================================================ Model C (no-show)
NS_ALL = ["age", "gender", "scholarship", "hypertension", "diabetes", "alcoholism", "handicap_level", "neighbourhood", "wait_days", "same_day", "scheduled_dow", "scheduled_hour",
          "appointment_dow", "prior_appointments", "prior_noshows", "prior_noshow_rate", "days_since_last_appointment"]
NS_SCHED = ["wait_days", "same_day", "scheduled_dow", "scheduled_hour", "appointment_dow", "prior_appointments", "prior_noshows", "prior_noshow_rate", "days_since_last_appointment"]
NS_CAT = ["gender", "neighbourhood"]


def model_c_noshow(lab: Lab) -> dict:
    """Secondary proof of concept on REAL appointment data. Label = attendance, not follow-up completion; never used to deny care."""
    from sklearn.metrics import average_precision_score
    df = load_noshow(lab.cfg)
    if lab.mode == "fast":
        keep = df.patient_key.drop_duplicates().sample(frac=0.4, random_state=lab.seed)
        df = df[df.patient_key.isin(keep)]
    tr, va, te = (df[df.split == s] for s in SPLITS)
    ytr, yva, yte = tr.no_show.values, va.no_show.values, te.no_show.values
    rows, best = [], {}
    for fs_name, cols in (("scheduling_only", NS_SCHED), ("with_demographics", NS_ALL)):
        for kind in [f for f in ("logreg", "hgb", "xgboost") if f != "xgboost" or HAS_XGB]:
            t = time.time()
            m, trials = tune_and_fit(kind, cols, tr, ytr, va, yva, max(2, lab.p["tuning_budget"] // 2), seed=lab.seed, name=f"{kind}/{fs_name}", cat_cols=NS_CAT)
            pc = m.predict(te)
            mm = binary_metrics(yte, pc, thr=float(np.quantile(pc, 0.8)))
            ci = grouped_bootstrap_ci(te.assign(_y=yte, _p=pc), "patient_key", lambda s: average_precision_score(s._y, s._p), n_boot=min(lab.p["bootstrap"], 100), seed=4)
            rows.append({"model": m.name, "features": fs_name, "val_pr_auc": float(trials.val_pr_auc.max()), "test_pr_auc": mm["pr_auc"], "pr_auc_ci_lo": ci["lo"], "pr_auc_ci_hi": ci["hi"],
                         "test_roc_auc": mm["roc_auc"], "test_brier": mm["brier"], "test_ece": mm["ece"], "prevalence": mm["prevalence"],
                         "precision@top20%": mm[f"precision@{float(np.quantile(pc, 0.8))}"], "recall@top20%": mm[f"recall@{float(np.quantile(pc, 0.8))}"], "fit_s": time.time() - t})
            best.setdefault(fs_name, {})[kind] = m
            lab.log.log("model_c_noshow", {"family": kind, "features": fs_name}, rows[-1], lab.seed, model=m.name, duration_s=rows[-1]["fit_s"], data_version="Kaggle noshow (real)",
                        warnings=["attendance label != follow-up completion; not for denying care"])
    res = pd.DataFrame(rows)
    lab.save_csv("model_c_comparison", res)
    top = res.sort_values("val_pr_auc", ascending=False).iloc[0]
    fs, kind = top.features, top.model.split("/")[0]
    m = best[fs][kind]
    pc = m.predict(te)
    age_band = pd.cut(te.age, [-1, 17, 34, 49, 64, 200], labels=["0-17", "18-34", "35-49", "50-64", "65+"])
    sub = []
    thr = float(np.quantile(pc, 0.8))
    for name, key in (("gender", te.gender), ("age_band", age_band), ("scholarship", te.scholarship)):
        for k in pd.unique(key.dropna()):
            mk = (key == k).values
            if mk.sum() >= 200 and 0 < yte[mk].sum() < mk.sum():
                sub.append({"group": name, "level": str(k), "n": int(mk.sum()), "prevalence": float(yte[mk].mean()), "mean_pred": float(pc[mk].mean()),
                            "pr_auc": float(average_precision_score(yte[mk], pc[mk])), "flagged_share_top20": float((pc[mk] >= thr).mean())})
    sub = pd.DataFrame(sub)
    lab.save_csv("model_c_subgroups", sub)
    viz.plot_calibration({f"{m.name}": calibration_curve_data(yte, pc)}, "No-show model calibration (real data, test)", lab.cfg, "model_c_calibration")
    m.save(lab.models / "noshow_model.joblib")
    out = {"comparison": res, "selected": m.name, "subgroups": sub, "prevalence_test": float(yte.mean()), "n": {"train": len(ytr), "validation": len(yva), "test": len(yte)},
           "caveats": "Brazilian public-hospital appointments (2016); attendance label only; demographic/geographic features can encode bias - the scheduling-only feature set is the recommended default; not for denying or deprioritising care"}
    lab.save_json("model_c_summary", {k: v for k, v in out.items() if k not in ("comparison", "subgroups")} | {"comparison": res.to_dict("records"), "subgroups": sub.to_dict("records")})
    return out
