"""Model B - probability that an obligation is still unresolved at its deadline, given only information available at assessment time.

* Training/evaluation frames are produced by the temporal evidence graph itself (`risk_frames`): features at each row's assessment time come
  from events with event_time <= assessment_time only. Labels come from the dataset's outcome derived from LATER events (never features).
* Families: logistic regression, HistGradientBoosting, random forest, XGBoost (when installed). Hyper-parameters are chosen by random
  search on train -> validation PR-AUC only; the test split is untouched until final reporting.
* Probabilities are calibrated on the validation split (isotonic when n >= 300 else sigmoid).
* Unverified and censored outcomes are NOT failures: binary training uses only completed (0) vs overdue_unresolved (1).
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .data_loader import SPLITS, iter_stream, load_risk
from .evidence_graph import TemporalEvidenceGraph, to_epoch
from .ontology import CATEGORICAL_FEATURES, FEATURE_GROUPS

try:
    from xgboost import XGBClassifier
except Exception:  # noqa: BLE001
    XGBClassifier = None

ALL_FEATURES = [c for g in FEATURE_GROUPS.values() for c in g]
META = ["obligation_id", "patient_id", "chain_id", "split", "source_dataset", "assessment_time_utc", "deadline_utc", "outcome_label", "label_binary"]


def build_graph_from_stream(cfg: dict, patients: set[str] | None = None, build_nx: bool = False, limit: int | None = None) -> TemporalEvidenceGraph:
    g = TemporalEvidenceGraph(cfg["engine"]["grace_hours"], build_nx=build_nx)
    for ev in iter_stream(cfg, patients=patients, limit=limit):
        g.ingest(ev)
    return g


def risk_frames(cfg: dict, patients: set[str] | None = None, graph: TemporalEvidenceGraph | None = None) -> dict[str, pd.DataFrame]:
    """Per-split DataFrames: META columns + all 49 graph-computed features at each row's assessment time."""
    rows = load_risk(cfg)
    g = graph or build_graph_from_stream(cfg, patients)
    out = {}
    for sp, df in rows.items():
        if patients is not None:
            df = df[df.patient_id.isin(patients)]
        feats = [g.features(o, to_epoch(a)) for o, a in zip(df.obligation_id, df.assessment_time_utc)]
        f = pd.DataFrame(feats, index=df.index)[ALL_FEATURES]
        out[sp] = pd.concat([df[META].reset_index(drop=True), f.reset_index(drop=True)], axis=1)
    return out


def binary(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    d = df[df.label_binary.notna()]
    return d, d.label_binary.astype(int).values


class _Calibrator:
    def __init__(self, method: str):
        self.method = method

    def fit(self, p, y):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        if self.method == "isotonic":
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        else:
            self.m = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(p / (1 - p)).reshape(-1, 1), y)
        return self

    def __call__(self, p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        if self.method == "isotonic":
            return np.clip(self.m.predict(p), 0.0, 1.0)
        return self.m.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]


def _preprocessor(kind: str, cols: list[str], cat_set=None) -> tuple[ColumnTransformer, list[int] | None]:
    cat_set = CATEGORICAL_FEATURES if cat_set is None else cat_set
    cat = [c for c in cols if c in cat_set]
    num = [c for c in cols if c not in cat_set]
    if kind == "logreg":
        return ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cat),
                                  ("n", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num)]), None
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    if kind == "rf":
        return ColumnTransformer([("c", make_pipeline(oe, SimpleImputer(strategy="constant", fill_value=-1)), cat),
                                  ("n", SimpleImputer(strategy="median", add_indicator=True), num)]), None
    return ColumnTransformer([("c", oe, cat), ("n", "passthrough", num)]), list(range(len(cat)))


SEARCH = {
    "logreg": lambda r: {"C": float(10 ** r.uniform(-3, 2))},
    "hgb": lambda r: {"max_depth": int(r.choice([3, 4, 6, 8])), "learning_rate": float(r.choice([0.03, 0.06, 0.1])), "max_iter": int(r.choice([100, 200, 300])),
                      "min_samples_leaf": int(r.choice([10, 20, 40])), "l2_regularization": float(r.choice([0.0, 0.1, 1.0]))},
    "rf": lambda r: {"n_estimators": int(r.choice([150, 300])), "max_depth": r.choice([6, 10, 14]).item(), "min_samples_leaf": int(r.choice([3, 5, 10])),
                     "max_features": r.choice(["sqrt", 0.5]).item() if False else (0.5 if r.random() < 0.5 else "sqrt")},
    "xgboost": lambda r: {"n_estimators": int(r.choice([150, 300, 500])), "max_depth": int(r.choice([3, 4, 5, 6])), "learning_rate": float(r.choice([0.03, 0.06, 0.1])),
                          "subsample": float(r.choice([0.7, 0.9])), "colsample_bytree": float(r.choice([0.6, 0.9])), "min_child_weight": int(r.choice([1, 5, 10]))},
}


class RiskModel:
    def __init__(self, kind: str, features: list[str], params: dict | None = None, seed: int = 0, name: str | None = None, cat_cols=None):
        self.cat_cols = list(CATEGORICAL_FEATURES if cat_cols is None else cat_cols)
        if kind == "xgboost" and XGBClassifier is None:
            raise RuntimeError("xgboost is not installed on this host")
        self.kind, self.features, self.params, self.seed = kind, list(features), params or {}, seed
        self.name = name or f"{kind}[{len(features)}f]"
        self.calibrator: _Calibrator | None = None

    def _make(self) -> Pipeline:
        pre, catidx = _preprocessor(self.kind, self.features, self.cat_cols)
        p = self.params
        if self.kind == "logreg":
            est = LogisticRegression(C=p.get("C", 1.0), max_iter=3000, random_state=self.seed)
        elif self.kind == "hgb":
            est = HistGradientBoostingClassifier(categorical_features=catidx, random_state=self.seed, early_stopping=False, **p)
        elif self.kind == "rf":
            est = RandomForestClassifier(n_jobs=2, random_state=self.seed, **p)
        else:
            est = XGBClassifier(tree_method="hist", n_jobs=2, random_state=self.seed, eval_metric="logloss", **p)
        return Pipeline([("pre", pre), ("est", est)])

    def fit(self, df: pd.DataFrame, y) -> "RiskModel":
        t = time.perf_counter()
        self.pipe = self._make().fit(df[self.features], y)
        self.train_seconds = time.perf_counter() - t
        return self

    def calibrate(self, df: pd.DataFrame, y, method: str | None = None) -> "RiskModel":
        raw = self.predict_raw(df)
        method = method or ("isotonic" if len(y) >= 300 else "sigmoid")
        self.calibrator = _Calibrator(method).fit(raw, np.asarray(y))
        return self

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict_proba(df[self.features])[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.predict_raw(df)
        return self.calibrator(raw) if self.calibrator else raw

    def predict_dicts(self, feats: list[dict]) -> np.ndarray:
        return self.predict(pd.DataFrame(feats, columns=None)[self.features] if feats else pd.DataFrame(columns=self.features))

    def save(self, path: Path):
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "RiskModel":
        return joblib.load(path)


def tune_and_fit(kind: str, features: list[str], train: pd.DataFrame, ytr, val: pd.DataFrame, yva, budget: int, seed: int = 0, name: str | None = None, cat_cols=None) -> tuple[RiskModel, pd.DataFrame]:
    """Random search over the family's space on validation PR-AUC (train never sees validation; test never used)."""
    rng = np.random.default_rng(seed)
    trials, best, best_s = [], None, -1.0
    for k in range(max(budget, 1)):
        params = SEARCH[kind](rng) if k > 0 or budget > 1 else {}
        try:
            m = RiskModel(kind, features, params, seed, name, cat_cols).fit(train, ytr)
            s = float(average_precision_score(yva, m.predict_raw(val)))
        except Exception as e:  # noqa: BLE001
            trials.append({"kind": kind, "params": str(params), "val_pr_auc": float("nan"), "error": str(e)[:80]})
            continue
        trials.append({"kind": kind, "params": str(params), "val_pr_auc": s, "train_s": m.train_seconds})
        if s > best_s:
            best, best_s = m, s
    best.calibrate(val, yva)
    return best, pd.DataFrame(trials).sort_values("val_pr_auc", ascending=False)


DENSE_FRACTIONS = (0.08, 0.2, 0.35, 0.5, 0.65, 0.8, 0.92, 0.98)


def dense_frames(cfg: dict, patients: set[str] | None = None, graph: TemporalEvidenceGraph | None = None, fractions=DENSE_FRACTIONS) -> dict[str, pd.DataFrame]:
    """Training/evaluation rows at SEVERAL assessment times across each obligation's window (features from events <= that time only).

    The dataset pipeline assessed each obligation once at 20-80% of its window; a live monitor scores obligations all the way to the deadline, so models are
    also trained on snapshots at `fractions` of the window. Population: non-root obligations still OPEN at the snapshot; label = the obligation's eventual
    outcome (completed=0, overdue_unresolved=1; unverified/censored excluded). Splits are the patient-level splits assigned before scenario generation."""
    from .data_loader import load_stream_truth
    from .evidence_graph import iso
    g = graph or build_graph_from_stream(cfg, patients)
    t = load_stream_truth(cfg)
    t = t[(~t.documented_root.astype(bool)) & t.obligation_id.isin(g.obs) & t.outcome_label.isin(["completed", "overdue_unresolved"])]
    rows, meta = [], []
    for r in t.itertuples(index=False):
        ob = g.obs[r.obligation_id]
        for f in fractions:
            A = ob.created + f * (ob.deadline - ob.created)
            if g.completion_time(ob.oid, A) is not None:
                break                                           # completed before this snapshot -> not at risk any more
            rows.append(g.features(ob.oid, A))
            meta.append((r.obligation_id, ob.patient_id, ob.chain_id, r.split, f, iso(A), iso(ob.deadline), r.outcome_label, 1 if r.outcome_label == "overdue_unresolved" else 0, ob.source_dataset))
    m = pd.DataFrame(meta, columns=["obligation_id", "patient_id", "chain_id", "split", "snapshot_fraction", "assessment_time_utc", "deadline_utc", "outcome_label", "label_binary", "source_dataset"])
    f = pd.DataFrame(rows)[ALL_FEATURES]
    df = pd.concat([m, f], axis=1)
    return {sp: df[df.split == sp].reset_index(drop=True) for sp in SPLITS}


def group_permutation_importance(model: RiskModel, df: pd.DataFrame, y, groups: dict[str, list[str]], repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """Drop in PR-AUC when a whole feature GROUP is shuffled across rows (model-agnostic, test-time; no refitting)."""
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    base = average_precision_score(y, model.predict(df))
    rows = []
    for name, cols in groups.items():
        cols = [c for c in cols if c in model.features]
        if not cols:
            continue
        drops = []
        for _ in range(repeats):
            d = df.copy()
            perm = rng.permutation(len(d))
            d[cols] = d[cols].values[perm]
            drops.append(base - average_precision_score(y, model.predict(d)))
        rows.append({"feature_group": name, "n_features": len(cols), "pr_auc_drop_mean": float(np.mean(drops)), "pr_auc_drop_std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("pr_auc_drop_mean", ascending=False).assign(base_pr_auc=base)


def feature_parity(cfg: dict, graph: TemporalEvidenceGraph, sample: int | None = None, patients: set[str] | None = None) -> dict:
    """Compare features recomputed by the evidence graph (from the replay stream) with the dataset pipeline's own feature tables at the same assessment times.
    Count/categorical features must match exactly; hour-valued features may differ by sub-second rounding of the exported timestamps (<= 0.001 h)."""
    import json as _json
    import math
    from .data_loader import path_of
    R = pd.concat(load_risk(cfg).values(), ignore_index=True)
    if patients is not None:
        R = R[R.patient_id.isin(patients)]
    if sample and len(R) > sample:
        R = R.sample(sample, random_state=0)
    spec = _json.load(open(path_of(cfg, "source", "processed/followup_risk/feature_spec.json")))
    cols = spec["feature_columns"]
    bad = {c: 0 for c in cols}
    for r in R.itertuples(index=False):
        f = graph.features(r.obligation_id, to_epoch(r.assessment_time_utc))
        for c in cols:
            a, b = f[c], getattr(r, c)
            if isinstance(a, str) or isinstance(b, str):
                ok = a == b
            elif isinstance(a, float) and math.isnan(a):
                ok = b != b
            else:
                tol = 1e-3 if c.startswith(("hours", "elapsed", "appointment_minus")) or c in ("window_hours",) else 1e-6
                ok = abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b))) + (1e-3 if tol > 1e-6 else 0)
            bad[c] += int(not ok)
    return {"rows_compared": int(len(R)), "features_compared": len(cols), "mismatches": {k: v for k, v in bad.items() if v}, "exact_or_within_tolerance": not any(bad.values())}
