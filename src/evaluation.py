"""Metrics and experiment tracking shared by all experiments (no model code here)."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_recall_fscore_support, precision_score,
                             recall_score, roc_auc_score)

from .clinical_extractor import ExtractionResult, interval_hours, unsupported_fields
from .ontology import CATEGORIES


# ----------------------------------------------------------------------------------------------- generic binary metrics
def ece_score(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def calibration_curve_data(y, p, bins: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    rows = [dict(bin=b, n=int((idx == b).sum()), mean_pred=float(p[idx == b].mean()), frac_pos=float(y[idx == b].mean())) for b in range(bins) if (idx == b).any()]
    return pd.DataFrame(rows)


def binary_metrics(y, p, thr: float = 0.5) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    pred = (p >= thr).astype(int)
    out = {"n": int(len(y)), "prevalence": float(y.mean()), "pr_auc": float(average_precision_score(y, p)) if y.sum() else float("nan"),
           "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"), "brier": float(brier_score_loss(y, p)), "ece": ece_score(y, p),
           f"precision@{thr}": float(precision_score(y, pred, zero_division=0)), f"recall@{thr}": float(recall_score(y, pred, zero_division=0)),
           f"f1@{thr}": float(f1_score(y, pred, zero_division=0))}
    return out


def precision_recall_at(y, p, thresholds=(0.3, 0.5, 0.7, 0.9)) -> pd.DataFrame:
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    rows = []
    for t in thresholds:
        pred = p >= t
        rows.append(dict(threshold=t, alerts=int(pred.sum()), precision=float(y[pred].mean()) if pred.any() else float("nan"), recall=float(pred[y == 1].mean()) if y.sum() else float("nan")))
    return pd.DataFrame(rows)


def grouped_bootstrap_ci(df: pd.DataFrame, group_col: str, metric_fn, n_boot: int = 200, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile CI of metric_fn(sub_df) resampling GROUPS (e.g. patients) with replacement."""
    rng = np.random.default_rng(seed)
    groups = {g: d for g, d in df.groupby(group_col)}
    keys = list(groups)
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(len(keys), len(keys), replace=True)
        sub = pd.concat([groups[keys[i]] for i in samp], ignore_index=True)
        try:
            v = metric_fn(sub)
        except Exception:  # noqa: BLE001
            continue
        if v == v:
            vals.append(v)
    vals = np.asarray(vals)
    point = metric_fn(df)
    if len(vals) < 10:
        return {"point": float(point), "lo": float("nan"), "hi": float("nan"), "n_boot": int(len(vals))}
    return {"point": float(point), "lo": float(np.quantile(vals, alpha / 2)), "hi": float(np.quantile(vals, 1 - alpha / 2)), "n_boot": int(len(vals))}


def paired_bootstrap_diff(df: pd.DataFrame, group_col: str, metric_a, metric_b, n_boot: int = 200, seed: int = 0) -> dict:
    """CI of metric_a - metric_b on the SAME resampled groups (paired comparison of two systems)."""
    rng = np.random.default_rng(seed)
    groups = {g: d for g, d in df.groupby(group_col)}
    keys = list(groups)
    diffs = []
    for _ in range(n_boot):
        sub = pd.concat([groups[keys[i]] for i in rng.choice(len(keys), len(keys), replace=True)], ignore_index=True)
        try:
            d = metric_a(sub) - metric_b(sub)
        except Exception:  # noqa: BLE001
            continue
        if d == d:
            diffs.append(d)
    diffs = np.asarray(diffs)
    return {"diff": float(metric_a(df) - metric_b(df)), "lo": float(np.quantile(diffs, 0.025)), "hi": float(np.quantile(diffs, 0.975)),
            "p_diff_le_0": float((diffs <= 0).mean()), "n_boot": int(len(diffs))}


# ----------------------------------------------------------------------------------------------- multilabel / classification
def multilabel_report(Y, P, labels) -> dict:
    Y, P = np.asarray(Y), np.asarray(P)
    pr, rc, f1, sup = precision_recall_fscore_support(Y, P, average=None, zero_division=0)
    per = {l: {"precision": float(a), "recall": float(b), "f1": float(c), "support": int(d)} for l, a, b, c, d in zip(labels, pr, rc, f1, sup)}
    mi = precision_recall_fscore_support(Y, P, average="micro", zero_division=0)[:3]
    ma = precision_recall_fscore_support(Y, P, average="macro", zero_division=0)[:3]
    return {"per_class": per, "micro": dict(zip(("precision", "recall", "f1"), map(float, mi))), "macro": dict(zip(("precision", "recall", "f1"), map(float, ma)))}


def single_label_confusion(y_true: list[str], y_pred: list[str], labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=[f"true:{l}" for l in labels], columns=[f"pred:{l}" for l in labels])


# ----------------------------------------------------------------------------------------------- extraction scoring
def _iou(a0, a1, b0, b1) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    return inter / max(a1 - a0 + b1 - b0 - inter, 1)


def _tok(s: str) -> set[str]:
    return set("".join(c if c.isalnum() else " " for c in (s or "").lower()).split())


def match_obligations(pred, truth, thr: float = 0.4, typed: bool = False) -> list[tuple[int, int]]:
    """Greedy one-to-one matching by character-span IoU (>= thr), optionally requiring equal task_type."""
    cands = []
    for i, p in enumerate(pred):
        for j, t in enumerate(truth):
            if typed and p.task_type != t["task_type"]:
                continue
            iou = _iou(p.evidence_span.start, p.evidence_span.end, t["start"], t["end"]) if p.evidence_span.start >= 0 else 0.0
            if iou >= thr:
                cands.append((iou, i, j))
    used_i, used_j, out = set(), set(), []
    for iou, i, j in sorted(cands, reverse=True):
        if i not in used_i and j not in used_j:
            used_i.add(i), used_j.add(j), out.append((i, j))
    return out


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def score_extraction(docs: list[dict], results: list[ExtractionResult]) -> dict:
    """Obligation-level scoring of extraction results against (synthetic-by-construction or developer-annotated) truth."""
    det = defaultdict(int)
    typed = defaultdict(int)
    per_cat = {c: defaultdict(int) for c in CATEGORIES}
    dl_ok = dl_n = dl_spur = dl_pred_none_truth_some = 0
    role_ok = role_n = cond_ok = cond_n = 0
    dep_tp = dep_fp = dep_fn = 0
    dm_tp = dm_fp = dm_fn = 0
    asserted = unsupported = ob_total = ob_unsupported = 0
    lat, toks, schema_ok, retries = [], [], [], []
    for d, r in zip(docs, results):
        truth, pred = d["obligations"], r.obligations
        m_det = match_obligations(pred, truth, typed=False)
        m_typ = match_obligations(pred, truth, typed=True)
        det["tp"] += len(m_det)
        det["fp"] += len(pred) - len(m_det)
        det["fn"] += len(truth) - len(m_det)
        typed["tp"] += len(m_typ)
        typed["fp"] += len(pred) - len(m_typ)
        typed["fn"] += len(truth) - len(m_typ)
        tc = {j for _, j in m_typ}
        pc = {i for i, _ in m_typ}
        for i, p in enumerate(pred):
            per_cat[p.task_type]["tp" if i in pc else "fp"] += 1
        for j, t in enumerate(truth):
            if j not in tc:
                per_cat[t["task_type"]]["fn"] += 1
        pmap = {i: j for i, j in m_typ}
        for i, j in m_typ:
            p, t = pred[i], truth[j]
            ph, th = interval_hours(p.due_interval), interval_hours(t["due_interval"])
            if th is not None:
                dl_n += 1
                dl_ok += int(ph is not None and abs(ph - th) < 1e-6)
            elif ph is not None:
                dl_spur += 1
            role_n += 1
            role_ok += int(p.responsible_role == t["role"])
            cond_n += 1
            cond_ok += int((p.conditional_requirements is None) == (t["conditional"] is None) and
                           (t["conditional"] is None or len(_tok(p.conditional_requirements) & _tok(t["conditional"])) / max(len(_tok(t["conditional"])), 1) >= 0.5))
        # dependency edges among matched obligations
        te = {(c, par) for c, t in enumerate(truth) for par in t["depends_on"]}
        pe = set()
        for i, p in enumerate(pred):
            for dep in p.dependencies:
                k = int(dep.rsplit("ob", 1)[1]) if "ob" in dep else -1
                if i in pmap and k in pmap and 0 <= k < len(pred):
                    pe.add((pmap[i], pmap[k]))
                else:
                    pe.add(("x", i, k))
        dep_tp += len(pe & te)
        dep_fp += len(pe - te)
        dep_fn += len(te - pe)
        mt = {j for _, j in m_typ}
        te_m = {(c, par) for (c, par) in te if c in mt and par in mt}
        pe_m = {e for e in pe if len(e) == 2 and e[0] in mt and e[1] in mt}
        dm_tp += len(pe_m & te_m)
        dm_fp += len(pe_m - te_m)
        dm_fn += len(te_m - pe_m)
        for p in pred:
            bad = unsupported_fields(p, d["text"])
            n_ass = 1 + (p.due_interval is not None) + (p.responsible_role is not None) + (p.conditional_requirements is not None)
            asserted += n_ass
            unsupported += len(bad)
            ob_total += 1
            ob_unsupported += int(bool(bad))
        lat.append(r.latency_s)
        toks.append(r.completion_tokens)
        schema_ok.append(r.schema_valid)
        retries.append(r.n_retries)
    lat = np.asarray(lat)
    return {
        "n_docs": len(docs), "n_truth_obligations": int(sum(len(d["obligations"]) for d in docs)), "n_pred_obligations": ob_total,
        "detection": _prf(det["tp"], det["fp"], det["fn"]), "typed": _prf(typed["tp"], typed["fp"], typed["fn"]),
        "category_f1": {c: _prf(v["tp"], v["fp"], v["fn"]) for c, v in per_cat.items()},
        "deadline": {"accuracy_on_matched_with_truth_deadline": dl_ok / dl_n if dl_n else float("nan"), "n": dl_n, "spurious_deadlines": dl_spur},
        "role_accuracy_on_matched": role_ok / role_n if role_n else float("nan"), "conditional_accuracy_on_matched": cond_ok / cond_n if cond_n else float("nan"),
        "dependency": _prf(dep_tp, dep_fp, dep_fn), "dependency_on_matched": _prf(dm_tp, dm_fp, dm_fn),
        "unsupported_field_rate": unsupported / asserted if asserted else 0.0, "obligations_with_unsupported_field_rate": ob_unsupported / ob_total if ob_total else 0.0,
        "schema_valid_rate": float(np.mean(schema_ok)), "mean_retries": float(np.mean(retries)),
        "latency_s": {"mean": float(lat.mean()), "p50": float(np.quantile(lat, 0.5)), "p95": float(np.quantile(lat, 0.95)), "max": float(lat.max())},
        "mean_completion_tokens": float(np.mean(toks)),
    }


def bootstrap_extraction_f1(docs, results, n_boot: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(len(docs), len(docs), replace=True)
        s = score_extraction([docs[i] for i in idx], [results[i] for i in idx])
        vals.append(s["typed"]["f1"])
    return {"typed_f1_lo": float(np.quantile(vals, 0.025)), "typed_f1_hi": float(np.quantile(vals, 0.975))}


# ----------------------------------------------------------------------------------------------- experiment tracking
def git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def hw_brief() -> dict:
    return {"machine": platform.machine(), "system": platform.platform(), "python": platform.python_version(), "byteorder": __import__("sys").byteorder,
            "cpus": __import__("os").cpu_count()}


class ExperimentLog:
    """Append-only JSONL log: one record per experiment run with config, seed, model id, data version, timings, metrics, hardware, git commit."""

    def __init__(self, results_dir: Path, root: Path):
        self.path = results_dir / "experiments.jsonl"
        self.root = root
        results_dir.mkdir(parents=True, exist_ok=True)

    def log(self, name: str, config: dict, metrics: dict, seed: int, model: str | None = None, duration_s: float | None = None, artifacts=None, warnings=None, data_version=None):
        rec = {"experiment": name, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "config": config, "seed": seed, "model": model, "data_version": data_version,
               "duration_s": duration_s, "metrics": metrics, "artifacts": [str(a) for a in (artifacts or [])], "warnings": warnings or [], "hardware": hw_brief(),
               "git_commit": git_commit(self.root)}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)) + "\n")
        return rec


def type_accuracy_on_detected(docs: list[dict], results: list[ExtractionResult]) -> dict:
    """Category assignment accuracy restricted to obligations whose SPAN was detected (type-agnostic IoU match) - comparable to span classification accuracy."""
    ok = n = 0
    for d, r in zip(docs, results):
        for i, j in match_obligations(r.obligations, d["obligations"], typed=False):
            n += 1
            ok += int(r.obligations[i].task_type == d["obligations"][j]["task_type"])
    return {"n_detected": n, "type_accuracy": ok / n if n else float("nan")}
