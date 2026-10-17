"""Experiment 1 (obligation extraction) and the end-to-end GenAI-extraction ablation.

Evaluation corpora (see eval_sets.py): (i) 40 hand-written, DEVELOPER-annotated texts (not independently reviewed), (ii) template documents with exact
truth by construction using phrasing structures held out from everything the rules/prompts were written against. No human-reviewed gold exists."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .action_classifier import ActionClassifier
from .clinical_extractor import EnsembleExtractor, HybridExtractor, LLMExtractor, RuleExtractor, ValidatedExtractor
from .eval_sets import challenge_docs, generate_template_docs
from .evaluation import bootstrap_extraction_f1, score_extraction
from .event_processor import EventProcessor
from .data_loader import ROOT
from .experiments import Lab
from .llm_backend import CachedBackend, get_backend
from .ontology import CATEGORIES
from . import scenario_lab as sl


def cached_backend(lab: Lab, which: str = "primary") -> CachedBackend:
    be = get_backend(lab.cfg, which)
    return CachedBackend(be, ROOT / "cache" / f"llm_cache_{be.name}.jsonl")


def make_extractors(be, clf=None) -> dict:
    rules = RuleExtractor()
    c2 = ValidatedExtractor(be, max_retries=1, fill=True)
    ex = {"A_rules": rules, "B_llm_raw": LLMExtractor(be), "C_llm_validated": ValidatedExtractor(be, max_retries=1), "C2_llm_validated_filled": c2,
          "E_rules_plus_llm": EnsembleExtractor(rules, ValidatedExtractor(be, max_retries=1, fill=True))}
    if clf is not None:
        ex["D_hybrid_llm_ml"] = HybridExtractor(be, clf, fill=True, max_retries=1)
    return ex


def _row(name, corpus, s, ci):
    return {"extractor": name, "corpus": corpus, "n_docs": s["n_docs"], "det_f1": s["detection"]["f1"], "typed_precision": s["typed"]["precision"], "typed_recall": s["typed"]["recall"],
            "typed_f1": s["typed"]["f1"], "typed_f1_lo": ci["typed_f1_lo"], "typed_f1_hi": ci["typed_f1_hi"], "deadline_acc": s["deadline"]["accuracy_on_matched_with_truth_deadline"],
            "spurious_deadlines": s["deadline"]["spurious_deadlines"], "role_acc": s["role_accuracy_on_matched"], "cond_acc": s["conditional_accuracy_on_matched"],
            "dep_precision": s["dependency"]["precision"], "dep_recall": s["dependency"]["recall"], "dep_f1": s["dependency"]["f1"], "dep_f1_on_matched": s["dependency_on_matched"]["f1"], "dep_matched_pairs": s["dependency_on_matched"]["tp"] + s["dependency_on_matched"]["fn"], "unsupported_field_rate": s["unsupported_field_rate"],
            "oblig_with_unsupported": s["obligations_with_unsupported_field_rate"], "schema_valid": s["schema_valid_rate"], "mean_retries": s["mean_retries"],
            "latency_mean_s": s["latency_s"]["mean"], "latency_p50_s": s["latency_s"]["p50"], "latency_p95_s": s["latency_s"]["p95"], "tokens": s["mean_completion_tokens"]}


def run_exp1(lab: Lab, which: str = "primary", n_chal: int | None = None, n_tpl: int | None = None, include: list[str] | None = None) -> dict:
    clf = ActionClassifier.load(lab.models / "action_classifier.joblib") if (include is None or "D_hybrid_llm_ml" in include) and (lab.models / "action_classifier.joblib").exists() else None
    be = cached_backend(lab, which)
    n_chal = n_chal or lab.p["llm_docs"]["challenge"]
    n_tpl = n_tpl or lab.p["llm_docs"]["template"]
    corpora = {"developer_annotated_challenge": challenge_docs()[:n_chal], "template_heldout_phrasing": generate_template_docs(n_tpl, lab.seed, "heldout")}
    exts = make_extractors(be, clf)
    if include:
        exts = {k: v for k, v in exts.items() if k in include}
    rows, per_cat, results = [], [], {}
    for cname, docs in corpora.items():
        for name, ex in exts.items():
            t = time.time()
            res = [ex.extract(d["text"], d["doc_id"]) for d in docs]
            s = score_extraction(docs, res)
            ci = bootstrap_extraction_f1(docs, res, n_boot=min(lab.p["bootstrap"], 200), seed=5)
            rows.append({**_row(name, cname, s, ci), "model": be.name if name != "A_rules" else "rules", "wall_s": time.time() - t})
            for c, v in s["category_f1"].items():
                per_cat.append({"extractor": name, "corpus": cname, "category": c, "f1": v["f1"], "precision": v["precision"], "recall": v["recall"], "tp": v["tp"], "fn": v["fn"], "fp": v["fp"]})
            results[(name, cname)] = res
            lab.log.log("exp1_extraction", {"extractor": name, "corpus": cname, "n_docs": len(docs), "llm": be.name}, {k: rows[-1][k] for k in ("typed_f1", "det_f1", "deadline_acc", "dep_f1", "unsupported_field_rate", "schema_valid", "latency_mean_s")},
                        lab.seed, model=be.name if name != "A_rules" else "rules", duration_s=rows[-1]["wall_s"], warnings=["no human-reviewed gold; corpora are developer-annotated or synthetic-by-construction"])
    tab = pd.DataFrame(rows)
    cat = pd.DataFrame(per_cat)
    lab.save_csv(f"exp1_extraction_{which}", tab); lab.save_csv(f"exp1_extraction_per_category_{which}", cat)
    return {"table": tab, "per_category": cat, "results": results, "backend": be, "classifier": clf, "corpora": corpora}


# ------------------------------------------------------------------------------------------------ end-to-end ablation
def run_exp5_genai(lab: Lab, which: str = "primary", n_episodes: int | None = None) -> dict:
    """Instruction text -> extractor -> monitored obligations -> evidence-graph replay -> alerts, versus simulated ground-truth outcomes.
    Removing GenAI extraction means obligations the rules cannot extract are never monitored; spurious/mis-timed extractions raise false alerts."""
    n = n_episodes or lab.p["scenario_episodes"]
    clf = ActionClassifier.load(lab.models / "action_classifier.joblib")
    be = cached_backend(lab, which)
    exts = make_extractors(be, clf)
    exts = {k: exts[k] for k in ("A_rules", "C2_llm_validated_filled", "D_hybrid_llm_ml", "E_rules_plus_llm")}
    docs = generate_template_docs(n, lab.seed + 7, "heldout")
    rng = np.random.default_rng(lab.seed + 99)
    episodes = []
    for i, d in enumerate(docs):
        beh = sl.BEHAVIOURS[int(rng.integers(len(sl.BEHAVIOURS)))]
        patient = f"lab:patient-{i:04d}"
        t0 = sl.T0 + float(rng.integers(0, 200)) * 86400
        tobs = sl.true_obligations(d, t0)
        ev, outcome = sl.simulate_events(tobs, patient, d["doc_id"], beh, np.random.default_rng(lab.seed + i), t0)
        episodes.append({"doc": d, "tobs": tobs, "events": ev, "outcome": outcome, "behaviour": beh, "patient": patient, "t0": t0})
    rows = []
    tot_fail = sum(1 for e in episodes for o in e["tobs"] if e["outcome"][o["id"]] == "overdue")
    for name in ["none (no extraction)"] + list(exts):
        agg = {"failures_total": tot_fail, "failures_surfaced": 0, "true_obligations": 0, "monitored": 0, "false_alerts": 0, "alerts_total": 0, "spurious_obligations": 0, "episodes": len(episodes)}
        for e in episodes:
            agg["true_obligations"] += len(e["tobs"])
            if name.startswith("none"):
                continue
            res = exts[name].extract(e["doc"]["text"], e["doc"]["doc_id"])
            stream, info = sl.build_monitoring_stream(e["doc"], res.obligations, e["tobs"], e["events"], e["patient"], e["t0"])
            p = EventProcessor(lab.cfg, ":memory:", scorers={}, enable_alerts=True, log_risk=False, build_nx=False)
            p.run(stream)
            p.finalize()
            al = p.alerts_df()
            reactive = al[al.rule.isin(["DEADLINE_MISSED_CONFIRMED", "COMPLETION_UNVERIFIED", "CONFLICTING_EVIDENCE", "RESULT_NOT_ACKNOWLEDGED"])]
            alerted = set(reactive.oid)
            mp = info["matched"]
            inv = {i: j for j, i in mp.items()}
            agg["monitored"] += len(mp)
            agg["spurious_obligations"] += info["n_extracted"] - len(mp)
            for tob in e["tobs"]:
                if e["outcome"][tob["id"]] == "overdue" and tob["idx"] in mp and f"{e['doc']['doc_id']}:x{mp[tob['idx']]}" in alerted:
                    agg["failures_surfaced"] += 1
            for oid in alerted:
                i = int(oid.rsplit(":x", 1)[1])
                if i not in inv or e["outcome"][e["tobs"][inv[i]]["id"]] == "completed":
                    agg["false_alerts"] += 1
            agg["alerts_total"] += len(alerted)
        agg.update({"extractor": name, "failure_recall": agg["failures_surfaced"] / max(tot_fail, 1), "monitored_fraction": agg["monitored"] / max(agg["true_obligations"], 1),
                    "false_alerts_per_episode": agg["false_alerts"] / len(episodes), "precision_of_alerts": 1 - agg["false_alerts"] / max(agg["alerts_total"], 1) if agg["alerts_total"] else float("nan")})
        rows.append(agg)
    tab = pd.DataFrame(rows)[["extractor", "failures_total", "failures_surfaced", "failure_recall", "monitored_fraction", "spurious_obligations", "false_alerts", "false_alerts_per_episode", "precision_of_alerts", "alerts_total", "episodes"]]
    lab.save_csv(f"exp5_genai_end_to_end_{which}", tab)
    lab.log.log("exp5_genai_ablation", {"episodes": len(episodes), "llm": be.name}, {"rows": tab.to_dict("records")}, lab.seed, model=be.name, warnings=["synthetic scenario-lab episodes"])
    return {"table": tab, "episodes": episodes}
