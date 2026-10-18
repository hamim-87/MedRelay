"""Single source of truth for every number in the deck: read from the archived result files of the executed LinuxONE notebook (no hand-typed metrics)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

EV = Path(__file__).resolve().parent / "evidence"
R = EV / "results"


def load() -> dict:
    F: dict = {}
    hw = json.loads((R / "hardware.json").read_text())
    b = json.loads((R / "benchmarks.json").read_text())
    dv = json.loads((R / "data_validation.json").read_text())
    e3 = json.loads((R / "exp3_summary.json").read_text())
    ew = pd.read_csv(R / "exp3_early_warning.csv")
    ew = ew[(ew.min_lead_h == 24) & (ew["target_recall(val)"] == 0.7)].set_index("system")
    ab = pd.read_csv(R / "exp5_feature_ablation_dense_test.csv").set_index("system")
    e4 = pd.read_csv(R / "exp4_evidence_uncertainty.csv").set_index("injection")
    g = pd.read_csv(R / "exp5_genai_end_to_end_primary.csv").set_index("extractor")
    x1 = pd.read_csv(R / "exp1_extraction_primary.csv")
    mb = pd.read_csv(R / "model_b_comparison.csv")
    mc = pd.read_csv(R / "model_c_comparison.csv")
    ma = json.loads((R / "model_a_summary.json").read_text())
    summ = pd.read_csv(R / "final_summary.csv").set_index("result")["measured value"]

    F["ew_all"] = pd.read_csv(R / "exp3_early_warning.csv"); F["lead_pr"] = pd.read_csv(R / "exp3_summary.csv").set_index("system"); F["hw"] = hw; F["bench"] = b; F["counts"] = dv["counts"]; F["e3"] = e3; F["ew"] = ew; F["ab"] = ab; F["e4"] = e4; F["genai"] = g; F["x1"] = x1
    F["mode"] = b["mode"]
    F["n_test_obl"] = e3["n"]["test_obligations"]; F["n_test_patients"] = e3["n"]["test_patients"]
    # risk model C (dense test) from the ablation table
    F["c_pr_auc"] = float(ab.loc["C_graph_ml", "pr_auc"]); F["b_pr_auc"] = float(ab.loc["B_independent_ml", "pr_auc"])
    m = re.search(r"PR-AUC ([0-9.]+) \[([0-9.]+),([0-9.]+)\]", summ["Follow-up risk (simulated outcomes)"])
    F["c_ci"] = (float(m.group(2)), float(m.group(3)))
    F["noshow_best"] = mc.sort_values("val_pr_auc", ascending=False).iloc[0].to_dict()
    F["noshow_prev"] = float(mc.prevalence.iloc[0])
    # state evaluation
    se = e3["state_eval"]
    F["state_acc_engine"] = se["engine_evidence_aware"]["accuracy"]; F["state_acc_naive"] = se["naive_absence_is_failure"]["accuracy"]
    F["state_acc_final"] = se["engine_final_after_reconciliation"]["accuracy"]
    F["nc_recall_engine"] = se["engine_evidence_aware"]["per_class_recall"]["not_confirmed"]; F["nc_recall_naive"] = se["naive_absence_is_failure"]["per_class_recall"]["not_confirmed"]
    F["missing_n"] = int(e4.loc["missing_completion", "affected"])
    F["missing_naive_overdue"] = float(e4.loc["missing_completion", "naive_declared_overdue"]); F["missing_engine_confirmed_overdue"] = float(e4.loc["missing_completion", "engine_declared_confirmed_overdue"])
    F["dup_ignored"] = int(e4.loc["duplicates", "duplicates_ignored_by_graph"]); F["dup_identical"] = float(e4.loc["duplicates", "engine_state_identical_to_clean_run"])
    # benchmarks
    thr = pd.DataFrame(b["throughput"]); F["thr"] = thr
    F["llm_tps"] = b["llm"]["tokens_per_s"]["mean"]; F["llm_model"] = b["llm"]["model"]; F["llm_mb"] = b["llm"]["model_size_mb"]
    mm = pd.DataFrame(b["nnpa_matmul"]["rows"]); F["mm"] = mm; F["zdnn"] = b["nnpa_matmul"]["zdnn_version"]
    F["queue"] = pd.DataFrame(b["queue"])
    F["mlat"] = pd.DataFrame(b["ml_latency"])
    import nbformat
    nb = nbformat.read(EV / "MedRelay_LinuxONE_End_to_End.executed_full.ipynb" if (EV / "MedRelay_LinuxONE_End_to_End.executed_full.ipynb").exists() else EV / "MedRelay_LinuxONE_End_to_End.executed_fast.ipynb", as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    txt = "\n".join((o.text if o.output_type == "stream" else o.get("data", {}).get("text/plain", "")) for c in code for o in c.get("outputs", []))
    m = re.search(r"notebook wall time: ([0-9.]+) min", txt); F["wall_min"] = float(m.group(1)) if m else None
    F["nb_cells"] = len(code); F["nb_errors"] = sum(any(o.output_type == "error" for o in c.get("outputs", [])) for c in code); F["nb_text"] = txt
    F["n_boot"] = int(e3["paired"]["C_minus_B"]["n_boot"])
    pm = R / "benchmark_peak_memory.csv"
    F["peakmem"] = pd.read_csv(pm) if pm.exists() else None
    F["model_sel"] = json.loads((R / "model_b_summary.json").read_text())["best_per_featureset"]["C_graph"]["model"] if (R / "model_b_summary.json").exists() else None
    return F


if __name__ == "__main__":
    F = load()
    for k, v in F.items():
        if not isinstance(v, (pd.DataFrame, dict)):
            print(k, "=", v)
