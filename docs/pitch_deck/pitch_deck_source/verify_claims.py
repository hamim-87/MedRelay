"""Check every number or phrase typed into the deck (and the video narration) against the archived evidence. Exits non-zero if a claim is not supported."""
import json
import re
import sys

import pandas as pd

import facts

F = facts.load(); EV = facts.EV; R = facts.R
txt = F["nb_text"]
import nbformat
nbf = EV / "MedRelay_LinuxONE_End_to_End.executed_full.ipynb"
nb = nbformat.read(nbf if nbf.exists() else EV / "MedRelay_LinuxONE_End_to_End.executed_fast.ipynb", as_version=4)
md_txt = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
dv = json.loads((R / "data_validation.json").read_text())
checks = []


def chk(name, ok, detail=""):
    checks.append((name, bool(ok), detail))


ew, e3, c, ab = F["ew"], F["e3"], F["counts"], F["ab"]
C, A, RM, B = "C_graph_ml", "A_deadline_only", "A_reminder_24h", "B_independent_ml"
chk("run mode is full", F["mode"] == "full", F["mode"])
chk("notebook: 56 code cells, 0 with errors", F["nb_cells"] == 56 and F["nb_errors"] == 0, (F["nb_cells"], F["nb_errors"]))
chk("notebook wall time parsed (251 min)", F["wall_min"] and abs(F["wall_min"] - 251.2) < 0.5, F["wall_min"])
chk("slides 3/10: graph model has the longest median lead (72 h)", ew.loc[C, "median_lead_h"] == 72 and ew.loc[C, "median_lead_h"] > max(ew.loc[x, "median_lead_h"] for x in (A, RM, B)))
chk("slides 7/9/10: graph model has the lowest false-alert rate of A, reminder and B", ew.loc[C, "false_alert_rate"] < min(ew.loc[x, "false_alert_rate"] for x in (A, RM, B)),
    {x: round(ew.loc[x, "false_alert_rate"], 3) for x in (C, A, RM, B)})
chk("slides 7/narration: graph model recall is lower than the other three", ew.loc[C, "recall"] < min(ew.loc[x, "recall"] for x in (A, RM, B)))
chk("slide 7: reminder has similar false-alert rate (within 2 points) and 24 h lead", abs(ew.loc[RM, "false_alert_rate"] - ew.loc[C, "false_alert_rate"]) < 0.02 and ew.loc[RM, "median_lead_h"] == 24)
chk("slide 7: graph vs independent ML significant on the dense test (CI excludes 0)", ab.loc[B, "diff_ci_hi"] < 0, (ab.loc[B, "diff_ci_lo"], ab.loc[B, "diff_ci_hi"]))
chk("slide 7: temporal features significant", ab.loc["C_no_temporal", "diff_ci_hi"] < 0)
chk("slide 7: evidence features not significant (CI spans 0)", ab.loc["C_no_evidence", "diff_ci_lo"] <= 0 <= ab.loc["C_no_evidence", "diff_ci_hi"] + 1e-9, (ab.loc["C_no_evidence", "diff_ci_lo"], ab.loc["C_no_evidence", "diff_ci_hi"]))
chk("slide 7: alert-level C-A and C-B differences not significant (CI spans 0)", all(e3["paired"][k]["lo"] <= 0 <= e3["paired"][k]["hi"] for k in ("C_minus_A", "C_minus_B")))
chk("slide 7: bootstrap resamples = 300", F["n_boot"] == 300)
chk("slide 7: lead-restricted PR-AUC of C beats A and B at 72 h ahead", F["lead_pr"].loc[C, "pr_auc_at_T-72h"] > max(F["lead_pr"].loc[A, "pr_auc_at_T-72h"], F["lead_pr"].loc[B, "pr_auc_at_T-72h"]))
chk("slide 7: max-score PR-AUC of A is within 0.01 of C", abs(F["lead_pr"].loc[A, "pr_auc_max_score"] - F["lead_pr"].loc[C, "pr_auc_max_score"]) < 0.01)
chk("slides 2/10/narration: 716 done-but-undocumented tasks; naive 100% overdue; engine 0 confirmed", F["missing_n"] == 716 and F["missing_naive_overdue"] == 1.0 and F["missing_engine_confirmed_overdue"] == 0.0)
chk("slide 7: naive rule recognises no unverified task (recall 0.0); engine > 0.95", F["nc_recall_naive"] == 0.0 and F["nc_recall_engine"] > 0.95, F["nc_recall_engine"])
chk("slides 7: state accuracy engine > naive; final = 1.000", F["state_acc_engine"] > F["state_acc_naive"] and round(F["state_acc_final"], 3) == 1.0)
chk("slide 9: 4,895 duplicates ignored, states identical to clean run", F["dup_ignored"] == 4895 and F["dup_identical"] == 1.0)
chk("slide 9: states restored identically 1.000 after rebuild from the event log", "stored states restored identically: 1.000" in txt)
chk("slide 5: 103,627 nodes / 223,925 edges; counts", (c["graph_nodes"], c["graph_edges"], c["patients"], c["obligations"]) == (103627, 223925, 3284, 24296))
chk("slide 5: 49 features", int(ab.loc[C, "n_features"]) == 49)
chk("slide 5: 6 feature groups in executed notebook", all(g in txt for g in ("graph_dependency", "graph_temporal", "own_events", "workload_history", "evidence", "task")))
chk("slide 5: 8 assessment snapshots per task (notebook text)", re.search(r"dense snapshots\*\* at 8 points", md_txt) is not None)
chk("slide 5: dependency graph acyclic + patient independence validated", dv["checks"]["dependency_graph_acyclic"]["ok"] and dv["checks"]["patient_level_independence"]["ok"])
chk("slide 5/README: validation warns there are no human-reviewed gold labels", any("no human-reviewed (gold)" in w for w in dv["warnings"]))
chk("slide 4/5: selected risk model family is random forest", (F["model_sel"] or "").startswith("rf/"), F["model_sel"])
chk("slide 6: VXE = 1 in llama.cpp build info", "VXE = 1" in txt)
chk("slide 6: NNPA flag + facility 165; no docker/podman", F["hw"]["nnpa_cpu_flag"] and F["hw"]["nnpa_facility_165_listed"] and F["hw"]["docker"] is None and F["hw"]["podman"] is None)
chk("slide 6: z/VM 7.3.0, s390x, big-endian, 2 vCPU", F["hw"]["hypervisor"] == "z/VM 7.3.0" and F["hw"]["uname_m"] == "s390x" and F["hw"]["byteorder"] == "big" and F["hw"]["nproc"] == 2)
q = F["queue"].set_index("scenario")
chk("slide 6: backpressure scenario loses no events", q.loc["2x capacity, block (backpressure)", "event_loss"] == 0)
chk("slide 6: bounded queue sheds events at 2x and 4x", q.loc["2x capacity, drop (bounded queue 200)", "loss_rate"] > 0.3 and q.loc["4x capacity, drop (bounded queue 200)", "loss_rate"] > 0.6)
thr = F["thr"]
chk("slide 6: graph+states throughput 12.4k at 100k events and 20.6k max", abs(thr[thr.config.str.startswith("graph+states")].events_per_s.max() - 20622) < 5 and thr[(thr.config.str.startswith("graph+states")) & (thr.workload_events == 100000)].events_per_s.iloc[0] > 12000)
chk("slide 6: ML-scoring throughput 88 to 122 events/s (batch-64 configurations)", 80 < thr[thr.config.str.contains("ML risk") & thr.config.str.contains("batch 64")].events_per_s.min() < 95 and 115 < thr[thr.config.str.contains("ML risk") & thr.config.str.contains("batch 64")].events_per_s.max() < 125)
chk("slide 6: zDNN speed-up about 339x at 1024^3", abs(F["mm"][F["mm"].M == 1024].speedup_total_vs_numpy.iloc[0] - 339) < 1)
chk("slide 6: LLM 15.0 tokens/s", abs(F["llm_tps"] - 15.0) < 0.1)
if F["peakmem"] is not None:
    pm = F["peakmem"]
    chk("slide 6: per-workload peak memory measured (VmHWM) for 100k ML run", ((pm.workload_events == 100000) & pm.config.str.contains("ML risk") & pm.peak_rss_mb.notna()).any())
chk("slide 7: GenAI end to end: rules + LLM surfaced more failures than rules only (45 vs 38 of 67, 120 episodes)", (int(F["genai"].loc["E_rules_plus_llm", "failures_surfaced"]), int(F["genai"].loc["A_rules", "failures_surfaced"]), int(F["genai"].failures_total.iloc[0]), int(F["genai"].episodes.iloc[0])) == (45, 38, 67, 120))
chk("slide 7: test set size (812 patients, 4,605 obligations)", (F["n_test_patients"], F["n_test_obl"]) == (812, 4605))
chk("slide 8: audit trail states appear verbatim in notebook", all(x in txt for x in ("censored_not_yet_assessable", "overdue_without_verified_completion", "R5_deadline_passed_open_confirmed", "R6_deadline_passed_awaiting_late_evidence")))
chk("slide 8: reconciliation verified line + demo note text + obligations", "RECONCILIATION VERIFIED" in txt and "Repeat the CBC within 48 hours. The treating physician must review the result." in txt and all(x in txt for x in ("PT48H", "P2W", "if the result is abnormal")))
bad = [x for x in checks if not x[1]]
for n, ok, d in checks:
    print(("PASS " if ok else "FAIL ") + n + (f"  [{d}]" if d != "" else ""))
print(f"\n{len(checks) - len(bad)}/{len(checks)} claims verified")
sys.exit(1 if bad else 0)
