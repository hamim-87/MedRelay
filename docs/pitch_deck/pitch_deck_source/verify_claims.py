"""Check every number or phrase typed into the deck text against the archived evidence. Exits non-zero if a claim is not supported."""
import json, re, sys
from pathlib import Path
import nbformat, pandas as pd
import facts

F = facts.load(); EV = facts.EV; R = facts.R
nb = nbformat.read(EV / "MedRelay_LinuxONE_End_to_End.executed_fast.ipynb", as_version=4)
code = [c for c in nb.cells if c.cell_type == "code"]
out_txt = "\n".join((o.text if o.output_type == "stream" else o.get("data", {}).get("text/plain", "")) for c in code for o in c.get("outputs", []))
md_txt = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
checks = []
def chk(name, ok, detail=""): checks.append((name, bool(ok), detail))

ew = F["ew"]; c = F["counts"]; e3 = F["e3"]
chk("56 code cells", len(code) == 56, len(code))
chk("0 cells with errors", sum(any(o.output_type == "error" for o in x.get("outputs", [])) for x in code) == 0)
chk("slide 3/10: median alert lead of graph model C = 72 h", ew.loc["C_graph_ml", "median_lead_h"] == 72, ew.loc["C_graph_ml", "median_lead_h"])
chk("slide 3: graph model has the highest lead among A/B/C", ew.loc["C_graph_ml", "median_lead_h"] > max(ew.loc["A_deadline_only", "median_lead_h"], ew.loc["B_independent_ml", "median_lead_h"]))
chk("slide 5: 103,627 nodes / 223,925 edges", (c["graph_nodes"], c["graph_edges"]) == (103627, 223925))
chk("slide 5: 49 features", int(F["ab"].loc["C_graph_ml", "n_features"]) == 49)
groups = ("graph_dependency", "graph_temporal", "own_events", "workload_history", "evidence", "task")
chk("slide 5: 6 feature groups in executed notebook", all(g in out_txt for g in groups))
chk("slide 5: 8 assessment snapshots per task (notebook text)", re.search(r"dense snapshots\*\* at 8 points", md_txt) is not None)
chk("slide 5: dependency graph acyclic validated", json.loads((R / "data_validation.json").read_text())["checks"]["dependency_graph_acyclic"]["ok"])
chk("slide 5: patient-grouped split / independence validated", json.loads((R / "data_validation.json").read_text())["checks"]["patient_level_independence"]["ok"])
dv = json.loads((R / "data_validation.json").read_text())
chk("slide 5: validation warns there are no human-reviewed (gold) action labels", any("no human-reviewed (gold)" in w for w in dv["warnings"]), dv["warnings"][:1])
chk("slide 6: VXE = 1 in llama.cpp build info", "VXE = 1" in out_txt)
chk("slide 6: NNPA flag + facility 165", F["hw"]["nnpa_cpu_flag"] and F["hw"]["nnpa_facility_165_listed"])
chk("slide 6: no docker/podman", F["hw"]["docker"] is None and F["hw"]["podman"] is None)
chk("slide 6: z/VM 7.3.0, s390x, big-endian, 2 vCPU", F["hw"]["hypervisor"] == "z/VM 7.3.0" and F["hw"]["uname_m"] == "s390x" and F["hw"]["byteorder"] == "big" and F["hw"]["nproc"] == 2)
q = F["queue"].set_index("scenario")
chk("slide 6: backpressure scenario loses no events", q.loc["2x capacity, block (backpressure)", "event_loss"] == 0)
chk("slide 7: max-score PR-AUC 0.981 (A) vs 0.969 (C)", abs(e3 and pd.read_csv(R / "exp3_summary.csv").set_index("system").loc["A_deadline_only", "pr_auc_max_score"] - 0.981) < 5e-4 and abs(pd.read_csv(R / "exp3_summary.csv").set_index("system").loc["C_graph_ml", "pr_auc_max_score"] - 0.969) < 5e-4)
chk("slide 7: bootstrap 100 resamples", e3["paired"]["C_minus_B"]["n_boot"] == 100)
chk("slide 7: A alerts on 167 of 283", (int(ew.loc["A_deadline_only", "alerts"]), int(ew.loc["A_deadline_only", "n_pos"] + ew.loc["A_deadline_only", "n_neg"])) == (167, 283))
chk("slide 7: GenAI e2e 17 failures, 24 episodes", (int(F["genai"].failures_total.iloc[0]), int(F["genai"].episodes.iloc[0])) == (17, 24))
chk("slide 8: audit trail states appear verbatim in notebook", all(x in out_txt for x in ("censored_not_yet_assessable", "overdue_without_verified_completion", "R5_deadline_passed_open_confirmed", "R6_deadline_passed_awaiting_late_evidence")))
chk("slide 8: reconciliation verified line", "RECONCILIATION VERIFIED" in out_txt)
chk("slide 8: demo note text verbatim", "Repeat the CBC within 48 hours. The treating physician must review the result." in out_txt)
chk("slide 8: obligations ob0..ob3 (PT48H, P2W, abnormal)", all(x in out_txt for x in ("PT48H", "P2W", "if the result is abnormal")))
chk("slide 9: states restored identically 1.000", "stored states restored identically: 1.000" in out_txt)
chk("slide 9: 406 duplicates ignored, identical to clean run", F["dup_ignored"] == 406 and F["dup_identical"] == 1.0)
chk("slide 2/10: 53 undocumented tasks; naive 100% overdue; engine 0 confirmed", F["missing_n"] == 53 and F["missing_naive_overdue"] == 1.0 and F["missing_engine_confirmed_overdue"] == 0.0)
chk("slide 7: 'naive never recognises an unverified task' (recall 0.0)", F["nc_recall_naive"] == 0.0)
chk("fast mode run", F["mode"] == "fast")
bad = [x for x in checks if not x[1]]
for n, ok, d in checks:
    print(("PASS " if ok else "FAIL ") + n + (f"  [{d}]" if d != "" else ""))
print(f"\n{len(checks) - len(bad)}/{len(checks)} claims verified")
sys.exit(1 if bad else 0)
