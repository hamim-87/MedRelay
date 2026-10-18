"""Notebook cells, part A: sections 1-6. Each entry is (kind, source)."""

CELLS = []
md = lambda s: CELLS.append(("markdown", s.strip("\n")))   # noqa: E731
code = lambda s: CELLS.append(("code", s.strip("\n")))     # noqa: E731

md("""
# MedRelay on IBM LinuxONE - Hybrid GenAI + ML + Temporal Evidence Graphs for Clinical Follow-up Obligations

**Notebook mode:** controlled by the environment variable `MEDRELAY_MODE` (`fast` = development subset, `full` = every available record). Run all cells top to bottom from a clean kernel.

## 1. Project introduction and research questions

**Problem.** An instruction being *issued* is not the same as it being *fulfilled*. "Repeat the blood test within 48 hours, have the physician review the result, and contact the patient if further action is needed" is a **chain** of dependent obligations. A recorded blood test says nothing about whether the result was reviewed, or the patient told.

**Architecture (implemented in `src/`, orchestrated here):**

```
clinical text / events -> GenAI extraction (local Granite via llama.cpp) -> schema + evidence-grounding validation -> structured obligations
  -> supervised action classifier (cross-check) -> Temporal Evidence Graph (NetworkX, deterministic state rules)
  -> streaming event matching (SQLite, idempotent) -> calibrated ML risk (unresolved at deadline) -> constraint-based alerts + explanations -> audit log
```

**Research question.** Can dependency-aware temporal evidence graphs plus hybrid AI predict unresolved clinical obligations *earlier and more accurately* than conventional independent-task monitoring? *(A proposed contribution that needs experimental validation - not a claim of being first.)*

**What is real and what is not (read this first).**

| item | status |
|---|---|
| Hardware measurements, latencies, throughput | measured on the machine that executes this notebook (section 2 states its architecture) |
| Follow-up outcomes (completed / overdue / unverified / censored) | **simulated** by a seeded generator from documented source events; results are *performance on simulated follow-up processes only* |
| Action-classification labels | **weak (rule-proposed, "silver")**; **no human-reviewed (gold) labels exist** |
| Obligation-extraction evaluation sets | a **developer-annotated** challenge set (annotator = implementer, not independent) and **synthetic-by-construction** template documents |
| Appointment no-show data | real de-identified appointment records; label = attendance, *not* follow-up completion |
| Everything | synthetic or de-identified; **not clinically validated; recommendations require qualified human review** |
""")

code("""
import os, sys, json, time, warnings, platform
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML, Markdown
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 60); pd.set_option("display.max_colwidth", 100); pd.set_option("display.precision", 3)
from src.data_loader import load_config
cfg = load_config(); MODE = cfg["mode"]
T_NB0 = time.time()
import psutil
_RSS_LOG = ROOT / "logs" / "rss_by_cell.csv"; _RSS_LOG.parent.mkdir(exist_ok=True); _RSS_LOG.write_text("cell,rss_mb,t_s\\n")
def _log_rss(result):                                       # memory of this kernel after every cell (the 4 GB VM has only ~2.7 GB usable)
    with open(_RSS_LOG, "a") as fh:
        fh.write(f"{result.execution_count},{psutil.Process().memory_info().rss / 1e6:.0f},{time.time() - T_NB0:.0f}\\n")
get_ipython().events.register("post_run_cell", _log_rss)
print(f"MedRelay | mode={MODE} | seed={cfg['seed']} | python {platform.python_version()} | machine {platform.machine()} | cwd {ROOT}")
""")

md("## 2. IBM LinuxONE hardware verification")
code("""
from src import benchmarks as bm
hw = bm.hardware_report()
slim = {k: v for k, v in hw.items() if k not in ("lscpu", "free_h", "df_h")}
Path("results").mkdir(exist_ok=True)
Path("results/hardware.json").write_text(json.dumps(slim, indent=2, default=str))
if hw["is_s390x"]:
    display(HTML("<div style='padding:8px;background:#e8f5e9;border-left:5px solid #2e7d32'><b>CONFIRMED:</b> this notebook is executing on IBM Z / LinuxONE "
                 f"(<code>uname -m</code> = <b>s390x</b>, {hw['byteorder']}-endian, hypervisor <b>{hw['hypervisor']}</b>).</div>"))
else:
    display(HTML(f"<div style='padding:8px;background:#ffebee;border-left:5px solid #c62828'><b>WARNING:</b> architecture is {hw['uname_m']}, not s390x. "
                 "Every measurement below comes from THIS machine and is NOT a LinuxONE result.</div>"))
print(hw["uname_a"]); print(); print(hw["lscpu"][:2200]); print("\\n/proc/cpuinfo features:", hw["cpu_features"])
keys = ["uname_m", "byteorder", "nproc", "memory_total_gb", "memory_available_gb", "disk_total_gb", "disk_free_gb", "python", "hypervisor", "has_vxe2", "nnpa_cpu_flag", "nnpa_facility_165_listed", "docker", "podman"]
display(pd.DataFrame({"value": [slim[k] for k in keys]}, index=keys))
print(hw["free_h"]); print(hw["df_h"])
""")
code("""
nn = bm.nnpa_probe()
display(pd.DataFrame({"value": nn}).astype(str))
print("NNPA is PRESENT in the CPU flags/facility list" if nn["cpu_flag_nnpa"] else "NNPA flag not present on this host")
print("Presence is not use: tree/linear models (sklearn, XGBoost) do NOT run on NNPA. A device-level zDNN test is run in section 16.")
""")

md("## 3. Environment and dependency configuration")
code("""
from src.llm_backend import gguf_byteorder, host_byteorder
display(pd.DataFrame([{"package": k, "version": v} for k, v in hw["packages"].items()]))
mods = sorted((ROOT / cfg["llm"]["models_dir"]).glob("*.gguf"))
display(pd.DataFrame([{"model_file": m.name, "size_mb": round(m.stat().st_size / 1e6, 1), "gguf_byteorder": gguf_byteorder(m), "host_byteorder": host_byteorder(),
                       "usable_without_conversion": gguf_byteorder(m) == host_byteorder()} for m in mods]))
import llama_cpp
print("llama.cpp via llama-cpp-python", llama_cpp.__version__); print(llama_cpp.llama_print_system_info().decode())
print("\\nSelected configuration (config.yaml):"); print(open("config.yaml").read())
print("Remote LLM backends:", "ENABLED" if cfg["llm"]["allow_remote"] else "disabled (all GenAI inference is local; no clinical text leaves this machine)")
""")
md("""
**Installation approach (see `docs/linuxone_setup.md`):** no Docker/Podman is required. Scientific Python comes from the distribution's precompiled `s390x` packages (PyPI has no `s390x`
wheel for NumPy/pandas/scikit-learn/pyarrow), `llama-cpp-python` is compiled from source for the z16 instruction set (VXE), Hugging Face GGUF files (little-endian) are converted once to
big-endian with `gguf-convert-endian`. Because `pyarrow` is unavailable, the dataset exports are consumed as gzip CSV.
""")

md("## 4. Dataset loading and integrity validation")
code("""
from src.data_loader import validate_datasets, build_stream, load_stream_truth
t = time.time()
rep = validate_datasets(cfg)
display(pd.DataFrame([{"check": k, "passed": v["ok"], "detail": str(v["detail"])[:120]} for k, v in rep["checks"].items()]))
print("hard failures:", rep["hard_failures"] or "none"); print("warnings:", *rep["warnings"], sep="\\n  - ")
assert not rep["hard_failures"], rep["hard_failures"]
print("\\nrecord counts:", rep["counts"]); print("validation took", round(time.time() - t, 1), "s")
""")
code("""
prov = rep["label_provenance"]
display(pd.DataFrame([{"dataset": "action classification", "label source": "weak rule-proposed (silver) + 622 train-only template (synthetic)", "human-reviewed gold": prov["gold_human_reviewed_action_labels"]},
                      {"dataset": "follow-up risk", "label source": prov["risk_labels"], "human-reviewed gold": 0},
                      {"dataset": "appointment no-show", "label source": prov["noshow_labels"], "human-reviewed gold": "n/a (real attendance record)"},
                      {"dataset": "evidence graph / event stream", "label source": "simulated scenario ground truth (hidden file, never in live payloads)", "human-reviewed gold": 0}]))
print("action label tiers:", prov["action_label_tiers"]); print("action label methods:", prov["action_label_methods"]); print("action sources:", prov["action_sources"])
cb = rep["class_balance"]
display(pd.DataFrame(cb["action"]).T.rename_axis("split")); display(pd.DataFrame(cb["risk_binary"]).T.rename_axis("split").rename(columns={"1": "overdue(1)", "0": "completed(0)"}))
display(pd.DataFrame(cb["risk_outcomes"]).T.rename_axis("split")); display(pd.DataFrame(cb["noshow"]).T.rename_axis("split"))
""")
code("""
from src.risk_predictor import build_graph_from_stream, feature_parity
from src.data_loader import subset_patients, read_source
pats = read_source(cfg, "interim/patients.csv.gz")
keep = subset_patients(cfg, pats)
t = time.time(); G0 = build_graph_from_stream(cfg, keep)
par = feature_parity(cfg, G0, sample=4000, patients=keep)
print("replay-stream graph:", G0.stats()); print("feature parity (graph recomputation vs dataset-pipeline tables):", par, "| %.1f s" % (time.time() - t))
assert par["exact_or_within_tolerance"], par
stream_paths = build_stream(cfg)
from collections import Counter
from src.data_loader import iter_stream
ev_counts, n_deliv, n_dup, n_late = Counter(), 0, 0, 0
from src.evidence_graph import to_epoch as _te
for ev_ in iter_stream(cfg):
    ev_counts[ev_["event_type"]] += 1; n_deliv += 1; n_dup += ev_["delivery_attempt"] > 1; n_late += (_te(ev_["ingest_time_utc"]) - _te(ev_["event_time_utc"])) > 7200
display(pd.DataFrame([ev_counts]).T.rename(columns={0: "deliveries"}).T)
an = pd.read_csv(stream_paths["anomalies"])
print("delivery imperfections injected into the replay stream:", an.anomalies.value_counts().to_dict(), "| deliveries:", n_deliv, "| duplicates:", n_dup, "| delayed >2h:", n_late)
del G0
import gc; gc.collect()
""")

md("## 5. Exploratory data analysis")
code("""
from src.data_loader import load_action, load_risk_hidden, load_noshow, load_interim
from src import visualization as viz
T = load_interim(cfg); A = load_action(cfg); NS = load_noshow(cfg); GT = load_stream_truth(cfg)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
ct = pd.crosstab(T["obligations"].category, T["ground_truth"].set_index("obligation_id").loc[T["obligations"].obligation_id].outcome_label.values)
ct.plot(kind="barh", stacked=True, ax=ax[0], width=.8); ax[0].set_title("Obligations by category and simulated outcome"); ax[0].legend(fontsize=7)
act = pd.DataFrame({sp: A[sp][[c for c in A[sp].columns if c.startswith("y_")]].sum() for sp in ("train", "validation", "test")}); act.index = [i[2:] for i in act.index]
act.plot(kind="barh", ax=ax[1]); ax[1].set_title("Action labels per split (silver)"); ax[1].set_xscale("log")
ns = NS.assign(band=pd.cut(NS.wait_days, [-1, 0, 3, 7, 14, 30, 400])).groupby("band").no_show.agg(["mean", "size"])
ax[2].bar(ns.index.astype(str), ns["mean"]); ax[2].set_title("No-show rate by waiting time (real data)"); ax[2].tick_params(axis="x", rotation=30)
plt.tight_layout(); viz.save(fig, cfg, "eda_overview"); plt.show()
""")
code("""
ob = T["obligations"]
win = (pd.to_datetime(ob.due_time_utc) - pd.to_datetime(ob.created_time_utc)).dt.total_seconds() / 86400
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
ax[0].hist(win.clip(upper=60), bins=40); ax[0].set_title("Deadline window (days, clipped at 60)")
depth = ob.groupby("chain_id").size().value_counts().sort_index(); ax[1].bar(depth.index.astype(str), depth.values); ax[1].set_title("Dependency chain length (obligations per chain)")
src = pd.concat([T["patients"].groupby(["source_dataset", "split"]).size().unstack(fill_value=0)]); src.plot(kind="bar", stacked=True, ax=ax[2]); ax[2].set_title("Patients per source and split"); ax[2].tick_params(axis="x", rotation=25)
plt.tight_layout(); viz.save(fig, cfg, "eda_structure"); plt.show()
print("encounters:", len(T["encounters"]), "| patients:", len(T["patients"]), "| obligations:", len(ob), "| chains:", ob.chain_id.nunique(), "| source events:", len(T["events"]))
print("share of obligations whose first step is a documented source event vs rule-extracted vs simulated:")
display(ob.groupby("seed_kind").size().rename("obligations").to_frame().T)
display(HTML("<i>Time precision: Synthetic Hospital dates are month-granular (all on the 15th); sub-day times are imputed. Synthea has clock times.</i>"))
""")

md("## 6. Clinical obligation schema")
code("""
import json as _json
from src.clinical_extractor import ExtractedObligation, EvidenceSpan
from src.ontology import STEP_SPEC, COMPLETION_EVENT, State, NODE_TYPES, EDGE_TYPES
sch = ExtractedObligation.model_json_schema()
print(_json.dumps({k: v for k, v in sch["properties"].items()}, indent=1)[:2600])
display(pd.DataFrame([{"workflow step": s, "category": c, "role": r, "completed only by event": COMPLETION_EVENT[s]} for s, (c, r) in STEP_SPEC.items()]))
display(pd.DataFrame([{"state": s.value} for s in State]).T)
print("graph node types:", NODE_TYPES); print("graph edge types:", EDGE_TYPES)
""")
code("""
from pydantic import ValidationError
bad_examples = [dict(task_type="laboratory_test", due_interval="two_days"), dict(task_type="telepathy"), dict(task_type="imaging", responsible_role="wizard")]
for kw in bad_examples:
    try:
        ExtractedObligation(obligation_id="x", description="x test", evidence_span=EvidenceSpan(start=0, end=1, text="x"), source_note_id="n", confidence=.5, **kw)
    except ValidationError as e:
        print("REJECTED:", kw, "->", e.errors()[0]["msg"][:110])
""")
