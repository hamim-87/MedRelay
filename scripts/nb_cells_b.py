"""Notebook cells, part B: sections 7-11."""

CELLS = []
md = lambda s: CELLS.append(("markdown", s.strip("\n")))   # noqa: E731
code = lambda s: CELLS.append(("code", s.strip("\n")))     # noqa: E731

md("""
## 7. GenAI obligation extraction

A local **IBM Granite 4.0** instruction model (GGUF, run with llama.cpp on this machine; Apache-2.0) converts text into schema-validated obligations. Four extraction strategies are compared:

| id | strategy |
|---|---|
| **A** | deterministic rule-based clause decomposition (no ML) |
| **B** | small local GenAI, grammar-constrained JSON, accepted after schema parsing only |
| **C / C2** | B + validation: evidence span must be verbatim in the source, unsupported deadlines/roles/conditionals are removed, dependencies must be acyclic; **C2** also fills missing deadline/role/dependency fields *only through deterministic parsers over the cited span* |
| **E** | rules + validated GenAI, overlapping obligations merged (rule fields win) |

**Evaluation data (no human-reviewed gold exists):** (i) 40 **developer-annotated** instruction texts (annotator = implementer; not independent), (ii) **template documents with exact ground truth by construction** whose sentence structures are *held out* from everything the rules and prompts were written against.
""")
code("""
from src.experiments import Lab
from src.exp_extraction import run_exp1, cached_backend, make_extractors
from src.scenario_lab import DEMO_TEXT
from src.clinical_extractor import RuleExtractor
lab = Lab(cfg)
print("fast/full parameters:", lab.p)
SPEC_TEXT = ("Patient should undergo repeat blood testing in two days. Results must be reviewed by the treating physician and follow-up arranged if abnormalities persist.")
be = cached_backend(lab, "primary")
print(f"GenAI model: {be.name} | size {be.size_bytes/1e6:.0f} MB | threads {be.n_threads} | load {be.load_s:.2f} s")
exts = make_extractors(be)
def show(res):
    rows = [{"id": o.obligation_id.split(":")[-1], "task_type": o.task_type, "due": o.due_interval, "role": o.responsible_role, "depends_on": [d.split(":")[-1] for d in o.dependencies],
             "conditional": o.conditional_requirements, "evidence_span": o.evidence_span.text[:70], "grounded": o.evidence_span.grounded, "missing": o.missing_information} for o in res.obligations]
    return pd.DataFrame(rows)
for name in ("A_rules", "B_llm_raw", "C2_llm_validated_filled", "E_rules_plus_llm"):
    r = exts[name].extract(SPEC_TEXT, "spec")
    print(f"--- {name}: {len(r.obligations)} obligations | latency {r.latency_s:.1f}s | schema_valid={r.schema_valid} | rejected={[x['reason'] for x in r.rejected]} | repairs={[(x.get('field'), x['action']) for x in r.repairs]}")
    display(show(r))
""")
code("""
t = time.time()
E1 = run_exp1(lab, "primary", include=["A_rules", "B_llm_raw", "C_llm_validated", "C2_llm_validated_filled", "E_rules_plus_llm"])
cols = ["extractor", "corpus", "n_docs", "typed_precision", "typed_recall", "typed_f1", "typed_f1_lo", "typed_f1_hi", "det_f1", "deadline_acc", "spurious_deadlines", "role_acc", "cond_acc",
        "dep_f1_on_matched", "dep_matched_pairs", "unsupported_field_rate", "oblig_with_unsupported", "schema_valid", "latency_mean_s", "latency_p95_s"]
display(E1["table"][cols]); print("Experiment 1 took", round(time.time() - t), "s | LLM cache hits/misses:", E1["backend"].hits, E1["backend"].misses)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
for k, (col, ttl) in enumerate((("typed_f1", "Obligation F1 (task type + span)"), ("unsupported_field_rate", "Unsupported-fact rate (lower is better)"), ("latency_mean_s", "Latency per document (s)"))):
    pv = E1["table"].pivot(index="extractor", columns="corpus", values=col); pv.plot(kind="bar", ax=ax[k]); ax[k].set_title(ttl); ax[k].tick_params(axis="x", rotation=30); ax[k].legend(fontsize=7)
plt.tight_layout(); viz.save(fig, cfg, "exp1_extraction"); plt.show()
""")
code("""
cat = E1["per_category"]
display(cat[cat.corpus == "developer_annotated_challenge"].pivot(index="category", columns="extractor", values="f1").round(2).assign(support=cat[(cat.corpus == "developer_annotated_challenge") & (cat.extractor == "A_rules")].set_index("category").eval("tp + fn")))
display(cat[cat.corpus == "template_heldout_phrasing"].pivot(index="category", columns="extractor", values="f1").round(2).assign(support=cat[(cat.corpus == "template_heldout_phrasing") & (cat.extractor == "A_rules")].set_index("category").eval("tp + fn")))
# what validation actually removed from raw GenAI output
rej, rep_ = {}, {}
for (name, corpus), res in E1["results"].items():
    if name == "C_llm_validated":
        for r in res:
            for x in r.rejected: rej[x["reason"]] = rej.get(x["reason"], 0) + 1
            for x in r.repairs: rep_[f"{x.get('field')}:{x['action']}"] = rep_.get(f"{x.get('field')}:{x['action']}", 0) + 1
print("validation rejected (obligations):", rej); print("validation repaired (fields):", rep_)
""")
code("""
big_ok = (ROOT / cfg["llm"]["models_dir"] / cfg["llm"]["large"].replace(".gguf", ".be.gguf")).exists() or (ROOT / cfg["llm"]["models_dir"] / "original_le" / cfg["llm"]["large"]).exists()
if big_ok:
    nl = lab.p["llm_large_docs"]
    t = time.time(); E1L = run_exp1(lab, "large", n_chal=nl["challenge"], n_tpl=nl["template"], include=["A_rules", "B_llm_raw", "C2_llm_validated_filled", "E_rules_plus_llm"])
    display(E1L["table"][["extractor", "model", "corpus", "n_docs", "typed_precision", "typed_recall", "typed_f1", "det_f1", "deadline_acc", "unsupported_field_rate", "schema_valid", "latency_mean_s", "latency_p95_s"]])
    print("larger-model comparison took", round(time.time() - t), "s on a SMALLER sample (n_docs above) because generation is ~5x slower")
    from src.llm_backend import release_backend
    release_backend(cfg, "large")                      # the 1B model held ~1.8 GB of the VM's ~2.7 GB usable RAM for the rest of the notebook
    print("larger model released from memory; kernel RSS now %.0f MB" % (psutil.Process().memory_info().rss / 1e6))
else:
    print("larger local model not installed; skipping the upper-bound comparison")
""")
code("""
# Model selection from MEASURED quality (not assumed): average typed-F1 over both corpora, restricted to GenAI-based extractors, plus latency.
sel = E1["table"][E1["table"].extractor.isin(["B_llm_raw", "C2_llm_validated_filled", "E_rules_plus_llm"])].groupby("extractor")[["typed_f1", "unsupported_field_rate", "latency_mean_s"]].mean()
display(sel)
best_ext = sel.typed_f1.idxmax(); print("highest mean typed F1 among GenAI-based extractors (350M):", best_ext)
rules_f1 = E1["table"][E1["table"].extractor == "A_rules"].typed_f1.mean()
print(f"rule-based baseline mean typed F1 = {rules_f1:.3f}; best GenAI-based = {sel.typed_f1.max():.3f}")
print("DECISION: Granite-4.0-350M (Q8_0) alone is too weak to replace the rules (it under-extracts, omits deadlines and fabricates roles). It is kept only behind the validator and in the rules+GenAI ensemble (E), which adds the "
      "obligations the rules miss while every field stays evidence-grounded. The 1B model (if present above) is the measured upper-bound comparison.")
SELECTED_EXTRACTOR = "E_rules_plus_llm"
""")

md("""
## 8. ML clinical action classification (Model A)

TF-IDF (word + character n-grams) with **logistic regression**, **linear SVM** and a **compact neural net** (TF-IDF -> SVD -> MLP). Hyper-parameters are chosen by random search on *train -> validation*
macro-F1 (budget in `config.yaml`); the test split is used once. Labels are **silver** (rule-proposed). `challenge_span_*` columns score the same classifier on **developer-annotated** obligation spans,
a different label source, to show how much the silver agreement overstates correctness.
""")
code("""
from src.experiments import model_a_action_classifier
t = time.time(); A_res = model_a_action_classifier(lab); print("Model A took", round(time.time() - t), "s")
display(A_res["comparison"].round(3)); print("best family:", A_res["best_kind"], A_res["best_params"])
display(A_res["per_class_test"].round(3)); print(A_res["synthetic_augmentation"]); print(A_res["label_provenance"])
from IPython.display import Image
for nm in ("model_a_confusion_silver", "model_a_confusion_spans"):
    display(Image(filename=str(ROOT / "figures" / f"{nm}.png")))
""")
code("""
from src.evaluation import type_accuracy_on_detected
rows = []
chal = E1["corpora"]["developer_annotated_challenge"]
for (name, corpus), res in E1["results"].items():
    if corpus == "developer_annotated_challenge":
        rows.append({"system": f"extraction {name}", **type_accuracy_on_detected(chal, res)})
rows.append({"system": f"ML classifier ({A_res['best_kind']}) on the TRUE spans", "n_detected": int(sum(len(d['obligations']) for d in chal)),
             "type_accuracy": float(A_res["comparison"].set_index("model").loc[A_res["best_kind"], "challenge_span_accuracy"])})
display(pd.DataFrame(rows).round(3))
print("Fair-comparison note: both rows score the CATEGORY assigned to a span against developer annotations. The classifier is given the gold span; the extractors must also find the span "
      "(restricted here to spans they detected), so this favours neither strongly. Silver-test agreement is NOT correctness.")
""")

md("""
## 9. Temporal evidence graph construction

Node types (Patient, Encounter, ClinicalNote, Obligation, Task, Result, Review, Acknowledgment, Notification, EvidenceEvent, ResponsibleRole), edge types (EXTRACTED_FROM, DEPENDS_ON, ASSIGNED_TO,
EVIDENCED_BY, COMPLETED_BY, REQUIRES_REVIEW, CONTRADICTS, BELONGS_TO). Workflow states come from deterministic rules R1-R7 that cite their evidence. The cells below *demonstrate* each required semantic on the scripted episode.
""")
code("""
from src.event_processor import EventProcessor
from src.evidence_graph import TemporalEvidenceGraph, iso, to_epoch
from src.scenario_lab import demo_events, demo_obligations, T0
from src import visualization_demo as vd
H = 3600.0
PID = "demo:patient-001"
gp = EventProcessor(cfg, ":memory:", scorers={}, build_nx=True)
gp.run(demo_events("delayed_ack")); gp.finalize()
g = gp.graph
print("graph:", g.stats())
from collections import Counter
print("node types:", dict(Counter(d["ntype"] for _, d in g.G.nodes(data=True)))); print("edge types:", dict(Counter(d["etype"] for _, _, d in g.G.edges(data=True))))
for when, hours in (("after the lab test (+31h)", 31), ("after deadline of the review, tracker open (+122h)", 122), ("after the delayed acknowledgment is documented (+152h)", 152)):
    now = T0 + hours * H
    print("\\n", when); display(pd.DataFrame([{"obligation": o.split(":")[-1], "step": g.obs[o].step_type, **{k: v for k, v in dict(state=g.derive_state(o, now, live=True).state.value, rule=g.derive_state(o, now, live=True).rule,
                                         evidence=g.derive_state(o, now, live=True).evidence_ids).items()}} for o in sorted(g.obs)]))
""")
code("""
fig = vd.plot_evidence_graph(gp, PID, now=T0 + 31 * H, title="Graph at +31h: the TEST is completed with evidence, the chain is NOT"); viz.save(fig, cfg, "graph_t31"); plt.show()
fig = vd.plot_evidence_graph(gp, PID, now=T0 + 152 * H, title="Graph at +152h: delayed acknowledgment (dotted red) reconciled the review obligation"); viz.save(fig, cfg, "graph_t152"); plt.show()
""")
code("""
# semantic checks, each executed live
from tests.conftest import mk_ev, mk_ob   # test helpers (synthetic records)
t0 = to_epoch("2024-03-04T09:00:00Z")
g1 = TemporalEvidenceGraph(12)
g1.ingest(mk_ob("o0", "lab_collected", t0, t0 + 48 * H, [], pos=0, n=3)); g1.ingest(mk_ob("o1", "result_reviewed", t0, t0 + 120 * H, ["o0"], pos=1, n=3)); g1.ingest(mk_ob("o2", "patient_notified", t0, t0 + 192 * H, ["o1"], pos=2, n=3))
g1.ingest(mk_ev("e1", "o0", "lab_collected", t0 + 10 * H))
checks = []
checks.append(("completed test does NOT complete the chain", [g1.derive_state(o, t0 + 20 * H).state.value for o in ("o0", "o1", "o2")]))
g1.ingest(mk_ev("e2", "o0", "result_posted", t0 + 14 * H)); g1.ingest(mk_ev("e3", "o1", "result_opened", t0 + 20 * H))
checks.append(("result available + opened is NOT an acknowledgment", (g1.derive_state("o1", t0 + 30 * H).state.value, g1.features("o1", t0 + 30 * H)["ack_state"])))
g1.ingest(mk_ev("e4", "o1", "result_acknowledged", t0 + 40 * H)); checks.append(("reviewed result does NOT notify the patient", (g1.derive_state("o1", t0 + 50 * H).state.value, g1.derive_state("o2", t0 + 50 * H).state.value)))
r = g1.ingest(mk_ev("e4", "o1", "result_acknowledged", t0 + 40 * H, t_ingest=t0 + 60 * H)); checks.append(("duplicate delivery ignored", (r.duplicate, g1.n_duplicates)))
g2 = TemporalEvidenceGraph(12); g2.ingest(mk_ob("a", "lab_collected", t0, t0 + H, ["b"])); rr = g2.ingest(mk_ob("b", "result_reviewed", t0, t0 + 2 * H, ["a"]))
checks.append(("cyclic dependency rejected", (rr.rejected, g2.rejected_cycles)))
g3 = TemporalEvidenceGraph(12); g3.ingest(mk_ob("x", "lab_collected", t0, t0 + 48 * H, [])); d = t0 + 48 * H
checks.append(("no evidence: grace -> unverified (not 'overdue')", [g3.derive_state("x", d + h * H).state.value for h in (1, 13)]))
g3.ingest(mk_ev("s", "x", "task_status_snapshot", d + H, details={"status": "open"})); checks.append(("tracker confirms open -> overdue", g3.derive_state("x", d + 2 * H).state.value))
g3.ingest(mk_ev("late", "x", "lab_collected", t0 + 40 * H, t_ingest=t0 + 90 * H)); checks.append(("late-documented completion reconciles (live view)", (g3.derive_state("x", d + 10 * H, live=True).state.value, g3.derive_state("x", d + 50 * H, live=True).state.value)))
g3.ingest(mk_ev("c", "x", "conflicting_record", t0 + 45 * H, ver="unverified")); checks.append(("contradictory evidence", g3.derive_state("x", d + 50 * H, live=True).state.value))
display(pd.DataFrame(checks, columns=["property", "observed"]))
""")

md("## 10. Follow-up risk prediction (Model B) - Experiment 2")
md("""
Target: probability that an obligation **remains unresolved at its deadline**, from features available at the assessment time only (feature code = the evidence graph; events after the cutoff are invisible).
Training uses **dense snapshots** at 8 points across each obligation's window (the dataset pipeline's single 20-80 % assessment point is also reported as `orig_protocol_*`), only for obligations still open at the snapshot.
**Unverified and censored obligations are never treated as failures.** Families: logistic regression, HistGradientBoosting, random forest, XGBoost; feature sets: *task-only*, *B independent tasks* (no dependency/evidence-graph features), *C graph-enhanced*.
All labels are **simulated** follow-up outcomes.
""")
code("""
from src.experiments import model_b_followup_risk
t = time.time(); B_res = model_b_followup_risk(lab); print("Model B took", round(time.time() - t), "s")
cols = ["model", "features", "val_pr_auc", "test_pr_auc", "test_pr_auc_ci_lo", "test_pr_auc_ci_hi", "test_roc_auc", "test_brier", "test_ece", "test_brier_uncalibrated", "test_ece_uncalibrated", "orig_protocol_test_pr_auc"]
display(B_res["comparison"][cols].round(3)); display(B_res["baselines"].round(3))
print("best per feature set:", B_res["summary"]["best_per_featureset"]); print("paired patient-bootstrap differences in PR-AUC:", B_res["summary"]["paired_bootstrap"])
display(B_res["thresholds"].round(3)); display(Image(filename=str(ROOT / "figures" / "model_b_calibration.png")))
""")
code("""
from src.risk_predictor import group_permutation_importance, binary
from src.ontology import FEATURE_GROUPS
te, yte = binary(lab.dense()["test"])
mC = B_res["best"]["C_graph"][0]
gi = group_permutation_importance(mC, te, yte, FEATURE_GROUPS, repeats=5 if MODE == "fast" else 10, seed=lab.seed)
display(gi.round(4))
ax = gi.set_index("feature_group").pr_auc_drop_mean.plot(kind="barh", xerr=gi.set_index("feature_group").pr_auc_drop_std, figsize=(6, 3)); ax.set_title(f"Group permutation importance ({mC.name}) - PR-AUC drop"); plt.tight_layout(); viz.save(plt.gcf(), cfg, "model_b_group_importance"); plt.show()
""")

md("""
### Secondary model C: appointment no-show on **real** data

The only real (non-simulated) outcome in the project is appointment attendance (public Kaggle no-show data, de-identified). It is a proof of concept that the same calibrated-model recipe works on real scheduling data;
the label is **attendance, not follow-up completion**, and the model must never be used to deny or deprioritise care. A scheduling-only feature set is compared with one that adds demographics (which can encode bias).
""")
code("""
from src.experiments import model_c_noshow
t = time.time(); C_res = model_c_noshow(lab); print("Model C took", round(time.time() - t), "s | n:", C_res["n"], "| test prevalence %.3f" % C_res["prevalence_test"])
display(C_res["comparison"].round(3)); print("selected on validation PR-AUC:", C_res["selected"]); display(C_res["subgroups"].round(3)); print(C_res["caveats"])
display(Image(filename=str(ROOT / "figures" / "model_c_calibration.png")))
""")

md("""
## 11. Hybrid GenAI + ML integration

The integration layer (`src/pipeline.py`) wires the pieces: the extractor *proposes* validated obligation records; the supervised classifier cross-checks the task type (strategy **D**); the evidence graph
turns records plus streaming events into deterministic states; the calibrated model forecasts risk. **The LLM never sets a workflow state.**
""")
code("""
from src.exp_replay import load_scorers, replay, split_patients
t = time.time(); scorers = load_scorers(lab, ablations=True); print("scorers:", list(scorers), "| trained/loaded in", round(time.time() - t), "s")
E1D = run_exp1(lab, "primary", include=["A_rules", "C2_llm_validated_filled", "D_hybrid_llm_ml", "E_rules_plus_llm"])
display(E1D["table"][["extractor", "corpus", "typed_precision", "typed_recall", "typed_f1", "det_f1", "deadline_acc", "unsupported_field_rate", "latency_mean_s"]].round(3))
ov = sum(1 for (n, c), res in E1D["results"].items() if n == "D_hybrid_llm_ml" for r in res for x in r.repairs if x.get("action") == "overridden_by_classifier")
print("task_type overrides made by the ML classifier in strategy D:", ov)
""")
code("""
from src.pipeline import HybridPipeline, script_events_for_created
pipe = HybridPipeline(cfg, make_extractors(cached_backend(lab, "primary"), E1D["classifier"])[SELECTED_EXTRACTOR], {"B_independent_ml": scorers["B_independent_ml"], "C_graph_ml": scorers["C_graph_ml"]}, build_nx=True)
res, created = pipe.ingest_note(DEMO_TEXT, "demo-note-1", "demo:patient-001")
print("extractor:", res.extractor, "| obligations:", len(res.obligations), "| latency %.1fs" % res.latency_s); print("DEMO TEXT:", DEMO_TEXT)
display(pd.DataFrame([{"obligation": o.obligation_id.split(":")[-1], "task_type": o.task_type, "due": o.due_interval, "role": o.responsible_role, "depends_on": [d.split(":")[-1] for d in o.dependencies],
                       "conditional": o.conditional_requirements, "span": o.evidence_span.text, "missing_information": o.missing_information, "confidence": round(o.confidence, 2)} for o in res.obligations]))
print("monitored obligations after extraction (all pending until evidence arrives):"); display(pipe.report()[["obligation", "step", "deadline", "depends_on", "state", "risk"]])
""")
code("""
script = script_events_for_created(created, "delayed_ack")
for k in (3, 6, len(script)):
    pipe2 = HybridPipeline(cfg, make_extractors(cached_backend(lab, "primary"), E1D["classifier"])[SELECTED_EXTRACTOR], {"B_independent_ml": scorers["B_independent_ml"], "C_graph_ml": scorers["C_graph_ml"]})
    pipe2.ingest_note(DEMO_TEXT, "demo-note-1", "demo:patient-001"); pipe2.feed(script[:k]);
    pipe2.proc._advance(to_epoch(script[k - 1]["ingest_time_utc"]) + 1); pipe2.proc.flush()
    print(f"\\nafter {k}/{len(script)} streamed events:"); display(pipe2.report()[["step", "state", "late", "risk", "rule"]])
print("alerts raised:", [(a.rule, a.status) for a in pipe2.proc.alert_log][:8])
""")
code("""
from src.alert_engine import verbalize_alert
a = next((x for x in pipe2.proc.alert_log), None)
if a is not None:
    print("DETERMINISTIC, evidence-grounded explanation (record of truth):\\n", a.explanation)
    v = verbalize_alert(cached_backend(lab, "primary").backend, a)
    print("\\nOptional GenAI paraphrase | grounded =", v["grounded"], "| foreign tokens:", v["foreign_tokens"], "| %.1fs" % v["latency_s"]); print(v["llm_text"]); print("\\nDISPLAYED:", v["text"][:600])
""")
