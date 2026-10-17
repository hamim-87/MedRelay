"""Notebook cells, part C: sections 12-18."""

CELLS = []
md = lambda s: CELLS.append(("markdown", s.strip("\n")))   # noqa: E731
code = lambda s: CELLS.append(("code", s.strip("\n")))     # noqa: E731

md("""
## 12. Real-time event processing

The replay engine streams the processed event stream (held-out **test** patients) through `EventProcessor`: per delivery it (1) validates schema/provenance, (2) resolves patient/obligation, (3) updates the temporal evidence graph,
(4) recomputes affected states, (5) builds current-time features, (6) scores risk with the live models, (7) applies deterministic alert rules, (8) builds an evidence-grounded explanation, (9) appends to the audit log.
State is persisted in **SQLite (WAL)**; processing is **idempotent**; event time and ingestion time are kept separately; the stream contains late, duplicated and out-of-order deliveries.
""")
code("""
from src.exp_replay import replay, split_patients, truth_frame
from src.event_processor import EventProcessor, deadline_only_score
from src.data_loader import iter_stream
db_path = ROOT / cfg["paths"]["db"]
for suf in ("", "-wal", "-shm"):
    (Path(str(db_path) + suf)).unlink(missing_ok=True)
pt = split_patients(lab, "test")
events_test = list(iter_stream(cfg, patients=pt))
t = time.time()
proc = replay(lab, pt, scorers, events=events_test, db=str(db_path), rescore_batch=1, track_latency=True, build_nx=True)
wall = time.time() - t
lat = np.array(proc.latencies) * 1000
print(f"replayed {len(events_test)} deliveries for {len(pt)} test patients in {wall:.1f}s -> {len(events_test)/wall:.0f} deliveries/s with per-event streaming inference (3 risk scorers) and SQLite persistence")
print("per-delivery latency ms p50/p95/p99:", np.percentile(lat, [50, 95, 99]).round(2)); print("processor counters:", dict(proc.stats))
print("graph:", proc.graph.stats()); print("final workflow states:", proc.state_counts()); print("database integrity:", proc.integrity_check())
""")
code("""
al = proc.alerts_df(); display(al.rule.value_counts().rename("alerts").to_frame().T); display(al.head(5))
display(proc.audit_df().head(8)[["oid", "t_ingest", "old_state", "new_state", "rule", "trigger_event_id"]])
print("example grounded alert:\\n", next(a for a in proc.alert_log if a.rule == "RESULT_NOT_ACKNOWLEDGED").explanation)
""")
code("""
# idempotency: re-delivering events changes nothing
before = {o: (s[0], s[2]) for o, s in proc.states.items()}; n_tr = len(proc.transitions)
proc.run(events_test[:3000]); proc.finalize()
after = {o: (s[0], s[2]) for o, s in proc.states.items()}
print("re-delivered 3000 events: states changed =", sum(before[o] != after[o] for o in before), "| new transitions =", len(proc.transitions) - n_tr, "| duplicates ignored so far =", proc.graph.n_duplicates)
# crash recovery: rebuild from the persisted event log only
proc.commit()
t = time.time(); rec = EventProcessor.recover(cfg, db_path, scorers={}); 
same = np.mean([rec.states[o][0] == proc.states[o][0] for o in proc.states if o in rec.states])
print(f"recovered {rec.graph.stats()['obligations']} obligations / {rec.graph.stats()['events_ingested']} events from SQLite in {time.time()-t:.1f}s; stored states restored identically: {same:.3f}")
# reproducibility: a second replay of the same stream gives identical audit trails
p2 = replay(lab, pt, scorers, events=events_test, rescore_batch=1, build_nx=False)
sig = lambda p: [(t["oid"], t["new_state"], t["rule"]) for t in p.transitions]
print("replay reproducibility (identical transition sequence in a second, independent run):", sig(proc) == sig(p2), f"| {len(sig(p2))} transitions")
""")

md("## 13. Closed-loop evidence verification")
code("""
from src.exp_replay import run_exp3, run_exp4, verification_metrics
t = time.time(); E3 = run_exp3(lab, scorers); print("Experiment 3 replay+evaluation took", round(time.time() - t), "s"); print(E3["n"])
se = E3["state_eval"]
display(pd.DataFrame({k: {"accuracy at t*=deadline+grace+1h": v["accuracy"], **{f"recall[{c}]": r for c, r in v["per_class_recall"].items()}} for k, v in se.items()}).T.round(3))
display(se["engine_evidence_aware"]["confusion"]); display(se["naive_absence_is_failure"]["confusion"])
print("t* is evaluated using only evidence INGESTED by t*; 'engine_final_after_reconciliation' = same engine after all late evidence arrived.")
""")
code("""
VM = verification_metrics(lab, E3["processor_test"], truth_frame(lab, E3["processor_test"]), list(iter_stream(cfg, patients=split_patients(lab, "test"))))
display(pd.DataFrame({"value": VM}).astype(str))
""")
code("""
t = time.time(); E4 = run_exp4(lab, rate=0.3); print("Experiment 4 took", round(time.time() - t), "s")
display(E4["table"].round(3).T)
print("Reading: 'missing_completion' - the task was done but its evidence was deleted: the evidence-aware engine must NOT call it a confirmed failure (naive deadline logic does). 'delayed_evidence' - interim state wrongly "
      "overdue under the naive rule, reconciled by the engine. 'contradictory' - naive logic counts it completed. 'unacknowledged_results' - the engine alerts before the deadline. 'duplicates' - states unchanged.")
""")

md("""
## 14. Baseline comparisons and ablations

**Experiment 3 (main).** Three systems on the *same* replayed stream of held-out patients: **A** deadline-only reminders, **B** independent tasks + ML, **C** dependency-aware evidence graph + ML.
Operating thresholds are chosen on *validation* patients for a target recall and applied unchanged to *test* patients. The key early-warning requirement is *minimum lead*: alerts must be raised at least 0/24/72 h before the deadline.
""")
code("""
ew = E3["early_warning"]; sy = E3["summary"]
display(sy.round(3))
for lead in (24.0, 72.0):
    print(f"--- alerts required at least {lead:.0f} h before the deadline; thresholds from VALIDATION at recall target 0.7, applied to TEST")
    display(ew[(ew.min_lead_h == lead) & (ew["target_recall(val)"] == 0.7)][["system", "threshold(val)", "alerts", "recall", "precision", "false_alert_rate", "median_lead_h", "alerts_on_unverified_or_censored"]].round(3))
print("paired patient-bootstrap differences in max-score PR-AUC:", E3["paired"])
""")
code("""
sub = ew[(ew["target_recall(val)"] == 0.7) & ew.system.isin(["A_deadline_only", "A_reminder_24h", "B_independent_ml", "C_graph_ml"])]
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
sy2 = sy[sy.system.isin(["A_deadline_only", "A_reminder_24h", "B_independent_ml", "C_graph_ml"])].set_index("system")
sy2[["pr_auc_at_T-72h", "pr_auc_at_T-24h", "pr_auc_max_score"]].plot(kind="bar", ax=ax[0]); ax[0].set_title("PR-AUC of the score known at T-72h / T-24h / max before deadline"); ax[0].tick_params(axis="x", rotation=20); ax[0].legend(fontsize=7)
pv = sub.pivot(index="system", columns="min_lead_h", values="false_alert_rate"); pv.plot(kind="bar", ax=ax[1]); ax[1].set_title("False-alert rate at recall target 0.7 (val), by required lead (h)"); ax[1].tick_params(axis="x", rotation=20)
pv = sub.pivot(index="system", columns="min_lead_h", values="recall"); pv.plot(kind="bar", ax=ax[2]); ax[2].set_title("Achieved TEST recall at the validation-chosen threshold"); ax[2].tick_params(axis="x", rotation=20)
plt.tight_layout(); viz.save(fig, cfg, "exp3_systems"); plt.show()
display(E3["alert_table"].round(2)); print("Engine alerts (system C default configuration, risk threshold %.2f) by rule:" % cfg["alerts"]["risk_threshold"])
""")
md("**Experiment 5 - ablations.** Remove one component at a time: dependency (graph) features, temporal features, evidence features, GenAI extraction, evidence verification. Feature-ablated models are retrained with the same family/hyper-parameters.")
code("""
from sklearn.metrics import average_precision_score, roc_auc_score
from src.evaluation import paired_bootstrap_diff, binary_metrics
te, yte = binary(lab.dense()["test"]); d = te.assign(_y=yte)
rows = []
for name in ("C_graph_ml", "C_no_graph_dependency", "C_no_temporal", "C_no_evidence", "C_no_graph_temporal", "B_independent_ml"):
    m = scorers[name]; p = m.predict(te); d[name] = p; mm = binary_metrics(yte, p)
    rows.append({"system": name, "n_features": len(m.features), "pr_auc": mm["pr_auc"], "roc_auc": mm["roc_auc"], "brier": mm["brier"], "ece": mm["ece"]})
tab = pd.DataFrame(rows)
tab["pr_auc_diff_vs_C"] = [0.0] + [paired_bootstrap_diff(d, "patient_id", lambda s, n=n: average_precision_score(s._y, s[n]), lambda s: average_precision_score(s._y, s["C_graph_ml"]), n_boot=lab.p["bootstrap"], seed=31)["diff"] for n in tab.system[1:]]
tab["diff_ci_lo"] = [0.0] + [paired_bootstrap_diff(d, "patient_id", lambda s, n=n: average_precision_score(s._y, s[n]), lambda s: average_precision_score(s._y, s["C_graph_ml"]), n_boot=lab.p["bootstrap"], seed=31)["lo"] for n in tab.system[1:]]
tab["diff_ci_hi"] = [0.0] + [paired_bootstrap_diff(d, "patient_id", lambda s, n=n: average_precision_score(s._y, s[n]), lambda s: average_precision_score(s._y, s["C_graph_ml"]), n_boot=lab.p["bootstrap"], seed=31)["hi"] for n in tab.system[1:]]
display(tab.round(4)); lab.save_csv("exp5_feature_ablation_dense_test", tab)
ewa = ew[(ew.min_lead_h == 24.0) & (ew["target_recall(val)"] == 0.7)][["system", "recall", "precision", "false_alert_rate", "median_lead_h"]]
print("Replay-level effect of each ablation (alerts >=24h before deadline, val-chosen threshold at recall 0.7):"); display(ewa[ewa.system.str.startswith(("C_", "B_"))].round(3))
""")
code("""
from src.exp_extraction import run_exp5_genai
t = time.time(); GEN = run_exp5_genai(lab, "primary"); print("GenAI end-to-end ablation took", round(time.time() - t), "s on", GEN["table"].episodes.iloc[0], "synthetic episodes")
display(GEN["table"].round(3))
print("'none' = no obligation extraction -> nothing is monitored; failures_surfaced counts true overdue obligations for which the monitoring system raised a reactive alert on the MATCHED extracted obligation. "
      "Spurious or mis-timed extractions raise false alerts. Outcomes in these episodes are simulated from explicit event timelines.")
""")
code("""
from src.exp_replay import run_priority
PRI = run_priority(lab, scorers); display(PRI["table"].round(3)); print("tuned weights:", {k: round(v, 3) for k, v in PRI["tuned_weights"].items()}, "| search budget:", PRI["search_budget"])
print("learned coefficients:", {k: round(v, 2) for k, v in PRI["learned_coefficients"].items()}); print("paired differences vs risk-only:", {k: (round(v["diff"], 3), round(v["lo"], 3), round(v["hi"], 3)) for k, v in PRI["paired"].items()}); print(PRI["note"])
""")
code("""
GENT = GEN["table"].set_index("extractor")
abl = pd.DataFrame([
    {"component removed": "graph dependency features", "evidence": f"PR-AUC {tab.set_index('system').loc['C_no_graph_dependency','pr_auc_diff_vs_C']:+.3f} (95% CI {tab.set_index('system').loc['C_no_graph_dependency','diff_ci_lo']:+.3f}..{tab.set_index('system').loc['C_no_graph_dependency','diff_ci_hi']:+.3f}) on dense test"},
    {"component removed": "temporal features", "evidence": f"PR-AUC {tab.set_index('system').loc['C_no_temporal','pr_auc_diff_vs_C']:+.3f} (95% CI {tab.set_index('system').loc['C_no_temporal','diff_ci_lo']:+.3f}..{tab.set_index('system').loc['C_no_temporal','diff_ci_hi']:+.3f})"},
    {"component removed": "evidence features", "evidence": f"PR-AUC {tab.set_index('system').loc['C_no_evidence','pr_auc_diff_vs_C']:+.3f} (95% CI {tab.set_index('system').loc['C_no_evidence','diff_ci_lo']:+.3f}..{tab.set_index('system').loc['C_no_evidence','diff_ci_hi']:+.3f})"},
    {"component removed": "evidence verification (engine -> naive absence=failure)", "evidence": f"state accuracy at t*: {se['engine_evidence_aware']['accuracy']:.3f} -> {se['naive_absence_is_failure']['accuracy']:.3f}; not-confirmed recall {se['engine_evidence_aware']['per_class_recall']['not_confirmed']:.2f} -> {se['naive_absence_is_failure']['per_class_recall']['not_confirmed']:.2f}"},
    {"component removed": "GenAI extraction (E -> rules only A)", "evidence": "failure recall %.3f -> %.3f; monitored fraction %.3f -> %.3f" % (GENT.loc["E_rules_plus_llm", "failure_recall"], GENT.loc["A_rules", "failure_recall"], GENT.loc["E_rules_plus_llm", "monitored_fraction"], GENT.loc["A_rules", "monitored_fraction"])},
])
display(abl); lab.save_csv("exp5_ablation_summary", abl)
""")

md("## 15. Explainability and case-level error analysis")
code("""
from src.exp_replay import error_analysis
EA = error_analysis(lab, E3, k=3)
print(f"threshold {EA['threshold']:.3f} (validation-chosen, recall target 0.7, >=24h lead); cases scored at T-{EA['horizon_h']:.0f}h")
display(EA["cases"].drop(columns=["events (type, event time, ingest time)"]))
for _, r in EA["cases"].head(2).iterrows(): print(r["case"], r["obligation"], "->", r["events (type, event time, ingest time)"])
display(EA["by_profile"].round(3))
print("The hidden generator profile is used ONLY here, to see which simulated behaviours the model gets wrong; it is never a feature.")
""")
code("""
gi_c = gi.set_index("feature_group").pr_auc_drop_mean.rename("PR-AUC drop when group shuffled")
display(gi_c.to_frame().round(4)); print("Most informative group:", gi_c.idxmax())
oid = EA["cases"].iloc[0]["obligation"]
print("Audit trail for", oid); display(E3["processor_test"].audit_df(oid)[["t_ingest", "old_state", "new_state", "rule", "trigger_event_id", "evidence"]])
print([ (a.rule, a.status, a.explanation[:260]) for a in E3["processor_test"].alert_log if a.oid == oid][:2])
""")

md("""
## 16. LinuxONE performance benchmarks

Methodology: warm-up discarded, repeated measurements, each throughput workload in a **fresh subprocess** (clean peak-RSS attribution). **ML-model latency and LLM generation latency are reported separately.**
Everything below is measured on the machine identified in section 2. "Event loss" is reported for a bounded queue under offered load.
""")
code("""
import gc, psutil
for _n in ("proc", "p2", "rec", "fin", "fm", "f", "pipe", "pipe2", "demo", "d", "gp", "g", "g1", "g2", "g3", "E4", "VM"):
    globals().pop(_n, None)
E3["processor_test"] = None; E3["scorers"] = None; gc.collect()
print("kernel RSS after releasing replay state: %.0f MB | system available: %.0f MB" % (psutil.Process().memory_info().rss / 1e6, psutil.virtual_memory().available / 1e6))
""")
code("""
from src import benchmarks as bm
from src.clinical_extractor import LLM_SCHEMA, SYSTEM_PROMPT
rows_f = te[mC.features].head(256).to_dict("records")
mlat = []
for nm in ("C_graph_ml", "B_independent_ml"):
    df_ = bm.bench_ml_latency(scorers[nm], rows_f, repeats=3 if MODE == "fast" else 5); df_.insert(0, "model", f"{nm} ({scorers[nm].kind})"); mlat.append(df_)
mlat = pd.concat(mlat); display(mlat.round(3))
conc = bm.bench_ml_concurrency(scorers["C_graph_ml"], rows_f, threads=(1, 2, 4, 8)); display(conc.round(2))
""")
code("""
prompts = [d["text"] for d in E1["corpora"]["developer_annotated_challenge"][:3 if MODE == "fast" else 6]]
llm_b = bm.bench_llm(be.backend, prompts, LLM_SCHEMA, SYSTEM_PROMPT, max_tokens=200)
display(pd.DataFrame({"value": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in {"model": llm_b["model"], "model_size_mb": llm_b["model_size_mb"], "threads": llm_b["threads"], "load_s": llm_b["load_s"],
      "prompt_tokens_mean": llm_b["prompt_tokens_mean"], "completion_tokens_mean": llm_b["completion_tokens_mean"], "prompt_eval_s (1-token call)": llm_b["prompt_eval_s_first_token_proxy"],
      "generation latency p50 (s)": llm_b["latency_s"]["p50"], "p95 (s)": llm_b["latency_s"]["p95"], "p99 (s)": llm_b["latency_s"]["p99"], "tokens/s mean": llm_b["tokens_per_s"]["mean"], "peak RSS MB": llm_b["peak_rss_mb"]}.items()}}))
lconc = bm.bench_llm_concurrency(be.backend, prompts[0], LLM_SCHEMA, SYSTEM_PROMPT, concurrency=(1, 2, 4), max_tokens=40); display(lconc.round(2))
print("ML scoring latency (ms, section above) vs LLM generation latency (s): about", round(llm_b['latency_s']['p50'] * 1000 / mlat[(mlat.batch == 1)].latency_ms_p50.iloc[0]), "x slower per call - the LLM is never on the per-event path.")
""")
code("""
spec = cfg["benchmarks"][MODE]; reps = spec["repeats"]
bench_rows = []
bm.run_workload({"n_events": 500, "scoring": "ml", "storage": "memory", "rescore_batch": 64, "mode": MODE})        # warm-up (discarded)
configs = [("graph+states only (in-memory)", dict(scoring="none", storage="memory", rescore_batch=64, alerts=False)),
           ("states+alerts (in-memory)", dict(scoring="none", storage="memory", rescore_batch=64, alerts=True)),
           ("states+alerts+ML risk (in-memory, batch 64)", dict(scoring="ml", storage="memory", rescore_batch=64, alerts=True)),
           ("states+alerts+ML risk (SQLite WAL, commit/200, batch 64)", dict(scoring="ml", storage="sqlite", rescore_batch=64, alerts=True, commit_every=200)),
           ("states+alerts+ML risk (SQLite, per-event rescoring, commit/200)", dict(scoring="ml", storage="sqlite", rescore_batch=1, alerts=True, commit_every=200))]
for n in spec["event_workloads"]:
    for label, c in configs:
        if n > 10000 and (c["rescore_batch"] == 1 or label.startswith("states+alerts (")):
            continue                                   # the 100k workload only runs the three most informative configurations (wall-clock budget)
        for r in range(reps if n < 100000 else 1):     # 100k events: a single measured run per configuration
            res = bm.run_workload({"n_events": n, "mode": MODE, **c})
            bench_rows.append({"workload_events": n, "config": label, "run": r, **{k: res.get(k) for k in ("events", "wall_s", "events_per_s", "peak_rss_mb", "error_rate", "rejected", "alerts", "transitions")},
                               "lat_p50_ms": res.get("latency_ms", {}).get("p50"), "lat_p95_ms": res.get("latency_ms", {}).get("p95"), "lat_p99_ms": res.get("latency_ms", {}).get("p99"), "error": res.get("error")})
BR = pd.DataFrame(bench_rows); BR.to_csv("results/throughput_benchmark_runs.csv", index=False)
agg = BR.groupby(["workload_events", "config"]).agg(events_per_s=("events_per_s", "median"), eps_min=("events_per_s", "min"), eps_max=("events_per_s", "max"), lat_p50_ms=("lat_p50_ms", "median"),
                                                  lat_p95_ms=("lat_p95_ms", "median"), lat_p99_ms=("lat_p99_ms", "median"), peak_rss_mb=("peak_rss_mb", "max"), error_rate=("error_rate", "max"), runs=("run", "count")).reset_index()
display(agg.round(2))
""")
code("""
ev_q = events_test[:2000]
def make_proc():
    return EventProcessor(cfg, ":memory:", scorers={"C_graph_ml": scorers["C_graph_ml"]}, primary="C_graph_ml", rescore_batch=64, enable_alerts=True, log_risk=False, build_nx=False)
# capacity of THIS consumer (states + alerts + calibrated ML risk, micro-batch 64) measured on the same events, single thread, no queue
caps = []
for _ in range(3):                                       # first pass is a warm-up (model/lib first-call costs); capacity = best warm pass
    _p = make_proc(); _t = time.perf_counter()
    for _e in ev_q: _p.process(_e)
    _p.flush(); caps.append(len(ev_q) / (time.perf_counter() - _t)); del _p
cap = max(caps[1:])
qrows = []
for label, mult, drop, maxsize in (("0.5x capacity, block", 0.5, False, 200), ("2x capacity, block (backpressure)", 2.0, False, 200), ("2x capacity, drop (bounded queue 200)", 2.0, True, 200), ("4x capacity, drop (bounded queue 200)", 4.0, True, 200)):
    p = make_proc(); r = bm.bench_queue(p.process, ev_q, rate_eps=cap * mult, maxsize=maxsize, drop_when_full=drop, finalize_fn=p.flush); r["scenario"] = label; qrows.append(r)
display(pd.DataFrame(qrows)[["scenario", "offered_rate_eps", "produced", "consumed", "dropped", "loss_rate", "max_backlog", "queue_latency_ms_p50", "queue_latency_ms_p95", "queue_latency_ms_p99", "produce_wall_s", "ideal_produce_wall_s", "total_wall_s", "policy"]].round(3))
print(f"consumer capacity measured on the same events (best of 2 warm passes {[round(c) for c in caps]}): {cap:.0f} events/s (states+alerts+ML risk). Offered rate = multiple of it. 'block' = backpressure (producer stalls, nothing lost: compare produce_wall_s with ideal_produce_wall_s); 'drop' = bounded queue sheds events.")
""")
code("""
mm = bm.nnpa_matmul_bench()
if mm["available"]:
    nd = pd.DataFrame(mm["rows"])[["M", "K", "N", "nnpa_op_ms", "nnpa_total_ms", "numpy_blas_ms", "cpu_naive_c_ms", "nnpa_op_gflops", "numpy_blas_gflops", "speedup_op_vs_numpy", "speedup_total_vs_numpy", "max_abs_err", "max_abs_ref"]]
    display(nd.round(4)); print("zDNN", mm["zdnn_version"], "-", mm["interpretation"])
else:
    print("NNPA device test not run on this host:", mm["reason"])
""")
code("""
import platform as _pl
bench_out = {"host": {k: slim[k] for k in ("uname_m", "byteorder", "nproc", "memory_total_gb", "hypervisor", "python")}, "ml_latency": mlat.to_dict("records"), "ml_concurrency": conc.to_dict("records"), "llm": llm_b,
             "llm_concurrency": lconc.to_dict("records"), "throughput": agg.to_dict("records"), "queue": qrows, "nnpa_matmul": mm, "nnpa_probe": nn, "mode": MODE}
Path("results/benchmarks.json").write_text(json.dumps(bench_out, indent=2, default=str))
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
a2 = agg.pivot(index="config", columns="workload_events", values="events_per_s"); a2.plot(kind="barh", ax=ax[0]); ax[0].set_title("Event-processing throughput (events/s, median)"); ax[0].legend(title="events", fontsize=7)
a3 = agg.pivot(index="config", columns="workload_events", values="peak_rss_mb"); a3.plot(kind="barh", ax=ax[1]); ax[1].set_title("Peak resident memory (MB)"); ax[1].legend(title="events", fontsize=7)
mlat.pivot(index="batch", columns="model", values="latency_ms_p50").plot(ax=ax[2], marker="o"); ax[2].set_xscale("log"); ax[2].set_title("ML scoring latency p50 (ms) vs batch size"); ax[2].set_ylabel("ms")
plt.tight_layout(); viz.save(fig, cfg, "benchmarks"); plt.show()
""")

md("""
## 17. Interactive live demonstration

**Required scenario (synthetic patient):** discharged with (1) a laboratory test within 48 h, (2) physician result review, (3) conditional patient notification, (4) a follow-up appointment.
The test completes, but the acknowledgment evidence is **delayed**. The system recognises the dependency chain, separates *completed* from *unverified*, updates the forecast risk as events stream in, raises an evidence-grounded alert,
shows the graph and timeline, **reconciles the state when the delayed acknowledgment is finally documented**, and keeps the full audit trail. The same engine, models and rules as the experiments are used (streaming inference at every event).
""")
code("""
from src.visualization_demo import LiveDemo, audit_table, alert_feed_html, plot_patient_journey, plot_evidence_graph, plot_risk_monitor
from src.scenario_lab import demo_events, DEMO_TEXT
live_scorers = {"B_independent_ml": scorers["B_independent_ml"], "C_graph_ml": scorers["C_graph_ml"]}
evs = demo_events("delayed_ack"); demo = LiveDemo(cfg, live_scorers, evs, PID)
idx = {e["event_id"]: i + 1 for i, e in enumerate(evs)}
print("scripted deliveries (ingest order):"); display(pd.DataFrame([{"#": i + 1, "event_type": e["event_type"], "obligation": e["obligation_id"].split(":")[-1], "event_time": e["event_time_utc"], "ingest_time": e["ingest_time_utc"]} for i, e in enumerate(evs)]))
""")
code("""
steps = sorted({7, 9, 10, len(evs)})
for k, p, fj, fg in demo.storyboard(steps, with_graph=True):
    print(f"=== frame after {k}/{len(evs)} deliveries (service clock {iso(p.clock)}) ===")
    display(fj); display(fg); plt.close(fj); plt.close(fg)
    display(pd.DataFrame([{"obligation": o.split(":")[-1], "state": p.graph.derive_state(o, p.clock, live=True).state.value, "risk(C)": round((p.last_risk.get(o) or {}).get("C_graph_ml", float("nan")), 2)} for o in sorted(p.graph.obs)]))
    display(HTML(alert_feed_html(p, PID, last=4)))
""")
code("""
fin = demo.run_to(len(evs)); fin.finalize()
fig = plot_risk_monitor(fin, PID, "C_graph_ml"); viz.save(fig, cfg, "demo_risk_monitor"); plt.show()
print("AUDIT TRAIL (every state change with event time/ingest time, rule and evidence):"); display(audit_table(fin, PID))
print("final alert feed (newest first):"); display(HTML(alert_feed_html(fin, PID, last=10)))
ob1 = [t for t in fin.transitions if t["oid"] == "demo:ob1"]
assert [t["new_state"] for t in ob1][-2:] == ["overdue_without_verified_completion", "completed_with_evidence"], "delayed acknowledgment was not reconciled"
print("RECONCILIATION VERIFIED: review obligation went overdue_without_verified_completion -> completed_with_evidence when the delayed acknowledgment arrived; late flag =", fin.states["demo:ob1"][2])
""")
code("""
# variants: acknowledgment NEVER documented, and the happy path
for v in ("missing_ack", "happy_path"):
    d = LiveDemo(cfg, live_scorers, demo_events(v), PID); f = d.run_to(len(d.events)); f.finalize()
    print("variant:", v, "->", {o.split(":")[-1]: f.states[o][0].value + (" (late)" if f.states[o][2] else "") for o in sorted(f.states)})
fig = plot_patient_journey(f, PID, "happy path"); plt.show()
fm = LiveDemo(cfg, live_scorers, demo_events("missing_ack"), PID).run_to(99); fm.finalize()
fig = plot_patient_journey(fm, PID, "Missing acknowledgment: downstream tasks stay unverified/overdue - the completed test does not complete the chain"); plt.show()
display(HTML(alert_feed_html(fm, PID, last=6)))
""")
code("""
# interactive widget (falls back to the static storyboard above when widgets cannot render, e.g. in a headless run)
try:
    import ipywidgets
    from IPython import get_ipython
    if get_ipython() is not None and os.environ.get("MEDRELAY_HEADLESS", "0") != "1":
        LiveDemo(cfg, live_scorers, demo_events("delayed_ack"), PID).widget()
    else:
        print("headless execution: interactive widget skipped; the storyboard frames above show the same states.")
except Exception as e:
    print("ipywidgets unavailable (", type(e).__name__, ") - using the static storyboard above")
""")
code("""
display(HTML("<b>Performance monitor (measured on %s, mode=%s)</b>" % (hw["uname_m"], MODE)))
best_row = agg.sort_values("events_per_s", ascending=False).iloc[0]
display(pd.DataFrame({"metric": ["ML scoring latency p50 / p95 / p99, batch 1 (ms)", "ML scoring throughput, batch 256 (rows/s)", "event processing throughput, best config (events/s)", "LLM generation p50 / p95 (s)", "LLM tokens/s", "peak RSS, event workload (MB)", "queue event loss, block policy"],
                      "value": [" / ".join(f"{x:.1f}" for x in mlat[(mlat.model.str.startswith("C_")) & (mlat.batch == 1)][["latency_ms_p50", "latency_ms_p95", "latency_ms_p99"]].iloc[0]), f"{mlat[(mlat.model.str.startswith('C_')) & (mlat.batch == 256)].rows_per_s.iloc[0]:.0f}",
                                f"{best_row.events_per_s:.0f} ({best_row.config}, {best_row.workload_events} events)", f"{llm_b['latency_s']['p50']:.1f} / {llm_b['latency_s']['p95']:.1f}", f"{llm_b['tokens_per_s']['mean']:.1f}",
                                f"{agg.peak_rss_mb.max():.0f}", str(qrows[0]['event_loss']) + " / " + str(qrows[1]['event_loss'])]}))
""")

md("## 18. Results, limitations and final conclusions")
code("""
import json as _j
def load(n): return _j.loads(Path(f"results/{n}.json").read_text())
mA = A_res["comparison"].sort_values("val_macro_f1", ascending=False).iloc[0]
cB = B_res["comparison"]; bC = cB[cB.features == "C_graph"].sort_values("val_pr_auc", ascending=False).iloc[0]; bB = cB[cB.features == "B_independent"].sort_values("val_pr_auc", ascending=False).iloc[0]
mC_ = load("model_c_summary"); cC = pd.DataFrame(mC_["comparison"]).sort_values("val_pr_auc", ascending=False).iloc[0]
r1 = E1["table"]; ew24 = ew[(ew.min_lead_h == 24.0) & (ew["target_recall(val)"] == 0.7)].set_index("system")
summary = pd.DataFrame([
    ["Action classification (silver labels)", f"{mA.model} macro-F1 {mA.test_macro_f1:.3f} / micro-F1 {mA.test_micro_f1:.3f} (test); developer-annotated spans: accuracy {mA.challenge_span_accuracy:.3f}"],
    ["Follow-up risk (simulated outcomes)", f"best graph model {bC.model}: PR-AUC {bC.test_pr_auc:.3f} [{bC.test_pr_auc_ci_lo:.3f},{bC.test_pr_auc_ci_hi:.3f}], ROC-AUC {bC.test_roc_auc:.3f}, Brier {bC.test_brier:.3f}, ECE {bC.test_ece:.3f}; independent-task {bB.model}: PR-AUC {bB.test_pr_auc:.3f}"],
    ["No-show (real data, secondary)", f"{cC.model}: PR-AUC {cC.test_pr_auc:.3f} (prevalence {cC.prevalence:.3f}), ROC-AUC {cC.test_roc_auc:.3f}, ECE {cC.test_ece:.3f}"],
    ["GenAI extraction (350M)", "typed F1 B/C2/E on challenge set: " + " / ".join(f"{r1[(r1.extractor == x) & (r1.corpus == 'developer_annotated_challenge')].typed_f1.iloc[0]:.2f}" for x in ('B_llm_raw', 'C2_llm_validated_filled', 'E_rules_plus_llm')) + f"; rules A {r1[(r1.extractor == 'A_rules') & (r1.corpus == 'developer_annotated_challenge')].typed_f1.iloc[0]:.2f}"],
    ["Early warning (>=24h lead, recall target 0.7 on val)", " | ".join(f"{s}: recall {ew24.loc[s,'recall']:.2f}, FAR {ew24.loc[s,'false_alert_rate']:.2f}, lead {ew24.loc[s,'median_lead_h']:.0f}h" for s in ("A_reminder_24h", "A_deadline_only", "B_independent_ml", "C_graph_ml"))],
    ["State correctness at t*", f"engine {se['engine_evidence_aware']['accuracy']:.3f} (final after reconciliation {se['engine_final_after_reconciliation']['accuracy']:.3f}) vs naive absence=failure {se['naive_absence_is_failure']['accuracy']:.3f}"],
    ["Throughput (best config)", f"{best_row.events_per_s:.0f} events/s on {hw['uname_m']} ({hw['nproc']} vCPU)"]], columns=["result", "measured value"])
display(summary); summary.to_csv("results/final_summary.csv", index=False)
print("notebook wall time: %.1f min | mode=%s | host=%s" % ((time.time() - T_NB0) / 60, MODE, hw["uname_m"]))
""")
md("""
### Interpretation, limitations and honest conclusions

* **Research question (simulated processes only).** The tables above quantify, with patient-grouped bootstrap intervals, whether dependency-aware graph features improve *early* warning over independent-task monitoring. Read the lead-restricted rows
  (>= 24 h / 72 h before the deadline): a deadline-only rule is hard to beat *late* in the window (it only fires once most of the window has elapsed), so any advantage of the graph approach is an *early-warning* advantage; see the confidence intervals before
  concluding anything, and note when an interval includes zero.
* **Labels.** Risk and state outcomes come from a **seeded simulator** whose causal structure we know; scores say how well the pipeline recovers that structure, **not** how it would perform on real hospital data. Hidden generator profiles are used only for post-hoc error analysis.
* **No gold annotations.** Action labels are rule-proposed; extraction evaluation uses a *developer-annotated* challenge set and synthetic-by-construction documents. Independent clinician review is the single most valuable missing piece.
* **GenAI.** The 350M Granite model is too weak to extract obligations on its own; validation removes fabricated fields but cannot add missing obligations. The ensemble with rules is the best measured GenAI-involving strategy; the 1B model (smaller sample) is the measured upper bound. Latency is dominated by the LLM, so it is kept off the per-event path.
* **Hardware.** Section 2 states the architecture that executed this run; throughput/latency are for that VM (2 vCPU, ~4 GB). NNPA is *present* and was device-tested with zDNN matmul; the tree/linear models do **not** use it.
* **Not a medical device.** All recommendations are for qualified human review; nothing here modifies orders, prescriptions, diagnoses or treatment.
""")
