"""Per-workload peak memory of the event engine (fresh subprocess per run, VmHWM) at 1k / 10k / 100k events, full-mode data and the full-mode selected risk model.
Writes results/benchmark_peak_memory.csv. Run: .venv/bin/python scripts/bench_peak_memory.py"""
import os, sys, time
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src import benchmarks as bm
CFG = [("graph+states only (in-memory)", dict(scoring="none", storage="memory", rescore_batch=64, alerts=False)),
       ("states+alerts (in-memory)", dict(scoring="none", storage="memory", rescore_batch=64, alerts=True)),
       ("states+alerts+ML risk (SQLite WAL, commit/200, batch 64)", dict(scoring="ml", storage="sqlite", rescore_batch=64, alerts=True, commit_every=200))]
rows = []
for n in (1000, 10000, 100000):
    for label, c in CFG:
        t = time.time(); r = bm.run_workload({"n_events": n, "mode": "full", **c})
        rows.append({"workload_events": n, "config": label, "events_per_s": r.get("events_per_s"), "peak_rss_mb": r.get("peak_rss_mb"), "rss_before_mb": r.get("rss_before_mb"),
                     "peak_increase_mb": (r.get("peak_rss_mb") or 0) - (r.get("rss_before_mb") or 0), "error": r.get("error"), "wall_s": round(time.time() - t, 1)})
        pd.DataFrame(rows).to_csv(ROOT / "results" / "benchmark_peak_memory.csv", index=False)
        print(rows[-1], flush=True)
