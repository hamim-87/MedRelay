"""Benchmark helpers: statistics, workload scaling (unique ids), queue loss accounting, hardware report honesty."""
import platform
import time

import numpy as np

from src import benchmarks as bm
from src.data_loader import iter_stream
from src.scenario_lab import demo_events


def test_percentile_stats():
    s = bm.pstats(np.arange(1, 101))
    assert s["p50"] == 50.5 and s["max"] == 100 and s["n"] == 100 and s["p99"] > s["p95"] > s["p50"]


def test_scale_events_keeps_ids_unique_and_ordered(cfg):
    base = list(iter_stream(cfg, limit=500))
    big = bm.scale_events(base, 1800)
    assert len(big) == 1800
    keys = [(e["event_id"], e.get("delivery_attempt", 1)) for e in big]
    assert len(set(keys)) == len(keys)                                  # replicas never collide on (event_id, delivery_attempt)
    t = [e["ingest_time_utc"] for e in big]
    assert t == sorted(t)
    copies = {e["patient_id"] for e in big if "~" in e["patient_id"]}
    assert copies and all(p.split("~")[0] in {e["patient_id"] for e in base} for p in copies)      # replicated patients are independent copies


def test_queue_reports_event_loss_when_bounded_and_dropping():
    slow = lambda e: time.sleep(0.004)   # noqa: E731
    ev = [{"i": i} for i in range(400)]
    lossless = bm.bench_queue(slow, ev[:120], rate_eps=400, maxsize=10, drop_when_full=False)
    lossy = bm.bench_queue(slow, ev[:400], rate_eps=2000, maxsize=5, drop_when_full=True)
    assert lossless["event_loss"] == 0 and lossless["consumed"] == 120
    assert lossy["event_loss"] > 0 and lossy["produced"] + lossy["event_loss"] == 400 and lossy["consumed"] == lossy["produced"]


def test_hardware_report_describes_the_running_machine():
    hw = bm.hardware_report()
    assert hw["uname_m"] == platform.machine() and hw["is_s390x"] == (platform.machine() == "s390x")
    assert hw["byteorder"] in ("little", "big") and hw["nproc"] >= 1
    if hw["is_s390x"]:
        assert hw["byteorder"] == "big"


def test_nnpa_bench_never_claims_availability_off_s390x():
    r = bm.nnpa_matmul_bench(sizes=((16, 16, 16, 2),))
    if platform.machine() != "s390x":
        assert r["available"] is False and "not s390x" in r["reason"]
    else:
        assert r["available"] in (True, False)
        if r["available"]:
            assert r["rows"][0]["nnpa_installed"] is True and "ZDNN_OK" in r["rows"][0]["status"]


def test_worker_workload_runs_and_reports(cfg):
    r = bm.run_workload({"n_events": 400, "scoring": "none", "storage": "memory", "rescore_batch": 64, "alerts": True, "mode": "fast"})
    assert "error" not in r, r
    assert r["events"] == 400 and r["events_per_s"] > 0 and r["error_rate"] == 0 and r["peak_rss_mb"] > 0
