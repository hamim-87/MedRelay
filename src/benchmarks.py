"""LinuxONE performance measurement: hardware evidence, ML inference latency, event-processing throughput, LLM generation speed, concurrency, queueing.

Methodology: warm-up runs are discarded, each workload is repeated, and workloads run in a fresh subprocess so peak RSS is attributable to that workload.
Hardware/arch fields come from the machine actually executing this code (nothing is assumed); ML-model latency and LLM generation latency are measured
and reported SEPARATELY. NNPA is only reported as used if a device-level test succeeded (see nnpa_probe)."""
from __future__ import annotations

import json
import os
import platform
import queue
import resource
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import ROOT


def _sh(cmd: str, timeout: int = 20) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"(unavailable: {e})"


def hardware_report() -> dict:
    """Mandatory hardware inspection, executed on the machine running the notebook."""
    import psutil
    pkgs = {}
    for m in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "networkx", "pydantic", "matplotlib", "plotly", "ipywidgets", "llama_cpp", "onnx", "onnxruntime", "joblib", "psutil"):
        try:
            mod = __import__(m)
            pkgs[m] = getattr(mod, "__version__", "?")
        except Exception:  # noqa: BLE001
            pkgs[m] = None
    cpuinfo = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    feats = ""
    for line in cpuinfo.splitlines():
        if line.lower().startswith(("features", "flags")):
            feats = line.split(":", 1)[1].strip()
            break
    fac = [l for l in cpuinfo.splitlines() if l.startswith("facilities")]
    fac_ids = set(fac[0].split(":", 1)[1].split()) if fac else set()
    vm = psutil.virtual_memory()
    du = shutil.disk_usage(str(ROOT))
    return {
        "uname_m": platform.machine(), "uname_a": _sh("uname -a"), "is_s390x": platform.machine() == "s390x", "byteorder": sys.byteorder,
        "os_release": {k: v.strip('"') for k, v in (l.split("=", 1) for l in Path("/etc/os-release").read_text().splitlines() if "=" in l) if k in ("PRETTY_NAME", "VERSION_ID")} if Path("/etc/os-release").exists() else {},
        "lscpu": _sh("lscpu"), "nproc": os.cpu_count(), "cpu_features": feats, "has_vector_facility_vx": " vx " in f" {feats} ", "has_vxe2": " vxe2 " in f" {feats} ",
        "nnpa_cpu_flag": " nnpa " in f" {feats} ", "nnpa_facility_165_listed": "165" in fac_ids,
        "memory_total_gb": round(vm.total / 1e9, 2), "memory_available_gb": round(vm.available / 1e9, 2), "disk_total_gb": round(du.total / 1e9, 1), "disk_free_gb": round(du.free / 1e9, 1),
        "python": platform.python_version(), "python_exe": sys.executable, "packages": pkgs, "docker": shutil.which("docker"), "podman": shutil.which("podman"),
        "free_h": _sh("free -h"), "df_h": _sh("df -h /"), "hypervisor": next((l.split(":", 1)[1].strip() for l in _sh("lscpu").splitlines() if l.startswith("Hypervisor:")), None)}


def pstats(x) -> dict:
    x = np.asarray(x, float)
    if len(x) == 0:
        return {}
    return {"n": int(len(x)), "mean": float(x.mean()), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)), "max": float(x.max())}


def peak_rss_mb() -> float:
    """Peak resident memory of THIS process (VmHWM from /proc/self/status). ru_maxrss is not used: a child process inherits its parent's peak across fork+exec,
    which made every benchmark worker report the notebook kernel's lifetime peak instead of its own."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ------------------------------------------------------------------------------------------------ ML inference latency (no LLM)
def bench_ml_latency(model, feature_rows: list[dict], batch_sizes=(1, 16, 64, 256), repeats: int = 3, warmup: int = 3) -> pd.DataFrame:
    out = []
    for bs in batch_sizes:
        rows = (feature_rows * (bs // len(feature_rows) + 1))[:bs]
        for _ in range(warmup):
            model.predict_dicts(rows)
        lat = []
        for _ in range(repeats * max(5, 200 // bs)):
            t = time.perf_counter()
            model.predict_dicts(rows)
            lat.append(time.perf_counter() - t)
        s = pstats(np.array(lat) * 1000)
        out.append({"batch": bs, "latency_ms_p50": s["p50"], "latency_ms_p95": s["p95"], "latency_ms_p99": s["p99"], "per_row_ms_p50": s["p50"] / bs, "rows_per_s": bs / (s["p50"] / 1000), "n_calls": s["n"]})
    return pd.DataFrame(out)


def bench_ml_concurrency(model, feature_rows: list[dict], threads=(1, 2, 4, 8), calls_per_thread: int = 40, batch: int = 16) -> pd.DataFrame:
    rows = (feature_rows * (batch // len(feature_rows) + 1))[:batch]
    out = []
    for nt in threads:
        lat: list[float] = []

        def work():
            local = []
            for _ in range(calls_per_thread):
                t = time.perf_counter()
                model.predict_dicts(rows)
                local.append(time.perf_counter() - t)
            return local
        t0 = time.perf_counter()
        with ThreadPoolExecutor(nt) as ex:
            for r in [f.result() for f in [ex.submit(work) for _ in range(nt)]]:
                lat += r
        wall = time.perf_counter() - t0
        s = pstats(np.array(lat) * 1000)
        out.append({"threads": nt, "requests": nt * calls_per_thread, "throughput_req_per_s": nt * calls_per_thread / wall, "latency_ms_p50": s["p50"], "latency_ms_p95": s["p95"], "latency_ms_p99": s["p99"], "errors": 0})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ LLM generation (separate from ML)
def bench_llm(backend, prompts: list[str], schema: dict, system: str, max_tokens: int = 200, repeats: int = 1) -> dict:
    lat, tps, toks, pt = [], [], [], []
    backend.generate_json([{"role": "system", "content": system}, {"role": "user", "content": prompts[0]}], schema, max_tokens=16)       # warm-up
    for _ in range(repeats):
        for p in prompts:
            r = backend.generate_json([{"role": "system", "content": system}, {"role": "user", "content": p}], schema, max_tokens=max_tokens)
            lat.append(r.latency_s); tps.append(r.tokens_per_s); toks.append(r.completion_tokens); pt.append(r.prompt_tokens)
    t = time.perf_counter()
    r = backend.generate_json([{"role": "system", "content": system}, {"role": "user", "content": prompts[0]}], {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}, max_tokens=1)
    prompt_eval = r.latency_s
    return {"model": backend.name, "model_size_mb": backend.size_bytes / 1e6, "threads": backend.n_threads, "latency_s": pstats(lat), "tokens_per_s": pstats(tps), "completion_tokens_mean": float(np.mean(toks)),
            "prompt_tokens_mean": float(np.mean(pt)), "prompt_eval_s_first_token_proxy": prompt_eval, "peak_rss_mb": peak_rss_mb(), "load_s": backend.load_s}


def bench_llm_concurrency(backend, prompt: str, schema: dict, system: str, concurrency=(1, 2, 4), max_tokens: int = 60) -> pd.DataFrame:
    out = []
    for c in concurrency:
        t0 = time.perf_counter()
        lat = []

        def one():
            t = time.perf_counter()
            backend.generate_json([{"role": "system", "content": system}, {"role": "user", "content": prompt}], schema, max_tokens=max_tokens)
            lat.append(time.perf_counter() - t)
        with ThreadPoolExecutor(c) as ex:
            [f.result() for f in [ex.submit(one) for _ in range(c)]]
        wall = time.perf_counter() - t0
        out.append({"concurrent_requests": c, "wall_s": wall, "throughput_req_per_s": c / wall, "latency_s_p50": float(np.percentile(lat, 50)), "latency_s_max": float(max(lat)),
                    "note": "llama.cpp context is serialised by a lock: concurrent requests queue (no parallel speed-up)"})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ event-processing workloads
def scale_events(events: list[dict], n: int) -> list[dict]:
    """First n deliveries; if the source stream is shorter, replicate it with suffixed patient/obligation/event ids (independent copies, same timing)."""
    if n <= len(events):
        return events[:n]
    out, k = [], 0
    while len(out) < n:
        sfx = f"~{k}"
        for e in events:
            if len(out) >= n:
                break
            if k == 0:
                out.append(e)
                continue
            c = dict(e)
            c["event_id"] = e["event_id"] + sfx
            c["delivery_id"] = (e.get("delivery_id") or e["event_id"]) + sfx
            c["patient_id"] = e["patient_id"] + sfx
            c["obligation_id"] = (e["obligation_id"] + sfx) if e.get("obligation_id") else None
            c["encounter_id"] = (e["encounter_id"] + sfx) if e.get("encounter_id") else None
            if e["event_type"] == "obligation_created":
                pl = dict(e["payload"])
                pl["depends_on"] = [d + sfx for d in pl["depends_on"]]
                pl["chain_id"] = pl["chain_id"] + sfx
                c["payload"] = pl
            out.append(c)
        k += 1
    out.sort(key=lambda e: (e["ingest_time_utc"], e["event_id"]))
    return out


def worker_run(spec: dict) -> dict:
    """Executed inside a fresh subprocess: one throughput workload."""
    from .data_loader import iter_stream, load_config
    from .event_processor import EventProcessor, deadline_only_score
    from .risk_predictor import RiskModel
    cfg = load_config()
    cfg["mode"] = spec.get("mode", cfg["mode"])
    n = spec["n_events"]
    base = list(iter_stream(cfg))
    ev = scale_events(base, n)
    scorers = {}
    if spec["scoring"] == "ml":
        scorers = {"C_graph_ml": RiskModel.load(ROOT / cfg["paths"]["models"] / "risk_C_graph.joblib")}
    elif spec["scoring"] == "deadline":
        scorers = {"A_deadline_only": deadline_only_score}
    db = ":memory:" if spec["storage"] == "memory" else str(ROOT / cfg["paths"]["results"] / f"bench_{os.getpid()}.sqlite")
    p = EventProcessor(cfg, db, scorers=scorers, primary="C_graph_ml", commit_every=spec.get("commit_every", 200), rescore_batch=spec.get("rescore_batch", 64), log_risk=False,
                       build_nx=spec.get("build_nx", True), track_latency=True, enable_alerts=spec.get("alerts", True))
    rss0 = peak_rss_mb()
    t0 = time.perf_counter()
    r = p.run(ev)
    p.finalize()
    wall = time.perf_counter() - t0
    lat = np.array(p.latencies) * 1000
    out = {**spec, "events": r["events"], "wall_s": wall, "events_per_s": r["events"] / wall, "latency_ms": pstats(lat), "peak_rss_mb": peak_rss_mb(), "rss_before_mb": rss0,
           "rejected": int(p.stats["rejected"]), "error_rate": p.stats["rejected"] / max(r["events"], 1), "duplicates": int(p.stats["duplicates"]), "transitions": int(p.stats["transitions"]),
           "alerts": int(sum(v for k, v in p.stats.items() if str(k).startswith("alert:"))), "graph": p.graph.stats(), "integrity": p.integrity_check() if spec["storage"] != "memory" else None}
    p.db.close()
    if spec["storage"] != "memory":
        for suf in ("", "-wal", "-shm"):
            try:
                os.remove(db + suf)
            except OSError:
                pass
    return out


def run_workload(spec: dict, timeout: int = 7200) -> dict:
    code = "import json,sys; from src.benchmarks import worker_run; print('@@RESULT@@' + json.dumps(worker_run(json.loads(sys.argv[1]))))"
    r = subprocess.run([sys.executable, "-c", code, json.dumps(spec)], cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
    for line in r.stdout.splitlines()[::-1]:
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    return {**spec, "error": (r.stderr or r.stdout)[-600:]}


def bench_queue(process_fn, events: list[dict], rate_eps: float, maxsize: int = 1000, drop_when_full: bool = False, finalize_fn=None) -> dict:
    """Producer feeds events at `rate_eps` into a bounded queue; one consumer processes. Reports max backlog, drops (event loss), drain time
    and the enqueue->processed latency (queueing delay + processing) of every consumed event."""
    q: queue.Queue = queue.Queue(maxsize=maxsize)
    stats = {"produced": 0, "dropped": 0, "consumed": 0, "max_backlog": 0}
    lat: list[float] = []
    done = threading.Event()

    def consumer():
        while not (done.is_set() and q.empty()):
            try:
                e, t_enq = q.get(timeout=0.05)
            except queue.Empty:
                continue
            process_fn(e)
            lat.append(time.perf_counter() - t_enq)
            stats["consumed"] += 1
    th = threading.Thread(target=consumer)
    th.start()
    t0 = time.perf_counter()
    for i, e in enumerate(events):
        target = t0 + i / rate_eps
        while time.perf_counter() < target:
            time.sleep(min(0.001, max(target - time.perf_counter(), 0)))
        try:
            if drop_when_full:
                q.put_nowait((e, time.perf_counter()))
            else:
                q.put((e, time.perf_counter()))
            stats["produced"] += 1
        except queue.Full:
            stats["dropped"] += 1
        stats["max_backlog"] = max(stats["max_backlog"], q.qsize())
    t_prod = time.perf_counter() - t0
    done.set()
    th.join()
    if finalize_fn is not None:
        finalize_fn()                                  # e.g. flush the last scoring micro-batch (counted in total_wall_s)
    pc = np.percentile(np.array(lat) * 1000, [50, 95, 99]) if lat else [float("nan")] * 3
    return {**stats, "offered_rate_eps": rate_eps, "produce_wall_s": t_prod, "ideal_produce_wall_s": len(events) / rate_eps, "total_wall_s": time.perf_counter() - t0,
            "event_loss": stats["dropped"], "loss_rate": stats["dropped"] / max(len(events), 1), "queue_latency_ms_p50": float(pc[0]), "queue_latency_ms_p95": float(pc[1]),
            "queue_latency_ms_p99": float(pc[2]), "policy": "drop" if drop_when_full else "block"}


# ------------------------------------------------------------------------------------------------ accelerator evidence
def nnpa_probe() -> dict:
    """Evidence about the IBM Z NNPA accelerator. Reports presence ONLY (cpu flag, facility bit, zDNN library); acceleration claims require a device test."""
    hw = hardware_report()
    zl = _sh("ldconfig -p | grep -i zdnn")
    return {"cpu_flag_nnpa": hw["nnpa_cpu_flag"], "facility_165_listed": hw["nnpa_facility_165_listed"], "libzdnn_installed": bool(zl), "libzdnn": zl.splitlines()[:2], "podman": hw["podman"], "docker": hw["docker"],
            "onnxruntime_available": hw["packages"].get("onnxruntime") is not None, "zdlc_note": "IBM zDLC (onnx-mlir) is distributed as a container image; it needs podman/docker + registry access",
            "xgboost_uses_nnpa": False, "sklearn_uses_nnpa": False}


def nnpa_matmul_bench(sizes=((64, 64, 64, 50), (256, 256, 256, 30), (512, 512, 512, 10), (1024, 1024, 1024, 5))) -> dict:
    """Device-level NNPA test via IBM zDNN (scripts/nnpa_matmul_bench.c): returns {'available': False, reason} on hosts without NNPA/libzdnn.
    Reports NNPA matmul vs a plain C loop AND vs numpy/BLAS float32 on the same host, plus the numerical difference (NNPA computes in 16-bit DLFLOAT)."""
    src = ROOT / "scripts" / "nnpa_matmul_bench.c"
    exe = ROOT / "scripts" / "nnpa_matmul_bench"
    if platform.machine() != "s390x":
        return {"available": False, "reason": f"host architecture is {platform.machine()}, not s390x"}
    if not _sh("ldconfig -p | grep -i libzdnn") or not shutil.which("gcc"):
        return {"available": False, "reason": "libzdnn and/or gcc not installed (sudo apt install libzdnn-dev build-essential)"}
    if not exe.exists() or exe.stat().st_mtime < src.stat().st_mtime:
        r = subprocess.run(["gcc", "-O2", "-o", str(exe), str(src), "-lzdnn", "-lm"], capture_output=True, text=True)
        if r.returncode:
            return {"available": False, "reason": "compile failed: " + r.stderr[-300:]}
    rows = []
    for m, k, n, it in sizes:
        r = subprocess.run([str(exe), str(m), str(k), str(n), str(it)], capture_output=True, text=True, timeout=300)
        line = [l for l in r.stdout.splitlines() if l.startswith("{")]
        if not line:
            return {"available": False, "reason": "zdnn run failed: " + (r.stdout + r.stderr)[-300:]}
        d = json.loads(line[-1])
        a = np.random.default_rng(0).random((m, k), dtype=np.float32) - 0.5
        b = np.random.default_rng(1).random((k, n), dtype=np.float32) - 0.5
        a @ b
        t = time.perf_counter()
        reps = 20 if m <= 256 else 5
        for _ in range(reps):
            a @ b
        np_ms = (time.perf_counter() - t) / reps * 1000
        d["numpy_blas_ms"] = np_ms
        d["numpy_blas_gflops"] = 2.0 * m * k * n / np_ms / 1e6
        d["speedup_op_vs_numpy"] = np_ms / d["nnpa_op_ms"]
        d["speedup_total_vs_numpy"] = np_ms / d["nnpa_total_ms"]
        rows.append(d)
    return {"available": True, "zdnn_version": rows[0]["zdnn_version"], "rows": rows,
            "interpretation": "Hardware NNPA execution is evidenced by a successful zDNN matmul at TFLOP/s-class rates with small 16-bit rounding differences. This does NOT mean the tree/linear risk models use NNPA: only dense neural layers do."}
