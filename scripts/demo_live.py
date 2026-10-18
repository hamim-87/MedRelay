"""Live terminal demo (about 10 s, no LLM wait): platform evidence + the 'delayed acknowledgement' scenario streamed through the real MedRelay engine.

    .venv/bin/python scripts/demo_live.py            # add --pause to wait for Enter between stages (for a screen recording)
Scripted events (synthetic). Every state change below comes from the deterministic rules in src/evidence_graph.py; nothing is typed by hand.
"""
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
PAUSE = "--pause" in sys.argv
C = {"pending": "\033[94m", "censored_not_yet_assessable": "\033[93m", "overdue_without_verified_completion": "\033[91m", "completed_with_evidence": "\033[92m",
     "unverified_insufficient_evidence": "\033[95m", "contradictory_evidence": "\033[95m"}
B, DIM, OFF, CY = "\033[1m", "\033[2m", "\033[0m", "\033[96m"


def stage(title):
    if PAUSE:
        input(f"\n{DIM}[Enter] {OFF}")
    print(f"\n{B}{CY}== {title}{OFF}")


def main():
    stage("1. Platform (this machine)")
    flags = set(open("/proc/cpuinfo").read().split())
    print(f"machine={platform.machine()}  byte order={sys.byteorder}-endian  cpus={os.cpu_count()}  python={platform.python_version()}")
    print(f"vector facility (vxe2): {'vxe2' in flags}   NNPA accelerator flag: {'nnpa' in flags}")
    try:
        import llama_cpp
        print("llama.cpp:", llama_cpp.llama_print_system_info().decode().strip().splitlines()[0][:110])
    except Exception as e:                                   # pragma: no cover
        print("llama.cpp not importable here:", type(e).__name__)

    stage("2. NNPA device test (IBM zDNN matmul 1024x1024x1024, vs NumPy BLAS on the CPU)")
    from src import benchmarks as bm
    r = bm.nnpa_matmul_bench(sizes=((1024, 1024, 1024, 3),))
    if r.get("available"):
        x = r["rows"][0]
        print(f"zDNN {r['zdnn_version']}: NNPA {x['nnpa_op_ms']:.2f} ms (op) / {x['nnpa_total_ms']:.2f} ms (with data conversion)   NumPy BLAS {x['numpy_blas_ms']:.0f} ms   naive C loop {x['cpu_naive_c_ms']:.0f} ms")
        print(f"speed-up incl. conversion: {x['speedup_total_vs_numpy']:.0f}x   max abs error {x['max_abs_err']:.3f} on values up to {x['max_abs_ref']:.1f} (16-bit arithmetic)")
        print(f"{DIM}Only dense neural layers could use the NNPA; the tree/linear risk models in MedRelay run on the CPU.{OFF}")
    else:
        print("NNPA test not available:", r.get("reason"))

    stage("3. Clinical note -> follow-up obligations (synthetic)")
    from src.data_loader import load_config
    from src.event_processor import EventProcessor
    from src.scenario_lab import demo_events
    print("Patient discharged after pneumonia admission. Repeat the CBC within 48 hours. The treating physician must review the result.")
    print("Contact the patient if the result is abnormal. Arrange a follow-up appointment in 2 weeks.")
    events = demo_events("delayed_ack")
    step = {e["obligation_id"]: e["payload"].get("step_type", "?") for e in events if e["event_type"] == "obligation_created"}
    print(f"{DIM}extracted + monitored obligations: " + ", ".join(f"{o.split(':')[-1]}={s}" for o, s in step.items()) + OFF)

    stage("4. Events stream in (late and out of order); states come from deterministic rules")
    cfg = load_config()
    p = EventProcessor(cfg, ":memory:", scorers={}, enable_alerts=True)
    seen = 0
    for e in sorted(events, key=lambda e: e["ingest_time_utc"]):
        p.process(e)
        label = e["obligation_id"].split(":")[-1] if e["obligation_id"] else "-"
        late = "  (documented late)" if e["ingest_time_utc"][:10] != e["event_time_utc"][:10] else ""
        print(f"{DIM}{e['ingest_time_utc']}  {e['event_type']:<22} {label}{late}{OFF}")
        for t in p.transitions[seen:]:
            ns = t["new_state"]
            print(f"    {B}{t['oid'].split(':')[-1]} {step.get(t['oid'], ''):<24}{OFF} {t['old_state'] or 'new':<34} -> {C.get(ns, '')}{ns}{OFF}   [{t['rule'].split('_')[0]}]")
        seen = len(p.transitions)
        time.sleep(0.15 if not PAUSE else 0)
    p.finalize()
    for t in p.transitions[seen:]:
        ns = t["new_state"]
        print(f"    {B}{t['oid'].split(':')[-1]} {step.get(t['oid'], ''):<24}{OFF} {t['old_state'] or 'new':<34} -> {C.get(ns, '')}{ns}{OFF}   [{t['rule'].split('_')[0]}]")

    stage("5. Audit trail of the result-review task (ob1): every change with its rule and evidence")
    a = p.audit_df("demo:ob1")
    for _, t in a.iterrows():
        ns = t["new_state"]
        ev = str(t["evidence"]).strip("[]").replace(chr(34), "") or "none"
        print(f"{t['t_ingest'][:16]}Z  {(t['old_state'] or 'new'):<36}-> {C.get(ns, '')}{ns:<36}{OFF}[{t['rule'].split('_')[0]}]  evidence={ev}")
    final = {o.split(":")[-1]: p.graph.derive_state(o, p.clock, live=True).state.value for o in sorted(p.graph.obs)}
    print(f"\n{B}final states:{OFF}", (", ".join(final) + " = " + next(iter(final.values()))) if len(set(final.values())) == 1 else final)
    ob1 = [t["new_state"] for _, t in a.iterrows()]
    ok = "overdue_without_verified_completion" in ob1 and ob1[-1] == "completed_with_evidence"
    print(f"{B}{'RECONCILED' if ok else 'CHECK'}:{OFF} review was correctly 'overdue' at the time, then 'completed_with_evidence' once the late acknowledgement was documented.")
    al = p.alert_log
    print(f"alerts raised: {', '.join(dict.fromkeys(x.rule for x in al))} ({sum(x.status == 'resolved' for x in al)}/{len(al)} resolved)")


if __name__ == "__main__":
    main()
