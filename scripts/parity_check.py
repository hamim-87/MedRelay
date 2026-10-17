"""Parity: features computed by the evidence graph from the replay stream == features in the dataset pipeline's tables."""
import sys, time, math, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from src.data_loader import load_config, iter_stream, load_risk
from src.evidence_graph import TemporalEvidenceGraph, to_epoch

cfg = load_config()
g = TemporalEvidenceGraph(grace_hours=12, build_nx=False)
t = time.time(); n = 0
for ev in iter_stream(cfg):
    g.ingest(ev); n += 1
print("ingested", n, "deliveries in", round(time.time() - t, 1), "s", g.stats())
R = pd.concat(load_risk(cfg).values(), ignore_index=True)
spec = json.load(open(Path(cfg["_root"]) / "data/source/processed/followup_risk/feature_spec.json"))
cols = spec["feature_columns"]
t = time.time(); bad = {c: 0 for c in cols}; tot = 0; examples = {}
for r in R.itertuples(index=False):
    f = g.features(r.obligation_id, to_epoch(r.assessment_time_utc)); tot += 1
    for c in cols:
        a, b = f[c], getattr(r, c)
        if isinstance(b, str) or isinstance(a, str):
            ok = a == b
        else:
            ok = (isinstance(a, float) and math.isnan(a) and (b != b)) or abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b))) + 1e-4
        if not ok:
            bad[c] += 1; examples.setdefault(c, (r.obligation_id, a, b))
print("computed", tot, "feature rows in", round(time.time() - t, 1), "s")
print("mismatching features:", {k: v for k, v in bad.items() if v} or "NONE")
for c, e in list(examples.items())[:8]: print(" ", c, e)
