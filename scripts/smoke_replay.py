import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from src.data_loader import *
from src.risk_predictor import *
from src.ontology import features_of, SYSTEM_FEATURES, FEATURE_GROUPS
from src.event_processor import EventProcessor, deadline_only_score
cfg = load_config(); cfg["mode"] = "fast"
F = risk_frames(cfg)
tr, ytr = binary(F["train"]); va, yva = binary(F["validation"])
fB = features_of(SYSTEM_FEATURES["B_independent_ml"]); fC = features_of(SYSTEM_FEATURES["C_graph_ml"])
mB, _ = tune_and_fit("hgb", fB, tr, ytr, va, yva, budget=1, seed=1)
mC, _ = tune_and_fit("hgb", fC, tr, ytr, va, yva, budget=1, seed=1)
pats = pd.read_csv(path_of(cfg, "source", "interim/patients.csv.gz"))
test_p = set(pats[pats.split == "test"].patient_id.sample(40, random_state=0))
ev = list(iter_stream(cfg, patients=test_p)); print("deliveries", len(ev))
p = EventProcessor(cfg, ":memory:", scorers={"A_deadline_only": deadline_only_score, "B_independent_ml": mB, "C_graph_ml": mC}, track_latency=True)
t = time.time(); r = p.run(ev); p.finalize()
print({k: v for k, v in r.items()}, "wall", round(time.time() - t, 1))
print("states", p.state_counts()); print("integrity", p.integrity_check())
lat = np.array(p.latencies) * 1000; print("event latency ms p50/p95/p99", np.percentile(lat, [50, 95, 99]).round(2))
print(p.alerts_df().rule.value_counts().to_dict()); print(p.graph.stats())
a = p.alert_log[0]; print(a.explanation)
