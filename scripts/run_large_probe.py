import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
os.environ["MEDRELAY_MODE"] = "fast"
from src.exp_extraction import *
lab = Lab(); t = time.time()
r = run_exp1(lab, which="large", n_chal=6, n_tpl=6, include=["A_rules", "B_llm_raw", "C2_llm_validated_filled", "E_rules_plus_llm"])
print(r["table"][["extractor", "corpus", "typed_precision", "typed_recall", "typed_f1", "det_f1", "deadline_acc", "unsupported_field_rate", "schema_valid", "latency_mean_s"]].round(2).to_string())
print("total", round(time.time() - t, 1))
