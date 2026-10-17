import sys, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["MEDRELAY_MODE"] = "fast"
from src.exp_extraction import *
lab = Lab(); t = time.time()
which = sys.argv[1] if len(sys.argv) > 1 else "primary"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
r = run_exp1(lab, which, n_chal=n, n_tpl=n, include=["A_rules", "B_llm_raw", "C_llm_validated", "C2_llm_validated_filled", "E_rules_plus_llm"])
print(r["table"][["extractor", "corpus", "typed_precision", "typed_recall", "typed_f1", "det_f1", "deadline_acc", "dep_f1_on_matched", "unsupported_field_rate", "schema_valid", "latency_mean_s"]].round(2).to_string())
print("total", round(time.time() - t, 1))
