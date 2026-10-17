"""Quick probe of extractors A/B/C on small samples (development aid, not the reported experiment)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_config
from src.eval_sets import challenge_docs, generate_template_docs
from src.clinical_extractor import RuleExtractor, LLMExtractor, ValidatedExtractor
from src.evaluation import score_extraction
from src.llm_backend import get_backend
cfg = load_config()
n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
which = sys.argv[2] if len(sys.argv) > 2 else "primary"
docs = challenge_docs()[:n]
be = get_backend(cfg, which)
for ex in (RuleExtractor(), LLMExtractor(be), ValidatedExtractor(be, max_retries=1)):
    t = time.time(); res = [ex.extract(d["text"], d["doc_id"]) for d in docs]
    s = score_extraction(docs, res)
    print(f"{ex.name:18s} n={n} detF1={s['detection']['f1']:.2f} typedF1={s['typed']['f1']:.2f} P={s['typed']['precision']:.2f} R={s['typed']['recall']:.2f} "
          f"dl={s['deadline']['accuracy_on_matched_with_truth_deadline']:.2f} dep_f1={s['dependency']['f1']:.2f} unsup={s['unsupported_field_rate']:.2f} "
          f"schema={s['schema_valid_rate']:.2f} lat={s['latency_s']['mean']:.1f}s  wall={time.time()-t:.0f}s", flush=True)
