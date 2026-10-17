import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_config
from src.eval_sets import challenge_docs
from src.clinical_extractor import LLMExtractor, ValidatedExtractor
from src.llm_backend import get_backend
cfg = load_config(); be = get_backend(cfg, sys.argv[2] if len(sys.argv) > 2 else "primary")
docs = challenge_docs()[int(sys.argv[1]):int(sys.argv[1]) + 4]
for d in docs:
    print("TEXT:", d["text"])
    print(" TRUTH:", [(o["task_type"], o["span"][:40], o["due_interval"], o["role"], o["depends_on"]) for o in d["obligations"]])
    r = ValidatedExtractor(be, max_retries=0).extract(d["text"], d["doc_id"])
    print(" RAW  :", r.raw_text[:600])
    print(" C    :", [(o.task_type, o.evidence_span.text[:40], o.due_interval, o.responsible_role, o.dependencies) for o in r.obligations], "rej", [x["reason"] for x in r.rejected], "rep", [(x["field"], x["action"]) for x in r.repairs])
