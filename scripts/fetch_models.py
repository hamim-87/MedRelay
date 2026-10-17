"""Download pinned GGUF models (Apache-2.0 IBM Granite 4.0) with checksums. Run once; idempotent."""
import hashlib, json, os, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
ROOT = Path(__file__).resolve().parent.parent
SPECS = [("ibm-granite/granite-4.0-350m-GGUF", "granite-4.0-350m-Q8_0.gguf", "b8208a86a58427e1739265318028eb5895b74bf2"),
         ("ibm-granite/granite-4.0-1b-GGUF", "granite-4.0-1b-Q4_0.gguf", "b27c2fe3f211b7f44e80fa620177aea371099aaa")]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 22): h.update(b)
    return h.hexdigest()
man = {}
for repo, fn, rev in SPECS:
    p = hf_hub_download(repo, fn, revision=rev, local_dir=ROOT / "models" / "llm" / "original_le")
    man[fn] = {"repo": repo, "revision": rev, "sha256_little_endian": sha(p), "bytes": os.path.getsize(p), "license": "apache-2.0"}
    print(fn, man[fn]["bytes"], flush=True)
(ROOT / "models" / "llm" / "MODELS.json").write_text(json.dumps(man, indent=2))
