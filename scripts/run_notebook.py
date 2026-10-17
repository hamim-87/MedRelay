"""Execute the master notebook with nbclient from a clean kernel.

  python scripts/run_notebook.py --mode fast            # development subset
  python scripts/run_notebook.py --mode full            # every available record (long)
  python scripts/run_notebook.py --mode fast --reset    # clear results/figures/models first (keeps the LLM cache)

The executed notebook (with outputs) is written back to MedRelay_LinuxONE_End_to_End.ipynb (or --output)."""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["fast", "full"], default="fast")
ap.add_argument("--notebook", default=str(ROOT / "MedRelay_LinuxONE_End_to_End.ipynb"))
ap.add_argument("--output", default=None)
ap.add_argument("--timeout", type=int, default=7200, help="per-cell timeout (s)")
ap.add_argument("--reset", action="store_true")
a = ap.parse_args()

os.environ["MEDRELAY_MODE"] = a.mode
os.environ["MEDRELAY_HEADLESS"] = "1"
os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"   # figures are embedded as PNG in the executed notebook
if a.reset:
    for d in ("results", "figures"):
        shutil.rmtree(ROOT / d, ignore_errors=True)
        (ROOT / d).mkdir(exist_ok=True)
KERNEL = "medrelay-linuxone"          # kernelspec pointing at THIS interpreter (the venv with llama-cpp-python etc.)
subprocess.run([sys.executable, "-m", "ipykernel", "install", "--user", "--name", KERNEL, "--display-name", "MedRelay (LinuxONE)"], check=True, capture_output=True)
nb = nbformat.read(a.notebook, as_version=4)
for c in nb.cells:                                   # start from a clean slate: drop stale outputs
    if c.cell_type == "code":
        c.outputs, c.execution_count = [], None
t = time.time()
client = NotebookClient(nb, timeout=a.timeout, kernel_name=KERNEL, resources={"metadata": {"path": str(ROOT)}}, allow_errors=False)
try:
    client.execute()
    status = "OK"
except Exception as e:  # noqa: BLE001
    status = f"FAILED: {type(e).__name__}: {str(e)[:800]}"
nbformat.write(nb, a.output or a.notebook)
print(f"[run_notebook] mode={a.mode} status={status} wall={(time.time()-t)/60:.1f} min -> {a.output or a.notebook}")
sys.exit(0 if status == "OK" else 1)
