"""Notebook structure + repeated clean-kernel execution (a mini notebook is executed twice and its outputs compared)."""
import os
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "MedRelay_LinuxONE_End_to_End.ipynb"


def test_master_notebook_has_all_18_sections_with_executable_code():
    nb = nbformat.read(NB, as_version=4)
    heads = [l for c in nb.cells if c.cell_type == "markdown" for l in c.source.splitlines() if l.startswith("## ")]
    for k in range(1, 19):
        assert any(h.startswith(f"## {k}. ") for h in heads), f"section {k} missing"
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code) >= 50 and all(c.source.strip() for c in code)
    # sections must not be placeholders: every section after the intro is followed by code before the next section heading
    cur, has = None, {}
    for c in nb.cells:
        if c.cell_type == "markdown":
            for l in c.source.splitlines():
                if l.startswith("## "):
                    cur = l
                    has[cur] = False
        elif cur:
            has[cur] = True
    assert all(has.values()), [k for k, v in has.items() if not v]


MINI = ['''
import os, sys, json
from pathlib import Path
ROOT = Path.cwd(); os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.data_loader import load_config
cfg = load_config()
from src.event_processor import EventProcessor
from src.scenario_lab import demo_events
''', '''
res = {}
for v in ("delayed_ack", "missing_ack"):
    p = EventProcessor(cfg, ":memory:", scorers={}); p.run(demo_events(v)); p.finalize()
    res[v] = [(t["oid"], t["new_state"], t["rule"]) for t in p.transitions]
print("RESULT", json.dumps(res, sort_keys=True))
''', '''
from src.data_loader import validate_datasets
rep = validate_datasets(cfg, write=False)
print("HARD", rep["hard_failures"], rep["counts"]["obligations"])
''']


def run_mini():
    subprocess.run([sys.executable, "-m", "ipykernel", "install", "--user", "--name", "medrelay-linuxone"], check=True, capture_output=True)
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(s) for s in MINI]
    NotebookClient(nb, timeout=600, kernel_name="medrelay-linuxone", resources={"metadata": {"path": str(ROOT)}}).execute()
    return [o["text"] for c in nb.cells for o in c.outputs if o.output_type == "stream"]


@pytest.mark.skipif(os.environ.get("MEDRELAY_SKIP_NB_EXEC") == "1", reason="notebook execution disabled")
def test_repeated_clean_kernel_execution_is_reproducible():
    a, b = run_mini(), run_mini()
    assert a == b                                   # identical outputs across two fresh kernels
    assert any("RESULT" in s for s in a) and any("HARD [] " in s for s in a)
