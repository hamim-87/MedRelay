"""Assemble MedRelay_LinuxONE_End_to_End.ipynb from the cell definitions (scripts/nb_cells_{a,b,c}.py)."""
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nb_cells_a, nb_cells_b, nb_cells_c  # noqa: E402

cells = nb_cells_a.CELLS + nb_cells_b.CELLS + nb_cells_c.CELLS
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src) for kind, src in cells]
nb.metadata["kernelspec"] = {"name": "medrelay-linuxone", "display_name": "MedRelay (LinuxONE)", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
out = HERE.parent / "MedRelay_LinuxONE_End_to_End.ipynb"
nbf.write(nb, out)
print(f"wrote {out.name}: {len(nb.cells)} cells ({sum(c.cell_type == 'code' for c in nb.cells)} code)")
