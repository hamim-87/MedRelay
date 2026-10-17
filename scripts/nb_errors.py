"""Print errors (and the cell index/source head) from an executed notebook."""
import sys
import nbformat
nb = nbformat.read(sys.argv[1], as_version=4)
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    for o in c.get("outputs", []):
        if o.output_type == "error":
            print(f"--- cell {i} ERROR {o.ename}: {o.evalue[:600]}")
            print("\n".join(l for l in o.traceback[-6:])[:1800])
            print("source head:", c.source[:300].replace("\n", " | "))
print("cells with outputs:", sum(1 for c in nb.cells if c.cell_type == "code" and c.get("outputs")), "of", sum(c.cell_type == "code" for c in nb.cells))
