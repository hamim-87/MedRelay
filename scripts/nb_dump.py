"""Dump text outputs of code cells (truncated) for review: nb_dump.py nb.ipynb first last [maxchars]"""
import sys, nbformat
nb = nbformat.read(sys.argv[1], as_version=4)
a, b = int(sys.argv[2]), int(sys.argv[3]); mx = int(sys.argv[4]) if len(sys.argv) > 4 else 1400
for i, c in enumerate(nb.cells):
    if c.cell_type != "code" or not (a <= i <= b):
        continue
    print(f"\n=== cell {i}: {c.source[:90].splitlines()[0]}")
    for o in c.get("outputs", []):
        if o.output_type == "stream":
            print(o.text[:mx])
        elif o.output_type in ("display_data", "execute_result"):
            if "text/plain" in o.data and "image/png" not in o.data:
                print(o.data["text/plain"][:mx])
            elif "image/png" in o.data:
                print("[image]")
        elif o.output_type == "error":
            print("ERROR", o.ename, o.evalue[:300])
