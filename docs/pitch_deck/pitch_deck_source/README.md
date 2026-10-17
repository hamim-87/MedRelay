# Pitch deck source

`build_deck.py` builds the deck from the archived results in `evidence/` (copied from the LinuxONE VM: `results/`, `figures/`, the executed notebook, README, docs, config).
`deckkit.py` is a small slide DSL with two backends (ReportLab for the PDF, python-pptx for the PPTX); `facts.py` reads every number from the result files; `charts.py` draws the charts;
`verify_claims.py` checks the typed claims against the evidence. Optional `team.json`: `{"name": "...", "members": ["..."]}`.
