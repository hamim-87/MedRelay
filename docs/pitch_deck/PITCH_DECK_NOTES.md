# MedRelay pitch deck: presenter notes, evidence and limits

Deck: `IBM_Z_Datathon_2026_Pitch_Deck.pdf` (10 slides, 16:9) and the editable `IBM_Z_Datathon_2026_Pitch_Deck.pptx` (same slides, speaker notes inside).
Matching competition theme (official page, starhacks.org/ibm-z-datathon-2026): **Real-Time AI for Critical Decisions, "Detect. Decide. Respond."** The event page lists no judging criteria or file limits, so none are claimed as official.
Target length: about 3 to 4 minutes (about 20 seconds per slide, 40 seconds on slides 6 and 8).
**Results shown are from the completed full-mode run** (4 h 11 min, 812 test patients, 4,605 test obligations); the earlier fast-mode run is archived in the repository (`reports/fast_run/`).

## Talking points

1. **Cover (10 s).** MedRelay finds clinical follow-up obligations in notes, tracks them against real evidence, and warns early about the ones likely to be missed. Fits the "Detect. Decide. Respond." theme. Built and run on an IBM LinuxONE VM. Team 4ier.
2. **Problem (25 s).** Obligations hide in free text, depend on each other, and documentation lags reality. Naive tools treat "no record" as "failed". In our simulated test the naive rule flagged 716 of 716 done-but-undocumented tasks as overdue; MedRelay confirmed none.
3. **Solution (25 s).** Extract, track, warn. The LLM only proposes evidence-grounded records; deterministic rules decide state; absence of evidence means "unverified", never "failed".
4. **Architecture (30 s).** Notes and the event stream enter the LinuxONE VM. Extractor (rules + Granite 350M via llama.cpp, Pydantic validation), evidence graph with state rules R1 to R7, idempotent event processor with SQLite, calibrated risk model (a random forest was selected on validation), alert engine with stated uncertainty, priority score, audit trail.
5. **Data and ML (25 s).** Synthetic and simulated data plus one real de-identified no-show set. Strict event-time features, dense snapshots, thresholds fixed on validation, patient-grouped bootstrap. Say clearly: no gold labels, outcomes simulated.
6. **LinuxONE (40 s).** s390x big-endian under z/VM 7.3, 2 vCPU, 4.2 GB (about 2.6 GB usable); llama.cpp built with VXE; Granite GGUF converted to big-endian; direct zDNN matmul on the NNPA (339x faster than NumPy BLAS on this VM, including data conversion). Be explicit: the tree models do not use NNPA; ML-scoring speed depends on the selected model (random forest: 88 to 122 events/s); event-engine peak memory 753 MB at 100,000 events.
7. **Results (35 s).** At 24 h or more of lead: 11.3% false-alert rate for the graph model vs 18.8% for a deadline-only rule, with 72 h median lead (a 24 h reminder: 12.0% with 24 h lead and higher recall). Over the whole window all three rankings are statistically indistinguishable; the graph model's edge is early (PR-AUC 0.74 vs 0.58 at 72 h ahead). Graph features beat independent tasks by 0.043 PR-AUC on the dense test (significant). Evidence-aware engine recognises 99% of unverified tasks; the naive rule 0%.
8. **Live prototype (40 s).** Walk the audit trail: pending, censored at the deadline, overdue when the tracker confirms it is open, completed when the delayed acknowledgement arrives. Real notebook output; the scenario events are scripted.
9. **Innovation and impact (25 s).** Left column measured, right column anticipated. Do not blur them.
10. **Conclusion (15 s).** Three achievements, why LinuxONE, next steps, repository link. End on: "Detect. Decide. Respond. And never mistake silence for failure."

## Evidence for every claim (`pitch_deck_source/evidence/`, copied from the VM)

All numbers are read by code from these files (`facts.py`, `charts.py`) and re-checked by `verify_claims.py` (40 of 40 checks pass).

| slide | claim | source file |
|---|---|---|
| 2, 7, 9, 10 | 716 done-but-undocumented tasks: naive 100% overdue, engine 0 confirmed; 99% of unverified tasks recognised vs 0% | `results/exp4_evidence_uncertainty.csv`, `results/exp3_summary.json` |
| 5 | 3,284 patients, 24,296 obligations, 34,441 events, 10,002 action examples, 110,521 appointments, 103,627 graph nodes, 223,925 edges | `results/data_validation.json` |
| 5 | 49 features in 6 groups, 8 dense snapshots, validation-chosen thresholds, selected model rf/C_graph | `results/exp5_feature_ablation_dense_test.csv`, `results/model_b_summary.json`, executed notebook |
| 6 | s390x, big-endian, z/VM 7.3.0, 2 vCPU, 4.18 GB, VXE2 and NNPA flags, no Docker or Podman | `results/hardware.json` |
| 6 | llama.cpp `VXE = 1`; Granite 350M Q8_0 at 15.0 tokens/s | executed notebook (section 7), `results/benchmarks.json` |
| 6 | zDNN 1.0.1 matmul 1024^3: NNPA 0.49 ms op / 1.15 ms with conversion, NumPy BLAS 389 ms, error 0.045 | `results/benchmarks.json` (`nnpa_matmul`) |
| 6 | 20.6k events/s (graph + states, 1k events), 12.4k at 100k; 88 to 122 events/s with ML risk; p99 queue latency 2.8 s at 2x; 39% / 67% shed at 2x / 4x | `results/benchmarks.json` (`throughput`, `queue`) |
| 6 | event-engine peak memory 753 MB at 100,000 events (ML run), per-workload `VmHWM` | `results/benchmark_peak_memory.csv` |
| 7 | false-alert rate 11.3% / 12.6% / 12.0% / 18.8% (C / B / reminder / deadline-only), recall, median lead, precision | `results/exp3_early_warning.csv` (lead >= 24 h, recall target 0.7) |
| 7 | ablation deltas with 95% CIs; PR-AUC 0.893 [0.882, 0.904]; lead-restricted PR-AUC | `results/exp5_feature_ablation_dense_test.csv`, `results/exp3_summary.csv`, `results/final_summary.csv` |
| 7 | no-show PR-AUC 0.358 vs 0.203 base rate | `results/model_c_comparison.csv` |
| 7 | rules + LLM surfaced 45 of 67 failures vs 38 for rules only (120 episodes) | `results/exp5_genai_end_to_end_primary.csv` |
| 8 | note text, extracted obligations, audit trail, risk figure | executed notebook, section 17 (`figures/demo_risk_monitor.png`) |
| 9 | 4,895 duplicates ignored, states identical; states restored 1.000 from the event log; 56 of 56 cells, 0 errors, 4 h 11 min | `results/exp4_evidence_uncertainty.csv`, executed notebook |

Dataset licences and origin URLs: `medrelay-dataset/docs/PROVENANCE.md` (Synthea, synthetic_hospital MIT, MTS-Dialog CC BY 4.0, ACI-Bench CC BY 4.0, Kaggle no-show CC-BY-NC-SA 4.0, which is non-commercial).

## Deliberately left out or not verified

* **Team**: "4ier" and the five members come from the team-page screenshot the user supplied (names re-capitalised, spelling unchanged; email addresses deliberately not shown). They live in `pitch_deck_source/team.json`.
* **Demo link**: no hosted demo exists; the demo is the notebook in the repository.
* **External statistics** on missed follow-ups or alert fatigue: not verified from a source in the repository, so none are quoted.
* **Run-to-run variation**: throughput and some injected-noise statistics differ between runs on the shared VM (the fast-mode run measured XGBoost scoring at about 350 to 470 events/s; the full-mode run selected a random forest at about 90 to 120). The deck shows the full-mode run only.
* **Benchmark memory column**: the `peak_rss_mb` values printed in the archived notebooks are the notebook kernel's lifetime peak, not per-workload memory; the deck uses the corrected per-workload measurement.
* **Not claimed**: clinical accuracy, NNPA acceleration of the ML models, security, cost or availability benefits, gold-label performance.
* **Small samples**: extraction was evaluated on 40 + 60 documents (wide intervals); lead-restricted PR-AUC values have no confidence intervals.
* The notebook's own graph figures (`graph_t31.png`, `graph_t152.png`) have clipped node labels, so they were not used; the audit trail is shown as text instead.

## Quality checks performed

* PDF: 10 pages, 960 x 540 pt, fonts embedded, each page rendered to an image and inspected; text-fit checker reports no overflow; no blank pages.
* PPTX: cannot be rendered in this environment (LibreOffice is not installed), so it was checked programmatically instead: 10 slides, every shape inside the slide, text boxes sized with the same Arial-compatible metrics as the PDF. Please open it once in PowerPoint or Keynote and look through the slides before presenting.
* Claims: `verify_claims.py` 40 of 40 pass.

## Regenerate

```bash
cd pitch_deck_source
python3 -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python verify_claims.py && .venv/bin/python build_deck.py   # writes the PDF and PPTX one folder up
```
