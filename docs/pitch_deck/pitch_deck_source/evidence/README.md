# Evidence for the pitch deck

Copied unmodified from the LinuxONE VM project (`~/medrelay-linuxone/reports/full_run/`): the results and figures of the executed **full-mode** notebook run
(clean kernel, 56 code cells, 0 errors, 251 min on the s390x VM; started 2026-10-17 23:48 UTC), plus the executed notebook and `final_report_full.md`.
`results/benchmark_peak_memory.csv` was produced afterwards by `scripts/bench_peak_memory.py` (per-workload `VmHWM`, because the notebook's own `peak_rss_mb` column is the kernel's lifetime peak).
The deck reads every number from `results/` (see `../facts.py`, `../verify_claims.py`). No credentials, host addresses or keys are included.
The earlier fast-mode run (10 min) is archived in the repository under `reports/fast_run/`.
