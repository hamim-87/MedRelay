#!/bin/bash
# Runs the FULL-mode notebook detached from any terminal and archives results when it finishes (started with: setsid nohup scripts/run_full_detached.sh &).
# Writes logs/full_run_status.json (exit code, times), logs/rss_by_cell.csv (kernel memory per cell) and, on success, reports/full_run/.
cd "$(dirname "$0")/.."
export MALLOC_ARENA_MAX=2
start=$(date -u +%FT%TZ)
echo "{\"started\": \"$start\", \"finished\": null, \"exit_code\": null}" > logs/full_run_status.json
.venv/bin/python scripts/run_notebook.py --mode full --reset --output results/nb_full.ipynb > logs/nb_full2.log 2>&1
rc=$?
echo "{\"started\": \"$start\", \"finished\": \"$(date -u +%FT%TZ)\", \"exit_code\": $rc}" > logs/full_run_status.json
if [ $rc -eq 0 ]; then
  rm -rf reports/full_run && mkdir -p reports/full_run
  cp -r results reports/full_run/results && cp -r figures reports/full_run/figures
  cp results/nb_full.ipynb reports/full_run/MedRelay_LinuxONE_End_to_End.executed_full.ipynb
  .venv/bin/python scripts/make_report.py > logs/make_report_full.log 2>&1 && cp reports/final_report.md reports/full_run/final_report_full.md
fi
