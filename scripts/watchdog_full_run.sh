#!/bin/bash
# Logs free memory and the notebook's last executed cell every 2 minutes until the full run reports it has finished (logs/watchdog.log).
cd "$(dirname "$0")/.."
while grep -q '"finished": null' logs/full_run_status.json 2>/dev/null; do
  echo "$(date -u +%FT%TZ) avail_mb=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) last_cell(exec,rss_mb,t_s)=$(tail -1 logs/rss_by_cell.csv 2>/dev/null)" >> logs/watchdog.log
  sleep 120
done
echo "$(date -u +%FT%TZ) run ended: $(cat logs/full_run_status.json)" >> logs/watchdog.log
