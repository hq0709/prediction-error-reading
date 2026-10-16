#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
PY=/raid/home/CAMCA/hj880/venvs/perr/bin/python
for c in harness harness_wm; do
  $PY -m perr.run_followup --model qwen3vl32b --ports 8001 --cond $c --offset 0 --n 60 --out runs/fu_pilot --concurrency 12 > logs/fupilot_$c.log 2>&1 &
done
wait
echo "FUPILOT DONE"
