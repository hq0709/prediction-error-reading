#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
PY=/raid/home/CAMCA/hj880/venvs/perr/bin/python
for c in harness harness_wm; do
  $PY -m perr.run --model qwen25vl72b --ports 8003 --cond $c --offset 160 --n 100 --max-calls 24 --out runs/fresh2 --concurrency 6 > logs/fresh2_$c.log 2>&1 &
  $PY -m perr.run_followup --model qwen25vl72b --ports 8003 --cond $c --offset 0 --n 60 --out runs/fu_pilot --concurrency 5 > logs/fu72_$c.log 2>&1 &
done
wait
echo "72B REDO DONE"
