#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
PY=/raid/home/CAMCA/hj880/venvs/perr/bin/python
for c in oneshot harness harness_wm; do
  $PY -m perr.run_followup --model qwen25vl72b --ports 8003 --cond $c --offset 0 --n 60 --out runs/fu_pilot --concurrency 6 > logs/fu72_$c.log 2>&1 &
done
$PY -m perr.run --model qwen25vl72b --ports 8003 --cond harness --offset 160 --n 100 --max-calls 24 --out runs/fresh2 --concurrency 6 > logs/fresh2_harness.log 2>&1 &
wait
echo "72B REST DONE"
