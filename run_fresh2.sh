#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
PY=/raid/home/CAMCA/hj880/venvs/perr/bin/python
for c in oneshot harness_wm; do
  $PY -m perr.run --model qwen25vl72b --ports 8003 --cond $c --offset 160 --n 100 --max-calls 24 --out runs/fresh2 --concurrency 12 > logs/fresh2_$c.log 2>&1 &
done
wait
echo "FRESH2 DONE"
