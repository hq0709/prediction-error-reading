#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
PY=/raid/home/CAMCA/hj880/venvs/perr/bin/python
for c in oneshot harness harness_wm; do
  $PY -m perr.run --model qwen3vl32b --ports 8001 --cond $c --offset 60 --n 100 --max-calls 24 --out runs/fresh --concurrency 14 > logs/fresh_$c.log 2>&1 &
done
wait
echo "FRESH DONE"
