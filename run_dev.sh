#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
PY=/raid/home/CAMCA/hj880/venvs/perr/bin/python
for c in oneshot harness harness_wm; do
  $PY -m perr.run --model qwen3vl32b --ports 8001 --cond $c --offset 50 --n 10 --out runs/dev1 --concurrency 10 > logs/dev1_$c.log 2>&1 &
done
wait
echo "DEV1 DONE"
