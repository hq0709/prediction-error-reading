#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
PY=/raid/home/CAMCA/hj880/venvs/perr/bin/python
for c in harness harness_wm; do
  $PY -m perr.run --model qwen3vl32b --ports 8001 --cond $c --offset 50 --n 10 --max-calls 24 --out runs/dev4 --concurrency 10 > logs/dev4_$c.log 2>&1 &
  $PY -m perr.run_followup --model qwen3vl32b --ports 8001 --cond $c --offset 60 --n 10 --out runs/fu_dev4 --concurrency 10 > logs/fudev4_$c.log 2>&1 &
done
wait
echo "DEV4 DONE"
