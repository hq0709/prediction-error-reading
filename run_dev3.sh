#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
PY=/raid/home/CAMCA/hj880/venvs/perr/bin/python
for c in harness harness_wm; do
  $PY -m perr.run --model qwen3vl32b --ports 8001 --cond $c --offset 50 --n 10 --max-calls 24 --out runs/dev3 --concurrency 10 > logs/dev3_$c.log 2>&1 &
  $PY -m perr.run_followup --model qwen3vl32b --ports 8001 --cond $c --offset 60 --n 10 --out runs/fu_dev3 --concurrency 10 > logs/fudev3_$c.log 2>&1 &
done
$PY -m perr.run_followup --model qwen3vl32b --ports 8001 --cond oneshot --offset 60 --n 10 --out runs/fu_dev3 --concurrency 10 > logs/fudev3_oneshot.log 2>&1 &
wait
echo "DEV3 DONE"
