#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
PY=/raid/home/CAMCA/hj880/venvs/perr/bin/python
for c in oneshot harness harness_wm; do
  $PY -m perr.run_followup --model qwen3vl32b --ports 8001 --cond $c --offset 60 --n 10 --out runs/fu_dev1 --concurrency 10 > logs/fudev1_$c.log 2>&1 &
done
wait
echo "FUDEV1 DONE"
