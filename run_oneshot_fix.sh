#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
PY=/raid/home/CAMCA/hj880/venvs/perr/bin/python
$PY -m perr.run --model qwen3vl32b --ports 8001 --cond oneshot --offset 0 --n 50 --out runs/pilot --concurrency 8 > logs/pilot_oneshot.log 2>&1 &
$PY -m perr.run_followup --model qwen3vl32b --ports 8001 --cond oneshot --offset 0 --n 60 --out runs/fu_pilot --concurrency 8 > logs/fupilot_oneshot.log 2>&1 &
wait
echo "ONESHOT DONE"
