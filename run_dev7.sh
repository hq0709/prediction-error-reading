#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
/raid/home/CAMCA/hj880/venvs/perr/bin/python -m perr.run --model qwen25vl72b --ports 8003 --cond harness_wm --offset 50 --n 10 --max-calls 24 --out runs/dev7 --concurrency 10 > logs/dev7_harness_wm.log 2>&1
echo "DEV7 DONE"
