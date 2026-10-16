#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
/raid/home/CAMCA/hj880/venvs/perr/bin/python -m perr.run --model qwen3vl32b --ports 8001 --cond harness_wm --offset 50 --n 10 --max-calls 24 --out runs/dev5 --concurrency 10 > logs/dev5_harness_wm.log 2>&1
echo "DEV5 DONE"
