#!/bin/bash
# SAT-Pro eval run: 200 LUNA16 scans (fresh1 + fresh2), then 53 MSD-liver cases. GPU 0 only.
cd /raid/home/CAMCA/hj880/perr/sota/sat_pro
export HF_HOME=/raid/home/CAMCA/hj880/hf_home CUDA_VISIBLE_DEVICES=0
PY=/raid/home/CAMCA/hj880/venvs/sota_sat/bin/python
date; $PY run_sat.py --ds luna --cases _work/eval_luna.txt --out /raid/home/CAMCA/hj880/perr/results/sota/sat_pro_luna.jsonl --port 29517 > logs/eval_luna.log 2>&1
echo "luna exit $?"; date
$PY run_sat.py --ds liver --cases ../eval_liver.txt --out /raid/home/CAMCA/hj880/perr/results/sota/sat_pro_liver.jsonl --port 29518 > logs/eval_liver.log 2>&1
echo "liver exit $?"; date
