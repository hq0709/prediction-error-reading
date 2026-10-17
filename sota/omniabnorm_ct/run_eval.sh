#!/bin/bash
# OmniAbnorm-CT eval run (the one run): 200 LUNA16 scans (fresh1 + fresh2) and 53 MSD-liver cases,
# two processes in parallel on GPU 4 only. run_omniabnorm.py appends one line per finished case and, if restarted,
# skips cases already written (no case is run twice).
cd /raid/home/CAMCA/hj880/perr/sota/omniabnorm_ct
export HF_HOME=/raid/home/CAMCA/hj880/hf_home CUDA_VISIBLE_DEVICES=4 PYTHONDONTWRITEBYTECODE=1
PY=/raid/home/CAMCA/hj880/venvs/sota_omniabnorm/bin/python
OUT=/raid/home/CAMCA/hj880/perr/results/sota
date
$PY run_omniabnorm.py --ds luna --cases ../eval_luna_all.txt --out $OUT/omniabnorm_ct_luna.jsonl > logs/eval_luna.log 2>&1 &
P1=$!
$PY run_omniabnorm.py --ds liver --cases ../eval_liver.txt --out $OUT/omniabnorm_ct_liver.jsonl > logs/eval_liver.log 2>&1 &
P2=$!
wait $P1; echo "luna exit $?"; date
wait $P2; echo "liver exit $?"; date
