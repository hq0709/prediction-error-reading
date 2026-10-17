#!/bin/bash
# SAT-Pro on the external liver set (HCC-TACE-Seg, 93 cases). GPU 3 only; CPU is shared and saturated.
cd /raid/home/CAMCA/hj880/perr/sota/sat_pro
export HF_HOME=/raid/home/CAMCA/hj880/hf_home CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8
date; /raid/home/CAMCA/hj880/venvs/sota_sat/bin/python run_sat.py --ds liver_ext --cases ../eval_liver_ext.txt \
    --out /raid/home/CAMCA/hj880/perr/results/sota/sat_pro_liver_ext.jsonl --num_workers 2 --port 29519 > logs/eval_liver_ext.log 2>&1
echo "liver_ext exit $?"; date
