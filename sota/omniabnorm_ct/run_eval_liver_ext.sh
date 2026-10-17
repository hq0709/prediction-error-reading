#!/bin/bash
# OmniAbnorm-CT on the external liver test set (HCC-TACE-Seg, 93 cases): the one eval run, exactly the MSD liver
# protocol (same query, window, every-slice sweep, mask -> answer rule, batch 32). GPU 3 only, OMP_NUM_THREADS=8.
cd /raid/home/CAMCA/hj880/perr/sota/omniabnorm_ct
export HF_HOME=/raid/home/CAMCA/hj880/hf_home CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
PY=/raid/home/CAMCA/hj880/venvs/sota_omniabnorm/bin/python
OUT=/raid/home/CAMCA/hj880/perr/results/sota
date
$PY run_omniabnorm.py --ds liver_ext --cases ../eval_liver_ext.txt --out $OUT/omniabnorm_ct_liver_ext.jsonl > logs/eval_liver_ext.log 2>&1
echo "liver_ext exit $?"; date
