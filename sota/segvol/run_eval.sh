#!/bin/bash
# SegVol eval run, one invocation per dataset: luna = 200 LUNA16 scans (eval_luna_all = fresh1 + fresh2), liver = 53 MSD
# Task03 cases, liver_ext = 93 HCC-TACE-Seg cases. One GPU, assigned by the queue (queue/jobs3.tsv, jobs ctb_segvol_*).
# run_segvol.py appends one line per finished case and skips cases already written (no case is run twice).
ds=$1
cd /raid/home/CAMCA/hj880/perr/sota/segvol
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY="/raid/home/CAMCA/hj880/venvs/sota_segvol/bin/python -I -B"
case $ds in luna) L=../eval_luna_all.txt ;; liver) L=../eval_liver.txt ;; liver_ext) L=../eval_liver_ext.txt ;; *) exit 2 ;; esac
date; echo "GPU $CUDA_VISIBLE_DEVICES"
$PY run_segvol.py --ds $ds --cases $L --out /raid/home/CAMCA/hj880/perr/results/sota/segvol_$ds.jsonl > logs/eval_$ds.log 2>&1
rc=$?; echo "$ds exit $rc"; date; exit $rc
