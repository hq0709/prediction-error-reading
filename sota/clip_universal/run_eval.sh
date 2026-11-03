#!/bin/bash
# CLIP-Driven Universal Model eval run, one invocation per dataset: luna = 200 LUNA16 scans (eval_luna_all = fresh1 + fresh2), liver = 53 MSD
# Task03 cases, liver_ext = 93 HCC-TACE-Seg cases. One GPU, assigned by the queue (queue/jobs3.tsv, jobs ctb_cu_*).
# run_clip_universal.py appends one line per finished case and skips cases already written (no case is run twice).
ds=$1
cd /raid/home/CAMCA/hj880/perr/sota/clip_universal
PY="/raid/home/CAMCA/hj880/venvs/sota_clip_universal/bin/python -I -B"
case $ds in luna) L=../eval_luna_all.txt ;; liver) L=../eval_liver.txt ;; liver_ext) L=../eval_liver_ext.txt ;; *) exit 2 ;; esac
date; echo "GPU $CUDA_VISIBLE_DEVICES"
$PY run_clip_universal.py --ds $ds --cases $L --out /raid/home/CAMCA/hj880/perr/results/sota/clip_universal_$ds.jsonl > logs/eval_$ds.log 2>&1
rc=$?; echo "$ds exit $rc"; date; exit $rc
