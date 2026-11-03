#!/bin/bash
# CLIP-Driven Universal Model: coordinate checks (README rule 3: LUNA16 pool scan, MSD pool case liver_0, HCC_001 at a fixed fraction of the
# array) and training-pool smoke tests (the 3 LUNA16 + 3 MSD pool cases used for SAT-Pro / VISTA3D / OmniAbnorm-CT).
# One GPU, assigned by the queue (queue/jobs3.tsv). Exit code 1 if any coordinate check fails or a run crashes.
cd /raid/home/CAMCA/hj880/perr/sota/clip_universal
PY="/raid/home/CAMCA/hj880/venvs/sota_clip_universal/bin/python -I -B"
rc=0
date; echo "GPU $CUDA_VISIBLE_DEVICES"
$PY coord_check.py --ds luna 1.3.6.1.4.1.14519.5.2.1.6279.6001.100953483028192176989979435275 > logs/coord_check.log 2>&1 || rc=1
$PY coord_check.py --ds liver liver_0 > logs/coord_check_liver.log 2>&1 || rc=1
$PY coord_check.py --ds liver_ext HCC_001 > logs/coord_check_liver_ext.log 2>&1 || rc=1
tail -n 1 logs/coord_check.log logs/coord_check_liver.log logs/coord_check_liver_ext.log
$PY run_clip_universal.py --ds luna --cases smoke/smoke_luna.txt --out smoke/smoke_luna.jsonl > logs/smoke_luna.log 2>&1 || rc=1
$PY run_clip_universal.py --ds liver --cases smoke/smoke_liver.txt --out smoke/smoke_liver.jsonl > logs/smoke_liver.log 2>&1 || rc=1
date; echo "checks exit $rc"; exit $rc
