#!/bin/bash
# VISTA3D eval run: 200 LUNA16 scans (fresh1 + fresh2), then 53 MSD-liver cases. GPU 1 only.
cd /raid/home/CAMCA/hj880/perr/sota/vista3d
export HF_HOME=/raid/home/CAMCA/hj880/hf_home CUDA_VISIBLE_DEVICES=1
PY=/raid/home/CAMCA/hj880/venvs/sota_vista3d/bin/python
date; $PY run_vista3d.py --ds luna --cases _work/eval_luna.txt --out /raid/home/CAMCA/hj880/perr/results/sota/vista3d_luna.jsonl > logs/eval_luna.log 2>&1
echo "luna exit $?"; date
$PY run_vista3d.py --ds liver --cases ../eval_liver.txt --out /raid/home/CAMCA/hj880/perr/results/sota/vista3d_liver.jsonl > logs/eval_liver.log 2>&1
echo "liver exit $?"; date
