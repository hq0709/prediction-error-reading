#!/bin/bash
# Two-stream readers with ten ~native-detail views: world-model focal addresses vs random thorax points (same 60k images).
cd /raid/home/CAMCA/hj880/perr
PY=/raid/home/CAMCA/hj880/venvs/sota_cxr_densenet121/bin/python
CUDA_VISIBLE_DEVICES=4 $PY train_cxr_evidence.py --crops focal --keys results/cxr_train60k_keys.txt --name ev60k_focal > logs/ev60k_focal.log 2>&1 &
until grep -q "images have crops" logs/crops_focal_random.log 2>/dev/null; do sleep 20; done
CUDA_VISIBLE_DEVICES=5 $PY train_cxr_evidence.py --crops focal_random --keys results/cxr_train60k_keys.txt --name ev60k_focal_random > logs/ev60k_focal_random.log 2>&1 &
wait
echo EV_FOCAL_DONE
