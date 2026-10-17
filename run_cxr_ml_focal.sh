#!/bin/bash
# Multi-label VLM reader, radiograph + six enlarged places (world-model focal addresses vs random thorax points), same 60k.
cd /raid/home/CAMCA/hj880/perr
export HF_HOME=/raid/home/CAMCA/hj880/hf_home
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
CUDA_VISIBLE_DEVICES=6 $PY train_cxr_ml.py --design focal --max-train 60000 --out runs_reader/cxr_ml60k_focal > logs/tr_cxr_ml60k_focal.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 $PY train_cxr_ml.py --design focal_random --max-train 60000 --out runs_reader/cxr_ml60k_focal_random > logs/tr_cxr_ml60k_focal_random.log 2>&1 &
wait
echo CXR_ML_FOCAL_DONE
