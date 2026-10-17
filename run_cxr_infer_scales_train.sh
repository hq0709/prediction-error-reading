#!/bin/bash
# Coarse-scale world models on the 60k training images (reader experiments).
cd /raid/home/CAMCA/hj880/perr
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
K=results/cxr_train60k_keys.txt
CUDA_VISIBLE_DEVICES=4 $PY infer_cxr_wm.py runs_wm/cxr_nll_s2/model.pt cxr2 $K --scale 2 > logs/infer_cxr2_train.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 $PY infer_cxr_wm.py runs_wm/cxr_nll_s4/model.pt cxr4 $K --scale 4 > logs/infer_cxr4_train.log 2>&1 &
wait
