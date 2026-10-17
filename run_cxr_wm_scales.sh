#!/bin/bash
# Multi-scale radiograph world models: the same U-Net on the radiograph downscaled 2x and 4x (hidden 8 / 16 cm).
cd /raid/home/CAMCA/hj880/perr
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
CUDA_VISIBLE_DEVICES=6 $PY train_cxr_wm.py --scale 2 --out runs_wm/cxr_nll_s2 > logs/train_cxr_nll_s2.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 $PY train_cxr_wm.py --scale 4 --out runs_wm/cxr_nll_s4 > logs/train_cxr_nll_s4.log 2>&1 &
wait
