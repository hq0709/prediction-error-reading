#!/bin/bash
# Dense reading of the evaluation radiographs (NIH boxes, ChestX-Det, MIMIC validation/test) at the two coarser scales.
cd /raid/home/CAMCA/hj880/perr
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
K=runs_wm/cxr_nll/scales_eval_keys.txt
CUDA_VISIBLE_DEVICES=6 $PY infer_cxr_wm.py runs_wm/cxr_nll_s2/model.pt cxr2 $K --scale 2 > logs/infer_cxr2_eval.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 $PY infer_cxr_wm.py runs_wm/cxr_nll_s4/model.pt cxr4 $K --scale 4 > logs/infer_cxr4_eval.log 2>&1 &
wait
