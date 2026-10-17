#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
export HF_HOME=/raid/home/CAMCA/hj880/hf_home
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
CUDA_VISIBLE_DEVICES=6 $PY infer_cxr_wm.py runs_wm/cxr_nll/model.pt cxr runs_wm/cxr_nll/cxdet_keys.txt > logs/infer_cxr_cxdet.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 $PY cxr_maps.py cxdet > logs/cxr_maps_cxdet.log 2>&1 &
wait
echo CXDET_MAPS_DONE
