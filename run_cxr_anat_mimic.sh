#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
PY=/raid/home/CAMCA/hj880/venvs/sota_cxr_xrv/bin/python
CUDA_VISIBLE_DEVICES=4 $PY cxr_anatomy.py mimic 0 2 > logs/cxr_anat_mimic0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 $PY cxr_anatomy.py mimic 1 2 > logs/cxr_anat_mimic1.log 2>&1 &
wait
