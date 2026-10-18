#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
export TORCH_HOME=/raid/home/CAMCA/hj880/torch_home
PY=/raid/home/CAMCA/hj880/venvs/sota_cxr_densenet121/bin/python
CUDA_VISIBLE_DEVICES=0 $PY train_cxr_detector.py --train-keys results/cxr_cxdet_boxes10_keys.txt --out runs_reader/cxr_detector_b10 > logs/tr_cxr_detector_b10.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 $PY train_cxr_detector.py --train-keys results/cxr_cxdet_boxes25_keys.txt --out runs_reader/cxr_detector_b25 > logs/tr_cxr_detector_b25.log 2>&1 &
wait
echo FEWBOX_DET_DONE
