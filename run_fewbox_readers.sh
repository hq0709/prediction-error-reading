#!/bin/bash
# Few-box focal readers (same recipe as runs_reader/cxr_focal, training images restricted), then their one-shot baselines.
cd /raid/home/CAMCA/hj880/perr
export HF_HOME=/raid/home/CAMCA/hj880/hf_home
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
for b in 10 25; do
  CUDA_VISIBLE_DEVICES=1 $PY train_cxr_focal_reader.py --out runs_reader/cxr_focal_b$b --train-keys results/cxr_cxdet_boxes${b}_keys.txt > logs/tr_cxr_focal_b$b.log 2>&1
done
for b in 10 25; do
  CUDA_VISIBLE_DEVICES=1 $PY train_cxr_oneshot.py --out runs_reader/cxr_oneshot_b$b --train-keys results/cxr_cxdet_boxes${b}_keys.txt > logs/tr_cxr_oneshot_b$b.log 2>&1
done
echo FEWBOX_READERS_DONE
