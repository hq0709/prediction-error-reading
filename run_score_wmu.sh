#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
export HF_HOME=/raid/home/CAMCA/hj880/hf_home
CUDA_VISIBLE_DEVICES=6 /raid/home/CAMCA/hj880/venvs/grl/bin/python train_cxr_focal_reader.py --out runs_reader/cxr_focal --score-only wmh/cxdet_test wmu/cxdet_test > logs/score_focal_d.log 2>&1
echo DONE
