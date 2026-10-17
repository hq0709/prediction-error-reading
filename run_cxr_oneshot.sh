#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
export HF_HOME=/raid/home/CAMCA/hj880/hf_home
CUDA_VISIBLE_DEVICES=1 /raid/home/CAMCA/hj880/venvs/grl/bin/python train_cxr_oneshot.py --out runs_reader/cxr_oneshot > logs/tr_cxr_oneshot.log 2>&1
