#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
export HF_HOME=/raid/home/CAMCA/hj880/hf_home
CUDA_VISIBLE_DEVICES=0 /raid/home/CAMCA/hj880/venvs/grl/bin/python train_cxr_focal_reader.py --out runs_reader/cxr_focal > logs/tr_cxr_focal.log 2>&1
