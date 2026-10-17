#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
export TORCH_HOME=/raid/home/CAMCA/hj880/torch_home
CUDA_VISIBLE_DEVICES=2 /raid/home/CAMCA/hj880/venvs/sota_cxr_densenet121/bin/python train_cxr_detector.py --out runs_reader/cxr_detector > logs/tr_cxr_detector.log 2>&1
