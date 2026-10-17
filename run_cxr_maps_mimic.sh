#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
export HF_HOME=/raid/home/CAMCA/hj880/hf_home
CUDA_VISIBLE_DEVICES=4 /raid/home/CAMCA/hj880/venvs/grl/bin/python cxr_maps.py mimic_eval > logs/cxr_maps_mimic_eval.log 2>&1
