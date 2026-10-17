#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
export HF_HOME=/raid/home/CAMCA/hj880/hf_home
CUDA_VISIBLE_DEVICES=6 /raid/home/CAMCA/hj880/venvs/grl/bin/python patchcore_pos.py > logs/patchcore_pos.log 2>&1
