#!/bin/bash
# Full training run: official train split, all frontal images, DDP over GPUs 3 and 4 (global batch 64).
cd /raid/home/CAMCA/hj880/perr/sota_cxr/densenet121
export CUDA_VISIBLE_DEVICES=3,4 OMP_NUM_THREADS=8 TORCH_HOME=/raid/home/CAMCA/hj880/hf_home/torch
/raid/home/CAMCA/hj880/venvs/sota_cxr_densenet121/bin/torchrun --nproc_per_node 2 --master_port 29531 \
  train.py --out /raid/home/CAMCA/hj880/perr/sota_cxr/densenet121/ckpt "$@"
