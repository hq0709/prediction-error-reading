#!/bin/bash
# Coordinate sanity check on 3 training pairs (GPU 1 only).
set -euo pipefail
cd /raid/home/CAMCA/hj880/perr/sota/lesionlocator
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8
/raid/home/CAMCA/hj880/venvs/sota_lesionlocator/bin/python -I run_lesionlocator.py plan_sanity.json _work/sanity
/raid/home/CAMCA/hj880/venvs/perr/bin/python -I sanity_train.py plan_sanity.json _work/sanity
