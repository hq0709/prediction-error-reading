#!/bin/bash
# The focal reader trained on the addresses it reads: ChestX-Det training rows at both world models' addresses (wmu),
# same recipe otherwise; launched on the emptiest of this project's GPUs once the views are rendered.
cd /raid/home/CAMCA/hj880/perr
export HF_HOME=/raid/home/CAMCA/hj880/hf_home OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
until grep -q "cxdet train:" logs/focal_views_cxdet_wmu_train.log; do sleep 30; done
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1!=1 && $1!=2 && $1!=3' | sort -t, -k2 -n | head -n 1 | cut -d, -f1)
echo "training on GPU $G"
CUDA_VISIBLE_DEVICES=$G $PY train_cxr_focal_reader.py --out runs_reader/cxr_focal_wmu --proposer wmu \
  --sets wmu/cxdet_test > logs/tr_cxr_focal_wmu.log 2>&1
echo WMU_READER_DONE
