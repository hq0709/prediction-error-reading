#!/bin/bash
# Cross-fitted readers on the training pool (for choosing the selection rule without touching evaluation scans).
#   bash run_reader_cv.sh <A|B> <gpu>
set -e
cd /raid/home/CAMCA/hj880/perr
Q8=/raid/home/CAMCA/hj880/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
export HF_HOME=/raid/home/CAMCA/hj880/hf_home
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=$2 $PY train_reader.py --model $Q8 --data data/reader_ctx_cv$1 --out runs_reader/q8_ctx_cv$1 --vision --epochs 3 > logs/tr_q8_ctx_cv$1.log 2>&1
CUDA_VISIBLE_DEVICES=$2 $PY score_reader.py hf runs_reader/q8_ctx_cv$1/merged q8_ctx_cv$1 data/reader_ctx_cv$1/heldout.jsonl
