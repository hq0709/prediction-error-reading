#!/bin/bash
# MIMIC-CXR reader for one input design (global | wm): LoRA (language + vision) on the training rows, best epoch by
# validation AUROC, scored on every test row.   bash run_cxr_reader.sh <design> <gpu>
set -e
cd /raid/home/CAMCA/hj880/perr
Q8=/raid/home/CAMCA/hj880/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
export HF_HOME=/raid/home/CAMCA/hj880/hf_home OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=$2 $PY train_reader.py --model $Q8 --data data/cxr_reader_$1 --out runs_reader/cxr_$1 --vision --epochs 2 --neg-ratio 1 > logs/tr_cxr_$1.log 2>&1
CUDA_VISIBLE_DEVICES=$2 $PY score_reader.py hf runs_reader/cxr_$1/merged cxr_$1 data/cxr_reader_$1/test.jsonl > logs/score_cxr_$1.log 2>&1
