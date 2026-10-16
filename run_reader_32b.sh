#!/bin/bash
# Larger reader on the final candidates: Qwen3-VL-32B, LoRA on language + vision, split over two GPUs; then scored.
#   bash run_reader_32b.sh "1,7"
set -e
cd /raid/home/CAMCA/hj880/perr
Q32=/raid/home/CAMCA/hj880/hf_cache/hub/models--Qwen--Qwen3-VL-32B-Instruct/snapshots/0cfaf48183f594c314753d30a4c4974bc75f3ccb
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
export HF_HOME=/raid/home/CAMCA/hj880/hf_home
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=$1 $PY train_reader.py --model $Q32 --data data/reader_ctx --out runs_reader/q32_ctx_vis --vision --epochs 3 --device-map auto > logs/tr_q32_ctx_vis.log 2>&1
CUDA_VISIBLE_DEVICES=$1 $PY score_reader.py hf runs_reader/q32_ctx_vis/merged q32_ctx_vis data/reader_ctx/eval_60_160.jsonl data/reader_ctx/eval_160_260.jsonl --device-map auto
