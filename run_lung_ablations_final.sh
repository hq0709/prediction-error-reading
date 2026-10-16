#!/bin/bash
# World-model ablations for the final lung reader (no measurement text): the reader is fixed, the candidates change;
# and the zero-shot reader on the final candidates with the same question.   bash run_lung_ablations_final.sh <gpu>
set -e
cd /raid/home/CAMCA/hj880/perr
Q8=/raid/home/CAMCA/hj880/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
export HF_HOME=/raid/home/CAMCA/hj880/hf_home OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
F=runs_reader/q8_ctx_nomeas/merged
CUDA_VISIBLE_DEVICES=$1 $PY score_reader.py hf $F nm_on_shape data/reader/eval_60_160.jsonl data/reader/eval_160_260.jsonl --variant nomeas
CUDA_VISIBLE_DEVICES=$1 $PY score_reader.py hf $F nm_on_int data/reader/eval_int_60_160.jsonl data/reader/eval_int_160_260.jsonl --variant nomeas
CUDA_VISIBLE_DEVICES=$1 $PY score_reader.py hf $F nm_on_ctxraw data/reader_ctxraw/eval_60_160.jsonl data/reader_ctxraw/eval_160_260.jsonl --variant nomeas
CUDA_VISIBLE_DEVICES=$1 $PY score_reader.py hf $F nm_k30 data/reader_ctx/eval_k30_60_160.jsonl data/reader_ctx/eval_k30_160_260.jsonl --variant nomeas
CUDA_VISIBLE_DEVICES=$1 $PY score_reader.py hf $Q8 zs_nm_on_ctx data/reader_ctx/eval_60_160.jsonl data/reader_ctx/eval_160_260.jsonl --variant nomeas
