#!/bin/bash
# Retrain the reader on the final candidates with one input/adapter change, then score it.
#   bash run_reader_variant.sh <name> <gpu> <train_reader args...>     e.g.  q8_ctx_axial 6 --vision --variant axial
set -e
cd /raid/home/CAMCA/hj880/perr
name=$1; gpu=$2; shift 2
Q8=/raid/home/CAMCA/hj880/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
export HF_HOME=/raid/home/CAMCA/hj880/hf_home
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=$gpu $PY train_reader.py --model $Q8 --data data/reader_ctx --out runs_reader/$name --epochs 3 "$@" > logs/tr_$name.log 2>&1
variant=full
for v in axial nomeas axial_nomeas; do [[ " $* " == *" $v "* ]] && variant=$v; done
CUDA_VISIBLE_DEVICES=$gpu $PY score_reader.py hf runs_reader/$name/merged $name data/reader_ctx/eval_60_160.jsonl data/reader_ctx/eval_160_260.jsonl --variant $variant
