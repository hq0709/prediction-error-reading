#!/bin/bash
# Cross-fitted readers for one reader variant (choose variants on the training pool, never on evaluation scans).
#   bash run_reader_cv_variant.sh <A|B> <gpu> <name> <train_reader args...>
set -e
cd /raid/home/CAMCA/hj880/perr
half=$1; gpu=$2; name=$3; shift 3
Q8=/raid/home/CAMCA/hj880/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
export HF_HOME=/raid/home/CAMCA/hj880/hf_home OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
variant=full
for v in axial nomeas; do [[ " $* " == *" $v "* ]] && variant=$v; done
CUDA_VISIBLE_DEVICES=$gpu $PY train_reader.py --model $Q8 --data data/reader_ctx_cv$half --out runs_reader/${name}_cv$half --epochs 3 "$@" > logs/tr_${name}_cv$half.log 2>&1
CUDA_VISIBLE_DEVICES=$gpu $PY score_reader.py hf runs_reader/${name}_cv$half/merged ${name}_cv$half data/reader_ctx_cv$half/heldout.jsonl --variant $variant
