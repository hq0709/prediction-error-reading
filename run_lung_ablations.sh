#!/bin/bash
# World-model ablations for the final lung reader (no retraining: the reader is fixed, the candidates change) and the
# zero-shot reader on the final candidates.   bash run_lung_ablations.sh <gpu>
set -e
cd /raid/home/CAMCA/hj880/perr
source /raid/home/CAMCA/hj880/venvs/perr/bin/activate
Q8=/raid/home/CAMCA/hj880/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
export HF_HOME=/raid/home/CAMCA/hj880/hf_home
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
for r in "60 160" "160 260"; do
  python make_reader_data.py --eval $r --proposal context_raw --out data/reader_ctxraw
done
F=runs_reader/q8_ctx_vis/merged
CUDA_VISIBLE_DEVICES=$1 $PY score_reader.py hf $F q8_ctx_vis_on_shape data/reader/eval_60_160.jsonl data/reader/eval_160_260.jsonl
CUDA_VISIBLE_DEVICES=$1 $PY score_reader.py hf $F q8_ctx_vis_on_int data/reader/eval_int_60_160.jsonl data/reader/eval_int_160_260.jsonl
CUDA_VISIBLE_DEVICES=$1 $PY score_reader.py hf $F q8_ctx_vis_on_ctxraw data/reader_ctxraw/eval_60_160.jsonl data/reader_ctxraw/eval_160_260.jsonl
CUDA_VISIBLE_DEVICES=$1 $PY score_reader.py hf $Q8 zs_qwen3vl8b_on_ctx data/reader_ctx/eval_60_160.jsonl data/reader_ctx/eval_160_260.jsonl
