#!/bin/bash
# Reader on the learned world model's candidates: pool strips, zero-shot scores for the rule choice, LoRA training
# (final recipe: language + vision, 3 epochs, best validation AUROC), evaluation at K = 10 and 30.
#   bash run_reader_ctx.sh <gpu for training> <gpu for zero-shot scoring>
set -e
cd /raid/home/CAMCA/hj880/perr
source /raid/home/CAMCA/hj880/venvs/perr/bin/activate
Q8=/raid/home/CAMCA/hj880/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
export HF_HOME=/raid/home/CAMCA/hj880/hf_home
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
[ -f data/reader_ctx/train.jsonl ] || python make_reader_data.py --gt --proposal context --out data/reader_ctx
CUDA_VISIBLE_DEVICES=$2 $PY score_reader.py hf $Q8 zs_qwen3vl8b_ctx data/reader_ctx/train.jsonl data/reader_ctx/val.jsonl > logs/zs_ctx_pool.log 2>&1 &
CUDA_VISIBLE_DEVICES=$1 $PY train_reader.py --model $Q8 --data data/reader_ctx --out runs_reader/q8_ctx_vis --vision --epochs 3 > logs/tr_q8_ctx_vis.log 2>&1
wait
python choose_rule.py results/reader_zs_qwen3vl8b_ctx_train.json results/reader_zs_qwen3vl8b_ctx_val.json > results/choose_rule_ctx.txt
CUDA_VISIBLE_DEVICES=$1 $PY score_reader.py hf runs_reader/q8_ctx_vis/merged q8_ctx_vis \
  data/reader_ctx/eval_60_160.jsonl data/reader_ctx/eval_160_260.jsonl data/reader_ctx/eval_k30_60_160.jsonl data/reader_ctx/eval_k30_160_260.jsonl
python summarize_reader.py results/reader_q8_ctx_vis_eval_60_160.json results/reader_q8_ctx_vis_eval_160_260.json --data data/reader_ctx
