#!/bin/bash
# Learned world model on the LUNA16 evaluation scans: candidate strips, recall, and the existing reader's scores.
set -e
cd /raid/home/CAMCA/hj880/perr
source /raid/home/CAMCA/hj880/venvs/perr/bin/activate
python make_reader_data.py --eval 60 160 --proposal context --out data/reader_ctx
python make_reader_data.py --eval 160 260 --proposal context --out data/reader_ctx
python make_reader_data.py --eval 60 160 --proposal context --out data/reader_ctx --k 30
python make_reader_data.py --eval 160 260 --proposal context --out data/reader_ctx --k 30
export HF_HOME=/raid/home/CAMCA/hj880/hf_home
CUDA_VISIBLE_DEVICES=$1 /raid/home/CAMCA/hj880/venvs/grl/bin/python score_reader.py hf runs_reader/q8_vis/merged q8_vis_on_ctx \
  data/reader_ctx/eval_60_160.jsonl data/reader_ctx/eval_160_260.jsonl data/reader_ctx/eval_k30_60_160.jsonl data/reader_ctx/eval_k30_160_260.jsonl
python summarize_reader.py results/reader_q8_vis_on_ctx_eval_60_160.json results/reader_q8_vis_on_ctx_eval_160_260.json --data data/reader_ctx
