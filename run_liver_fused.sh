#!/bin/bash
# Liver reader on the fused world model's candidates: strips, the old reader's scores, retraining (final recipe), scores.
#   bash run_liver_fused.sh <gpu>
set -e
cd /raid/home/CAMCA/hj880/perr
source /raid/home/CAMCA/hj880/venvs/perr/bin/activate
Q8=/raid/home/CAMCA/hj880/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
export HF_HOME=/raid/home/CAMCA/hj880/hf_home
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
[ -f data/liver_reader_fused/train.jsonl ] || python make_liver_data.py --proposal fused --out data/liver_reader_fused
CUDA_VISIBLE_DEVICES=$1 $PY score_reader.py hf runs_reader/q8_liver_vis/merged q8_liver_vis_on_fused data/liver_reader_fused/eval.jsonl
CUDA_VISIBLE_DEVICES=$1 $PY train_reader.py --model $Q8 --data data/liver_reader_fused --out runs_reader/q8_liver_fused_vis --vision --epochs 3 > logs/tr_q8_liver_fused_vis.log 2>&1
CUDA_VISIBLE_DEVICES=$1 $PY score_reader.py hf runs_reader/q8_liver_fused_vis/merged q8_liver_fused_vis data/liver_reader_fused/eval.jsonl
