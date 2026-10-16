#!/bin/bash
# Score the MSD-trained liver readers on the external set (no retraining).   bash run_liver_ext_readers.sh <gpu>
set -e
cd /raid/home/CAMCA/hj880/perr
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
export HF_HOME=/raid/home/CAMCA/hj880/hf_home OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p data/liver_ext_reader_liver/s data/liver_ext_reader_fused/s
cp data/liver_ext_reader_liver/eval.jsonl data/liver_ext_reader_liver/s/eval_ext_liverprior.jsonl
cp data/liver_ext_reader_fused/eval.jsonl data/liver_ext_reader_fused/s/eval_ext_fused.jsonl
CUDA_VISIBLE_DEVICES=$1 $PY score_reader.py hf runs_reader/q8_liver_vis/merged q8_liver_vis data/liver_ext_reader_liver/s/eval_ext_liverprior.jsonl
CUDA_VISIBLE_DEVICES=$1 $PY score_reader.py hf runs_reader/q8_liver_fused_vis/merged q8_liver_fused_vis data/liver_ext_reader_fused/s/eval_ext_fused.jsonl
