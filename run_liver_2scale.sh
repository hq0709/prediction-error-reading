#!/bin/bash
# Liver reader input: one row of 60 mm views vs two rows (60 mm + 150 mm). Both readers train on MSD pool + CRLM with
# 40 CRLM cases held out for validation; the choice is made there, never on MSD eval or the external set.
#   bash run_liver_2scale.sh <gpu one-scale> <gpu two-scale>
set -e
cd /raid/home/CAMCA/hj880/perr
source /raid/home/CAMCA/hj880/venvs/perr/bin/activate
Q8=/raid/home/CAMCA/hj880/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
export HF_HOME=/raid/home/CAMCA/hj880/hf_home OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python make_liver_data.py --proposal fused --tag lcrlm --context --out data/liver_reader_fused_lcrlm_2s
python make_liver_crlm_data.py lcrlm data/liver_reader_fused_lcrlm_2s data/liver_reader_all2_h context holdout
python make_liver_crlm_data.py lcrlm data/liver_reader_fused_lcrlm data/liver_reader_all_h plain holdout
LIVER_TAG=lcrlm LIVER_CONTEXT=1 python make_liver_ext_data.py strips fused
CUDA_VISIBLE_DEVICES=$1 $PY train_reader.py --model $Q8 --data data/liver_reader_all_h --out runs_reader/q8_liver_h1 --vision --epochs 3 > logs/tr_q8_liver_h1.log 2>&1 &
CUDA_VISIBLE_DEVICES=$2 $PY train_reader.py --model $Q8 --data data/liver_reader_all2_h --out runs_reader/q8_liver_h2 --vision --epochs 3 > logs/tr_q8_liver_h2.log 2>&1
wait
grep best logs/tr_q8_liver_h1.log logs/tr_q8_liver_h2.log
