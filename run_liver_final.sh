#!/bin/bash
# Final liver reader: two-row strips (60 + 150 mm), all training data (MSD pool + all CRLM), scored once on the MSD
# evaluation half and on the external set (HCC-TACE-Seg).   bash run_liver_final.sh <gpu>
set -e
cd /raid/home/CAMCA/hj880/perr
source /raid/home/CAMCA/hj880/venvs/perr/bin/activate
Q8=/raid/home/CAMCA/hj880/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
export HF_HOME=/raid/home/CAMCA/hj880/hf_home OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python make_liver_crlm_data.py lcrlm data/liver_reader_fused_lcrlm_2s data/liver_reader_all2 context
CUDA_VISIBLE_DEVICES=$1 $PY train_reader.py --model $Q8 --data data/liver_reader_all2 --out runs_reader/q8_liver_final --vision --epochs 3 > logs/tr_q8_liver_final.log 2>&1
mkdir -p data/liver_ext_reader_fused_lcrlm_2scale/s
cp data/liver_ext_reader_fused_lcrlm_2scale/eval.jsonl data/liver_ext_reader_fused_lcrlm_2scale/s/eval_ext_final.jsonl
CUDA_VISIBLE_DEVICES=$1 $PY score_reader.py hf runs_reader/q8_liver_final/merged q8_liver_final data/liver_reader_fused_lcrlm_2s/eval.jsonl data/liver_ext_reader_fused_lcrlm_2scale/s/eval_ext_final.jsonl
