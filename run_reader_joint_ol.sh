#!/bin/bash
# One reader for all three tasks, with the prior findings outlined on the disease-course strips: lung nodules (anatomy, chest CT), liver tumours (anatomy, abdominal CT) and new or
# grown lesions (disease course, whole-body CT pairs), trained on the union of the three training sets with the final
# recipe, scored on each evaluation set.   bash run_reader_joint_ol.sh <gpu>
set -e
cd /raid/home/CAMCA/hj880/perr
Q8=/raid/home/CAMCA/hj880/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
export HF_HOME=/raid/home/CAMCA/hj880/hf_home OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p data/reader_joint3_ol
cat data/reader_ctx/train.jsonl data/liver_reader_all2/train.jsonl data/course_reader_ol/train.jsonl > data/reader_joint3_ol/train.jsonl
cat data/reader_ctx/val.jsonl data/liver_reader_all2/val.jsonl data/course_reader_ol/val.jsonl > data/reader_joint3_ol/val.jsonl
CUDA_VISIBLE_DEVICES=$1 $PY train_reader.py --model $Q8 --data data/reader_joint3_ol --out runs_reader/q8_joint3_ol --vision --variant nomeas --epochs 3 > logs/tr_q8_joint3_ol.log 2>&1
F=runs_reader/q8_joint3_ol/merged
CUDA_VISIBLE_DEVICES=$1 $PY score_reader.py hf $F q8_joint3_ol data/reader_ctx/eval_60_160.jsonl data/reader_ctx/eval_160_260.jsonl --variant nomeas
mkdir -p data/joint3ol_eval
cp data/liver_reader_fused_lcrlm_2s/eval.jsonl data/joint3ol_eval/eval_liver.jsonl
cp data/liver_ext_reader_fused_lcrlm_2scale/eval.jsonl data/joint3ol_eval/eval_liver_ext.jsonl
cp data/course_reader_ol/eval.jsonl data/joint3ol_eval/eval_course.jsonl
CUDA_VISIBLE_DEVICES=$1 $PY score_reader.py hf $F q8_joint3_ol data/joint3ol_eval/eval_liver.jsonl data/joint3ol_eval/eval_liver_ext.jsonl data/joint3ol_eval/eval_course.jsonl
