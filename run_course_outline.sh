#!/bin/bash
# Disease-course strips with the prior findings outlined vs without: same candidates, same recipe; the choice is made on
# the validation pairs of training patients.   bash run_course_outline.sh <gpu>
set -e
cd /raid/home/CAMCA/hj880/perr
source /raid/home/CAMCA/hj880/venvs/perr/bin/activate
Q8=/raid/home/CAMCA/hj880/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
export HF_HOME=/raid/home/CAMCA/hj880/hf_home OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
OMP_NUM_THREADS=4 python make_course_data.py pool --outline --out data/course_reader_ol
OMP_NUM_THREADS=4 python make_course_data.py eval --outline --out data/course_reader_ol
CUDA_VISIBLE_DEVICES=$1 $PY train_reader.py --model $Q8 --data data/course_reader_ol --out runs_reader/q8_course_ol --vision --epochs 3 > logs/tr_q8_course_ol.log 2>&1
grep best logs/tr_q8_course_vis.log logs/tr_q8_course_ol.log
