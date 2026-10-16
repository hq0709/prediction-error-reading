#!/bin/bash
# Disease-course reader: LoRA (language + vision) on the training pairs' strips, then scored on the evaluation pairs;
# also the zero-shot reader on the same strips.   bash run_course_reader.sh <gpu> <gpu for zero-shot>
set -e
cd /raid/home/CAMCA/hj880/perr
Q8=/raid/home/CAMCA/hj880/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
export HF_HOME=/raid/home/CAMCA/hj880/hf_home OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=$2 $PY score_reader.py hf $Q8 zs_course data/course_reader/eval.jsonl > logs/zs_course.log 2>&1 &
CUDA_VISIBLE_DEVICES=$1 $PY train_reader.py --model $Q8 --data data/course_reader --out runs_reader/q8_course_vis --vision --epochs 3 > logs/tr_q8_course_vis.log 2>&1
wait
CUDA_VISIBLE_DEVICES=$1 $PY score_reader.py hf runs_reader/q8_course_vis/merged q8_course_vis data/course_reader/eval.jsonl
source /raid/home/CAMCA/hj880/venvs/perr/bin/activate
python summarize_course.py results/reader_zs_course_eval.json results/reader_q8_course_vis_eval.json
