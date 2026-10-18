#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
CUDA_VISIBLE_DEVICES=0 $PY infer_cxr_wm.py runs_wm/cxr_course/model.pt cxrc runs_wm/cxr_course/todo_keys.txt --shard 0 2 > logs/infer_cxrc_pairs0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY infer_cxr_wm.py runs_wm/cxr_course/model.pt cxrc runs_wm/cxr_course/todo_keys.txt --shard 1 2 > logs/infer_cxrc_pairs1.log 2>&1 &
wait
echo COURSE_INFER_DONE
