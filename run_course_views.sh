#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
PY=/raid/home/CAMCA/hj880/venvs/sota_cxr_densenet121/bin/python
CUDA_VISIBLE_DEVICES=1 $PY train_cxr_course.py --views wm --name course_wm > logs/course_wm.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 $PY train_cxr_course.py --views random --name course_random > logs/course_random.log 2>&1 &
wait
echo COURSE_VIEWS_DONE
