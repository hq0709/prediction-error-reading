#!/bin/bash
# Interval-change readers: comparison only now; with world-model / random change views once the course maps exist.
cd /raid/home/CAMCA/hj880/perr
PY=/raid/home/CAMCA/hj880/venvs/sota_cxr_densenet121/bin/python
CUDA_VISIBLE_DEVICES=1 $PY train_cxr_course.py --views none --name course_none > logs/course_none.log 2>&1 &
until [ -z "$(ps -u hj880 -o args= | grep 'cxr_course/model.pt' | grep -v grep)" ]; do sleep 30; done
$PY train_cxr_course.py addresses > logs/course_addresses.log 2>&1
CUDA_VISIBLE_DEVICES=1 $PY train_cxr_course.py --views wm --name course_wm > logs/course_wm.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 $PY train_cxr_course.py --views random --name course_random > logs/course_random.log 2>&1 &
wait
echo COURSE_READERS_DONE
