#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
PY=/raid/home/CAMCA/hj880/venvs/perr/bin/python
$PY make_cxr_crops.py focal > logs/crops_focal.log 2>&1
$PY make_cxr_crops.py focal_random > logs/crops_focal_random.log 2>&1
echo FOCAL_CROPS_DONE
