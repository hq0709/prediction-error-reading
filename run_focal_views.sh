#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
PY=/raid/home/CAMCA/hj880/venvs/perr/bin/python
$PY cxr_focal_data.py nih > logs/focal_views_nih.log 2>&1
$PY cxr_focal_data.py cxdet --proposer patchcore > logs/focal_views_cxdet_pc.log 2>&1
$PY cxr_focal_data.py cxdet --proposer random > logs/focal_views_cxdet_rand.log 2>&1
$PY cxr_focal_data.py nih --proposer patchcore > logs/focal_views_nih_pc.log 2>&1
$PY cxr_focal_data.py nih --proposer random > logs/focal_views_nih_rand.log 2>&1
$PY cxr_focal_data.py mimic > logs/focal_views_mimic.log 2>&1
echo FOCAL_VIEWS_DONE
