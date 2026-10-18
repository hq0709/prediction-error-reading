#!/bin/bash
# Detector baselines on ChestX-Det focal findings (same data as the readers): YOLO11m and RT-DETR-l (ultralytics),
# RetinaNet and FCOS (torchvision). Packed on GPUs 6, 0 and 4.
cd /raid/home/CAMCA/hj880/perr
export TORCH_HOME=/raid/home/CAMCA/hj880/torch_home
Y=/raid/home/CAMCA/hj880/venvs/sota_cxr_yolo/bin/python
G=/raid/home/CAMCA/hj880/venvs/grl/bin/python
CUDA_VISIBLE_DEVICES=6 $Y train_cxr_yolo.py --model yolo11m.pt --out runs_reader/cxr_yolo11m > logs/tr_cxr_yolo11m.log 2>&1 &
sleep 90   # the first run writes the shared PNGs
CUDA_VISIBLE_DEVICES=6 $Y train_cxr_yolo.py --model rtdetr-l.pt --out runs_reader/cxr_rtdetr > logs/tr_cxr_rtdetr.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 $G train_cxr_detector.py --arch retinanet --out runs_reader/cxr_retinanet > logs/tr_cxr_retinanet.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 $G train_cxr_detector.py --arch fcos --out runs_reader/cxr_fcos > logs/tr_cxr_fcos.log 2>&1 &
wait
