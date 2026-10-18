#!/bin/bash
# Report-label training of the focal reader on MIMIC, step 2: once the MIMIC training addresses are scored
# (run_mil_stage.sh), retrain the reader on ChestX-Det boxes + MIMIC multiple-instance rows on the emptiest of this
# project's GPUs, then score the evaluation sets.
cd /raid/home/CAMCA/hj880/perr
export HF_HOME=/raid/home/CAMCA/hj880/hf_home OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
until grep -q MIL_SCORED logs/run_mil_stage.log; do sleep 60; done
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1!=2 && $1!=3' | sort -t, -k2 -n | head -n 1 | cut -d, -f1)
echo "training on GPU $G"
CUDA_VISIBLE_DEVICES=$G $PY train_cxr_focal_reader.py --out runs_reader/cxr_focal_mil \
  --mimic-rows results/cxr_focal/cxr_focal/wm_mimic_train.jsonl \
  --sets wm/cxdet_test wm/mimic_validate wm/mimic_test wm/nih_test > logs/tr_cxr_focal_mil.log 2>&1
echo MIL_TRAINED
