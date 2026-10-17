#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
CUDA_VISIBLE_DEVICES=5 $PY train_cxr_wm.py --canvas 1024 --out runs_wm/cxr_nll_1024 > logs/train_cxr_nll_1024.log 2>&1
python3 -c "
import pandas as pd
k = pd.read_csv('results/cxr_cxdet.csv').key.tolist() + pd.read_csv('results/cxr_nih.csv').key.tolist() + open('runs_wm/cxr_nll/mimic_eval_keys.txt').read().split()
open('runs_wm/cxr_nll_1024/eval_keys.txt','w').write('\n'.join(k))"
CUDA_VISIBLE_DEVICES=5 $PY infer_cxr_wm.py runs_wm/cxr_nll_1024/model.pt cxrh runs_wm/cxr_nll_1024/eval_keys.txt --canvas 1024 --shard 0 2 > logs/infer_cxrh_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 $PY infer_cxr_wm.py runs_wm/cxr_nll_1024/model.pt cxrh runs_wm/cxr_nll_1024/eval_keys.txt --canvas 1024 --shard 1 2 > logs/infer_cxrh_1.log 2>&1 &
wait
echo WM1024_DONE
