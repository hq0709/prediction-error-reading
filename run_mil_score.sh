#!/bin/bash
# Report-label reader, step 3: score the evaluation sets in shards on this project's free GPUs while the training
# process scores ChestX-Det test and MIMIC validate itself; then join the shards.
cd /raid/home/CAMCA/hj880/perr
export HF_HOME=/raid/home/CAMCA/hj880/hf_home OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
until grep -q "epoch 1 validation" logs/tr_cxr_focal_mil.log; do sleep 60; done
sleep 120   # the adapter is saved, the scoring model reloaded
free_gpu() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1!=2 && $1!=3' | sort -t, -k2 -n | head -n 1 | cut -d, -f1; }
for s in wm/mimic_test_s0 wm/mimic_test_s1 wmh/mimic_validate wmh/mimic_test_s0 wmh/mimic_test_s1 wm/nih_test; do
  G=$(free_gpu); echo "$s on GPU $G"
  CUDA_VISIBLE_DEVICES=$G $PY train_cxr_focal_reader.py --out runs_reader/cxr_focal_mil --score-only $s > logs/score_mil_$(echo $s | tr / _).log 2>&1 &
  sleep 90
done
wait
R=results/cxr_focal/cxr_focal_mil
cat $R/wm_mimic_test_s0.jsonl $R/wm_mimic_test_s1.jsonl > $R/wm_mimic_test.jsonl
cat $R/wmh_mimic_test_s0.jsonl $R/wmh_mimic_test_s1.jsonl > $R/wmh_mimic_test.jsonl
echo MIL_SCORE_DONE
