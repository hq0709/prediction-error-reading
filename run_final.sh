#!/bin/bash
# Train a focal reader on GPU <train gpu> (it scores the ChestX-Det test images and the held-out tenth itself), then,
# once its best adapter is saved, score NIH and the MIMIC union shards on six more GPUs and join the shards.
#   bash run_final.sh <run> "<training args>" <train gpu> <gpu> <gpu> <gpu> <gpu> <gpu> <gpu>
cd /raid/home/CAMCA/hj880/perr
export HF_HOME=/raid/home/CAMCA/hj880/hf_home OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
RUN=$1; ARGS=$2; GT=$3; shift 3
G=("$@")
CUDA_VISIBLE_DEVICES=$GT $PY train_cxr_focal_reader.py --out runs_reader/$RUN $ARGS --sets wmu/cxdet_test wmu/cxdet_val \
  > logs/tr_$RUN.log 2>&1 &
until grep -q "epoch 1 validation" logs/tr_$RUN.log; do sleep 60; done
sleep 120   # the adapter is saved, the scoring model reloaded
n=0
for s in wmu/nih_test wmu/mimic_validate_s0 wmu/mimic_validate_s1 wmu/mimic_test_s0 wmu/mimic_test_s1 wmu/mimic_test_s2; do
  CUDA_VISIBLE_DEVICES=${G[$n]} $PY train_cxr_focal_reader.py --out runs_reader/$RUN --score-only $s \
    > logs/score_${RUN}_$(echo $s | tr / _).log 2>&1 &
  n=$((n + 1)); sleep 20
done
wait
R=results/cxr_focal/$RUN
cat $R/wmu_mimic_validate_s0.jsonl $R/wmu_mimic_validate_s1.jsonl > $R/wmu_mimic_validate.jsonl
cat $R/wmu_mimic_test_s0.jsonl $R/wmu_mimic_test_s1.jsonl $R/wmu_mimic_test_s2.jsonl > $R/wmu_mimic_test.jsonl
echo "FINAL_DONE $RUN"
