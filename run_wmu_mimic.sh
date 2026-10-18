#!/bin/bash
# Score a focal reader on the union addresses of the MIMIC evaluation splits (make_wmu_mimic.py), one shard per GPU,
# then join the shards.   bash run_wmu_mimic.sh <run> <gpu> <gpu> <gpu> <gpu> <gpu>
cd /raid/home/CAMCA/hj880/perr
export HF_HOME=/raid/home/CAMCA/hj880/hf_home OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
RUN=$1; shift
G=("$@")
n=0
for s in wmu/mimic_validate_s0 wmu/mimic_validate_s1 wmu/mimic_test_s0 wmu/mimic_test_s1 wmu/mimic_test_s2; do
  CUDA_VISIBLE_DEVICES=${G[$n]} $PY train_cxr_focal_reader.py --out runs_reader/$RUN --score-only $s \
    > logs/score_${RUN}_$(echo $s | tr / _).log 2>&1 &
  n=$((n + 1)); sleep 20
done
wait
R=results/cxr_focal/$RUN
cat $R/wmu_mimic_validate_s0.jsonl $R/wmu_mimic_validate_s1.jsonl > $R/wmu_mimic_validate.jsonl
cat $R/wmu_mimic_test_s0.jsonl $R/wmu_mimic_test_s1.jsonl $R/wmu_mimic_test_s2.jsonl > $R/wmu_mimic_test.jsonl
echo "WMU_MIMIC_DONE $RUN"
