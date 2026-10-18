#!/bin/bash
# Hard negatives for the reader that reads both world models' addresses: (1) once runs_reader/cxr_focal_wmu is
# trained, it scores its own training addresses (4 shards on free GPUs); (2) the reader is retrained with 5 of its 10
# negatives per image and epoch taken from the addresses that reader scored highest; (3) test scoring.
cd /raid/home/CAMCA/hj880/perr
export HF_HOME=/raid/home/CAMCA/hj880/hf_home OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
free_gpu() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1!=2 && $1!=3' | sort -t, -k2 -n | head -n 1 | cut -d, -f1; }
until grep -q "epoch 1 validation" logs/tr_cxr_focal_wmu.log; do sleep 60; done
sleep 60
for i in 0 1 2 3; do
  G=$(free_gpu); echo "shard $i on GPU $G"
  CUDA_VISIBLE_DEVICES=$G $PY train_cxr_focal_reader.py --out runs_reader/cxr_focal_wmu --score-only wmu/cxdet_trainwm_s$i > logs/score_hard_s$i.log 2>&1 &
  sleep 90
done
wait
R=results/cxr_focal/cxr_focal_wmu
cat $R/wmu_cxdet_trainwm_s{0,1,2,3}.jsonl > $R/wmu_cxdet_trainwm.jsonl
G=$(free_gpu); echo "round 2 on GPU $G"
CUDA_VISIBLE_DEVICES=$G $PY train_cxr_focal_reader.py --out runs_reader/cxr_focal_wmu_hard --proposer wmu \
  --hard-from $R/wmu_cxdet_trainwm.jsonl --hard 5 --sets wmu/cxdet_test wmu/cxdet_val > logs/tr_cxr_focal_wmu_hard.log 2>&1
echo HARDNEG_DONE
