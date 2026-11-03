#!/bin/bash
# GPU smoke test of train_madad.py: throughput and peak memory for fp32 (TF32, as the repo) and bf16 autocast.
PY="/raid/home/CAMCA/hj880/venvs/sota_madad/bin/python -I -B"
cd /raid/home/CAMCA/hj880/perr
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
for cfg in "bf16 8" "fp32 8" "bf16 16"; do
  set -- $cfg
  extra=""; [ "$1" = "bf16" ] && extra="--bf16"
  rm -rf sota/madad/runs/smoke_$1_$2
  $PY sota/madad/train_madad.py --out sota/madad/runs/smoke_$1_$2 --train-limit 2400 --max-steps 20 --log-every 5 \
      --micro $2 --epochs 200 --warmup-epochs 0 --workers 12 $extra 2>&1 | grep -v "^making\|Working with\|warn"
done
