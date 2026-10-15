#!/bin/bash
# launch Qwen2.5-VL-72B on GPUs 4-7 once each has >= 50 GB free (checks every 60 s)
cd /raid/home/CAMCA/hj880/perr/serve
Q72=hf_cache/models--Qwen--Qwen2.5-VL-72B-Instruct/snapshots/89c86200743eec961a297729e7990e8f2ddbc4c5
while true; do
  ok=1
  for g in 4 5 6 7; do
    used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
    [ $((81559 - used)) -lt 51200 ] && ok=0
  done
  if [ $ok -eq 1 ]; then
    echo "launching 72B at $(date)"
    GPU_UTIL=0.60 ./serve.sh qwen25vl72b 4,5,6,7 4 8003 $Q72 32768 > ../logs/serve_q72.log 2>&1
    echo "72B server exited at $(date)"
    exit 0
  fi
  sleep 60
done
