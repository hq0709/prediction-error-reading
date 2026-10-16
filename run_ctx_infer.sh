#!/bin/bash
# Dense inference of a learned world-model checkpoint on a uid list, sharded over GPUs; then exits.
#   bash run_ctx_infer.sh <ckpt> <tag> <uids.txt> "0 1 2 3 5 6" [organ]
ckpt=$1; tag=$2; uids=$3; gpus=($4); organ=${5:-lung}
n=${#gpus[@]}
for i in "${!gpus[@]}"; do
  CUDA_VISIBLE_DEVICES=${gpus[$i]} /raid/home/CAMCA/hj880/venvs/grl/bin/python infer_context_wm.py $ckpt $tag $uids --shard $i $n --organ $organ > logs/infer_${tag}_$i.log 2>&1 &
done
wait
