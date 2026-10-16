#!/bin/bash
# Dense inference of the disease-course world model over pair keys, sharded over GPUs.
#   bash run_course_infer.sh <ckpt> <tag> <keys.txt> "2 3 5 6"
ckpt=$1; tag=$2; keys=$3; gpus=($4)
n=${#gpus[@]}
for i in "${!gpus[@]}"; do
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=${gpus[$i]} /raid/home/CAMCA/hj880/venvs/grl/bin/python infer_course_wm.py $ckpt $tag $keys --shard $i $n > logs/infer_${tag}_$i.log 2>&1 &
done
wait
