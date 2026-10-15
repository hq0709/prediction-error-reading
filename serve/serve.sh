#!/bin/bash
# usage: serve.sh <name> <gpus> <tp> <port> <model_path> <max_len>
NAME=$1; GPUS=$2; TP=$3; PORT=$4; MODEL=$5; MAXLEN=$6
export CUDA_VISIBLE_DEVICES=$GPUS
export HF_HOME=/raid/home/CAMCA/hj880/hf_home
export VLLM_CACHE_ROOT=/raid/home/CAMCA/hj880/perr/serve/vllm_cache
export TMPDIR=/raid/home/CAMCA/hj880/perr/serve/tmp
export PATH=/raid/home/CAMCA/hj880/venvs/grl/bin:$PATH
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME=/raid/home/CAMCA/hj880/venvs/grl/lib/python3.13/site-packages/nvidia/cu13
export PATH=/raid/home/CAMCA/hj880/venvs/grl/lib/python3.13/site-packages/nvidia/cu13/bin:$PATH
mkdir -p $TMPDIR
exec /raid/home/CAMCA/hj880/venvs/grl/bin/vllm serve /raid/home/CAMCA/hj880/$MODEL \
  --served-model-name $NAME --tensor-parallel-size $TP --port $PORT \
  --max-model-len $MAXLEN --gpu-memory-utilization ${GPU_UTIL:-0.78} --max-num-seqs 64 \
  --limit-mm-per-prompt '{"image": 40}' --trust-remote-code \
  --compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}'
