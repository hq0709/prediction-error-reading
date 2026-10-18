#!/bin/bash
cd /raid/home/CAMCA/hj880/perr
export HF_HOME=/raid/home/CAMCA/hj880/hf_home
export VLLM_CACHE_ROOT=/raid/home/CAMCA/hj880/perr/serve/vllm_cache
export TMPDIR=/raid/home/CAMCA/hj880/perr/serve/tmp
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME=/raid/home/CAMCA/hj880/venvs/grl/lib/python3.13/site-packages/nvidia/cu13
export PATH=/raid/home/CAMCA/hj880/venvs/grl/lib/python3.13/site-packages/nvidia/cu13/bin:/raid/home/CAMCA/hj880/venvs/grl/bin:$PATH
export GPU_UTIL=0.4
CUDA_VISIBLE_DEVICES=0 /raid/home/CAMCA/hj880/venvs/grl/bin/python label_cxr_change.py > logs/label_cxr_change.log 2>&1
echo LABEL_DONE
