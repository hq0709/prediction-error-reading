#!/bin/bash
cd /raid/home/CAMCA/hj880/perr/serve
screen -dmS vq3 bash -c "GPU_UTIL=0.48 ./serve.sh qwen3vl32b 0,1,2,3 4 8001 hf_cache/hub/models--Qwen--Qwen3-VL-32B-Instruct/snapshots/0cfaf48183f594c314753d30a4c4974bc75f3ccb 65536 > ../logs/serve_q3.log 2>&1"
screen -dmS vq72 bash -c "GPU_UTIL=0.62 ./serve.sh qwen25vl72b 4,5,6,7 4 8003 hf_cache/models--Qwen--Qwen2.5-VL-72B-Instruct/snapshots/89c86200743eec961a297729e7990e8f2ddbc4c5 32768 > ../logs/serve_q72.log 2>&1"
