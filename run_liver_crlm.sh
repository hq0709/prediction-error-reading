#!/bin/bash
# Liver with CRLM added to training: world-model inference (tag lcrlm), strips, reader training, scores on MSD and
# the external set.   bash run_liver_crlm.sh "<gpus for inference>" <gpu for the reader>
set -e
cd /raid/home/CAMCA/hj880/perr
source /raid/home/CAMCA/hj880/venvs/perr/bin/activate
Q8=/raid/home/CAMCA/hj880/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
export HF_HOME=/raid/home/CAMCA/hj880/hf_home OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
M=runs_wm/liver_crlm_nll/model.pt
cat runs_wm/liver_nll/all_cases.txt results/crlm_cases.txt > runs_wm/liver_crlm_nll/msd_crlm_cases.txt
bash run_ctx_infer.sh $M lcrlm runs_wm/liver_crlm_nll/msd_crlm_cases.txt "$1" liver
bash run_ctx_infer.sh $M lcrlm sota/eval_liver_ext.txt "$1" liver_ext
python liver_ctx_recall.py lcrlm runs_wm/liver_crlm_nll/split.json | grep -E "liverprior|ctx3pool|ctx2pool|ctx4pool"
python make_liver_data.py --proposal fused --tag lcrlm --out data/liver_reader_fused_lcrlm
python make_liver_crlm_data.py lcrlm data/liver_reader_fused_lcrlm data/liver_reader_all
LIVER_TAG=lcrlm python make_liver_ext_data.py strips fused
CUDA_VISIBLE_DEVICES=$2 $PY train_reader.py --model $Q8 --data data/liver_reader_all --out runs_reader/q8_liver_all_vis --vision --epochs 3 > logs/tr_q8_liver_all_vis.log 2>&1
mkdir -p data/liver_ext_reader_fused_lcrlm/s
cp data/liver_ext_reader_fused_lcrlm/eval.jsonl data/liver_ext_reader_fused_lcrlm/s/eval_ext_fused_lcrlm.jsonl
CUDA_VISIBLE_DEVICES=$2 $PY score_reader.py hf runs_reader/q8_liver_all_vis/merged q8_liver_all_vis data/liver_reader_fused_lcrlm/eval.jsonl data/liver_ext_reader_fused_lcrlm/s/eval_ext_fused_lcrlm.jsonl
