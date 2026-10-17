#!/bin/bash
# The one evaluation run: 64 pairs (sota/eval_course.txt) on GPU 1 only, split into NSHARDS processes balanced by
# lesion count (resumable: finished BL pieces have done.json); then the fixed answer rule.
set -uo pipefail
cd /raid/home/CAMCA/hj880/perr/sota/lesionlocator
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8
PY=/raid/home/CAMCA/hj880/venvs/sota_lesionlocator/bin/python
NSHARDS=${NSHARDS:-5}
date +"start %F %T"
for s in $(seq 0 $((NSHARDS - 1))); do
    $PY -I run_lesionlocator.py plan_eval.json _work/eval --shard $s --nshards $NSHARDS > logs/eval_shard$s.log 2>&1 &
done
wait
date +"shards done %F %T"
$PY -I answers.py plan_eval.json _work/eval /raid/home/CAMCA/hj880/perr/results/sota/lesionlocator_course.jsonl
