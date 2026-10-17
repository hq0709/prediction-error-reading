#!/bin/bash
# Continuation of the evaluation run as a work queue: NPROC processes on GPU 1 take BL pieces via atomic claim files
# (largest pieces first); finished pieces (done.json) are skipped. Same configuration as run_eval.sh. Then the fixed
# answer rule.
set -uo pipefail
cd /raid/home/CAMCA/hj880/perr/sota/lesionlocator
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8
PY=/raid/home/CAMCA/hj880/venvs/sota_lesionlocator/bin/python
NPROC=${NPROC:-4}
date +"start %F %T"
for s in $(seq 0 $((NPROC - 1))); do
    $PY -I run_lesionlocator.py plan_eval.json _work/eval --claim > logs/eval_claim$s.log 2>&1 &
    sleep 20
done
wait
date +"workers done %F %T"
$PY -I answers.py plan_eval.json _work/eval /raid/home/CAMCA/hj880/perr/results/sota/lesionlocator_course.jsonl
