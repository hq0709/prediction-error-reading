#!/bin/bash
# Report-label training of the focal reader on MIMIC, step 1: the box-trained reader scores the world-model addresses of
# the MIMIC training subset (results/cxr_mil_keys.txt; 4 shards on GPUs 0, 4, 5, 6).
cd /raid/home/CAMCA/hj880/perr
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8   # the box is shared; default torch threading oversubscribes it
until grep -q "mimic_train train:" logs/focal_views_mimic_train.log; do sleep 30; done
$PY - <<'P'
import json
rows = [json.loads(l) for l in open("data/cxr_focal/wm/mimic_train.jsonl")]
keys = sorted({r["key"] for r in rows})
shard = {k: i % 4 for i, k in enumerate(keys)}
fs = [open(f"data/cxr_focal/wm/mimic_train_s{i}.jsonl", "w") for i in range(4)]
for r in rows:
    fs[shard[r["key"]]].write(json.dumps(r) + "\n")
P
i=0
for g in 0 4 5 6; do
  CUDA_VISIBLE_DEVICES=$g $PY train_cxr_focal_reader.py --out runs_reader/cxr_focal --score-only wm/mimic_train_s$i > logs/score_mil_s$i.log 2>&1 &
  i=$((i+1))
done
wait
cat results/cxr_focal/cxr_focal/wm_mimic_train_s{0,1,2,3}.jsonl > results/cxr_focal/cxr_focal/wm_mimic_train.jsonl
echo MIL_SCORED
