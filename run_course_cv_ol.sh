#!/bin/bash
# Cross-validation fold of the disease-course reader with the prior findings outlined: the fold's world model and
# candidates are those of run_course_cv.sh; only the strips and the reader change.   bash run_course_cv_ol.sh <fold> <gpu>
set -e
cd /raid/home/CAMCA/hj880/perr
f=$1; g=$2
Q8=/raid/home/CAMCA/hj880/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
PY=/raid/home/CAMCA/hj880/venvs/grl/bin/python
export HF_HOME=/raid/home/CAMCA/hj880/hf_home OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
source /raid/home/CAMCA/hj880/venvs/perr/bin/activate
OMP_NUM_THREADS=4 python make_course_data.py eval --keys runs_wm/course_cv${f}_keys.txt --tag cv$f --outline --out data/course_cv${f}_ol
python - <<PYEOF
import json, os
import numpy as np
fold = set(k.rsplit("_", 1)[0] for k in open("runs_wm/course_cv${f}_keys.txt").read().split())
rows = [json.loads(l) for p in ("data/course_reader_ol/train.jsonl", "data/course_reader_ol/val.jsonl") for l in open(p)]
rows = [r for r in rows if r["uid"].rsplit("_", 1)[0] not in fold]
pts = sorted({r["uid"].rsplit("_", 1)[0] for r in rows})
val = set(np.random.default_rng(0).choice(pts, size=24, replace=False).tolist())
os.makedirs("data/course_cv${f}_ol_reader", exist_ok=True)
for name, sel in (("train", lambda r: r["uid"].rsplit("_", 1)[0] not in val), ("val", lambda r: r["uid"].rsplit("_", 1)[0] in val)):
    with open(f"data/course_cv${f}_ol_reader/{name}.jsonl", "w") as fh:
        for r in rows:
            if sel(r):
                fh.write(json.dumps(r) + "\n")
PYEOF
CUDA_VISIBLE_DEVICES=$g $PY train_reader.py --model $Q8 --data data/course_cv${f}_ol_reader --out runs_reader/q8_course_cv${f}_ol --vision --epochs 3 > logs/tr_q8_course_cv${f}_ol.log 2>&1
CUDA_VISIBLE_DEVICES=$g $PY score_reader.py hf runs_reader/q8_course_cv${f}_ol/merged q8_course_cv${f}_ol data/course_cv${f}_ol/eval.jsonl
python summarize_course.py results/reader_q8_course_cv${f}_ol_eval.json --data data/course_cv${f}_ol
