#!/bin/bash
# GPU smoke test of dinomaly_cxr.py: 200 iterations on 4,000 images, then maps of 16 held-out images (not written to
# cache/cxr_maps; raw maps only, inside the smoke run).
PY="/raid/home/CAMCA/hj880/venvs/sota_dinomaly/bin/python -I -B"
cd /raid/home/CAMCA/hj880/perr
rm -rf sota/dinomaly/runs/smoke
$PY sota/dinomaly/dinomaly_cxr.py train --out sota/dinomaly/runs/smoke --iters 200 --train-limit 4000 2>&1 | grep -v "xFormers\|warnings.warn" && \
$PY sota/dinomaly/dinomaly_cxr.py maps --run sota/dinomaly/runs/smoke --keys sota/dinomaly/keys_smoke.txt --save-raw 2>&1 | grep -v "xFormers\|warnings.warn"
