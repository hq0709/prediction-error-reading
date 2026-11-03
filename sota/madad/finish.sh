#!/bin/bash
# After the final MAD-AD maps: candidates of the held-out tenth with the project's own script (same peak finding,
# suppression, K = 45 and thorax search mask as every proposer), and their focal-box recall@45.
set -e
cd /raid/home/CAMCA/hj880/perr
/raid/home/CAMCA/hj880/venvs/perr/bin/python cxr_focal_data.py cxdet --proposer map:madad --k 45 --keys sota/madad/keys_dev.txt --name madad_dev
/raid/home/CAMCA/hj880/venvs/perr/bin/python -I -B sota/madad/jsonl_recall.py data/cxr_focal/madad_dev/cxdet_train.jsonl | tee sota/madad/dev_recall_final.json
ls cache/cxr_maps/madad/cxdet | wc -l
