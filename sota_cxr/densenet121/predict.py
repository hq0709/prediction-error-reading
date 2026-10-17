"""Test-split predictions of the supervised DenseNet-121 at the epoch chosen on the official validate split.

    python predict.py [--ckpt ckpt] [--epoch E]
Writes results/sota_cxr/densenet121.csv (official test split, frontal images) and densenet121_validate.csv.
"""
import argparse
import json
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common import FINDINGS, OUT, split_table  # noqa: E402
from train import CXR, build_model, predict, val_auc, worker_init  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{HERE}/ckpt")
    ap.add_argument("--epoch", type=int, default=-1, help="default: best epoch by validate mean AUROC")
    ap.add_argument("--name", default="densenet121")
    a = ap.parse_args()
    hist = json.load(open(f"{a.ckpt}/history.json"))
    ep = a.epoch if a.epoch >= 0 else max(hist, key=lambda h: h["val_mean_auroc"])["epoch"]
    ck = torch.load(f"{a.ckpt}/epoch{ep}.pt", map_location="cpu", weights_only=False)
    res = ck["args"]["res"]
    dev = torch.device("cuda")
    model = build_model()
    model.load_state_dict(ck["model"])
    model = model.to(dev).to(memory_format=torch.channels_last).eval()
    print(f"epoch {ep} (validate mean AUROC {ck['hist'][-1]['val_mean_auroc']:.4f}), input {res} px")
    for split, suffix in (("validate", "_validate"), ("test", "")):
        df = split_table(split)
        ds = CXR(df)
        assert ds.cached.all(), "validate / test images are all in the cache"
        prob = predict(model, DataLoader(ds, batch_size=128, num_workers=8, worker_init_fn=worker_init), res, dev)
        out = pd.DataFrame(prob, columns=list(FINDINGS))
        out.insert(0, "key", df.key.values)
        out.to_csv(f"{OUT}/{a.name}{suffix}.csv", index=False)
        if split == "validate":
            print(f"validate mean AUROC recomputed: {val_auc(prob, ds)[0]:.4f} ({len(df)} images)")
        print(f"wrote {OUT}/{a.name}{suffix}.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
