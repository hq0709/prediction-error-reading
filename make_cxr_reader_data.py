"""MIMIC-CXR reader rows: one (image, finding) question each, labelled from the CheXpert-labeler labels (1 = yes;
0 or not mentioned = no; uncertain = skipped).

  sample   choose the training images (per finding up to 1500 positives and 1500 negatives, the negatives including
           "No Finding" images), 300 validation images and every test image; writes results/cxr_reader_images.csv
  rows     render one input per image for a design (global | wm) and write data/cxr_reader_<design>/{train,val,test}.jsonl

    python make_cxr_reader_data.py sample
    python make_cxr_reader_data.py rows global
    python make_cxr_reader_data.py rows wm --tag cxr      # needs the world model's error maps for these images
    python make_cxr_reader_data.py rows grid              # control: six fixed places enlarged
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from perr.cxr import CACHE, FINDINGS
from perr.cxr_reader import NAMES, Q_GLOBAL, Q_WM, grid_points, render_global, render_wm
from perr.cxr_wm import error_map, peaks, valid_region

WM = "/raid/home/CAMCA/hj880/perr/cache/cxr_wm"
LABELS = [f for f in FINDINGS if f != "No Finding"]
DESIGN, TAG, SIGMA, SIGN = "global", "cxr", 8.0, 0


def sample():
    a = pd.read_csv("results/cxr_mimic.csv")
    b = pd.read_csv("results/cxr_mimic_train.csv")
    tr = pd.concat([b, a[a.split == "train"]]).drop_duplicates("key")
    rng = np.random.default_rng(0)
    keys = set()
    for f in LABELS:
        pos = tr[tr[f] == 1].key.tolist()
        neg = tr[(tr[f] == 0) | tr[f].isna()].key.tolist()
        keys |= set(rng.choice(pos, min(1500, len(pos)), replace=False))
        keys |= set(rng.choice(neg, min(1500, len(neg)), replace=False))
    va = a[a.split == "validate"]
    va = va.iloc[rng.permutation(len(va))[:300]]
    te = a[a.split == "test"]
    out = pd.concat([tr[tr.key.isin(keys)].assign(part="train"), va.assign(part="val"), te.assign(part="test")])
    out.to_csv("results/cxr_reader_images.csv", index=False)
    open("results/cxr_reader_keys.txt", "w").write("\n".join(out.key) + "\n")
    print(out.part.value_counts().to_dict())


def render(row):
    key = row["key"]
    path = f"data/cxr_reader_{DESIGN}/img/{key.replace('/', '_')}.png"
    if not os.path.exists(path):
        if DESIGN == "global":
            im = render_global(key)
        elif DESIGN == "grid":
            im = render_wm(key, grid_points(np.load(f"{CACHE}/{key}.npy")), row["path"], row)
        else:
            img = np.load(f"{CACHE}/{key}.npy")
            R = np.load(f"{WM}/{key}.{TAG}R.npy").astype(np.float32)
            S = np.load(f"{WM}/{key}.{TAG}S.npy").astype(np.float32)
            pk = peaks(error_map(R, S, SIGMA, SIGN), valid_region(img), k=6, nms=40)
            im = render_wm(key, pk, row["path"], row)
        im.save(path)
    return os.path.abspath(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["sample", "rows"])
    ap.add_argument("design", nargs="?", default="global", choices=["global", "wm", "grid"])
    ap.add_argument("--tag", default="cxr")
    a = ap.parse_args()
    if a.what == "sample":
        sample()
    else:
        DESIGN, TAG = a.design, a.tag
        os.makedirs(f"data/cxr_reader_{DESIGN}/img", exist_ok=True)
        d = pd.read_csv("results/cxr_reader_images.csv")
        with Pool(32) as p:
            d["image"] = p.map(render, d.to_dict("records"), chunksize=16)
        q = Q_GLOBAL if DESIGN == "global" else Q_WM                # wm and grid share the wording
        rng = np.random.default_rng(1)
        for part in ("train", "val", "test"):
            rows = []
            for r in d[d.part == part].to_dict("records"):
                for i, f in enumerate(LABELS):
                    v = r[f]
                    if v == -1:
                        continue
                    rows.append({"uid": r["key"], "peak": i + 1, "finding": f, "image": r["image"], "question": q.format(f=NAMES[f]),
                                 "label": int(v == 1), "wm": 1.0, "source": "wm"})
            if part == "train":                         # keep the per-finding balance chosen in sample()
                keep = []
                for f in LABELS:
                    fr = [x for x in rows if x["finding"] == f]
                    pos = [x for x in fr if x["label"]]
                    neg = [x for x in fr if not x["label"]]
                    keep += pos[:1500] + [neg[j] for j in rng.permutation(len(neg))[:1500]]
                rows = keep
            with open(f"data/cxr_reader_{DESIGN}/{part}.jsonl", "w") as fh:
                for x in rows:
                    fh.write(json.dumps(x) + "\n")
            print(DESIGN, part, len(rows), "rows,", sum(x["label"] for x in rows), "positive")
