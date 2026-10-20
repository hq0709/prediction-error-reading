"""Candidate views for the focal radiograph reader, the chest-radiograph counterpart of the lung CT candidate strips:
the world model's top focal addresses inside the thorax (precision-weighted error, sigma 2, brighter than expected,
16 px suppression on the 512 canvas; the configuration chosen on ChestX-Det train) and, for training, the centres of
the radiologist boxes. Each address is rendered from the original image as one 448 x 224 view: a context crop of 0.3 of
the longer side with the detail region outlined, and that detail region (0.1 of the longer side) at ~native resolution.

Labels (ChestX-Det boxes): an address carries a finding when it lies inside one of that finding's boxes; Nodule, Mass,
Calcification, Fracture.

    python cxr_focal_data.py cxdet            # ChestX-Det train (world-model candidates + box centres) and test
    python cxr_focal_data.py nih              # NIH box images
    python cxr_focal_data.py mimic            # MIMIC validation and test images
    python cxr_focal_data.py mimic_train      # MIMIC training images for report-label training (results/cxr_mil_keys.txt)
    python cxr_focal_data.py cxdet --proposer patchcore|random   # ablations: other address sources, same views
    python cxr_focal_data.py cxdet --proposer grid    # every place (25 px lattice over the thorax): reading without a proposer
    python cxr_focal_data.py cxdet --proposer wmh|wmu # the 1024-grid world model / both world models in turn
    python cxr_focal_data.py cxdet --proposer wmr     # ablation: the world model's addresses over the whole radiograph
    python cxr_focal_data.py cxdet --proposer wmn --split test   # the 512 world model trained on NIH normals (another site)
    python cxr_focal_data.py cxdet --proposer wma --split test   # the 512 world model trained on MIMIC images of any label

Writes data/cxr_focal/<proposer>/<set>.jsonl (one row per address) and the views under data/cxr_focal/views/.
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from cxr_recall import boxes
from perr.cxr import CACHE
from perr.cxr_anat import search_domain
from perr.cxr_wm import error_map, peaks, valid_region

ROOT = "/raid/home/CAMCA/hj880/perr"
OUT = f"{ROOT}/data/cxr_focal"
TYPES = ("Nodule", "Mass", "Calcification", "Fracture")
K, NMS, SIDE = 30, 16, 224
CTX_FRAC, DET_FRAC = 0.3, 0.1
PROPOSER = "wm"


def addresses(key):
    img = np.load(f"{CACHE}/{key}.npy")
    D = search_domain(key, valid_region(img)) if PROPOSER not in ("wmr", "wmur") else valid_region(img)
    if PROPOSER in ("wm", "wmh", "wmr", "wmn", "wma"):                   # wmh: the 1024-grid world model (train_cxr_wm.py --canvas 1024)
        t = {"wmh": "cxrh", "wmn": "cxrn", "wma": "cxra"}.get(PROPOSER, "cxr")  # wmn: trained on NIH normals; wma: on any MIMIC image
        R = np.load(f"{ROOT}/cache/cxr_wm/{key}.{t}R.npy").astype(np.float32)
        S = np.load(f"{ROOT}/cache/cxr_wm/{key}.{t}S.npy").astype(np.float32)
        W = error_map(R, S, 2.0, 1)
    elif PROPOSER in ("wmu", "wmu3", "wmuraw", "wmur"):            # wmur: both, over the whole radiograph                   # both (wmu3: and the site's own normal-radiograph model;
        lists = []                                                # wmuraw: raw residual, no precision weighting):
        for t in (("cxr", "cxrh", "cxrn") if PROPOSER == "wmu3" else ("cxr", "cxrh")):   # in turn, repeats within 16 px dropped
            R = np.load(f"{ROOT}/cache/cxr_wm/{key}.{t}R.npy").astype(np.float32)
            S = np.load(f"{ROOT}/cache/cxr_wm/{key}.{t}S.npy").astype(np.float32)
            W = ndi.gaussian_filter(R, 2.0) if PROPOSER == "wmuraw" else error_map(R, S, 2.0, 1)
            lists.append(peaks(W, D, k=K, nms=NMS))
        out = []
        for ps in zip(*lists):
            for p in ps:
                if all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) > NMS for q in out):
                    out.append(p)
        return out[:len(lists) * K]
    elif PROPOSER == "patchcore":
        W = ndi.zoom(np.load(f"{ROOT}/cache/cxr_maps/patchcore/{key}.npy").astype(np.float32), 512 / 37, order=1)
    elif PROPOSER == "grid":                                       # every place: a 25 px lattice over the thorax
        return [(r, c, 0.0) for r in range(12, 512, 25) for c in range(12, 512, 25) if D[r, c]]
    else:
        W = np.random.default_rng(abs(hash(key)) % 2 ** 32).random((512, 512)).astype(np.float32)
    return peaks(W, D, k=K, nms=NMS)


def render(orig, x, y):
    """448 x 224: context crop (detail region outlined) | detail crop; x, y in original pixels."""
    L = max(orig.size)
    cs, ds = CTX_FRAC * L, DET_FRAC * L
    ctx = orig.crop((int(x - cs / 2), int(y - cs / 2), int(x + cs / 2), int(y + cs / 2))).resize((SIDE, SIDE), Image.LANCZOS)
    det = orig.crop((int(x - ds / 2), int(y - ds / 2), int(x + ds / 2), int(y + ds / 2))).resize((SIDE, SIDE), Image.LANCZOS)
    ctx = ctx.convert("RGB")
    h = SIDE * DET_FRAC / CTX_FRAC / 2
    ImageDraw.Draw(ctx).rectangle([SIDE / 2 - h, SIDE / 2 - h, SIDE / 2 + h, SIDE / 2 + h], outline=(255, 255, 0), width=2)
    out = Image.new("RGB", (2 * SIDE, SIDE))
    out.paste(ctx, (0, 0))
    out.paste(det.convert("RGB"), (SIDE, 0))
    return out


def work(item):
    key, path, geo, bx, train = item
    try:
        pts = [(r, c, v, "wm") for r, c, v in addresses(key)]
    except FileNotFoundError:
        return []
    if train:                                                       # box centres, each with a jittered copy
        rng = np.random.default_rng(abs(hash(key)) % 2 ** 32)
        for b in bx:
            if b["finding"] in TYPES:
                cr, cc = b["y"] + b["h"] / 2, b["x"] + b["w"] / 2
                for jit in (0.0, 0.25):
                    pts.append((cr + jit * b["h"] * rng.uniform(-1, 1), cc + jit * b["w"] * rng.uniform(-1, 1), 0.0, "gt"))
    orig = Image.open(path).convert("L")
    rows = []
    vdir = f"{OUT}/views/{PROPOSER}/{key}"
    os.makedirs(vdir, exist_ok=True)
    for i, (r, c, v, src) in enumerate(pts):
        x, y = (c - geo["pad_x"]) / geo["scale"], (r - geo["pad_y"]) / geo["scale"]
        p = f"{vdir}/{i}.png"
        if not os.path.exists(p):
            render(orig, x, y).save(p)
        lab = {t: int(any(b["finding"] == t and b["x"] <= c <= b["x"] + b["w"] and b["y"] <= r <= b["y"] + b["h"] for b in bx))
               for t in TYPES}
        rows.append({"key": key, "i": i, "r": float(r), "c": float(c), "wm": float(v), "src": src, "rank": i if src == "wm" else -1,
                     "view": p, **lab})
    return rows


def items(which):
    if which in ("cxdet", "nih"):
        g = pd.read_csv(f"{ROOT}/results/cxr_{which}.csv")
        b = boxes(which)
        by = {k: v[["finding", "x", "y", "w", "h"]].to_dict("records") for k, v in b.groupby("key")}
        split = g.split if "split" in g else pd.Series(["test"] * len(g))
        for (_, r), s in zip(g.iterrows(), split):
            path = r.path if "path" in g else f"/raid/home/CAMCA/hj880/data_ext/chestx_det/images/{s}/{r.image}"
            yield s, (r.key, path, {"scale": r.scale, "pad_x": r.pad_x, "pad_y": r.pad_y}, by.get(r.key, []),
                      s == "train" and PROPOSER in ("wm", "wmu"))
    else:
        from perr.cxr import letterbox, mimic_frontal                     # noqa: F401
        m = mimic_frontal()
        m["key"] = "mimic/" + m.dicom_id
        if which == "mimic_train":
            m = m[(m.split == "train") & m.key.isin(set(open(f"{ROOT}/results/cxr_mil_keys.txt").read().split()))]
        else:
            m = m[m.split.isin(["validate", "test"])]
        for r in m.itertuples():
            w, h = r.Columns, r.Rows
            s = 512 / max(w, h)
            geo = {"scale": s, "pad_x": (512 - max(1, round(w * s))) // 2, "pad_y": (512 - max(1, round(h * s))) // 2}
            yield r.split, (r.key, r.path, geo, [], False)


def main():
    global PROPOSER
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["cxdet", "nih", "mimic", "mimic_train"])
    ap.add_argument("--proposer", choices=["wm", "wmh", "wmu", "wmu3", "wmuraw", "wmur", "wmr", "wmn", "wma", "patchcore", "random", "grid"], default="wm")
    ap.add_argument("--split", default=None, help="only this split (e.g. test)")
    a = ap.parse_args()
    PROPOSER = a.proposer
    its = [it for it in items(a.which) if a.split is None or it[0] == a.split]
    with Pool(48) as p:
        res = p.map(work, [it for _, it in its], chunksize=2)
    os.makedirs(f"{OUT}/{PROPOSER}", exist_ok=True)
    for split in sorted({s for s, _ in its}):
        rows = [r for (s, _), rr in zip(its, res) if s == split for r in rr]
        with open(f"{OUT}/{PROPOSER}/{a.which.split('_')[0]}_{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        pos = {t: sum(r[t] for r in rows) for t in TYPES}
        print(f"{a.which} {split}: {len({r['key'] for r in rows})} images, {len(rows)} addresses, positives {pos}", flush=True)


if __name__ == "__main__":
    main()
