"""open_set_cxr.py on a second dataset: the 10 CheXlocalize pathologies (radiologist segmentations, CheXpert validation
radiographs, build_chexloc.py). No method was trained on any of them; every model is the one used on ChestX-Det,
unchanged (no training, no tuning; same canvas, world models, addresses, detector runs and K).

A method points at a finding when one of its first K addresses lies inside the finding's region. Regions are the
connected components (8-connectivity) of the letterboxed segmentation; the "mask" columns test the pixel mask, the
"box" columns the component's bounding box (the ChestX-Det table uses boxes). "< 2% of the image" keeps components whose
box covers less than 2% of the 512 x 512 canvas, as on ChestX-Det; an image counts once per finding.

    python open_set_chexloc.py
Writes results/open_set_chexloc.csv and results/open_set_chexloc.json.
"""
import json
import os
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from open_set_cxr import DETECTORS, KS, random_addresses, union_addresses

ROOT = "/raid/home/CAMCA/hj880/perr"
MASKS = f"{ROOT}/cache/chexloc_masks"
FINDINGS = ("Airspace Opacity", "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
            "Lung Lesion", "Pleural Effusion", "Pneumothorax", "Support Devices")


def regions(key):
    """{finding: [(component mask, (x0, y0, x1, y1), box area fraction), ...]} on the 512 canvas."""
    z = np.load(f"{MASKS}/{key.split('/', 1)[1]}.npz")
    out = {}
    for f in FINDINGS:
        m = z[f.replace(" ", "_")]
        if not m.any():
            continue
        lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
        comps = []
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            ys, xs = sl
            comps.append((lab[sl] == i, (xs.start, ys.start, xs.stop - 1, ys.stop - 1), sl,
                          (xs.stop - xs.start) * (ys.stop - ys.start) / 512 ** 2))
        out[f] = comps
    return out


def inside(r, c, comp, use_box):
    m, (x0, y0, x1, y1), sl, _ = comp
    if not (x0 <= c <= x1 and y0 <= r <= y1):
        return False
    if use_box:
        return True
    rr, cc = int(round(r)) - sl[0].start, int(round(c)) - sl[1].start
    return 0 <= rr < m.shape[0] and 0 <= cc < m.shape[1] and bool(m[rr, cc])


def main():
    keys = open(f"{ROOT}/results/chexloc_keys.txt").read().split()
    reg = {k: regions(k) for k in keys}
    methods = {}
    with Pool(24) as p:
        methods["world model, one-sided (focal setting)"] = dict(p.starmap(union_addresses, [(k, 1) for k in keys]))
        methods["world model, two-sided"] = dict(p.starmap(union_addresses, [(k, 0) for k in keys]))
        methods["random thorax points"] = dict(p.map(random_addresses, keys))
    for label, run in DETECTORS.items():
        dump = f"{ROOT}/results/cxr_boxes/{run}_chexloc.json"
        if os.path.exists(dump):
            methods[label] = {k: [(r, c) for _, r, c in v] for k, v in json.load(open(dump)).items()}
            continue
        rows = [json.loads(l) for l in open(f"{ROOT}/results/cxr_focal/{run}/chexloc.jsonl")]
        per = defaultdict(list)
        for r in rows:
            per[r["key"]].append((max(v for k_, v in r.items() if k_.startswith("p_")), r["r"], r["c"]))
        methods[label] = {k: [(r, c) for _, r, c in sorted(v, reverse=True)] for k, v in per.items()}
    out = []
    for size, cap in (("all boxes", 1.0), ("boxes < 2% of the image", 0.02)):
        for use_box, region in ((False, "mask"), (True, "box")):
            for label, addr in methods.items():
                for f in FINDINGS:
                    imgs = {k: [c for c in reg[k].get(f, []) if c[3] < cap] for k in keys}
                    imgs = {k: v for k, v in imgs.items() if v}
                    if not imgs:
                        continue
                    for K in KS:
                        h = [any(inside(r, c, comp, use_box) for r, c in addr.get(k, [])[:K] for comp in comps)
                             for k, comps in imgs.items()]
                        out.append({"boxes": size, "region": region, "method": label, "finding": f, "K": K,
                                    "images": len(imgs), "hit": float(np.mean(h))})
    df = pd.DataFrame(out)
    df.to_csv(f"{ROOT}/results/open_set_chexloc.csv", index=False)
    json.dump(out, open(f"{ROOT}/results/open_set_chexloc.json", "w"), indent=0)
    pd.set_option("display.width", 250)
    for size in df.boxes.unique():
        for region in ("mask", "box"):
            d = df[(df.boxes == size) & (df.region == region) & (df.K == 10)]
            print(f"\n{size}, hit inside {region}, K = 10")
            t = d.pivot(index="finding", columns="method", values="hit")
            t["images"] = d.groupby("finding").images.first()
            print(t.round(2).to_string())
            print("mean over types:", d.groupby("method").hit.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
