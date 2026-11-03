"""Test-time views for the focal radiograph verifier: the same addresses as data/cxr_focal/<src>/<set>.jsonl, re-rendered
from the original image with both crop sides (context 0.3, detail 0.1 of the longer side) divided by a zoom factor f
(f < 1 shows more surroundings, f > 1 less). This is augment_view's scale jitter done from the original pixels, so no
black border enters the view. The address stays at the centre (no translation, no flips).

    python make_cxr_tta_views.py wmu cxdet_val 0.9      # -> data/cxr_focal/tta_z0.9_wmu/cxdet_val.jsonl
"""
import json
import os
import sys
from multiprocessing import Pool

import pandas as pd
from PIL import Image

import cxr_focal_data as cfd

ROOT = "/raid/home/CAMCA/hj880/perr"


def work(item):
    key, rows, path, geo, f, vdir = item
    orig = Image.open(path).convert("L")
    os.makedirs(f"{vdir}/{key}", exist_ok=True)
    out = []
    for r in rows:
        x, y = (r["c"] - geo["pad_x"]) / geo["scale"], (r["r"] - geo["pad_y"]) / geo["scale"]
        p = f"{vdir}/{key}/{r['i']}.png"
        if not os.path.exists(p):
            cfd.render(orig, x, y).save(p)
        out.append({**r, "view": p})
    return out


def main():
    src, which, f = sys.argv[1], sys.argv[2], float(sys.argv[3])
    cfd.CTX_FRAC, cfd.DET_FRAC = 0.3 / f, 0.1 / f       # the outline keeps its ratio (DET_FRAC / CTX_FRAC)
    name = f"tta_z{f:g}_{src}"
    rows = [json.loads(l) for l in open(f"{cfd.OUT}/{src}/{which}.jsonl")]
    g = pd.read_csv(f"{ROOT}/results/cxr_cxdet.csv").set_index("key")
    by = {}
    for r in rows:
        by.setdefault(r["key"], []).append(r)
    items = []
    for key, rs in by.items():
        gr = g.loc[key]
        path = gr.path if "path" in g else f"/raid/home/CAMCA/hj880/data_ext/chestx_det/images/{gr.split}/{gr.image}"
        items.append((key, rs, path, {"scale": gr.scale, "pad_x": gr.pad_x, "pad_y": gr.pad_y}, f,
                      f"{cfd.OUT}/views/{name}"))
    with Pool(32) as p:
        res = p.map(work, items, chunksize=2)
    os.makedirs(f"{cfd.OUT}/{name}", exist_ok=True)
    with open(f"{cfd.OUT}/{name}/{which}.jsonl", "w") as fo:
        for rs in res:
            for r in rs:
                fo.write(json.dumps(r) + "\n")
    print(f"{name}/{which}: {sum(map(len, res))} views", flush=True)


if __name__ == "__main__":
    main()
