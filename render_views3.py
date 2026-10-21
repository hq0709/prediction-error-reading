"""Three-panel candidate views for the radiograph verifier: the whole radiograph with the context region outlined, the
context crop with the detail region outlined (as in the two-panel view), and the enlarged detail region. The candidate
rows of an existing set are kept unchanged (same images, places and labels); only the view is re-rendered.

    python render_views3.py wmu cxdet_train cxdet_val cxdet_test      # -> data/cxr_focal/wmu_g/<set>.jsonl

Global panel: the original image letterboxed into SIDE x SIDE (aspect kept, black padding), the context crop (0.3 of the
longer side) outlined in yellow.
"""
import json
import os
import sys
from multiprocessing import Pool

import pandas as pd
from PIL import Image, ImageDraw

import cxr_focal_data as cfd

ROOT = "/raid/home/CAMCA/hj880/perr"
SIDE = cfd.SIDE


def render3(orig, x, y):
    two = cfd.render(orig, x, y)                                   # 448 x 224: context (detail outlined) | detail
    L = max(orig.size)
    s = SIDE / L
    g = orig.convert("RGB").resize((max(1, round(orig.size[0] * s)), max(1, round(orig.size[1] * s))), Image.LANCZOS)
    glob = Image.new("RGB", (SIDE, SIDE))
    ox, oy = (SIDE - g.size[0]) // 2, (SIDE - g.size[1]) // 2
    glob.paste(g, (ox, oy))
    h = cfd.CTX_FRAC * L * s / 2
    cx, cy = ox + x * s, oy + y * s
    ImageDraw.Draw(glob).rectangle([cx - h, cy - h, cx + h, cy + h], outline=(255, 255, 0), width=2)
    out = Image.new("RGB", (3 * SIDE, SIDE))
    out.paste(glob, (0, 0))
    out.paste(two, (SIDE, 0))
    return out


def work(item):
    key, path, geo, rows, vdir = item
    orig = Image.open(path).convert("L")
    out = []
    for r in rows:
        p = f"{vdir}/{key}/{r['i']}.png"
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            x, y = (r["c"] - geo["pad_x"]) / geo["scale"], (r["r"] - geo["pad_y"]) / geo["scale"]
            render3(orig, x, y).save(p)
        out.append({**r, "view": p})
    return out


def main():
    """Arguments: <proposer>/<set> pairs (e.g. wmu/mimic_test_s0); a set name's prefix (cxdet, nih, mimic) selects the
    image source. A bare proposer followed by set names is also accepted (wmu cxdet_train cxdet_val)."""
    args = sys.argv[1:]
    pairs = [a.split("/", 1) for a in args if "/" in a] or [(args[0], s) for s in args[1:]]
    meta = {}
    for which in sorted({s.split("_")[0] for _, s in pairs}):
        for _, (key, path, geo, _bx, _t) in cfd.items(which):
            meta[key] = (path, geo)
    jobs = []
    for proposer, s in pairs:
        name = f"{proposer}_g"
        os.makedirs(f"{cfd.OUT}/{name}", exist_ok=True)
        rows = [json.loads(line) for line in open(f"{cfd.OUT}/{proposer}/{s}.jsonl")]
        by = {}
        for r in rows:
            by.setdefault(r["key"], []).append(r)
        vdir = f"{cfd.OUT}/views/{name}/{s}"
        jobs.append((name, s, rows, [(k, *meta[k], v, vdir) for k, v in by.items()]))
    with Pool(int(os.environ.get("RENDER_PROCS", "32"))) as p:
        for name, s, rows, items in jobs:
            res = p.map(work, items, chunksize=4)
            done = {(r["key"], r["i"]): r for rr in res for r in rr}
            with open(f"{cfd.OUT}/{name}/{s}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(done[(r["key"], r["i"])]) + "\n")
            print(f"{name}/{s}: {len(rows)} rows, {len(items)} images", flush=True)


if __name__ == "__main__":
    main()
