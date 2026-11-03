"""Candidate views that show the world model's expectation next to what is there: a 2 x 2 view (448 x 448). Top row, what
is there (the two-panel view of cxr_focal_data.render: context crop of 0.3 of the longer side with the detail region
outlined | the detail region, 0.1 of the longer side, enlarged). Bottom row: the same two crops rendered from the image
the world model expected, expected = observed - residual. The residual R = observed - predicted (cache/cxr_wm/<key>.
<tag>R.npy, intensity / 255 on the 512 canvas, infer_cxr_wm.py) is resampled bilinearly from the canvas onto the original
pixel grid and subtracted from the original image (x 255), so the expected image keeps the detail finer than the canvas
and loses what the model did not predict. The residual is first smoothed (Gaussian, --sigma 1 canvas px): the dense
prediction is tiled from independent 16 px blocks, and the raw residual carries their seams into the expected image;
at sigma 1 the seams go, ribs and vessels stay, and a focal departure is still removed (sigma 2 leaves much of it).
The candidate rows of an existing set are kept unchanged (same images, places
and labels); only the view is re-rendered.

    python render_views_expect.py wmu cxdet_train cxdet_val cxdet_test     # -> data/cxr_focal/wmu_x/<set>.jsonl
    python render_views_expect.py wmu cxdet_val --tags cxr --name wmu_x512   # one world model's expectation only

--tags: the residuals averaged into one expectation (default both world models of the wmu proposer, the 512 and the
1024 grid model, so every candidate shows the same expectation whichever model proposed it).
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

import cxr_focal_data as cfd

ROOT = "/raid/home/CAMCA/hj880/perr"
SIDE = cfd.SIDE
TAGS = ("cxr", "cxrh")


def expected_image(orig, key, geo, tags, sigma=1.0):
    """The world model's expected image on the original pixel grid (uint8 'L')."""
    R = np.mean([np.load(f"{ROOT}/cache/cxr_wm/{key}.{t}R.npy").astype(np.float32) for t in tags], 0)
    if sigma > 0:
        R = ndi.gaussian_filter(R, sigma)
    w, h = orig.size
    s = geo["scale"]
    yy = (np.arange(h) + 0.5) * s - 0.5 + geo["pad_y"]          # original pixel centre -> canvas coordinate
    xx = (np.arange(w) + 0.5) * s - 0.5 + geo["pad_x"]
    Ro = ndi.map_coordinates(R, np.meshgrid(yy, xx, indexing="ij"), order=1, mode="nearest")
    E = np.asarray(orig, np.float32) - 255.0 * Ro
    return Image.fromarray(np.clip(np.round(E), 0, 255).astype(np.uint8))


def render_expect(orig, exp, x, y):
    top, bot = cfd.render(orig, x, y), cfd.render(exp, x, y)
    out = Image.new("RGB", (2 * SIDE, 2 * SIDE))
    out.paste(top, (0, 0))
    out.paste(bot, (0, SIDE))
    d = ImageDraw.Draw(out)                                        # a thin dark separator between the two rows
    d.line([(0, SIDE), (2 * SIDE, SIDE)], fill=(0, 0, 0), width=2)
    return out


def work(item):
    key, path, geo, rows, vdir, tags, sigma = item
    orig = Image.open(path).convert("L")
    exp = None
    out = []
    for r in rows:
        p = f"{vdir}/{key}/{r['i']}.png"
        if not os.path.exists(p):
            if exp is None:
                exp = expected_image(orig, key, geo, tags, sigma)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            x, y = (r["c"] - geo["pad_x"]) / geo["scale"], (r["r"] - geo["pad_y"]) / geo["scale"]
            render_expect(orig, exp, x, y).save(p)
        out.append({**r, "view": p})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proposer")
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--tags", nargs="+", default=list(TAGS))
    ap.add_argument("--sigma", type=float, default=1.0, help="Gaussian smoothing of the residual (canvas px)")
    ap.add_argument("--name", default=None, help="output name (default <proposer>_x)")
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    name = a.name or f"{a.proposer}_x"
    meta = {}
    for which in sorted({s.split("_")[0] for s in a.sets}):
        for _, (key, path, geo, _bx, _t) in cfd.items(which):
            meta[key] = (path, geo)
    os.makedirs(f"{cfd.OUT}/{name}", exist_ok=True)
    with Pool(a.procs) as p:
        for s in a.sets:
            rows = [json.loads(line) for line in open(f"{cfd.OUT}/{a.proposer}/{s}.jsonl")]
            by = {}
            for r in rows:
                by.setdefault(r["key"], []).append(r)
            vdir = f"{cfd.OUT}/views/{name}/{s}"
            res = p.map(work, [(k, *meta[k], v, vdir, tuple(a.tags), a.sigma) for k, v in by.items()], chunksize=2)
            done = {(r["key"], r["i"]): r for rr in res for r in rr}
            with open(f"{cfd.OUT}/{name}/{s}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(done[(r["key"], r["i"])]) + "\n")
            print(f"{name}/{s}: {len(rows)} views", flush=True)


if __name__ == "__main__":
    main()
