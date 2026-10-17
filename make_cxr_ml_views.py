"""Evidence views for the multi-label reader (train_cxr_ml.py --design wm | grid): perr.cxr_reader.render_wm (radiograph
with six numbered places circled, each place enlarged from the original image) at

  wm     the world model's top-6 addresses (precision-weighted error, sigma 8, both signs, 40 px suppression, as the
         single-finding reader and make_cxr_crops.py);
  grid   six fixed lung-field places (control).

For the reader's 60k training images (results/cxr_train60k_keys.txt) and every validation and test image; writes
data/cxr_ml_<design>/<key>.png.

    python make_cxr_ml_views.py wm
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

from perr.cxr import CACHE, mimic_frontal
from perr.cxr_reader import grid_points, render_wm
from perr.cxr_wm import error_map, peaks, valid_region

WM = "/raid/home/CAMCA/hj880/perr/cache/cxr_wm"
OUT = "/raid/home/CAMCA/hj880/perr/data/cxr_ml_{}"
DESIGN = "wm"


def work(item):
    key, path = item
    out = f"{OUT.format(DESIGN)}/{key.replace('/', '_')}.png"
    if os.path.exists(out):
        return 1
    try:
        img = np.load(f"{CACHE}/{key}.npy")
        if DESIGN == "grid":
            pts = grid_points(img)
        else:
            R = np.load(f"{WM}/{key}.cxrR.npy").astype(np.float32)
            S = np.load(f"{WM}/{key}.cxrS.npy").astype(np.float32)
            pts = peaks(error_map(R, S, 8.0, 0), valid_region(img), k=6, nms=40)
        w, h = Image.open(path).size
        s = 512 / max(w, h)
        geo = {"scale": s, "pad_x": (512 - max(1, round(w * s))) // 2, "pad_y": (512 - max(1, round(h * s))) // 2}
        render_wm(key, pts, path, geo).save(out)
        return 1
    except (FileNotFoundError, OSError) as e:
        print(f"skip {key}: {e!r}", flush=True)
        return 0


if __name__ == "__main__":
    DESIGN = sys.argv[1]
    os.makedirs(OUT.format(DESIGN), exist_ok=True)
    m = mimic_frontal()
    m["key"] = "mimic/" + m.dicom_id
    keys = set(open("results/cxr_train60k_keys.txt").read().split())
    m = m[m.key.isin(keys) | m.split.isin(["validate", "test"])]
    with Pool(64) as p:
        ok = p.map(work, list(zip(m.key, m.path)), chunksize=16)
    print(f"{DESIGN}: {sum(ok)} of {len(m)} views rendered", flush=True)
