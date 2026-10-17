"""Full-resolution evidence views for the two-stream radiograph reader (train_cxr_evidence.py): for every image, K
square crops of the original MIMIC JPEG (side = CROP_FRAC of the longer side, ~600 px), resized to 224 x 224, centred at

  wm     the world model's top-K addresses (precision-weighted error peaks, the same as the VLM reader's: sigma 8,
         both signs, 40 px suppression on the 512 grid);
  grid   control: six fixed places (upper, middle, lower zones of each lung field), i.e. enlargement without the
         world model.
  focal  the world model's ten focal addresses inside the thorax (perr.cxr_anat.search_domain; precision-weighted
         error, sigma 2, brighter than expected, 16 px suppression), crops of 0.1 of the longer side (~native detail);
  focal_random  control: ten random thorax points, same crops (enlargement at places the world model did not choose).

A crop at 224 px shows its region at ~2.2x the detail of the 512 px canonical image (~3.5x the 320 px input of the
global reader). Writes cache/cxr_crops_<design>/<key>.npy (K, 224, 224) uint8 and the centres (512 grid) to
cache/cxr_crops_<design>/<key>.pts.npy, for the 60k training images and every validation and test image.

    python make_cxr_crops.py wm
    python make_cxr_crops.py grid
    python make_cxr_crops.py grid all      # every training image (full-data runs)
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

from perr.cxr import CACHE, mimic_frontal
from perr.cxr_reader import grid_points
from perr.cxr_anat import search_domain
from perr.cxr_wm import error_map, peaks, valid_region

K, SIDE, CROP_FRAC = 6, 224, 0.2
SETTINGS = {"focal": (10, 0.1), "focal_random": (10, 0.1)}           # design -> (K, CROP_FRAC); others use the default
WM = "/raid/home/CAMCA/hj880/perr/cache/cxr_wm"
OUT = "/raid/home/CAMCA/hj880/perr/cache/cxr_crops_{}"
DESIGN = "wm"


def addresses(key, img):
    if DESIGN == "grid":
        return grid_points(img)
    if DESIGN == "focal_random":
        rng = np.random.default_rng(int(key.split("-")[-1], 16) % 2 ** 32)
        return peaks(rng.random((512, 512)).astype(np.float32), search_domain(key, valid_region(img)), k=K, nms=16)
    if DESIGN == "focal":
        R = np.load(f"{WM}/{key}.cxrR.npy").astype(np.float32)
        S = np.load(f"{WM}/{key}.cxrS.npy").astype(np.float32)
        return peaks(error_map(R, S, 2.0, 1), search_domain(key, valid_region(img)), k=K, nms=16)
    R = np.load(f"{WM}/{key}.cxrR.npy").astype(np.float32)
    S = np.load(f"{WM}/{key}.cxrS.npy").astype(np.float32)
    return peaks(error_map(R, S, 8.0, 0), valid_region(img), k=K, nms=40)


def work(item):
    key, path = item
    out = f"{OUT.format(DESIGN)}/{key}.npy"
    if os.path.exists(out):
        return 1
    try:
        img = np.load(f"{CACHE}/{key}.npy")
        pts = addresses(key, img)
        orig = Image.open(path).convert("L")
    except (FileNotFoundError, OSError) as e:
        print(f"skip {key}: {e!r}", flush=True)
        return 0
    w, h = orig.size
    s = 512 / max(w, h)
    px, py = (512 - max(1, round(w * s))) // 2, (512 - max(1, round(h * s))) // 2
    side = CROP_FRAC * max(w, h)
    crops = np.zeros((K, SIDE, SIDE), np.uint8)
    for n, (r, c, _) in enumerate(pts[:K]):
        x, y = (c - px) / s, (r - py) / s
        box = (int(x - side / 2), int(y - side / 2), int(x + side / 2), int(y + side / 2))
        crops[n] = np.asarray(orig.crop(box).resize((SIDE, SIDE), Image.LANCZOS), np.uint8)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out[:-4] + ".pts.npy", np.array([(r, c) for r, c, _ in pts[:K]], np.int16))
    np.save(out, crops)
    return 1


if __name__ == "__main__":
    DESIGN = sys.argv[1]
    K, CROP_FRAC = SETTINGS.get(DESIGN, (K, CROP_FRAC))
    m = mimic_frontal()
    m["key"] = "mimic/" + m.dicom_id
    if len(sys.argv) > 2 and sys.argv[2] == "all":
        m = m[[os.path.exists(f"{CACHE}/{k}.npy") for k in m.key]]
    else:
        keys = set(open("results/cxr_train60k_keys.txt").read().split())
        m = m[m.key.isin(keys) | m.split.isin(["validate", "test"])]
    items = list(zip(m.key, m.path))
    with Pool(64) as p:
        ok = p.map(work, items, chunksize=16)
    print(f"{DESIGN}: {sum(ok)} of {len(items)} images have crops", flush=True)
