"""CheXlocalize validation set (Stanford AIMI; radiologist segmentations of 10 CheXpert pathologies on the CheXpert
validation radiographs; images from CheXpert Plus PNG_valid) on the canonical grids, for the unseen-finding test
(open_set_chexloc.py). Licensed data: everything stays under cache/ and results/ (both untracked).

  cache/cxr/chexloc/<id>.npy, cache/cxr1024/chexloc/<id>.npy   512 / 1024 letterbox (perr/cxr.py), frontal views
  cache/chexloc_masks/<id>.npz                                  one bool 512 x 512 mask per pathology (letterboxed)
  results/cxr_chexloc.csv                                       geometry; results/cxr_chexloc_masks.csv: pixel counts

Masks are COCO RLE in the original pixel grid (size = [h, w]); they are decoded here without pycocotools.

    python -I build_chexloc.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, "/raid/home/CAMCA/hj880/perr")
from perr.cxr import build, letterbox  # noqa: E402

ROOT = "/raid/home/CAMCA/hj880/perr"
SEG = "/raid/home/CAMCA/hj880/dl_chexlocalize/CheXlocalize/gt_segmentations_val.json"
IMG = "/raid/home/CAMCA/hj880/dl_chexpertplus_valid"
MASKS = f"{ROOT}/cache/chexloc_masks"


def rle_counts(s: str) -> list[int]:
    """COCO compressed RLE string -> run lengths (maskApi.c rleFrString)."""
    cnts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, 1
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    return cnts


def decode(rle) -> np.ndarray:
    h, w = rle["size"]
    flat = np.zeros(h * w, bool)
    pos, val = 0, False
    for n in rle_counts(rle["counts"]):
        if val:
            flat[pos:pos + n] = True
        pos += n
        val = not val
    assert pos == h * w, (pos, h * w)
    return flat.reshape((w, h)).T                     # column-major


def main():
    seg = json.load(open(SEG))
    geo, cnt = [], []
    os.makedirs(MASKS, exist_ok=True)
    for name in sorted(k for k in seg if k.endswith("frontal")):
        pat, study, view = name.split("_", 2)
        path = f"{IMG}/{pat}/{study}/{view}.png"
        key = f"chexloc/{name}"
        g = build(key, path)
        out1024 = f"{ROOT}/cache/cxr1024/{key}.npy"
        if not os.path.exists(out1024):
            os.makedirs(os.path.dirname(out1024), exist_ok=True)
            np.save(out1024, letterbox(Image.open(path), 1024)[0])
        geo.append(g)
        ms = {}
        for path_name, rle in seg[name].items():
            m = decode(rle)
            assert m.shape == (g["h"], g["w"]), (name, m.shape, g)
            # letterbox the mask exactly as the image (nearest), then binarize
            mm, _, _ = letterbox(Image.fromarray(m.astype(np.uint8) * 255), 512)
            ms[path_name] = mm > 127
            cnt.append({"key": key, "finding": path_name, "pixels_orig": int(m.sum()), "pixels_512": int(ms[path_name].sum())})
        np.savez_compressed(f"{MASKS}/{name}.npz", **{k.replace(" ", "_"): v for k, v in ms.items()})
    pd.DataFrame(geo).to_csv(f"{ROOT}/results/cxr_chexloc.csv", index=False)
    c = pd.DataFrame(cnt)
    c.to_csv(f"{ROOT}/results/cxr_chexloc_masks.csv", index=False)
    print(len(geo), "frontal images")
    print(c[c.pixels_512 > 0].groupby("finding").key.count().to_string())


if __name__ == "__main__":
    main()
