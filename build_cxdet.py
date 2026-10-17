"""ChestX-Det (Deepwise, Apache 2.0; 3,578 NIH ChestX-ray14 images, 13 findings with radiologist boxes and polygons) on
the canonical 512 grid: cache/cxr/cxdet/<name>.npy, geometry results/cxr_cxdet.csv and boxes results/cxr_cxdet_boxes.csv
(image, finding, x, y, w, h in original pixels, split), the format of the NIH box table (cxr_recall.boxes).

    python build_cxdet.py
"""
import json
import os

import pandas as pd

from perr.cxr import build

SRC = "/raid/home/CAMCA/hj880/data_ext/chestx_det"


def main():
    geo, rows = [], []
    for split, ann in (("train", "ChestX_Det_train.json"), ("test", "ChestX_Det_test.json")):
        if not os.path.isdir(f"{SRC}/images/{split}"):
            print(f"{split}: images not extracted, skipped")
            continue
        for r in json.load(open(f"{SRC}/ChestX-Det-Dataset/{ann}")):
            name = r["file_name"]
            g = build(f"cxdet/{name[:-4]}", f"{SRC}/images/{split}/{name}")
            geo.append({"image": name, **g, "split": split, "n_boxes": len(r["boxes"])})
            for sym, (x1, y1, x2, y2) in zip(r["syms"], r["boxes"]):
                rows.append({"image": name, "finding": sym, "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "split": split})
    pd.DataFrame(geo).to_csv("results/cxr_cxdet.csv", index=False)
    pd.DataFrame(rows).to_csv("results/cxr_cxdet_boxes.csv", index=False)
    print(f"{len(geo)} images, {len(rows)} boxes")


if __name__ == "__main__":
    main()
