"""Cache chest radiographs on the canonical 512 x 512 grid (perr/cxr.py) and write the image tables.

    python build_cxr.py mimic      # frontal MIMIC-CXR: training "No Finding" images, all validation and test images
    python build_cxr.py nih        # NIH ChestX-ray14 images that carry a box
    python build_cxr.py mimic_train   # the reader's training subset (results/cxr_mimic_reader_train_list.csv)
    python build_cxr.py mimic_all     # every frontal image of every split (for readers trained on all data)
Writes results/cxr_<set>.csv with each image's key, labels and geometry.
"""
import sys
from multiprocessing import Pool

import pandas as pd

from perr.cxr import NIH, build, mimic_frontal


def _build(args):
    try:
        return build(*args)
    except Exception as e:                       # unreadable file: keep going, report at the end
        return {"key": args[0], "error": repr(e)[:200]}


def run(df, name):
    with Pool(32) as p:
        geo = p.map(_build, list(zip(df.key, df.path)), chunksize=16)
    g = pd.DataFrame(geo)
    bad = g[g.get("error").notna()] if "error" in g else g.iloc[0:0]
    out = df.merge(g[g.columns.difference(["error"])] if "error" in g else g, on="key")
    out = out[~out.key.isin(bad.key)] if len(bad) else out
    out.to_csv(f"results/cxr_{name}.csv", index=False)
    print(name, len(out), "images cached;", len(bad), "unreadable", bad.head(3).to_dict("records") if len(bad) else "")


if __name__ == "__main__":
    if sys.argv[1] == "mimic_all":
        m = mimic_frontal()
        m["key"] = "mimic/" + m.dicom_id
        run(m, "mimic_all")
    elif sys.argv[1] == "mimic_train":
        lst = pd.read_csv("results/cxr_mimic_reader_train_list.csv")
        m = mimic_frontal()
        m["key"] = "mimic/" + m.dicom_id
        run(m[m.key.isin(set(lst.key))].copy(), "mimic_train")
    elif sys.argv[1] == "mimic":
        m = mimic_frontal()
        keep = (m.split.isin(["validate", "test"])) | ((m.split == "train") & (m["No Finding"] == 1.0))
        m = m[keep].copy()
        m["key"] = "mimic/" + m.dicom_id
        run(m, "mimic")
    else:
        b = pd.read_csv(f"{NIH}/BBox_List_2017.csv").iloc[:, :6]
        b.columns = ["image", "finding", "x", "y", "w", "h"]
        imgs = pd.DataFrame({"image": sorted(b.image.unique())})
        imgs["key"] = "nih/" + imgs.image.str.replace(".png", "", regex=False)
        imgs["path"] = f"{NIH}/png/images/" + imgs.image
        run(imgs, "nih")
        b.to_csv("results/cxr_nih_boxes.csv", index=False)
