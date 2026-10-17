"""Which official MIMIC-CXR test images could the torchxrayvision MIMIC models have been trained on?

The released mimic_ch / mimic_nb weights (and the MIMIC part of "all") were published on 2020-03-30 (commit 13358b1),
before the training script appeared in the repository.  The official split was never used.  Two reconstructions:

  H2 (code as of the weight release + first public train_model.py, 2020-05): MIMIC_Dataset kept PA views only and,
     with unique_patients=True (default), one row per patient (groupby(subject_id).first()); the script then split
     each dataset 50/50 with torch.manual_seed(0) + torch.utils.data.random_split (image level = patient level here).
  H1 (train_model.py since 2020-12): PA + AP, every image, GroupShuffleSplit(train 0.8, test 0.2, random_state=0)
     grouped by patient.

Writes key lists (official test frontal images) for each hypothesis into this directory:
  test_keys_<H>_trainpat.txt  images of test patients assigned to the xrv training part
  test_keys_<H>_heldpat.txt   images of test patients assigned to the xrv held-out part
  test_keys_H2_trainimg.txt   (H2) the single image per patient that was itself a training image
  test_keys_H2_heldimg.txt    (H2) the same image slot (first PA image) of the held-out patients
The reconstruction is only trusted if the xrv models show a memorisation gap on it that a model trained on the
official train split does not show (see NOTES.md).
"""
import os
import sys

import numpy as np
import pandas as pd
import sklearn.model_selection
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import split_table  # noqa: E402

CSV = "/home/local/PARTNERS/hj880/dMLLM_in_Medicine/data/mimic"


def joined():
    lab = pd.read_csv(f"{CSV}/mimic-cxr-2.0.0-chexpert (1).csv").set_index(["subject_id", "study_id"])
    meta = pd.read_csv(f"{CSV}/mimic-cxr-2.0.0-metadata.csv").set_index(["subject_id", "study_id"])
    return lab.join(meta).reset_index()


def main():
    test = split_table("test")
    test["pid"] = test.subject_id.astype(int)
    out = {}
    j = joined()
    print("chexpert rows", j[["subject_id", "study_id"]].drop_duplicates().shape[0], "joined image rows", len(j))

    # H2: PA only, one row per patient, 50/50 random_split with torch.manual_seed(0)
    pa = j[j.ViewPosition == "PA"]
    first = pa.groupby("subject_id").first().reset_index()           # sorted by subject_id, as in the old loader
    n = len(first)
    torch.manual_seed(0)
    perm = torch.randperm(n).tolist()
    tr = set(first.subject_id.iloc[perm[: int(0.5 * n)]])
    trimg = set(first.dicom_id.iloc[perm[: int(0.5 * n)]])
    heldimg = set(first.dicom_id.iloc[perm[int(0.5 * n):]])
    out["H2"] = (tr, set(first.subject_id) - tr)
    print(f"H2: {n} patients with a PA image; {len(tr)} in the xrv training half")
    with open(f"{HERE}/test_keys_H2_trainimg.txt", "w") as f:
        f.write("\n".join(test.key[test.dicom_id.isin(trimg)]) + "\n")
    with open(f"{HERE}/test_keys_H2_heldimg.txt", "w") as f:
        f.write("\n".join(test.key[test.dicom_id.isin(heldimg)]) + "\n")
    print("H2: official test images that were themselves xrv training images:", int(test.dicom_id.isin(trimg).sum()),
          "; the same image slot of held-out patients:", int(test.dicom_id.isin(heldimg).sum()))

    # H1: PA + AP, all images, GroupShuffleSplit 80/20 by patient, seed 0
    d = j[j.ViewPosition.isin(["PA", "AP"])].reset_index(drop=True)
    pid = d.subject_id.astype(str)
    gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.8, test_size=0.2, random_state=0)
    tri, tei = next(gss.split(X=range(len(d)), groups=pid))
    out["H1"] = (set(d.subject_id.iloc[tri]), set(d.subject_id.iloc[tei]))
    print(f"H1: {len(d)} images, {pid.nunique()} patients; {len(out['H1'][0])} patients in the xrv training part")

    for h, (trp, hep) in out.items():
        a, b = test.pid.isin(trp), test.pid.isin(hep)
        with open(f"{HERE}/test_keys_{h}_trainpat.txt", "w") as f:
            f.write("\n".join(test.key[a]) + "\n")
        with open(f"{HERE}/test_keys_{h}_heldpat.txt", "w") as f:
            f.write("\n".join(test.key[b]) + "\n")
        print(f"{h}: official test patients {test.pid.nunique()}: in xrv train {test.pid[a].nunique()} "
              f"({int(a.sum())} images), held out {test.pid[b].nunique()} ({int(b.sum())} images), "
              f"in neither {test.pid[~a & ~b].nunique()} ({int((~a & ~b).sum())} images)")


if __name__ == "__main__":
    main()
