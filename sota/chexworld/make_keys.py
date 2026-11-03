"""Image lists for the world-model anomaly maps (CheXWorld, RadJEPA): the verifier's held-out tenth of the ChestX-Det
training images, reproduced exactly as train_cxr_oneshot.py builds it (sorted keys of data/cxr_focal/wm/cxdet_train.jsonl,
random.Random(0).shuffle, first len // 10), and the ChestX-Det test images (results/cxr_cxdet.csv, split == "test").

    python sota/chexworld/make_keys.py
Writes sota/{chexworld,radjepa}/heldout_keys.txt (302) and test_keys.txt (553); only keys, no labels are read.
"""
import json
import os
import random

import pandas as pd

ROOT = "/raid/home/CAMCA/hj880/perr"


def main():
    fkeys = sorted({json.loads(l)["key"] for l in open(f"{ROOT}/data/cxr_focal/wm/cxdet_train.jsonl")})
    random.Random(0).shuffle(fkeys)
    held = sorted(fkeys[:len(fkeys) // 10])
    g = pd.read_csv(f"{ROOT}/results/cxr_cxdet.csv")
    test = sorted(g[g.split == "test"].key)
    assert not set(held) & set(test)
    assert all(os.path.exists(f"{ROOT}/cache/cxr/{k}.npy") for k in held + test)
    for d in ("chexworld", "radjepa"):
        open(f"{ROOT}/sota/{d}/heldout_keys.txt", "w").write("\n".join(held) + "\n")
        open(f"{ROOT}/sota/{d}/test_keys.txt", "w").write("\n".join(test) + "\n")
    print(f"held-out tenth {len(held)} images, test {len(test)} images")


if __name__ == "__main__":
    main()
