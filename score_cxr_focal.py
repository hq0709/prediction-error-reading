"""Image-level reading of focal findings from scored addresses (train_cxr_focal_reader.py output): per image and finding,
the score is the reader's maximum P(yes) over the top-K addresses (or the product with the world model's normalised
error), and the pointed address is that maximum. Against the radiologist boxes:

  AUROC    presence (an image is positive when it has a box of the finding; ChestX-Det annotates its 13 findings on
           every image, so images without a box are negatives)
  hit      among positive images, the pointed address lies in a box of the finding
  LROC     area under the localisation ROC: a positive counts only when its pointed address is in a box, i.e. the
           probability that a random positive image is both ranked above a random negative image and correctly
           pointed (evidence-gated AUROC)

    python score_cxr_focal.py results/cxr_focal/cxr_focal/wm_cxdet_test.jsonl cxdet test
"""
import json
import sys

import numpy as np
import pandas as pd

from cxr_recall import boxes

TYPES = ("Nodule", "Mass", "Calcification", "Fracture")


def lroc(score, pos, hit):
    """Area under the LROC curve: P(score_pos > score_neg and hit_pos) with ties counted half."""
    s, pos, hit = np.asarray(score, float), np.asarray(pos, bool), np.asarray(hit, bool)
    sn = np.sort(s[~pos])
    if not len(sn) or not pos.any():
        return float("nan")
    below = np.searchsorted(sn, s[pos], "left") + 0.5 * (np.searchsorted(sn, s[pos], "right") - np.searchsorted(sn, s[pos], "left"))
    return float(np.sum((below / len(sn)) * hit[pos]) / pos.sum())


def auroc(score, pos):
    return lroc(score, pos, np.ones(len(pos), bool))


def evaluate(rows, b, types, ks=(10, 30), rules=("max-p", "wm x reader")):
    df = pd.DataFrame(rows)
    df = df[df.src == "wm"]
    out = []
    for k in ks:
        d = df[df["rank"] < k].copy()
        d["wmn"] = d.groupby("key").wm.transform(lambda v: v / max(v.max(), 1e-6))
        for rule in rules:
            for t in types:
                d["s"] = d[f"p_{t}"] if rule == "max-p" else d[f"p_{t}"] * d.wmn
                top = d.loc[d.groupby("key").s.idxmax()].set_index("key")
                bt = b[b.finding == t]
                keys = sorted(df.key.unique())
                pos = np.array([k_ in set(bt.key) for k_ in keys])
                score = top.s.reindex(keys).fillna(0).to_numpy()
                hit = np.array([any(r.x <= top.c.get(k_, -1) <= r.x + r.w and r.y <= top.r.get(k_, -1) <= r.y + r.h
                                    for r in bt[bt.key == k_].itertuples()) for k_ in keys])
                out.append({"K": k, "rule": rule, "finding": t, "pos": int(pos.sum()), "neg": int((~pos).sum()),
                            "AUROC": auroc(score, pos), "hit": hit[pos].mean() if pos.any() else np.nan, "LROC": lroc(score, pos, hit)})
    return pd.DataFrame(out)


def main():
    path, which = sys.argv[1], sys.argv[2]
    b = boxes(which)
    if len(sys.argv) > 3:
        b = b[b.split == sys.argv[3]]
    rows = [json.loads(l) for l in open(path)]
    types = [t for t in TYPES if t in set(b.finding)]
    res = evaluate(rows, b, types)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False, float_format="%.3f"))
    print(res.groupby(["K", "rule"])[["AUROC", "hit", "LROC"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
