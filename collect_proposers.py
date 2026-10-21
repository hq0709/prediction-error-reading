"""Models of normal appearance as candidate proposers on ChestX-Det test, all read by the kept PER verifier
(choose_cxr_aug.py) with max-p: candidate recall (share of positive image-finding pairs with a candidate inside a box
of that finding among the first K) and the radiograph metrics of score_cxr_focal.evaluate. Prints one line per
numbers.tex macro; sets that are not scored yet are reported as missing.

    python collect_proposers.py [run]
"""
import collections
import json
import os
import sys

from choose_cxr_aug import chosen
from cxr_recall import boxes
from score_cxr_focal import TYPES, evaluate

RUN = sys.argv[1] if len(sys.argv) > 1 else chosen("PER (ours)")
R = f"results/cxr_focal/{RUN}"
# macro stem: (candidate set, K) -- the world models give about 45 candidates per image after suppression, so every
# single-map proposer is read at K = 45
SETS = {
    "Per": ("wmu", 60),                 # both PER world models, union, about 45 per image
    "PatchCore": ("patchcore45", 45),   # PatchCore (CVPR 2022), memory bank of normal features
    "Dinomaly": ("dinomaly45", 45),     # Dinomaly (CVPR 2025), feature reconstruction, one-class on MIMIC No Finding
    "CheXWorld": ("chexworld45", 45),   # CheXWorld (CVPR 2025), latent world model, released weights
    "RadJepa": ("radjepa45", 45),       # RadJEPA (EMNLP 2026), latent world model, released weights
    "IterMask": ("itermask245", 45),    # IterMask2 (MICCAI 2024), masked-pixel prediction, MIMIC No Finding
    "MadAd": ("madad45", 45),           # MAD-AD (IPMI 2025), masked latent diffusion, MIMIC No Finding
    "Random": ("random45s", 45),        # random thorax points
    "Lattice": ("grid", 1000),          # every 25 px lattice point in the thorax
}


def recall(rows, b, k):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["key"]].append(r)
    hit = n = 0
    for t in TYPES:
        for key in b[b.finding == t].key.unique():
            n += 1
            hit += any(r[t] for r in by.get(key, []) if r["rank"] < k)
    return hit / n


def main():
    b = boxes("cxdet")
    b = b[b.split == "test"]
    print(f"% run {RUN}")
    for stem, (name, k) in SETS.items():
        path = f"{R}/{name}_cxdet_test.jsonl"
        if not os.path.exists(path):
            print(f"% {stem}: missing {path}")
            continue
        rows = [json.loads(line) for line in open(path)]
        e = evaluate(rows, b, TYPES, ks=(k,), rules=("max-p",))
        print(f"\\newcommand{{\\Prop{stem}Recall}}{{{recall(rows, b, k):.3f}}} % collect_proposers.py -> {name}, K={k}")
        for m in ("AUROC", "hit", "LROC"):
            tag = {"AUROC": "Auc", "hit": "Hit", "LROC": "Lroc"}[m]
            print(f"\\newcommand{{\\Prop{stem}{tag}}}{{{e[m].mean():.3f}}} % collect_proposers.py -> {name}, K={k}, {RUN}")


if __name__ == "__main__":
    main()
