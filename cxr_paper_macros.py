"""Radiograph main-text values that depend on the kept full-data verifier (choose_cxr_aug.py), from its scored candidate
sets on ChestX-Det test; max-p, mean over the four focal findings. Prints one line per numbers.tex macro.

    python cxr_paper_macros.py [run]          # default: the kept PER run
"""
import json
import os
import sys

from choose_cxr_aug import chosen
from cxr_recall import boxes
from score_cxr_focal import TYPES, evaluate

RUN = sys.argv[1] if len(sys.argv) > 1 else chosen("PER (ours)")
R = f"results/cxr_focal/{RUN}"
# macro: (candidate set, K, metric); K is the reading budget of each list
MACROS = {
    "CxrPerAuc": ("wmu", 60, "AUROC"), "CxrPerHit": ("wmu", 60, "hit"), "CxrPerLroc": ("wmu", 60, "LROC"),
    "AblCxrNoWm": ("random45s", 45, "LROC"),          # 45 random thorax points (per-image crc32 seeds)
    "AblCxrNoPrec": ("wmuraw", 60, "LROC"),           # raw one-sided error from both world models
    "AblCxrNoDomain": ("wmur", 60, "LROC"),           # no thorax search mask
    "AblCxrOneGrid": ("wm", 30, "LROC"),              # 512-pixel world model only
    "AblCxrTwoSided": ("hp_two", 60, "LROC"),         # two-sided error
    "SrcCxrPatchCore": ("patchcore45", 45, "LROC"),   # 45 PatchCore candidates
    "SrcCxrLattice": ("grid", 1000, "LROC"),          # every 25 px lattice point in the thorax mask
}


def main():
    b = boxes("cxdet")
    b = b[b.split == "test"]
    cache = {}
    print(f"% run {RUN}")
    for name, (stem, k, metric) in MACROS.items():
        path = f"{R}/{stem}_cxdet_test.jsonl"
        if not os.path.exists(path):
            print(f"{name}: missing {path}")
            continue
        if (stem, k) not in cache:
            rows = [json.loads(line) for line in open(path)]
            cache[stem, k] = evaluate(rows, b, TYPES, ks=(k,), rules=("max-p",))
        print(f"{name}: {cache[stem, k][metric].mean():.3f}   ({stem}, K={k})")


if __name__ == "__main__":
    main()
