"""Disease-course world: recall of candidate addresses for new or growing lesions (autoPET IV Longitudinal-CT).

A pair's targets are its new lesions and the lesions that grew by at least GROWTH in volume (default x1.728, i.e. +20%
in diameter as in RECIST; GROWTH=1.2 gives autoPET's own definition), with an equivalent diameter >= MIN_DIAM mm. A candidate hits when it lies within 3 mm of a target's voxels.
World models compared:
  no-change (mechanical): the follow-up minus the registered baseline, smoothed; read on both signs;
  learned: the disease-course model's error R, read raw or precision-weighted ("pool"), on the positive sign,
  the negative sign, or both.

    python course_recall.py <tag> runs_wm/course_nll/split.json        # world-model dev pairs
    python course_recall.py <tag> sota/eval_course.txt                  # evaluation pairs
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy import ndimage as ndi

from perr.autopet import load_pair
from perr.context_wm import peaks_from_error
from perr.course_candidates import selected_targets
from perr.course_wm import H, ISO, load_iso, norm, to_original

MIN_DIAM = 10.0
TOL_MM = 3.0
GROWTH = float(os.environ.get("GROWTH", "1.728"))    # RECIST-like: +20% diameter = x1.728 volume; autoPET's own: 1.2
TAG = "cnll"
KS = (1, 5, 10, 20, 30)


def selected(t):
    return selected_targets(t, MIN_DIAM, GROWTH)


def target_map(key):
    """Selected targets grown by TOL_MM, on the native follow-up grid (cached)."""
    path = f"{ISO}/{key}.targets{int(MIN_DIAM)}_g{GROWTH:g}.npy"
    if os.path.exists(path):
        return np.load(path, mmap_mode="r")
    P = load_pair(key)
    t = selected(P.targets)
    labs = sorted(int(v) for v in t.fu_label) if len(t) else []
    m = np.isin(np.asarray(P.fu_lab), labs) if labs else np.zeros(P.shape, bool)
    if m.any():
        m = ndi.distance_transform_edt(~m, sampling=P.spacing) <= TOL_MM
    np.save(path, m.astype(np.uint8))
    return np.load(path, mmap_mode="r")


def has_target(key):
    return bool(len(selected(load_pair(key).targets)))


def first_hit(T, pk):
    for k, (z, y, x, *_) in enumerate(pk):
        i = tuple(int(round(v)) for v in (z, y, x))
        if all(0 <= i[d] < T.shape[d] for d in range(3)) and T[i]:
            return k + 1
    return None


def prior_findings(key):
    """Addresses of the baseline's annotated lesions, registered onto the follow-up grid (the patient's history)."""
    P = load_pair(key)
    lab = np.asarray(P.bl_lab)
    ids = [i for i in np.unique(lab) if i]
    if not ids:
        return []
    cs = ndi.center_of_mass(np.ones(lab.shape, np.uint8), lab, ids)
    return [(*c, 1.0) for c in cs]


def work(key):
    T = target_map(key)
    FU, BL, D, meta = load_iso(key)
    res = {"key": key, "n_prior": len(prior_findings(key))}
    res["prior_only"] = first_hit(T, prior_findings(key))
    nc = norm(FU) - norm(BL)                                     # the no-change world model's error
    for sign, name in ((1, "pos"), (-1, "neg")):
        pk = peaks_from_error(sign * nc, D, k=50, sigma=2.0, nms_mm=10.0 / H)
        res[f"nochange_{name}"] = first_hit(T, [(*to_original(p[:3], meta), p[3]) for p in pk])
    if os.path.exists(f"{ISO}/{key}.{TAG}R.npy"):
        R = np.load(f"{ISO}/{key}.{TAG}R.npy").astype(np.float32)
        S = np.load(f"{ISO}/{key}.{TAG}S.npy")
        for sg in (1.0, 2.0, 3.0):
            for w in ("none", "pool"):
                lists = {}
                for sign, name in ((1, "pos"), (-1, "neg")):
                    lists[name] = [(*to_original(p[:3], meta), p[3])
                                   for p in peaks_from_error(sign * R, D, k=50, sigma=sg, nms_mm=10.0 / H, S=S, weight=w)]
                    res[f"learned{sg:g}{w}_{name}"] = first_hit(T, lists[name])
                both = sorted(lists["pos"] + lists["neg"], key=lambda p: -p[3])
                res[f"learned{sg:g}{w}_both"] = first_hit(T, both)
                # history first (every prior finding is re-read), then the world model's surprises
                res[f"prior+learned{sg:g}{w}_pos"] = first_hit(T, prior_findings(key) + lists["pos"])
    return res


if __name__ == "__main__":
    TAG, src = sys.argv[1], sys.argv[2]
    keys = json.load(open(src))["dev"] if src.endswith(".json") else open(src).read().split()
    keys = [k for k in keys if not np.load(f"{ISO}/{k}.meta.npy", allow_pickle=True).item().get("empty") and has_target(k)]
    with Pool(24) as p:
        res = p.map(work, keys, chunksize=1)
    out = os.path.basename(src).split(".")[0]
    json.dump(res, open(f"results/course_recall_{TAG}_{out}_g{GROWTH:g}.json", "w"))
    print(f"{len(res)} pairs with a target >= {MIN_DIAM:g} mm (new, or volume x{GROWTH:g}); prior findings per pair: "
          f"median {np.median([r['n_prior'] for r in res]):.0f}, max {max(r['n_prior'] for r in res)}")
    for k in res[0]:
        if k in ("key", "n_prior"):
            continue
        f = [r.get(k) for r in res]
        print(f"{k:24s}", " ".join(f"K={K}:{np.mean([x is not None and x <= K for x in f]):.2f}" for K in KS))
