"""Disease-course reader data (autoPET IV Longitudinal-CT): baseline / follow-up strips at the world model's top-K
addresses, labelled from the lesion masks.

Labels: 1 inside a new or grown lesion >= 10 mm (grown by 3 mm); 0 elsewhere, including stable or shrinking lesions
and smaller targets (which are skipped in training, as for the liver).
Training pool: pairs of training patients (the world-model dev patients included: they were only held out of the
world model); "wm" rows at the top-K addresses, plus "gt" rows: four strips per target (centre jittered by up to 3 mm)
and one per non-target lesion >= 10 mm (a hard negative). Evaluation: validation and test pairs, "wm" rows only.

    python make_course_data.py pool --proposal learned
    python make_course_data.py eval --proposal learned [--k 30]
"""
import argparse
import json
import os
import zlib
from multiprocessing import Pool

import numpy as np

from perr.autopet import load_pair, pairs
from perr.context_wm import peaks_from_error
from perr.course_candidates import QUESTION, course_strips
from perr.course_wm import H, ISO, load_iso, norm, to_original

OUT = "/raid/home/CAMCA/hj880/perr/data/course_reader"
MIN_DIAM = 10.0
K, PROPOSAL, TAG, SIGMA, SIGN, GT = 10, "learned", "cnll", 2.0, "both", False


def proposal_top(key, k):
    FU, BL, D, meta = load_iso(key)
    if PROPOSAL == "learned":
        R = np.load(f"{ISO}/{key}.{TAG}R.npy").astype(np.float32)
        S = np.load(f"{ISO}/{key}.{TAG}S.npy")
        sign_lists = {"pos": [1], "neg": [-1], "both": [1, -1]}[SIGN]
        pk = [p for sg in sign_lists for p in peaks_from_error(sg * R, D, k=k, sigma=SIGMA, nms_mm=10.0 / H, S=S, weight="pool")]
    else:                                                      # no change: follow-up minus registered baseline
        nc = norm(FU) - norm(BL)
        sign_lists = {"pos": [1], "neg": [-1], "both": [1, -1]}[SIGN]
        pk = [p for sg in sign_lists for p in peaks_from_error(sg * nc, D, k=k, sigma=SIGMA, nms_mm=10.0 / H)]
    pk = sorted(pk, key=lambda p: -p[3])
    out, sp = [], H                                            # merge the two signs: 10 mm suppression on the iso grid
    for p in pk:
        if all(np.linalg.norm((np.array(p[:3]) - np.array(q[:3])) * sp) > 10.0 for q in out):
            out.append(p)
    return [(*to_original(p[:3], meta), p[3]) for p in out[:k]]


def gt_points(P, rng):
    """Four jittered strips per target >= MIN_DIAM; one per other lesion >= MIN_DIAM on this grid (hard negatives)."""
    t, les = P.targets, P.lesions
    pts = []
    for r in t[t.diam_mm >= MIN_DIAM].itertuples():
        for _ in range(4):
            j = rng.uniform(-1, 1, 3) * 3.0 / np.array(P.spacing)
            pts.append((r.fu_z + j[0], r.fu_y + j[1], r.fu_x + j[2], 0.0))
    other = les[(~les.target.astype(bool)) & les.fu_in_grid.astype(bool) & (les.diam_mm >= MIN_DIAM) & les.fu_z.notna()]
    pts += [(r.fu_z, r.fu_y, r.fu_x, 0.0) for r in other.itertuples()]
    return pts


def work(key):
    try:
        P = load_pair(key)
        rows = [dict(r, source="wm") for r in course_strips(P, proposal_top(key, K), MIN_DIAM)]
        if GT:
            rows += [dict(r, source="gt") for r in course_strips(P, gt_points(P, np.random.default_rng(zlib.crc32(key.encode()))), MIN_DIAM)]
        out = []
        t = P.targets
        small = set(int(v) for v in t[t.diam_mm < MIN_DIAM].fu_label) if len(t) else set()
        for r in rows:
            i = tuple(int(round(c)) for c in r["zyx"])
            inb = all(0 <= i[d] < P.shape[d] for d in range(3))
            on_small = inb and int(P.fu_lab[i]) in small
            if GT and on_small and r["label"] != "target":
                continue                                       # smaller targets: skipped in training
            path = f"{OUT}/img/{key}_{r['source']}{r['peak']}.png"
            r.pop("image").save(path)
            out.append({**r, "uid": key, "image": path, "small": bool(on_small), "label": int(r["label"] == "target"),
                        "lesion": r["label"] == "lesion", "question": QUESTION})
        return out
    except Exception as e:
        return [{"uid": key, "error": repr(e)[:300]}]


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["pool", "eval"])
    ap.add_argument("--proposal", choices=["learned", "nochange"], default="learned")
    ap.add_argument("--tag", default="cnll")
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--sign", choices=["pos", "neg", "both"], default="both")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    K, PROPOSAL, TAG, SIGMA, SIGN, OUT = a.k, a.proposal, a.tag, a.sigma, a.sign, os.path.abspath(a.out)
    GT = a.part == "pool"
    os.makedirs(f"{OUT}/img", exist_ok=True)
    keys = pairs("train") if a.part == "pool" else pairs("val") + pairs("test")
    keys = [k for k in keys if not np.load(f"{ISO}/{k}.meta.npy", allow_pickle=True).item().get("empty")]
    with Pool(16) as p:
        res = p.map(work, keys, chunksize=1)
    rows = [r for rs in res for r in rs if "error" not in r]
    errs = [r for rs in res for r in rs if "error" in r]
    if a.part == "eval":
        write(f"{OUT}/eval{'' if K == 10 else f'_k{K}'}.jsonl", rows)
        parts = {"eval": rows}
    else:
        held = set(np.random.default_rng(0).choice(sorted({r["uid"].rsplit("_", 1)[0] for r in rows}), size=24, replace=False))
        parts = {"train": [r for r in rows if r["uid"].rsplit("_", 1)[0] not in held],
                 "val": [r for r in rows if r["uid"].rsplit("_", 1)[0] in held]}
        for name, rs in parts.items():
            write(f"{OUT}/{name}.jsonl", rs)
    for name, rs in parts.items():
        for src in ("wm", "gt"):
            sub = [r for r in rs if r["source"] == src]
            if sub:
                print(f"{name} {src}: {len(sub)} rows, {sum(r['label'] for r in sub)} positive, "
                      f"{sum(r['lesion'] for r in sub)} on other lesions, {len({r['uid'] for r in sub})} pairs", flush=True)
    print(f"pairs {len(keys)}, errors {len(errs)} {errs[:3]}", flush=True)
