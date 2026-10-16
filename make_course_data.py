"""Disease-course reader data (autoPET IV Longitudinal-CT): baseline / follow-up strips at the world model's top-K
addresses, labelled from the lesion masks.

Labels: 1 inside a selected target (new, or grown to x1.728 volume, >= 10 mm; grown by 3 mm); 0 elsewhere, including
stable or shrinking lesions and targets below the size or growth bar (those are skipped in training, as for the liver).
Candidates: every prior finding (the baseline's annotated lesions registered onto the follow-up: the patient's
history, re-read for growth) and the world model's top-K surprises ("prior" and "wm" rows; --no-history: "wm" only).
Training pool: pairs of training patients (the world-model dev patients included: they were only held out of the
world model); the candidate rows plus "gt" rows: four strips per target (centre jittered by up to 3 mm)
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
from scipy import ndimage as ndi

from perr.context_wm import error_map, peaks_from_error
from perr.course_candidates import QUESTION, QUESTION_OUTLINE, course_strips, selected_targets
from perr.course_wm import H, ISO, load_iso, norm, to_original

OUT = "/raid/home/CAMCA/hj880/perr/data/course_reader"
MIN_DIAM = 10.0
K, PROPOSAL, TAG, SIGMA, SIGN, GT, HISTORY, OUTLINE = 10, "learned", "cnll", 3.0, "pos", False, True, False


def prior_findings(P):
    """The baseline's annotated lesions, registered onto the follow-up grid: the patient's history."""
    lab = np.asarray(P.bl_lab)
    ids = [i for i in np.unique(lab) if i]
    return [tuple(c) for c in ndi.center_of_mass(np.ones(lab.shape, np.uint8), lab, ids)] if ids else []


def with_history(P, key, k):
    """Every prior finding (re-read for growth), then the world model's top surprises not within 10 mm of one. Each
    address carries the world model's precision-weighted error there, so the selection rule treats both alike."""
    FU, BL, D, meta = load_iso(key)
    R = np.load(f"{ISO}/{key}.{TAG}R.npy").astype(np.float32)
    S = np.load(f"{ISO}/{key}.{TAG}S.npy")
    W = error_map(R, S, SIGMA, True, "pool")
    to_iso = lambda zyx: tuple(int(np.clip(round(v), 0, n - 1)) for v, n in
                               zip((np.asarray(zyx) * meta["spacing"] - meta["lo_mm"]) / H, W.shape))
    prior = [(*c, float(max(W[to_iso(c)], 1e-6))) for c in prior_findings(P)]
    sp = np.array(P.spacing)
    wm = [p for p in proposal_top(key, k)
          if all(np.linalg.norm((np.array(p[:3]) - np.array(q[:3])) * sp) > 10.0 for q in prior)]
    return [dict(zyx=p, source="prior") for p in prior] + [dict(zyx=p, source="wm") for p in wm[:k]]


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
    t, les = selected_targets(P.targets, MIN_DIAM), P.lesions
    pts = []
    for r in t.itertuples():
        for _ in range(4):
            j = rng.uniform(-1, 1, 3) * 3.0 / np.array(P.spacing)
            pts.append((r.fu_z + j[0], r.fu_y + j[1], r.fu_x + j[2], 0.0))
    other = les[(~les.target.astype(bool)) & les.fu_in_grid.astype(bool) & (les.diam_mm >= MIN_DIAM) & les.fu_z.notna()]
    pts += [(r.fu_z, r.fu_y, r.fu_x, 0.0) for r in other.itertuples()]
    return pts


def work(key):
    try:
        P = load_pair(key)
        if HISTORY:
            cands = with_history(P, key, K)
            rows = [dict(r, source=c["source"]) for r, c in
                    zip(course_strips(P, [c["zyx"] for c in cands], MIN_DIAM, prior_outline=OUTLINE), cands)]
        else:
            rows = [dict(r, source="wm") for r in course_strips(P, proposal_top(key, K), MIN_DIAM, prior_outline=OUTLINE)]
        if GT:
            rows += [dict(r, source="gt") for r in course_strips(P, gt_points(P, np.random.default_rng(zlib.crc32(key.encode()))),
                                                                 MIN_DIAM, prior_outline=OUTLINE)]
        out = []
        t, sel = P.targets, selected_targets(P.targets, MIN_DIAM)
        small = set(int(v) for v in t.fu_label) - set(int(v) for v in sel.fu_label) if len(t) else set()
        for r in rows:
            i = tuple(int(round(c)) for c in r["zyx"])
            inb = all(0 <= i[d] < P.shape[d] for d in range(3))
            on_small = inb and int(P.fu_lab[i]) in small
            if GT and on_small and r["label"] != "target":
                continue                                       # targets below the size or growth bar: skipped in training
            path = f"{OUT}/img/{key}_{r['source']}{r['peak']}.png"
            r.pop("image").save(path)
            out.append({**r, "uid": key, "image": path, "small": bool(on_small), "label": int(r["label"] == "target"),
                        "lesion": r["label"] == "lesion", "question": QUESTION_OUTLINE if OUTLINE else QUESTION})
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
    ap.add_argument("--sigma", type=float, default=3.0, help="smoothing in 1.5 mm voxels (fixed on the dev pairs)")
    ap.add_argument("--sign", choices=["pos", "neg", "both"], default="pos")
    ap.add_argument("--no-history", action="store_true", help="world-model addresses only (no prior findings)")
    ap.add_argument("--outline", action="store_true", help="outline the baseline's annotated lesions on both rows")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--keys", help="evaluation pair keys (default: validation + test pairs)")
    a = ap.parse_args()
    K, PROPOSAL, TAG, SIGMA, SIGN, OUT = a.k, a.proposal, a.tag, a.sigma, a.sign, os.path.abspath(a.out)
    HISTORY, OUTLINE = not a.no_history, a.outline
    GT = a.part == "pool"
    os.makedirs(f"{OUT}/img", exist_ok=True)
    keys = pairs("train") if a.part == "pool" else (open(a.keys).read().split() if a.keys else pairs("val") + pairs("test"))
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
        for src in ("prior", "wm", "gt"):
            sub = [r for r in rs if r["source"] == src]
            if sub:
                print(f"{name} {src}: {len(sub)} rows, {sum(r['label'] for r in sub)} positive, "
                      f"{sum(r['lesion'] for r in sub)} on other lesions, {len({r['uid'] for r in sub})} pairs", flush=True)
    print(f"pairs {len(keys)}, errors {len(errs)} {errs[:3]}", flush=True)
