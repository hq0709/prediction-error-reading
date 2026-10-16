"""Build the autoPET IV Longitudinal-CT disease-course cache (resumable), its summary JSON and QC sheets.

usage: python build_autopet.py [--workers 48] [--threads 4] [--pids pid ...] [--summary-only] [--force]

Each patient is one job: every BL piece is registered to every FU piece, then each FU piece is written as a pair under
cache/autopet/ (<pair>.meta.npy last, so an interrupted run resumes at the first unfinished patient). Licence CC BY-NC
4.0: the cache and everything derived from it stay local and non-commercial.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from multiprocessing import get_context

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from perr import autopet as ap  # noqa: E402

SUMMARY = f"{HERE}/results/autopet_summary.json"
QC_DIR = f"{HERE}/results/autopet_qc"
LOG = f"{HERE}/logs/autopet_build.log"


def _work(job):
    pid, threads, force = job
    t = time.time()
    try:
        if not force and ap._is_done(pid):
            return pid, "cached", 0.0, None
        ap.build_patient(pid, threads=threads, force=force)
        return pid, "built", time.time() - t, None
    except Exception:
        return pid, "error", time.time() - t, traceback.format_exc()


def _voxels(pid: str) -> int:
    import SimpleITK as sitk
    n = 0
    for side in ("BL", "FU"):
        for k in ap.pieces(pid, side):
            r = sitk.ImageFileReader()
            r.SetFileName(ap._paths(pid, side, k)[0])
            r.ReadImageInformation()
            n += int(np.prod(r.GetSize())) * (len(ap.pieces(pid, "FU")) if side == "BL" else 1)
    return n


def build(pids, workers, threads, force):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    todo = [p for p in pids if force or not ap._is_done(p)]
    print(f"{len(pids)} patients, {len(pids) - len(todo)} cached, {len(todo)} to build", flush=True)
    if not todo:
        return []
    todo.sort(key=_voxels, reverse=True)          # largest first, so no straggler starts last
    errors = []
    t0 = time.time()
    with get_context("spawn").Pool(workers, maxtasksperchild=1) as pool, open(LOG, "a") as log:
        for i, (pid, status, dt, err) in enumerate(pool.imap_unordered(_work, [(p, threads, force) for p in todo]), 1):
            line = (f"{time.strftime('%H:%M:%S')} [{i}/{len(todo)}] {pid} {status} {dt:.0f}s "
                    f"(elapsed {time.time() - t0:.0f}s)")
            print(line, flush=True)
            log.write(line + "\n")
            if err:
                errors.append({"pid": pid, "error": err})
                log.write(err + "\n")
            log.flush()
    return errors


def _pct(x, q):
    x = [v for v in x if np.isfinite(v)]
    return float(np.percentile(x, q)) if x else float("nan")


def summarize(errors=()):
    metas = {}
    for k in ap.pairs():
        path = f"{ap.CACHE}/{k}.meta.npy"
        if os.path.exists(path):
            metas[k] = np.load(path, allow_pickle=True).item()
    D = pd.concat([ap.csv(p).assign(pid=p, split=ap.split_of(p)) for p in ap.patients()], ignore_index=True)
    L = pd.concat([pd.DataFrame(m["lesions"]).assign(pair=k, split=m["split"]) for k, m in metas.items()],
                  ignore_index=True)
    own = L[L.owned]
    qa = own[np.isfinite(own.qa_mm) & ~own.linking_unclear]
    T = own[own.target]

    def split_block(s):
        ks = [k for k, m in metas.items() if m["split"] == s]
        t = T[T.split == s]
        cov = [metas[k]["bl_body_cover"] for k in ks]
        return {
            "patients": len(ap.patients(s)), "pairs": len(ks),
            "pairs_with_target": int(sum(1 for k in ks if metas[k]["targets"])),
            "targets": int(len(t)), "targets_new": int((t.topology_class == "NEWLYAPPEARING").sum()),
            "targets_growing": int((t.topology_class != "NEWLYAPPEARING").sum()),
            "target_diam_mm_median": _pct(t.diam_mm, 50),
            "target_diam_mm_p10_p90": [_pct(t.diam_mm, 10), _pct(t.diam_mm, 90)],
            "targets_ge_10mm": int((t.diam_mm >= 10).sum()),
            "pairs_with_target_ge_10mm": int(t[t.diam_mm >= 10].pair.nunique()),
            "targets_bl_cover_lt_0.5": int((t.bl_cover < 0.5).sum()),
            "targets_ge_10mm_bl_cover_ge_0.5": int(((t.diam_mm >= 10) & (t.bl_cover >= 0.5)).sum()),
            "bl_body_cover_median": _pct(cov, 50),
        }

    per_pair_qa = {k: m["qa"] for k, m in metas.items()}
    flagged = {k: v for k, v in per_pair_qa.items() if v["flag"]}
    low_cov = {k: round(m["bl_body_cover"], 3) for k, m in metas.items() if m["bl_body_cover"] < 0.8}
    stage = {s: _pct(qa[f"pt_{s}_mm"], 50) for s in ("init", "rigid", "affine", "final") if f"pt_{s}_mm" in qa}
    stage90 = {s: _pct(qa[f"pt_{s}_mm"], 90) for s in ("init", "rigid", "affine", "final") if f"pt_{s}_mm" in qa}
    kept = pd.Series([r["kept"] for m in metas.values() for q, r in m["registrations"].items()
                      if q in m["bl_order"]]).value_counts().to_dict()
    try:
        size = int(subprocess.check_output(["du", "-sb", ap.CACHE]).split()[0])
    except Exception:
        size = -1
    topo = D.topology_class.value_counts().to_dict()
    multi = [p for p in ap.patients() if len(ap.pieces(p, "FU")) > 1 or len(ap.pieces(p, "BL")) > 1]
    out = {
        "dataset": {"root": ap.ROOT, "licence": "CC BY-NC 4.0 (non-commercial; never redistribute derived data)",
                    "patients": {s: len(ap.patients(s)) for s in ("train", "val", "test")}, "csv_lesions": len(D)},
        "conventions": {
            "array": "[z, y, x] after SimpleITK.DICOMOrient(img, 'LPS'): x -> patient left, y -> posterior, "
                     "z -> superior (all source images already have identity LPS direction)",
            "cog": "cog_bl / cog_fu / cog_propagated / cog_backpropagated are continuous voxel coordinates 'x y z' in "
                   "ITK = NIfTI (i, j, k) order on the piece img_id_bl / img_id_fu, voxel centres at +0.5: "
                   "cog = centroid(mask == label) + 0.5 per axis (verified on 4079 BL and 3231 FU entries; a handful "
                   "differ by < 0.4 voxel where the mask was edited, one by 4.3 voxels). Not millimetres.",
            "volume": "volume_bl / volume_fu are mm^3 (= voxel count x voxel volume); cache tables use ml",
            "mask": "mask value = lesion_id; a MERGING group appears in the FU mask once, under merged_into",
            "topology_class": {"counts": topo, "counts_by_split": D.groupby("split").topology_class.value_counts()
                               .unstack(fill_value=0).to_dict(orient="index"),
                               "meaning": {"UNCHANGED": "one-to-one BL-FU match (size may change)",
                                           "NEWLYAPPEARING": "FU only",
                                           "DISAPPEARING": "BL only (volume_fu 0)",
                                           "MERGING": "several BL lesions -> one FU lesion (merged_into = FU label)"},
                               "no_split_class": True},
            "linking_unclear": {"True": int((D.linking_unclear == True).sum()),  # noqa: E712
                                "False": int((D.linking_unclear == False).sum()),  # noqa: E712
                                "missing (2 CSVs lack the column; treated as not unclear)":
                                    int(D.linking_unclear.isna().sum())},
            "target_rule": "owned FU lesion, not linking_unclear (NaN = not unclear), NEWLYAPPEARING or "
                           f"volume_fu / volume_bl >= {ap.GROWTH} (MERGING: against the summed BL volume of the group)",
        },
        "pieces": {
            "rule": "each FU piece is its own pair <pid>_<NN> on its native grid; owned lesions = img_id_fu == NN "
                    "(CSV-exact); lesions of an overlapping same-frame sibling FU piece are projected into fu_lab as "
                    "owned = False; the BL side composites all BL pieces, each registered separately, ordered by "
                    "similarity to the FU, later pieces only filling empty voxels; a secondary piece is used if its "
                    "mapping agrees with the anchor piece (<= 15 mm) or its NCC >= 0.7",
            "patients_multi_piece": len(multi),
            "patients_multi_fu": int(sum(len(ap.pieces(p, "FU")) > 1 for p in ap.patients())),
            "patients_multi_bl": int(sum(len(ap.pieces(p, "BL")) > 1 for p in ap.patients())),
            "pairs": len(metas),
            "projected_sibling_pieces": int(sum(len(m["projected"]) for m in metas.values())),
            "projection_skipped_frame_mismatch": {k: m["projection_skipped"] for k, m in metas.items()
                                                  if m["projection_skipped"]},
            "bl_pieces_dropped": {k: sorted(set(m["registrations"]) - set(m["bl_order"])) for k, m in metas.items()
                                  if set(m["registrations"]) - set(m["bl_order"])},
        },
        "registration": {
            "method": "per BL piece -> FU piece, images only: 4 mm canvases + body masks; global translation by masked "
                      "NCC over all 3-D shifts at 8 mm (+ main-pair affine as a candidate for secondary pieces); "
                      "rigid (Mattes MI) per candidate, best NCC kept; rigid -> affine (8/4 mm); cubic B-spline "
                      f"(control mesh {ap.MESH_MM:.0f} mm, coefficients bounded to +-{ap.BOUND_MM:.0f} mm, 8/4 mm "
                      "levels, LBFGS-B, mean squared difference of HU clipped to [-1000, 1000]) with body masks; "
                      "rigid/affine use Mattes MI; "
                      "rigid kept if the affine lowers canvas NCC, linear kept if the B-spline lowers it or folds; "
                      "native resampling via a 2 mm displacement field (linear HU, nearest labels)",
            "qa_definition": "owned UNCHANGED lesions, not linking_unclear, warped BL label present and not cut by "
                             "the FU grid boundary: distance (mm) between the centroid of the warped BL label and "
                             "the FU lesion centroid on the FU grid (includes real shifts of changing lesions)",
            "qa_lesions": int(len(qa)), "qa_median_mm": _pct(qa.qa_mm, 50), "qa_p90_mm": _pct(qa.qa_mm, 90),
            "qa_by_split": {s: {"n": int((qa.split == s).sum()), "median_mm": _pct(qa[qa.split == s].qa_mm, 50),
                                "p90_mm": _pct(qa[qa.split == s].qa_mm, 90)} for s in ("train", "val", "test")},
            "qa_by_lesion_type": {t: {"n": int(len(g)), "median_mm": _pct(g.qa_mm, 50), "p90_mm": _pct(g.qa_mm, 90)}
                                  for t, g in qa.groupby("lesion_type")},
            "point_error_by_stage_median_mm": stage, "point_error_by_stage_p90_mm": stage90,
            "organiser_propagated_point_error_same_lesions": {"median_mm": _pct(qa.prop_mm, 50),
                                                              "p90_mm": _pct(qa.prop_mm, 90),
                                                              "n": int(np.isfinite(qa.prop_mm).sum())},
            "final_stage_kept": kept,
            "flag_rule_mm": ap.QA_FLAG_MM, "flagged_pairs": flagged,
            "pairs_without_qa_lesions": int(sum(1 for v in per_pair_qa.values() if v["n"] == 0)),
        },
        "fov": {"bl_body_cover_median": _pct([m["bl_body_cover"] for m in metas.values()], 50),
                "pairs_bl_body_cover_lt_0.8": low_cov},
        "splits": {s: split_block(s) for s in ("train", "val", "test")},
        "cache": {"dir": ap.CACHE, "bytes": size, "gb": round(size / 1e9, 1)},
        "missing_pairs": sorted(set(ap.pairs()) - set(metas)),
        "errors": list(errors),
    }
    os.makedirs(os.path.dirname(SUMMARY), exist_ok=True)
    with open(SUMMARY, "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: out[k] for k in ("splits", "cache")}, indent=1, default=str))
    print("QA", {k: out["registration"][k] for k in ("qa_lesions", "qa_median_mm", "qa_p90_mm")},
          "flagged", len(flagged))
    return out


def qc_sheets(keys_lesions, out_dir=QC_DIR):
    """Axial slice through a target: FU | registered BL | FU - BL, lesion outlines (FU green, warped BL magenta)."""
    from PIL import Image, ImageDraw
    from scipy import ndimage as ndi
    os.makedirs(out_dir, exist_ok=True)

    def win(a, lo=-160, hi=240):
        return (np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)

    paths = []
    for key, lesion in keys_lesions:
        P = ap.load_pair(key)
        L = P.lesions
        t = L[(L.fu_label == lesion) & L.owned].sort_values("target", ascending=False).iloc[0]
        z, y, x = (int(round(t[c])) for c in ("fu_z", "fu_y", "fu_x"))
        F, B = np.asarray(P.fu[z]), np.asarray(P.bl[z])
        V, BD = np.asarray(P.bl_valid[z]) > 0, np.asarray(P.body[z]) > 0
        bl_ids = [int(v) for v in L[(L.fu_label == lesion) & (L.bl_label > 0)].bl_label]   # MERGING: all members
        FL, BLb = np.asarray(P.fu_lab[z]) == lesion, np.isin(np.asarray(P.bl_lab[z]), bl_ids)

        def edge(m):
            return m & ~ndi.binary_erosion(m)
        f = np.stack([win(F)] * 3, -1)
        b = np.stack([win(B)] * 3, -1)
        b[~V] = (70, 0, 0)
        d = np.clip(np.where(V, F.astype(np.float32) - B, 0) / 300.0, -1, 1)
        dd = np.zeros(F.shape + (3,), np.uint8)
        dd[..., 0] = (np.clip(d, 0, 1) * 255).astype(np.uint8)
        dd[..., 2] = (np.clip(-d, 0, 1) * 255).astype(np.uint8)
        for im in (f, b, dd):
            im[edge(BD)] = (255, 200, 0)
            im[edge(FL)] = (0, 255, 0)
        if bl_ids:
            b[edge(BLb)] = (255, 0, 255)
        sy, sx = P.spacing[1:]
        tiles = []
        for im, lab in ((f, "FU"), (b, "BL registered"), (dd, "FU - BL (red +, blue -)")):
            pim = Image.fromarray(im).resize((int(im.shape[1] * sx / min(sx, sy)), int(im.shape[0] * sy / min(sx, sy))),
                                             Image.NEAREST)
            dr = ImageDraw.Draw(pim)
            dr.rectangle([0, 0, 7 * len(lab) + 6, 13], fill=(0, 0, 0))
            dr.text((3, 1), lab, fill=(255, 255, 0))
            tiles.append(pim)
        W = sum(p.width for p in tiles) + 8
        S = Image.new("RGB", (W, tiles[0].height + 18), (30, 30, 30))
        xx = 0
        for p in tiles:
            S.paste(p, (xx, 0))
            xx += p.width + 4
        ImageDraw.Draw(S).text(
            (4, tiles[0].height + 3),
            f"{key} z={z} lesion {lesion} {t.topology_class} {t.lesion_type} d={t.diam_mm:.1f} mm "
            f"ratio={t.ratio:.2f}  pair QA median {P.meta['qa']['median_mm']:.1f} mm  "
            f"outlines: body (orange), FU lesion (green), warped BL lesion (magenta)", fill=(255, 255, 255))
        path = f"{out_dir}/{key}_lesion{lesion}_z{z}.png"
        S.save(path)
        paths.append(path)
    return paths


def pick_qc(n=3):
    """Deterministic QC choice: a large new lesion, a large growing lesion, a lesion in a multi-piece pair; all with
    good BL coverage, pair QA below the flag."""
    rows = []
    for k in ap.pairs():
        path = f"{ap.CACHE}/{k}.meta.npy"
        if not os.path.exists(path):
            continue
        m = np.load(path, allow_pickle=True).item()
        for t in m["targets"]:
            if t["diam_mm"] >= 15 and t["bl_cover"] >= 0.99 and not m["qa"]["flag"]:
                rows.append((k, t["fu_label"], t["topology_class"], t["diam_mm"], len(ap.pieces(m["pid"], "FU")) > 1))
    R = pd.DataFrame(rows, columns=["pair", "label", "topo", "diam", "multi"])
    picks = []
    for sel in (R[(R.topo == "NEWLYAPPEARING") & ~R.multi], R[(R.topo != "NEWLYAPPEARING") & ~R.multi], R[R.multi]):
        sel = sel[~sel.pair.isin([p for p, _ in picks])].sort_values(["diam", "pair"])
        if len(sel):
            r = sel.iloc[len(sel) // 2]           # median-sized candidate, not the most dramatic
            picks.append((r.pair, int(r.label)))
    return picks[:n]


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--workers", type=int, default=48)
    a.add_argument("--threads", type=int, default=4)
    a.add_argument("--pids", nargs="*")
    a.add_argument("--summary-only", action="store_true")
    a.add_argument("--force", action="store_true")
    args = a.parse_args()
    errors = []
    if not args.summary_only:
        errors = build(args.pids or ap.patients(), args.workers, args.threads, args.force)
    summarize(errors)
    picks = pick_qc()
    print("QC:", qc_sheets(picks))


if __name__ == "__main__":
    main()
