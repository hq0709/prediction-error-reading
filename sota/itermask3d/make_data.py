"""Verifier data for IterMask3D's addresses on Set 1 and Set 2, exactly as for the PatchCore ablation
(`make_reader_data.py --eval LO HI --proposal patchcore --out data/abl_free_pc`): the top K = 10 addresses of each scan
(2 mm smoothing, peaks inside the grown lung domain, 10 mm suppression; itermask3d_lung.map_peaks) rendered as candidate
strips by make_reader_data.work through its _Saved proposal. make_reader_data.py is imported, not edited: the saved
peaks are registered as one more proposal and its --eval main loop is repeated here.

    python sota/itermask3d/make_data.py --infer eval --gamma 0.01
Writes results/itermask3d_lung_peaks.json (rows uid, peaks, as ct_patchcore.py's) and
data/abl_free_imk3d/eval_60_160.jsonl, data/abl_free_imk3d/eval_160_260.jsonl.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

ROOT = "/raid/home/CAMCA/hj880/perr"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import make_reader_data as mrd  # noqa: E402
from perr.study import pilot_cases  # noqa: E402

RUNS = f"{ROOT}/sota/itermask3d/runs"
PEAKS = f"{ROOT}/results/itermask3d_lung_peaks.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer", default="eval", help="inference folder under sota/itermask3d/runs")
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--out", default="data/abl_free_imk3d")
    a = ap.parse_args()
    sets = [(lo, hi, pilot_cases(hi)[lo:hi]) for lo, hi in ((60, 160), (160, 260))]
    assert sets[0][2] == open("sota/eval_luna_fresh1.txt").read().split()
    assert sets[1][2] == open("sota/eval_luna_fresh2.txt").read().split()
    rows = []
    for _, _, uids in sets:
        for u in uids:
            r = json.load(open(f"{RUNS}/{a.infer}/{u}.g{a.gamma:g}.json"))
            rows.append({"uid": u, "peaks": r["peaks"]})
    json.dump(rows, open(PEAKS, "w"))
    print(f"{len(rows)} scans' peaks in {PEAKS}", flush=True)
    mrd.PROPOSALS["itermask3d"] = lambda: mrd._Saved(PEAKS)
    mrd.K, mrd.OUT, mrd.PROPOSAL, mrd.GT = 10, os.path.abspath(a.out), "itermask3d", False
    os.makedirs(f"{mrd.OUT}/img", exist_ok=True)
    for lo, hi, uids in sets:
        with Pool(24) as p:
            results = p.map(mrd.work, uids, chunksize=1)
        rs = [r for rr in results for r in rr if "error" not in r]
        errs = [r for rr in results for r in rr if "error" in r]
        mrd.write(f"{mrd.OUT}/eval_{lo}_{hi}.jsonl", rs)
        print(f"eval {lo}-{hi}: {len(rs)} rows on {len({r['uid'] for r in rs})} scans, errors {len(errs)} {errs[:3]}",
              flush=True)


if __name__ == "__main__":
    main()
