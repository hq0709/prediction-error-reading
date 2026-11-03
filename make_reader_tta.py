"""Test-time views for the lung CT verifier: each candidate strip of a reader data file re-rendered (same harness measure
tool, perr.candidates.strips) with its centre moved by a small fixed offset in millimetres. The candidate keeps its
address, label and world-model value; only the picture the verifier reads changes.

Offsets: the four vertices of a regular tetrahedron at 1 mm from the address (t0..t3; unit directions
(1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1) / sqrt(3) in (z, y, x) mm), so their mean is the address itself.

    python make_reader_tta.py data/reader_ctxfree_cvA/heldout.jsonl t0 --uids results/pool_scans_with_big.json
    -> data/reader_tta/<stem of the data dir>_t0/heldout.jsonl  (images under .../img/)
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from perr.candidates import strips
from perr.study import load_luna

DIRS = {f"t{i}": np.array(d, float) / np.sqrt(3) for i, d in
        enumerate([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)])}
DIRS["z"] = np.zeros(3)          # no offset: reproduces the stored strips (check)
MM = 1.0
OUT = None


def work(item):
    uid, rows, off = item
    try:
        s = load_luna(uid)
        d = DIRS[off] * MM / np.array(s.spacing)
        pts = [(*(np.array(r["zyx"], float) + d), r["wm"]) for r in rows]
        res = {x["peak"]: x for x in strips(s, pts)}
        out = []
        for i, r in enumerate(rows):
            x = res[i + 1]
            path = f"{OUT}/img/{uid}_{r['source']}{r['peak']}.png"
            x["image"].save(path)
            out.append({**r, "image": path, "tta": off, "tta_measurement": x["measurement"]})
        return out
    except Exception as e:
        return [{"uid": uid, "error": repr(e)[:200]}]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("offset", choices=sorted(DIRS))
    ap.add_argument("--uids", default=None, help="JSON list: only these scans")
    ap.add_argument("--mm", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    MM = a.mm
    rows = [json.loads(l) for l in open(a.data)]
    if a.uids:
        keep = set(json.load(open(a.uids)))
        rows = [r for r in rows if r["uid"] in keep]
    tag = os.path.basename(os.path.dirname(os.path.abspath(a.data)))
    OUT = os.path.abspath(f"data/reader_tta/{tag}_{a.offset}" + ("" if a.mm == 1.0 else f"_{a.mm:g}mm"))
    os.makedirs(f"{OUT}/img", exist_ok=True)
    by = {}
    for r in rows:
        by.setdefault(r["uid"], []).append(r)
    with Pool(a.workers) as p:
        res = p.map(work, [(u, rs, a.offset) for u, rs in by.items()], chunksize=1)
    good = [r for rs in res for r in rs if "error" not in r]
    errs = [r for rs in res for r in rs if "error" in r]
    stem = os.path.basename(a.data)
    with open(f"{OUT}/{stem}", "w") as f:
        for r in good:
            f.write(json.dumps(r) + "\n")
    print(f"{OUT}/{stem}: {len(good)} rows of {len(rows)}, errors {len(errs)} {errs[:2]}", flush=True)
