"""Liver world model: recall of its top-K addresses for tumours >= 10 mm (and a densest/darkest-point baseline)."""
import json
from multiprocessing import Pool

import numpy as np
from scipy import ndimage as ndi

from perr.liver import LiverPrior, cases, has_target, label_at, load_liver


def work(case):
    try:
        s = load_liver(case)
        if not has_target(s):
            return {"case": case, "target": False}
        pk = LiverPrior().top(s, 30)
        first = next((k + 1 for k, p in enumerate(pk) if label_at(case, p[:3]) == "big"), None)
        return {"case": case, "target": True, "first": first, "n": len(pk)}
    except Exception as e:
        return {"case": case, "error": repr(e)[:200]}


if __name__ == "__main__":
    with Pool(24) as p:
        res = p.map(work, cases(), chunksize=1)
    json.dump(res, open("results/liver_recall.json", "w"))
    err = [r for r in res if "error" in r]
    t = [r for r in res if r.get("target")]
    print(f"cases {len(res)}, with target {len(t)}, errors {len(err)} {err[:2]}")
    print(" ".join(f"K={k}:{np.mean([r['first'] is not None and r['first'] <= k for r in t]):.2f}" for k in (1, 5, 10, 20, 30)))
