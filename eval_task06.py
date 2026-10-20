"""External lung test on MSD Task06 (prep_task06.py): systems trained on LUNA16 nodules, run unchanged.

    python eval_task06.py strips              # PER candidates: the label-free lung world model's top 10, reader strips
    python eval_task06.py detect <run> <rule> # a trained 3-D RetinaNet (train_ct_retinanet.py) on every case
    python eval_task06.py score               # hit (reported point inside the tumour grown by 3 mm), by tumour size
"""
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from perr.candidates import candidate_rows
from perr.context_wm import ContextPrior
from perr.study import load_luna

CASES = "results/t06_cases.txt"
OUT = "data/reader_t06"
CACHE = "/raid/home/CAMCA/hj880/perr/cache/luna"
TAG = "nllfree"


def strip_rows(case):
    s = load_luna(case)
    rows = []
    for r in candidate_rows(s, 10, proposal=ContextPrior(TAG)):
        path = os.path.abspath(f"{OUT}/img/{case}_wm{r['peak']}.png")
        r.pop("image").save(path)
        rows.append({**r, "uid": case, "image": path, "source": "wm", "label": int(inside(case, r["zyx"]))})
    return rows


def inside(case, zyx):
    T = np.load(f"{CACHE}/{case}.tumour.npy", mmap_mode="r")
    i = [int(round(v)) for v in zyx]
    return all(0 <= i[d] < T.shape[d] for d in range(3)) and bool(T[i[0], i[1], i[2]])


def largest(case):
    m = np.load(f"{CACHE}/{case}.meta.npy", allow_pickle=True).item()
    return max(d for _, d in m["nodules"])


def main():
    cmd = sys.argv[1]
    cases = open(CASES).read().split()
    if cmd == "strips":
        os.makedirs(f"{OUT}/img", exist_ok=True)
        with Pool(16) as p:
            rows = [r for rs in p.map(strip_rows, cases, chunksize=1) for r in rs]
        with open(f"{OUT}/eval_t06.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        by = defaultdict(list)
        for r in rows:
            by[r["uid"]].append(r)
        print(len(rows), "rows; oracle@10", np.mean([any(c["label"] for c in v) for v in by.values()]))
    elif cmd == "detect":
        import torch
        from train_ct_retinanet import Scan, detect, make_detector
        run, rule = sys.argv[2], sys.argv[3]
        dev = torch.device("cuda")
        det = make_detector(dev, pretrained="retina3" in run)
        det.network.load_state_dict(torch.load(f"{run}/model.pt", map_location=dev))
        out = {}
        for c in cases:
            rows = detect(det, Scan(c), dev)
            cand = [r for r in rows if rule == "top" or r["size_mm"] >= 6.0] or rows[:1]
            out[c] = {"zyx": cand[0]["zyx"] if cand else None, "boxes": rows[:20]}
        json.dump(out, open(f"{run}/eval_t06.json", "w"))
        print(run, "done", len(out))
    else:
        res = {}
        p = json.load(open("results/reader_q8_ctxfree_nomeas_eval_t06.json"))
        meta = {(json.loads(l)["uid"], json.loads(l)["peak"]): json.loads(l) for l in open(f"{OUT}/eval_t06.jsonl")}
        by = defaultdict(list)
        for r in p:
            by[r["uid"]].append({**r, "zyx": meta[(r["uid"], r["peak"])]["zyx"]})
        res["PER (label-free world model + reader, WM x reader)"] = {
            u: inside(u, max(cs, key=lambda c: c["wm"] / sorted(cs, key=lambda c: c["peak"])[0]["wm"] * c["p"])["zyx"]) for u, cs in by.items()}
        res["PER, reader max-p"] = {u: inside(u, max(cs, key=lambda c: c["p"])["zyx"]) for u, cs in by.items()}
        res["world model alone (top address)"] = {u: inside(u, sorted(cs, key=lambda c: c["peak"])[0]["zyx"]) for u, cs in by.items()}
        res["oracle: a tumour among the top 10"] = {u: any(inside(u, c["zyx"]) for c in cs) for u, cs in by.items()}
        for name, run in (("3-D RetinaNet, from scratch", "runs_det/ct_retina2_b100"), ("3-D RetinaNet, MedicalNet", "runs_det/ct_retina3_b100")):
            if os.path.exists(f"{run}/eval_t06.json"):
                d = json.load(open(f"{run}/eval_t06.json"))
                res[name] = {u: bool(v["zyx"]) and inside(u, v["zyx"]) for u, v in d.items()}
        size = {u: largest(u) for u in cases}
        print(f"{'method':55s} all ({len(cases)})   largest <= 30 mm   > 30 mm")
        for name, h in res.items():
            a = [h[u] for u in cases if u in h]
            s = [h[u] for u in cases if u in h and size[u] <= 30]
            l = [h[u] for u in cases if u in h and size[u] > 30]
            print(f"{name:55s} {np.mean(a):.2f}          {np.mean(s):.2f} ({len(s)})        {np.mean(l):.2f} ({len(l)})")


if __name__ == "__main__":
    main()
