"""Where the final PER configurations miss, and how well their confidence ranks their answers (paper analysis).

For each case with a target: hit (the chosen address lies in a target), search miss (no target among the K addresses
read) and verification miss (a target among them, another address chosen); hit + search miss + verification miss = 1.
Selective answering: cases ranked by the chosen address's score; accuracy of the most confident 50 / 70 / 90 / 100 %,
and the AUROC of that score for separating hits from misses.

Final configurations: lung, label-free world model + reader q8_ctxfree_nomeas, WM x reader, K = 10 (Set 1, Set 2);
liver, two-sided fused list + reader q8_liver_t08h3, max-p, K = 30 (MSD evaluation half, HCC-TACE-Seg); disease course,
prior findings + world model top 10 + reader q8_course_vis, max-p (pairs with a target); radiographs, both world models'
60 addresses + the kept full-data reader (choose_cxr_aug.py), max-p, per focal finding among positive ChestX-Det test images.

    python analysis_per.py            # writes results/analysis_per.json
"""
import json
from collections import defaultdict

import numpy as np

from perr.score import score_case
from perr.study import load_luna

TYPES = ("Nodule", "Mass", "Calcification", "Fracture")


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def summary(cases):
    """cases: list of (hit, oracle, confidence)."""
    h = np.array([c[0] for c in cases], float)
    o = np.array([c[1] for c in cases], float)
    q = np.array([c[2] for c in cases], float)
    order = np.argsort(-q, kind="stable")
    sel = {f"{int(f * 100)}": float(h[order[:max(1, int(round(f * len(h))))]].mean()) for f in (0.5, 0.7, 0.9, 1.0)}
    return {"n": len(h), "hit": float(h.mean()), "oracle": float(o.mean()), "search_miss": float(1 - o.mean()),
            "verify_miss": float(o.mean() - h.mean()), "selective": sel, "conf_auroc": auroc(q[h == 1], q[h == 0])}


def lung(lo, hi, name="q8_ctxfree_nomeas", data="data/reader_ctxfree", k=10):
    meta = {(r["uid"], r["peak"]): r for r in map(json.loads, open(f"{data}/eval_{lo}_{hi}.jsonl"))}
    by = defaultdict(list)
    for r in json.load(open(f"results/reader_{name}_eval_{lo}_{hi}.json")):
        by[r["uid"]].append({**r, "zyx": meta[(r["uid"], r["peak"])]["zyx"]})
    cases = []
    for u, cs in by.items():
        s = load_luna(u)
        cs = sorted(cs, key=lambda c: c["peak"])[:k]
        q = [c["wm"] / cs[0]["wm"] * c["p"] for c in cs]
        j = int(np.argmax(q))
        hits = [score_case(s, {"present": True, "zyx": c["zyx"]})["hit6"] for c in cs]
        cases.append((hits[j], max(hits), q[j]))
    return summary(cases)


def liver(which, name="q8_liver_t08h3", lst="fused2", k=30):
    meta = {(r["uid"], r["peak"]): r for r in map(json.loads, open(f"data/liver_k30_{which}_{lst}/eval.jsonl"))}
    by = defaultdict(list)
    for r in json.load(open(f"results/reader_{name}_eval_{which}_{lst}.json")):
        by[r["uid"]].append({**r, "label": meta[(r["uid"], r["peak"])]["label"]})
    cases = []
    for cs in by.values():
        cs = sorted(cs, key=lambda c: c["peak"])[:k]
        c = max(cs, key=lambda c: c["p"])
        cases.append((c["label"], max(x["label"] for x in cs), c["p"]))
    return summary(cases)


def course(name="q8_course_vis", data="data/course_reader", k=10):
    from perr.autopet import load_pair
    from perr.course_candidates import selected_targets
    meta = {(r["uid"], r["peak"]): r for r in map(json.loads, open(f"{data}/eval.jsonl"))}
    by = defaultdict(list)
    for r in json.load(open(f"results/reader_{name}_eval.json")):
        by[r["uid"]].append({**r, "source": meta[(r["uid"], r["peak"])].get("source", "wm")})
    cases = []
    for key, cs in by.items():
        cs = sorted(cs, key=lambda c: c["peak"])
        cs = [c for c in cs if c["source"] == "prior"] + [c for c in cs if c["source"] == "wm"][:k]
        P = load_pair(key)
        if not len(selected_targets(P.targets, 10.0)):
            continue
        c = max(cs, key=lambda c: c["p"])
        cases.append((c["label"], max(x["label"] for x in cs), c["p"]))
    return summary(cases)


def radiograph(path=None, k=60):
    from choose_cxr_aug import chosen
    from cxr_recall import boxes
    path = path or f"results/cxr_focal/{chosen('PER (ours)')}/wmu_cxdet_test.jsonl"
    b = boxes("cxdet")
    rows = [r for r in map(json.loads, open(path)) if r["src"] == "wm" and r["rank"] < k]
    by = defaultdict(list)
    for r in rows:
        by[r["key"]].append(r)
    out = {}
    for t in TYPES:
        pos = set(b[b.finding == t].key)
        cases = []
        for key, cs in by.items():
            if key not in pos:
                continue
            c = max(cs, key=lambda c: c[f"p_{t}"])
            cases.append((c[t], max(x[t] for x in cs), c[f"p_{t}"]))
        out[t] = summary(cases)
    out["mean"] = {m: float(np.mean([out[t][m] for t in TYPES])) for m in ("hit", "oracle", "search_miss", "verify_miss")}
    return out


def main():
    res = {"lung_set1": lung(60, 160), "lung_set2": lung(160, 260), "liver_msd": liver("msd"), "liver_ext": liver("ext"),
           "course": course(), "radiograph": radiograph()}
    for k, v in res.items():
        if k == "radiograph":
            for t, s in v.items():
                print(f"radiograph {t:13s} " + " ".join(f"{m} {s[m]:.3f}" for m in ("hit", "oracle", "search_miss", "verify_miss"))
                      + (f" n {s['n']}" if "n" in s else ""))
        else:
            print(f"{k:10s} n {v['n']:3d} hit {v['hit']:.3f} oracle {v['oracle']:.3f} search {v['search_miss']:.3f} "
                  f"verify {v['verify_miss']:.3f} selective {v['selective']} conf AUROC {v['conf_auroc']:.3f}")
    json.dump(res, open("results/analysis_per.json", "w"), indent=1)


if __name__ == "__main__":
    main()
