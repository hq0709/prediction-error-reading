"""Disease course, split by what changed: a target lesion that is NEW at follow-up (topology NEWLYAPPEARING: nothing at
that place before) or one that GREW (matched to a baseline lesion, x1.728 volume). A tracker such as LesionLocator
starts from the baseline lesions, so a new lesion is outside what it can follow; the disease-course world model predicts
the follow-up from the registered baseline and flags any departure, new or grown.

Lesion recall: a target is found when a candidate lies within 3 mm of it (PER: a candidate the reader accepts,
P(yes) >= 0.5, among the prior findings and the world model's top 10; LesionLocator: any tracked lesion's follow-up
location). Pair hit: pairs whose targets are all new / include a grown one, the reported location lies in a target.

    python course_new_vs_grown.py
"""
import json
from collections import defaultdict

import numpy as np

from perr.autopet import load_pair
from perr.course_candidates import selected_targets
from summarize_course import MIN_DIAM, lesion_of


def per_candidates(path="results/reader_q8_course_vis_eval.json", data="data/course_reader", k=10, sources=("prior", "wm")):
    meta = {(r["uid"], r["peak"]): r for r in map(json.loads, open(f"{data}/eval.jsonl"))}
    by = defaultdict(list)
    for r in json.load(open(path)):
        m = meta[(r["uid"], r["peak"])]
        by[r["uid"]].append({**r, "zyx": m["zyx"], "source": m.get("source", "wm")})
    out = {}
    for key, cs in by.items():
        cs = sorted(cs, key=lambda c: c["peak"])
        wm = [c for c in cs if c["source"] == "wm"][:k] if "wm" in sources else []
        out[key] = ([c for c in cs if c["source"] == "prior"] if "prior" in sources else []) + wm
    return out


def main():
    pairs = open("sota/eval_course.txt").read().split()
    per = per_candidates()
    per_wm = per_candidates(sources=("wm",))
    ll = {json.loads(l)["case"]: json.loads(l) for l in open("results/sota/lesionlocator_course.jsonl")}
    rec = defaultdict(lambda: defaultdict(list))
    hit = defaultdict(lambda: defaultdict(list))
    for key in pairs:
        P = load_pair(key)
        t = selected_targets(P.targets, MIN_DIAM)
        if not len(t):
            continue
        kind = {int(r.fu_label): ("new" if str(r.topology_class).upper().startswith("NEW") else "grown") for r in t.itertuples()}
        group = "only new" if set(kind.values()) == {"new"} else "with grown"
        methods = {
            "PER: prior findings + world model + reader": [c["zyx"] for c in per.get(key, []) if c["p"] >= 0.5],
            "PER: world model + reader (no history)": [c["zyx"] for c in per_wm.get(key, []) if c["p"] >= 0.5],
            "LesionLocator (tracks baseline lesions)": [c["zyx"] for c in ll.get(key, {}).get("candidates", [])],
        }
        picks = {
            "PER: prior findings + world model + reader": max(per.get(key, []), key=lambda c: c["p"], default=None),
            "PER: world model + reader (no history)": max(per_wm.get(key, []), key=lambda c: c["p"], default=None),
            "LesionLocator (tracks baseline lesions)": ll.get(key) if ll.get(key, {}).get("present") else None,
        }
        for name, pts in methods.items():
            got = {lesion_of(P, z) for z in pts}
            for lab, kd in kind.items():
                rec[name][kd].append(lab in got)
            pk = picks[name]
            hit[name][group].append(bool(pk) and lesion_of(P, pk["zyx"]) in kind)
    print("lesion recall (targets found)")
    for name in rec:
        print(f"  {name:45s} " + "  ".join(f"{kd}: {np.mean(v):.2f} ({sum(v)}/{len(v)})" for kd, v in sorted(rec[name].items())))
    print("pair hit (the reported location lies in a target)")
    for name in hit:
        print(f"  {name:45s} " + "  ".join(f"{g}: {np.mean(v):.2f} ({sum(v)}/{len(v)})" for g, v in sorted(hit[name].items())))


if __name__ == "__main__":
    main()
