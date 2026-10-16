"""Disease-course evaluation (autoPET IV Longitudinal-CT validation + test pairs).

From reader scores on the evaluation strips (results/reader_<name>_eval*.json, rows: uid = pair, peak, wm, label, p):
  pair hit rate   over pairs with a target (new, or grown to x1.728 volume, >= 10 mm): the chosen candidate lies in one;
  lesion recall   over all targets: a target is found when a candidate with P(yes) >= 0.5 among the top K lies in it;
  progression AUROC over all pairs: does the pair contain a target, scored by the largest P(yes) among its candidates.

    python summarize_course.py results/reader_<name>_eval.json [--data data/course_reader] [--k 10]
"""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy import ndimage as ndi

from perr.autopet import load_pair
from perr.course_candidates import selected_targets

MIN_DIAM, TOL = 10.0, 3.0


def lesion_of(P, zyx):
    """FU label of the target lesion within TOL mm of a point, or 0."""
    t = selected_targets(P.targets, MIN_DIAM)
    labs = sorted(int(v) for v in t.fu_label) if len(t) else []
    if not labs:
        return 0
    lab = np.where(np.isin(np.asarray(P.fu_lab), labs), np.asarray(P.fu_lab), 0)
    i = tuple(int(round(c)) for c in zyx)
    if not all(0 <= i[d] < lab.shape[d] for d in range(3)):
        return 0
    r = [int(np.ceil(TOL / s)) + 1 for s in P.spacing]
    sl = tuple(slice(max(0, i[d] - r[d]), i[d] + r[d] + 1) for d in range(3))
    sub = lab[sl]
    if not sub.any():
        return 0
    g = np.stack(np.meshgrid(*[np.arange(s.start, s.start + n) for s, n in zip(sl, sub.shape)], indexing="ij"), -1)
    d = np.linalg.norm((g - np.array(i)) * np.array(P.spacing), axis=-1)
    d[sub == 0] = np.inf
    j = np.unravel_index(np.argmin(d), d.shape)
    return int(sub[j]) if d[j] <= TOL else 0


def main(path, data="data/course_reader", k=10, sources="prior,wm"):
    """Candidates per pair: every prior finding (if the data has them) plus the world model's top k addresses;
    sources restricts them (ablations: "prior" = history only, "wm" = world model only)."""
    keep = set(sources.split(","))
    res = json.load(open(path))
    meta = {(r["uid"], r["peak"]): r for r in map(json.loads, open(f"{data}/eval{'' if k <= 10 else '_k30'}.jsonl"))}
    by = defaultdict(list)
    for r in res:
        m = meta[(r["uid"], r["peak"])]
        by[r["uid"]].append({**r, "zyx": m["zyx"], "source": m.get("source", "wm")})
    score = lambda c: c["wm"] * c["p"]                     # WM x reader; raw, so that pairs are comparable
    rules = {"wm_x_reader": lambda cs: max(cs, key=score), "reader_maxp": lambda cs: max(cs, key=lambda c: c["p"])}
    out, y, s, found, total = defaultdict(list), [], [], 0, 0
    for key, cs in by.items():
        cs = sorted(cs, key=lambda c: c["peak"])
        wm = [c for c in cs if c["source"] == "wm"][:k] if "wm" in keep else []
        cs = ([c for c in cs if c["source"] == "prior"] if "prior" in keep else []) + wm
        if not cs:
            cs = [{"p": 0.0, "wm": 0.0, "label": 0, "zyx": [0, 0, 0], "source": "none", "peak": 0}]
        P = load_pair(key)
        t = selected_targets(P.targets, MIN_DIAM)
        tl = set(int(v) for v in t.fu_label) if len(t) else set()
        y.append(int(bool(tl)))
        s.append(max(c["p"] for c in cs))                  # pair score for the progression AUROC: max P(yes)
        if tl:
            for name, pick in rules.items():
                out[name].append(pick(cs)["label"])
            out["wm_top1"].append(wm[0]["label"] if wm else 0)
            out["oracle"].append(int(any(x["label"] for x in cs)))
            got = {lesion_of(P, x["zyx"]) for x in cs if x["p"] >= 0.5 and x["label"]}
            found += len(got & tl)
            total += len(tl)
    y, s = np.array(y), np.array(s)
    pos, neg = s[y == 1], s[y == 0]
    auc = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]) if len(pos) and len(neg) else float("nan")
    print(f"{path}  K={k}: pairs with a target {len(out['oracle'])} | " +
          " | ".join(f"{n} {np.mean(v):.2f}" for n, v in out.items()) +
          f" | lesion recall {found}/{total} = {found / max(total, 1):.2f} | progression AUROC {auc:.3f} ({len(pos)} vs {len(neg)} pairs)"
          f" [{sources}]")


if __name__ == "__main__":
    args = sys.argv[1:]
    kw = {}
    for flag in ("--data", "--k", "--sources"):
        if flag in args:
            i = args.index(flag)
            kw[flag[2:]] = int(args[i + 1]) if flag == "--k" else args[i + 1]
            args = args[:i] + args[i + 2:]
    for p in args:
        main(p, **kw)
