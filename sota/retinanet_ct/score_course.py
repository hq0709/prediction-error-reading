"""Score a longitudinal answer file (results/sota/<method>_course.jsonl) with the code PER's own numbers come from.

Each evaluation pair's reported box becomes one candidate row (peak 1, source "wm", wm 1, p = the box's score), labelled
by perr/course_candidates.course_strips exactly as PER's candidate rows are labelled (the point rounded to the nearest
follow-up voxel, inside a target grown by 3 mm). summarize_course.summary then gives the pair hit ("reader_maxp": the
row with the largest p, i.e. the reported box, over pairs with a target) and the progression AUROC (all pairs, scored
by the largest p, i.e. the reported box's score). A pair without an answer is a miss with score 0. Also: the pair hit
split as in course_new_vs_grown.py (pairs whose targets are all new / include a grown one).

    /raid/home/CAMCA/hj880/venvs/perr/bin/python sota/retinanet_ct/score_course.py <method> [<method> ...]
"""
import json
import os
import sys

import numpy as np

ROOT = "/raid/home/CAMCA/hj880/perr"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from perr.autopet import load_pair  # noqa: E402
from perr.course_candidates import course_strips, selected_targets  # noqa: E402
from summarize_course import MIN_DIAM, lesion_of, summary  # noqa: E402


def main(method):
    ans = {r["case"]: r for r in map(json.loads, open(f"results/sota/{method}_course.jsonl"))}
    keys = open("sota/eval_course.txt").read().split()
    out = f"sota/retinanet_ct/course_{method}"
    os.makedirs(out, exist_ok=True)
    meta, scores, split = [], [], {"only new": [], "with grown": []}
    for k in keys:
        a = ans.get(k)
        P = load_pair(k)
        if a and a.get("present") and a.get("zyx") is not None:
            z, y, x = a["zyx"]
            label = course_strips(P, [(z, y, x, 1.0)], MIN_DIAM)[0]["label"]
            zyx, p = [float(z), float(y), float(x)], float(a.get("score") or 0.0)
        else:
            zyx, label, p = [0.0, 0.0, 0.0], "none", 0.0
        meta.append({"uid": k, "peak": 1, "zyx": zyx, "source": "wm", "label": int(label == "target")})
        scores.append({"uid": k, "peak": 1, "wm": 1.0, "p": p, "label": int(label == "target")})
        t = selected_targets(P.targets, MIN_DIAM)
        if len(t):
            kind = {int(r.fu_label): ("new" if str(r.topology_class).upper().startswith("NEW") else "grown")
                    for r in t.itertuples()}
            group = "only new" if set(kind.values()) == {"new"} else "with grown"
            split[group].append(bool(a and a.get("present")) and lesion_of(P, zyx) in kind)
    with open(f"{out}/eval.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in meta)
    json.dump(scores, open(f"{out}/scores.json", "w"))
    m = summary([(f"{out}/scores.json", out)], k=10, sources="prior,wm")
    print(f"{method}: pair hit {m['reader_maxp']:.3f} ({int(round(m['reader_maxp'] * m['pairs_with_target']))}/"
          f"{m['pairs_with_target']} pairs with a target) | progression AUROC {m['auroc']:.3f} ({m['pos']} vs {m['neg']} pairs) | "
          + " | ".join(f"{g}: {np.mean(v):.2f} ({sum(v)}/{len(v)})" for g, v in split.items() if v), flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        main(name)
