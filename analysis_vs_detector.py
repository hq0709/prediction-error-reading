"""PER against the 3D RetinaNets trained on the same scans (lung: train_ct_retinanet.py, runs_det/ct_retina{2,3}_b100;
liver: sota/retinanet_ct, results/sota/retinanet_ct_{scratch,mednet}_liver[_ext].jsonl).

  selective answering: cases ranked by each method's own confidence (PER: the selection score of its answer, as in
      analysis_per.py; RetinaNet: the score of its reported box); accuracy of the most confident half, and the AUROC with
      which the confidence separates hits from misses
  lung hit by the size of the scan's largest nodule

Prints one numbers.tex line per macro.

    python analysis_vs_detector.py
"""
import json
from collections import defaultdict

import numpy as np

from analysis_per import liver as per_liver
from analysis_per import lung as per_lung
from analysis_per import summary
from perr.liver import label_at
from perr.liver_ext import label_at as ext_label_at
from perr.score import score_case
from perr.study import load_luna

SETS = (("A", 60, 160), ("B", 160, 260))
DET = {"Scr": "ct_retina2_b100", "Med": "ct_retina3_b100"}
LIVER = {"Scr": "retinanet_ct_scratch", "Med": "retinanet_ct_mednet"}
BINS = (("Small", 6, 10), ("Mid", 10, 15), ("Large", 15, 1000))


def lung_detector(run, lo, hi):
    """{uid: (hit, confidence)} for the top-scoring box, the rule chosen on validation ("top")."""
    out = {}
    for u, b in json.load(open(f"runs_det/{run}/eval_{lo}_{hi}.json")).items():
        if not b:
            out[u] = (0, 0.0)
            continue
        top = max(b, key=lambda x: x["score"])
        out[u] = (int(score_case(load_luna(u), {"present": True, "zyx": top["zyx"]})["hit6"]), float(top["score"]))
    return out


def per_lung_cases(lo, hi, name="q8_ctxfree_nomeas", data="data/reader_ctxfree", k=10):
    """{uid: hit} for PER's answer (world model x verifier over the top 10), as analysis_per.lung."""
    meta = {(r["uid"], r["peak"]): r for r in map(json.loads, open(f"{data}/eval_{lo}_{hi}.jsonl"))}
    by = defaultdict(list)
    for r in json.load(open(f"results/reader_{name}_eval_{lo}_{hi}.json")):
        by[r["uid"]].append({**r, "zyx": meta[(r["uid"], r["peak"])]["zyx"]})
    out = {}
    for u, cs in by.items():
        cs = sorted(cs, key=lambda c: c["peak"])[:k]
        c = cs[int(np.argmax([x["wm"] / cs[0]["wm"] * x["p"] for x in cs]))]
        out[u] = int(score_case(load_luna(u), {"present": True, "zyx": c["zyx"]})["hit6"])
    return out


def liver_detector(method, ext):
    name = "liver_ext" if ext else "liver"
    pred = {r["case"]: r for r in map(json.loads, open(f"results/sota/{method}_{name}.jsonl"))}
    lab = ext_label_at if ext else label_at
    cases = []
    for c in open(f"sota/eval_{name}.txt").read().split():
        p = pred.get(c)
        h = int(bool(p and p.get("present") and p.get("zyx") is not None and lab(c, p["zyx"]) == "big"))
        cases.append((h, h, float(p.get("score") or 0.0) if p else 0.0))
    return summary(cases)


def line(name, value, note):
    print(f"\\newcommand{{\\{name}}}{{{value}}} % analysis_vs_detector.py -> {note}")


def main():
    for tag, lo, hi in SETS:
        s = per_lung(lo, hi)
        line(f"SelPerLung{tag}", f"{s['selective']['50']:.2f}", f"PER lung {lo}-{hi}, accuracy of the most confident half")
        line(f"ConfPerLung{tag}", f"{s['conf_auroc']:.2f}", f"PER lung {lo}-{hi}, confidence AUROC (hits vs misses)")
        for d, run in DET.items():
            cases = lung_detector(run, lo, hi)
            s = summary([(h, h, q) for h, q in cases.values()])
            line(f"SelRet{d}Lung{tag}", f"{s['selective']['50']:.2f}", f"{run} {lo}-{hi}, most confident half")
            line(f"ConfRet{d}Lung{tag}", f"{s['conf_auroc']:.2f}", f"{run} {lo}-{hi}, confidence AUROC")
    for which, ext in (("Msd", False), ("Ext", True)):
        s = per_liver("ext" if ext else "msd")
        line(f"SelPerLiver{which}", f"{s['selective']['50']:.2f}", f"PER liver {which}, most confident half")
        line(f"ConfPerLiver{which}", f"{s['conf_auroc']:.2f}", f"PER liver {which}, confidence AUROC")
        for d, m in LIVER.items():
            s = liver_detector(m, ext)
            line(f"SelRet{d}Liver{which}", f"{s['selective']['50']:.2f}", f"{m} liver {which}, most confident half")
            line(f"ConfRet{d}Liver{which}", f"{s['conf_auroc']:.2f}", f"{m} liver {which}, confidence AUROC")
    for tag, lo, hi in SETS:
        per = per_lung_cases(lo, hi)
        det = {d: lung_detector(run, lo, hi) for d, run in DET.items()}
        size = {u: max(n.diameter_mm for n in load_luna(u).nodules) for u in per}
        for b, a, z in BINS:
            us = [u for u in per if a <= size[u] < z]
            line(f"SizePer{b}{tag}", f"{np.mean([per[u] for u in us]):.2f}", f"PER lung {lo}-{hi}, largest nodule {a}-{z} mm (n={len(us)})")
            for d in DET:
                line(f"SizeRet{d}{b}{tag}", f"{np.mean([det[d][u][0] for u in us]):.2f}", f"{DET[d]} {lo}-{hi}, {a}-{z} mm (n={len(us)})")


if __name__ == "__main__":
    main()
