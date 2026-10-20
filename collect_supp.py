"""Numbers for the supplementary tables, in one JSON (results/supp_tables.json), read by the paper's table generator
(perr-paper/scripts/make_supp_tables.py). Every entry is recomputed from the scored outputs:

  cxr_per_finding   ChestX-Det test, per focal finding: presence AUROC, pointing hit, LROC (score_cxr_focal.evaluate)
  nih_per_finding   NIH ChestX-ray14 boxes (nodule, mass): the same, readers and detectors applied unchanged
  cxr_curve         radiograph label efficiency, every method and budget (results/cxr_curve.csv, collect_cxr_curve.py)
  liver_grid        liver reading budget x address list x rule, reader q8_liver_t08h3, on the 100 Task08 development cases,
                    the MSD evaluation half and HCC-TACE-Seg (collect_liver_kdev.py)
  open_set          unseen ChestX-Det finding types (results/open_set_cxr.csv, open_set_cxr.py)
  analysis          miss decomposition and selective answering (results/analysis_per.json, analysis_per.py)
  sweeps            one-at-a-time hyperparameter sweeps (hp_recall.py, make_reader_data.py, cxr_focal_data.py and
                    make_liver_kdev.py variants scored by the final readers)
  lung_ablation     lung hit per candidate source (world model, raw error, intensity, blob shape, PatchCore,
                    two-sided error) under four rules, final reader fixed; and the world-model candidates read by a
                    reader trained on blob-filter candidates
  cxr_aug           validation scores with and without training augmentation, and the chosen run (choose_cxr_aug.py)

    python collect_supp.py
"""
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from choose_cxr_aug import chosen
from cxr_recall import boxes
from score_cxr_focal import TYPES, evaluate

R = "results/cxr_focal"
PER = chosen("PER (ours)")   # the kept full-data verifier run (with or without augmentation), choose_cxr_aug.py
CXR_METHODS = [  # key, file stem under results/cxr_focal (cxdet_test / nih_test), K; augmentation: choose_cxr_aug.py
    ("frcnn", f"{chosen('Faster R-CNN')}/", 1), ("retinanet", f"{chosen('RetinaNet')}/", 1), ("fcos", f"{chosen('FCOS')}/", 1),
    ("yolo", "cxr_yolo11m/", 1), ("rtdetr", f"{chosen('RT-DETRv2-R50')}/", 1), ("defdetr", f"{chosen('Deformable DETR')}/", 1),
    ("oneshot", f"{chosen('one-shot Qwen3-VL-8B')}/", 1), ("per", f"{chosen('PER (ours)')}/wmu_", 60)]


def rows_of(path):
    return [json.loads(l) for l in open(path)]


NIH_CLEAN = set(open("results/cxr_nih_clean_keys.txt").read().split())   # patients outside ChestX-Det training


def keep_clean(rows, which):
    return rows if which != "nih" else [r for r in rows if r["key"] in NIH_CLEAN]


def per_finding(which, split):
    b = boxes(which)
    if split:
        b = b[b.split == split]
    if which == "nih":
        b = b[b.key.isin(NIH_CLEAN)]
    types = [t for t in TYPES if t in set(b.finding)]
    out = {}
    for key, stem, k in CXR_METHODS:
        path = f"{R}/{stem}{which}_test.jsonl"
        if not os.path.exists(path):
            continue
        res = evaluate(keep_clean(rows_of(path), which), b, types, ks=(k,), rules=("max-p",))
        out[key] = {r.finding: {"AUROC": r.AUROC, "hit": r.hit, "LROC": r.LROC, "pos": int(r.pos)} for r in res.itertuples()}
    # PER with the NIH-normal world model's 30 addresses added (same final reader, max-p over 90 addresses)
    add = f"{R}/{PER}/wmn_{which}_test.jsonl"
    if os.path.exists(add):
        extra = rows_of(add)
        for r in extra:
            r["rank"] += 60
        res = evaluate(keep_clean(rows_of(f"{R}/{PER}/wmu_{which}_test.jsonl") + extra, which), b, types, ks=(90,),
                       rules=("max-p",))
        out["per_site"] = {r.finding: {"AUROC": r.AUROC, "hit": r.hit, "LROC": r.LROC, "pos": int(r.pos)} for r in res.itertuples()}
    return out


def site():
    """New site from normal radiographs, on ChestX-Det test and the NIH boxes of patients outside ChestX-Det training:
    single-model reader (runs_reader/cxr_focal) at the 30 addresses of each 512 world model; the final method with the
    NIH-normal world model's 30 addresses added, and the control that reads 15 more addresses from each MIMIC model."""
    out = {}
    for which, split in (("cxdet", "test"), ("nih", None)):
        b = boxes(which)
        if split:
            b = b[b.split == split]
        if which == "nih":
            b = b[b.key.isin(NIH_CLEAN)]
        types = [t for t in TYPES if t in set(b.finding)]
        ent = {}
        for pre in ("wm", "wmn", "wma"):
            path = f"{R}/cxr_focal/{pre}_{which}_test.jsonl"
            if os.path.exists(path):
                res = evaluate(keep_clean(rows_of(path), which), b, types, ks=(30,), rules=("max-p",))
                ent[pre] = {m: float(res[m].mean()) for m in ("AUROC", "hit", "LROC")}
        base = keep_clean(rows_of(f"{R}/{PER}/wmu_{which}_test.jsonl"), which)
        res = evaluate(base, b, types, ks=(60,), rules=("max-p",))
        ent["final"] = {m: float(res[m].mean()) for m in ("AUROC", "hit", "LROC")}
        add = f"{R}/{PER}/wmn_{which}_test.jsonl"
        if os.path.exists(add):
            extra = keep_clean(rows_of(add), which)
            for r in extra:
                r["rank"] += 60
            res = evaluate(base + extra, b, types, ks=(90,), rules=("max-p",))
            ent["final+nih"] = {m: float(res[m].mean()) for m in ("AUROC", "hit", "LROC")}
        ctl = f"{R}/{PER}/hp_k45_{which}_test.jsonl"
        if os.path.exists(ctl):
            res = evaluate(keep_clean(rows_of(ctl), which), b, types, ks=(90,), rules=("max-p",))
            ent["final_k45"] = {m: float(res[m].mean()) for m in ("AUROC", "hit", "LROC")}
        out[which] = ent
    return out


def liver_grid(name="q8_liver_t08h3"):
    from collect_liver_kdev import KS, LISTS, RULES, hit, load
    out = defaultdict(dict)
    for which in ("t08", "msd", "ext"):
        for lst in LISTS:
            if not os.path.exists(f"results/reader_{name}_eval_{which}_{lst}.json"):
                continue
            by = load(name, which, lst)
            for k in KS:
                for rule in RULES:
                    out[which][f"{lst}|{k}|{rule}"] = hit(by, k, rule)
    return out


def lung_hit(name, data, lo, hi, k=10, rule="wm_x_reader"):
    from collect_ct_curve import reader_hit
    return reader_hit(name, lo, hi, data=data, k=k, rule=rule)


def sweeps():
    out = {"lung": {}, "radiograph": {}, "liver": {}}
    # lung: address recall (dev split and Set 1 + Set 2) and PER hit with the final reader, one change at a time
    rec = {}
    for f in os.listdir("results"):
        if f.startswith("hp_recall_") and f.endswith(".json"):
            tag, which = f[len("hp_recall_"):-5].rsplit("_", 1)
            rec[(tag, which)] = json.load(open(f"results/{f}"))
    lung = {}
    retrained = {"hpb0": ("beta", 0.0), "hpb025": ("beta", 0.25), "hpb1": ("beta", 1.0), "hph16": ("hidden", 16),
                 "hph32": ("hidden", 32), "hpf10": ("normal share", 0.10), "hpf25": ("normal share", 0.25),
                 "hpf50": ("normal share", 0.50)}
    final = {"recall_dev": rec.get(("nllfree", "dev"), {}).get("s2_n10"), "recall_eval": rec.get(("nllfree", "eval"), {}).get("s2_n10"),
             "hit": [lung_hit("q8_ctxfree_nomeas", "data/reader_ctxfree", lo, hi) for lo, hi in ((60, 160), (160, 260))]}
    lung["final"] = final
    for tag, (param, val) in retrained.items():
        lung[f"{param}={val}"] = {"recall_dev": rec.get((tag, "dev"), {}).get("s2_n10"),
                                  "recall_eval": rec.get((tag, "eval"), {}).get("s2_n10"),
                                  "hit": [lung_hit(f"hp_{tag}", f"data/hp_{tag}", lo, hi) for lo, hi in ((60, 160), (160, 260))]}
    for tag, (param, val, var) in {"hps1": ("sigma", 1, "s1_n10"), "hps3": ("sigma", 3, "s3_n10"), "hps4": ("sigma", 4, "s4_n10"),
                                   "hpn5": ("nms", 5, "s2_n5"), "hpn15": ("nms", 15, "s2_n15"), "hpn20": ("nms", 20, "s2_n20")}.items():
        lung[f"{param}={val}"] = {"recall_dev": rec.get(("nllfree", "dev"), {}).get(var),
                                  "recall_eval": rec.get(("nllfree", "eval"), {}).get(var),
                                  "hit": [lung_hit(tag, f"data/{tag}", lo, hi) for lo, hi in ((60, 160), (160, 260))]}
    for k in (5, 10, 20, 30):     # reading budget: the top 30 addresses scored once, the first K read
        path = "results/reader_hpk30_eval_k30_60_160.json"
        if os.path.exists(path):
            lung[f"K={k}"] = {"recall_dev": rec.get(("nllfree", "dev"), {}).get("s2_n10", {}).get(str(k)),
                              "recall_eval": rec.get(("nllfree", "eval"), {}).get("s2_n10", {}).get(str(k)),
                              "hit": [_lung_k30(k, lo, hi) for lo, hi in ((60, 160), (160, 260))]}
    out["lung"] = lung
    # radiographs: ChestX-Det test and the reader's held-out tenth of the training images (dev), final reader fixed
    b = boxes("cxdet")
    rad = {}
    for name, k in (("hp_final", 60), ("hp_s1", 60), ("hp_s3", 60), ("hp_s4", 60), ("hp_n8", 60), ("hp_n24", 60),
                    ("hp_n32", 60), ("hp_k10", 20), ("hp_k20", 40), ("hp_k40", 80), ("hp_two", 60)):
        ent = {}
        for split, path in (("dev", f"{R}/{PER}/{name}_cxdet_train.jsonl"),
                            ("test", f"{R}/{PER}/{name if name != 'hp_final' else 'wmu'}_cxdet_test.jsonl")):
            if os.path.exists(path):
                bs = b[b.split == ("train" if split == "dev" else "test")]
                res = evaluate(rows_of(path), bs, TYPES, ks=(k,), rules=("max-p",))
                ent[split] = {m: float(res[m].mean()) for m in ("AUROC", "hit", "LROC")}
        rad[name] = ent
    out["radiograph"] = rad
    # liver: Task08 development cases, MSD evaluation half and HCC-TACE-Seg, reader q8_liver_t08h3, K = 30, max-p
    from collect_liver_kdev import hit
    liv = {}
    for suf in ("", "_s2", "_s4", "_c5", "_c20", "_n10", "_n20"):
        ent = {}
        for which in ("t08", "msd", "ext"):
            path = f"results/reader_q8_liver_t08h3_eval_{which}_fused2{suf}.json"
            if not os.path.exists(path):
                continue
            meta = {(r["uid"], r["peak"]): r for r in map(json.loads, open(f"data/liver_k30_{which}_fused2{suf}/eval.jsonl"))}
            by = defaultdict(list)
            for r in json.load(open(path)):
                by[r["uid"]].append({**r, "label": meta[(r["uid"], r["peak"])]["label"]})
            ent[which] = {"hit": hit(by, 30, "maxp"),
                          "recall30": float(np.mean([any(c["label"] for c in v) for v in by.values()]))}
        liv[suf or "final"] = ent
    out["liver"] = liv
    return out


def _lung_k30(k, lo, hi):
    from perr.score import score_case
    from perr.study import load_luna
    meta = {(r["uid"], r["peak"]): r for r in map(json.loads, open(f"data/hpk30/eval_k30_{lo}_{hi}.jsonl"))}
    by = defaultdict(list)
    for r in json.load(open(f"results/reader_hpk30_eval_k30_{lo}_{hi}.json")):
        by[r["uid"]].append({**r, "zyx": meta[(r["uid"], r["peak"])]["zyx"]})
    h = []
    for u, cs in by.items():
        cs = sorted(cs, key=lambda c: c["peak"])[:k]
        c = max(cs, key=lambda c: c["wm"] / cs[0]["wm"] * c["p"])
        h.append(score_case(load_luna(u), {"present": True, "zyx": c["zyx"]})["hit6"])
    return float(np.mean(h))


LUNG_ABLATION = {   # candidate sources and error variants, final reader fixed (abl_free_data, abl2_data in queue/jobs3.tsv)
    "final": ("reader_q8_ctxfree_nomeas_eval_{}_{}.json", "data/reader_ctxfree/eval_{}_{}.jsonl"),
    "raw_error": ("reader_abl_free_raw_eval_{}_{}.json", "data/abl_free_raw/eval_{}_{}.jsonl"),
    "intensity": ("reader_abl_free_int_eval_int_{}_{}.json", "data/abl_free_int/eval_int_{}_{}.jsonl"),
    "shape": ("reader_abl_free_shape_eval_{}_{}.json", "data/abl_free_shape/eval_{}_{}.jsonl"),
    "patchcore": ("reader_abl_free_pc_eval_{}_{}.json", "data/abl_free_pc/eval_{}_{}.jsonl"),
    "two_sided": ("reader_hpt2_eval_{}_{}.json", "data/hpt2/eval_{}_{}.jsonl"),
    # world-model candidates read by a verifier trained on blob-filter candidates (shtr_reader in queue/jobs3.tsv)
    "verifier_on_blob": ("reader_q8_shapetrain_on_ctxfree_eval_{}_{}.json", "data/reader_ctxfree/eval_{}_{}.jsonl"),
    # world model trained with nodule cubes excluded: read by the final verifier (lungx_score) and by its own (nm_k30)
    "excluded_wm": ("reader_ctxfree_on_ctx_eval_{}_{}.json", "data/reader_ctx/eval_{}_{}.jsonl"),
    "excluded_wm_own": ("reader_nm_k30_eval_k30_{}_{}.json", "data/reader_ctx/eval_k30_{}_{}.jsonl"),
}


def lung_ablation(k=10):
    """Hit on Set 1 / Set 2 for each candidate source under four rules: the source's top address (no reader), source
    score x reader probability, reader probability alone, and the oracle (any of the K addresses within the radius)."""
    from perr.score import score_case
    from perr.study import load_luna
    out = {}
    for name, (res, data) in LUNG_ABLATION.items():
        ent = {}
        for lo, hi in ((60, 160), (160, 260)):
            rp, dp = "results/" + res.format(lo, hi), data.format(lo, hi)
            if not (os.path.exists(rp) and os.path.exists(dp)):
                continue
            meta = {(r["uid"], r["peak"]): r for r in map(json.loads, open(dp))}
            by = defaultdict(list)
            for r in json.load(open(rp)):
                by[r["uid"]].append({**r, "zyx": meta[(r["uid"], r["peak"])]["zyx"]})
            h = defaultdict(list)
            for u, cs in by.items():
                cs = sorted(cs, key=lambda c: c["peak"])[:k]
                s = load_luna(u)
                hit = [score_case(s, {"present": True, "zyx": c["zyx"]})["hit6"] for c in cs]
                h["no_reader"].append(hit[0])
                h["wm_x_reader"].append(hit[max(range(len(cs)), key=lambda i: cs[i]["wm"] / cs[0]["wm"] * cs[i]["p"])])
                h["maxp"].append(hit[max(range(len(cs)), key=lambda i: cs[i]["p"])])
                h["oracle"].append(any(hit))
            ent[f"set{1 if lo == 60 else 2}"] = {r: float(np.mean(v)) for r, v in h.items()}
        out[name] = ent
    return out


def course():
    """Longitudinal CT at K = 10: the final per-task verifier on the evaluation pairs (8 pairs re-registered), the
    three-fold cross-validation over training patients (pooled), and the world model retrained without lesion
    annotations (when scored). Pair hit = verifier max-p (the selected rule)."""
    from summarize_course import summary

    def ent(sets):
        if not all(os.path.exists(p) for p, _ in sets):
            return None
        m = summary(sets)
        return {"hit": m["reader_maxp"], "wm_x_reader": m["wm_x_reader"], "wm_top1": m["wm_top1"], "oracle": m["oracle"],
                "lesion_recall": m["lesion_recall"], "auroc": m["auroc"], "pairs_with_target": m["pairs_with_target"],
                "pairs": m["pos"] + m["neg"]}
    return {"final": ent([("results/reader_q8_course_vis_fix_eval.json", "data/course_reader_fix")]),
            "cv": ent([(f"results/reader_q8_course_cv{i}_eval.json", f"data/course_cv{i}") for i in range(3)]),
            "label_free": ent([("results/reader_q8_course_free_eval.json", "data/course_free_reader")]),
            "label_free_same": ent([("results/reader_q8_course_vis_on_free_eval.json", "data/course_free_reader")])}


def label_free_liver(verifier="q8_liver_free"):
    """Liver world model retrained without lesion annotations (tag lt08free), K = 30, max-p; read by its own verifier
    (q8_liver_free) or by the final verifier trained on the lesion-excluding world model's candidates (lfx_score)."""
    from collect_liver_kdev import hit
    out = {}
    for which in ("t08", "msd", "ext"):
        path = f"results/reader_{verifier}_eval_{which}_fused2_free.json"
        data = next((d for d in (f"data/liver_k30_{which}_fused2_free/eval.jsonl",
                                 f"data/liver_k30_{which}_fused2_free/s/eval_{which}_fused2_free.jsonl") if os.path.exists(d)), None)
        if not (os.path.exists(path) and data):
            continue
        meta = {(r["uid"], r["peak"]): r for r in map(json.loads, open(data))}
        by = defaultdict(list)
        for r in json.load(open(path)):
            by[r["uid"]].append({**r, "label": meta[(r["uid"], r["peak"])]["label"]})
        out[which] = {"hit": hit(by, 30, "maxp"), "recall30": float(np.mean([any(c["label"] for c in v) for v in by.values()]))}
    return out


def mimic_densenet(run=PER, glob="densenet121"):
    """PER + DenseNet-121 on the MIMIC-CXR test split, fixed rule: equal-weight average of the two rank-normalised
    image scores (lesion = nodule or mass, every candidate read), as in score_cxr_focal_mimic.py."""
    from score_cxr_focal_mimic import image_scores, labels, ranks, read
    from score_cxr_focal_mimic import auroc as auc_of
    path = f"{R}/{run}/wmu_mimic_test.jsonl"
    if not os.path.exists(path):
        return {}
    sc = image_scores(read(path), (60,))
    gs = pd.read_csv(f"results/sota_cxr/{glob}.csv").set_index("key")
    lab = labels("test")
    out = {}
    for finding, kind, name in (("Lung Lesion", "lesion_nm", "lesion"), ("Fracture", "fracture", "fracture")):
        keys = [x for x in lab.index if x in gs.index and x in sc.index]
        y = lab.loc[keys, finding]
        keys = [x for x, o in zip(keys, y.isin([0, 1]) | y.isna()) if o]
        y = lab.loc[keys, finding].fillna(0).to_numpy() == 1
        out[name] = float(auc_of(0.5 * ranks(sc.loc[keys, (kind, 60)]) + 0.5 * ranks(gs.loc[keys, finding]), y))
    return out


def cxr_sources():
    """Other candidate sources read by the final radiograph verifier on ChestX-Det test (max-p over all candidates)."""
    b = boxes("cxdet")
    bs = b[b.split == "test"]
    out = {}
    for name, stem, k in (("patchcore", "patchcore", 30), ("patchcore45", "patchcore45", 45),
                          ("random45", "random45s", 45), ("grid", "grid", 1000)):   # random45s: per-image crc32 seeds
        path = f"{R}/{PER}/{stem}_cxdet_test.jsonl"
        if os.path.exists(path):
            res = evaluate(rows_of(path), bs, TYPES, ks=(k,), rules=("max-p",))
            out[name] = {m: float(res[m].mean()) for m in ("AUROC", "hit", "LROC")}
    path = f"{R}/{PER}/wmu_cxdet_test.jsonl"     # no verifier: the world-model score of the union list
    rows = rows_of(path)
    for r in rows:
        for t in TYPES:
            r[f"p_{t}"] = r["wm"]
    res = evaluate(rows, bs, TYPES, ks=(60,), rules=("max-p",))
    out["no_verifier_union"] = {m: float(res[m].mean()) for m in ("AUROC", "hit", "LROC")}
    return out


def main():
    out = {"cxr_per_finding": per_finding("cxdet", "test"), "nih_per_finding": per_finding("nih", None), "site": site()}
    if os.path.exists("results/cxr_curve.csv"):
        out["cxr_curve"] = pd.read_csv("results/cxr_curve.csv").to_dict("records")
    out["liver_grid"] = liver_grid()
    if os.path.exists("results/open_set_cxr.csv"):
        out["open_set"] = pd.read_csv("results/open_set_cxr.csv").to_dict("records")
    if os.path.exists("results/analysis_per.json"):
        out["analysis"] = json.load(open("results/analysis_per.json"))
    out["sweeps"] = sweeps()
    out["lung_ablation"] = lung_ablation()
    out["course"] = course()
    out["label_free"] = {"liver": label_free_liver(), "course": out["course"]["label_free"],
                         "liver_same": label_free_liver("q8_liver_t08h3_on_free"),
                         "course_same": out["course"]["label_free_same"]}
    out["mimic"] = {"per_densenet": mimic_densenet()}
    out["cxr_sources"] = cxr_sources()
    out["cxr_aug"] = json.load(open("results/cxr_aug_choice.json"))     # choose_cxr_aug.py
    json.dump(out, open("results/supp_tables.json", "w"), indent=1, default=float)
    print(json.dumps({k: (list(v)[:12] if isinstance(v, dict) else len(v)) for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
