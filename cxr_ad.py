"""MIMIC-CXR normal vs abnormal from address maps, read inside the thorax (the protocol of train_cxr_latent_wm.py:
validation and test images that are "No Finding" or have a positive finding; configuration chosen on validation).

Image score of a map = mean of its top q fraction inside the domain. Maps: the pixel world model at each scale
(precision-weighted error, smoothing sigma, brighter-than-expected or both signs) and the scales combined (each scale's
map divided by its 99th percentile over the domain of "No Finding" validation images, then the maximum over scales);
the multi-scale latent world model; PatchCore.

    python cxr_ad.py cxr,cxr2,cxr4
"""
import json
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from perr.cxr import CACHE, FINDINGS
from perr.cxr_anat import search_domain
from perr.cxr_wm import error_map, valid_region

ROOT = "/raid/home/CAMCA/hj880/perr"
WM, MAPS = f"{ROOT}/cache/cxr_wm", f"{ROOT}/cache/cxr_maps"
QS = (0.005, 0.02, 0.05, 0.1, 0.2)
SIGMAS = (4, 8, 16)
TAGS = ("cxr",)
EXTRA = ()
DOMS = {"radiograph": None, "thorax": ("organs", 8)}


def auroc(s, y):
    """Mann-Whitney AUROC (train_cxr_latent_wm.auroc, without its torch import)."""
    s, y = np.asarray(s, float), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def up(a):
    a = np.asarray(a, np.float32)
    return ndi.zoom(a, 512 / a.shape[-1], order=1)


def maps(key):
    out = {}
    for t in TAGS:
        R = np.load(f"{WM}/{key}.{t}R.npy").astype(np.float32)
        S = np.load(f"{WM}/{key}.{t}S.npy").astype(np.float32)
        for sg in SIGMAS:
            out[f"{t}_s{sg}_pos"] = error_map(R, S, sg, 1)
            out[f"{t}_s{sg}_abs"] = error_map(R, S, sg, 0)
    lat = np.load(f"{MAPS}/latent_ms/{key}.npy").astype(np.float32)
    out["latent_raw_sum"] = up(sum(lat[si, 0] / np.median(lat[si, 0]) for si in range(3)))
    out["latent_raw6"] = up(lat[1, 0])
    out["patchcore"] = up(np.load(f"{MAPS}/patchcore/{key}.npy"))
    for name in EXTRA:                                  # further 37 x 37 maps in cache/cxr_maps/<name>
        out[name] = up(np.load(f"{MAPS}/{name}/{key}.npy"))
    return out


def topq(v, q):
    v = np.sort(v)[::-1]
    return float(v[:max(1, int(len(v) * q))].mean())


def work(key):
    img = np.load(f"{CACHE}/{key}.npy")
    V = valid_region(img)
    doms = {d: V if spec is None else search_domain(key, V, *spec) for d, spec in DOMS.items()}
    row, vals = {"key": key}, {}
    for name, W in maps(key).items():
        for d, D in doms.items():
            v = W[D]
            vals[(name, d)] = v
            for q in QS:
                row[f"{name}|{d}|{q}"] = topq(v, q)
    # 99th percentiles for the scale combination (used for "No Finding" validation images only)
    row.update({f"p99|{n}|{d}": float(np.percentile(v, 99)) for (n, d), v in vals.items()})
    return row


def combined(key, norms, names):
    M, doms = work_maps(key)
    out = {}
    for combo, parts in names.items():
        for d, D in doms.items():
            C = np.max([M[p] / norms[(p, d)] for p in parts], 0)[D]
            for q in QS:
                out[f"{combo}|{d}|{q}"] = topq(C, q)
    return {"key": key, **out}


def work_maps(key):
    img = np.load(f"{CACHE}/{key}.npy")
    V = valid_region(img)
    doms = {d: V if spec is None else search_domain(key, V, *spec) for d, spec in DOMS.items()}
    M = {k: v for k, v in maps(key).items() if k.startswith(tuple(TAGS))}
    return M, doms


def main():
    global TAGS, EXTRA
    if "--extra" in sys.argv:
        i = sys.argv.index("--extra")
        EXTRA = tuple(sys.argv[i + 1].split(","))
        del sys.argv[i:i + 2]
    TAGS = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else TAGS
    d = pd.read_csv("results/cxr_mimic.csv")
    others = [f for f in FINDINGS if f != "No Finding"]
    ev = d[d.split.isin(["validate", "test"])].copy()
    ev["abnormal"] = (ev[others] == 1).any(axis=1).astype(int)
    ev = ev[(ev["No Finding"] == 1) | (ev.abnormal == 1)].reset_index(drop=True)
    with Pool(48) as p:
        rows = list(p.imap(work, ev.key.tolist(), chunksize=4))
    sc = ev.merge(pd.DataFrame(rows), on="key")
    # scale combinations: per scale normaliser = median over "No Finding" validation images of their 99th percentile
    nfv = sc[(sc.split == "validate") & (sc["No Finding"] == 1)]
    if len(TAGS) > 1:
        combos = {}
        for sg in SIGMAS:
            for sign in ("pos", "abs"):
                parts = [f"{t}_s{sg}_{sign}" for t in TAGS]
                combos[f"max({'+'.join(TAGS)})_s{sg}_{sign}"] = parts
        norms = {(p, dd): float(nfv[f"p99|{p}|{dd}"].median()) for parts in combos.values() for p in parts for dd in DOMS}
        json.dump({f"{p}|{dd}": v for (p, dd), v in norms.items()}, open("results/cxr_scale_norms.json", "w"), indent=1)
        with Pool(48) as p:
            crow = p.starmap(combined, [(k, norms, combos) for k in sc.key], chunksize=4)
        sc = sc.merge(pd.DataFrame(crow), on="key")
    feats = [c for c in sc.columns if c.count("|") == 2 and not c.startswith("p99")]
    sc[["key", "split", "abnormal", "No Finding"] + others + feats].to_csv(f"results/cxr_ad_{'_'.join(TAGS + EXTRA)}.csv", index=False)
    val, test = sc[sc.split == "validate"], sc[sc.split == "test"]
    res = pd.DataFrame([{"feature": c, "method": c.split("|")[0], "domain": c.split("|")[1], "q": c.split("|")[2],
                         "val": auroc(val[c], val.abnormal), "test": auroc(test[c], test.abnormal)} for c in feats])
    res["family"] = [m.split("_s")[0] if m.startswith(TAGS) or m.startswith("max") else m for m in res.method]
    best = res.loc[res.groupby(["family", "domain"]).val.idxmax()].sort_values(["domain", "val"], ascending=[True, False])
    pd.set_option("display.width", 200)
    print("best configuration per family and domain (chosen on validation):")
    print(best[["family", "domain", "method", "q", "val", "test"]].to_string(index=False, float_format="%.3f"))
    nfr = test[test["No Finding"] == 1]
    for _, r in best[best.domain == "thorax"].iterrows():
        line = []
        for f in others:
            pos = test[test[f] == 1]
            if len(pos) >= 20:
                sub = pd.concat([pos.assign(y=1), nfr.assign(y=0)])
                line.append(f"{f[:8]}={auroc(sub[r.feature], sub.y):.2f}")
        print(f"  {r.family:22s}", " ".join(line))


if __name__ == "__main__":
    main()
