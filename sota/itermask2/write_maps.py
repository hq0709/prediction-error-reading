"""Read-out variants of the IterMask2 anomaly map (the official map: squared error between the radiograph and its final
reconstruction, cache/cxr_maps/itermask2_raw/<key>.npy, 512 x 512) for the address search: Gaussian smoothing of sigma
px on the 512 canvas before peak finding (sigma 0 = the official map unchanged). The only free inference setting; it is
chosen on the verifier's held-out tenth by focal-box recall@45 (dev_recall.py), and the chosen variant is written to
cache/cxr_maps/itermask2/<key>.npy for every image.

    python sota/itermask2/write_maps.py --sigmas 0 2 4 8 --keys sota/itermask2/dev_keys.txt
    python sota/itermask2/write_maps.py --final 4 --keys sota/itermask2/dev_keys.txt sota/itermask2/test_keys.txt
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
from scipy import ndimage as ndi

MAPS = "/raid/home/CAMCA/hj880/perr/cache/cxr_maps"


def one(job):
    key, sigma, name, src = job
    M = np.load(f"{MAPS}/{src}/{key}.npy").astype(np.float32)
    if sigma > 0:
        M = ndi.gaussian_filter(M, sigma)
    p = f"{MAPS}/{name}/{key}.npy"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p, M.astype(np.float16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="+", required=True)
    ap.add_argument("--sigmas", type=float, nargs="*", default=[])
    ap.add_argument("--final", type=float, default=None, help="the chosen sigma: written as cache/cxr_maps/itermask2")
    ap.add_argument("--src", default="itermask2_raw", help="maps to read (checkpoint checks: another folder)")
    ap.add_argument("--prefix", default="itermask2", help="output folder prefix of the variants")
    a = ap.parse_args()
    keys = [k for f in a.keys for k in open(f).read().split()]
    jobs = [(k, s, f"{a.prefix}_s{s:g}", a.src) for s in a.sigmas for k in keys]
    if a.final is not None:
        jobs += [(k, a.final, "itermask2", a.src) for k in keys]
    with Pool(32) as p:
        p.map(one, jobs, chunksize=8)
    print(f"{len(jobs)} maps written", flush=True)


if __name__ == "__main__":
    main()
