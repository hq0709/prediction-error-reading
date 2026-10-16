"""Helper for precache_train.py: walks the training list from the other end and skips any scan already started."""
import os
from multiprocessing import Pool
from perr.study import CACHE, load_luna
from perr.worldmodels import ShapePrior, lung_domain

def work(uid):
    if os.path.exists(f"{CACHE}/{uid}.vol.npy"):
        return uid, "skip"
    try:
        s = load_luna(uid); lung_domain(s); ShapePrior().peaks(s); return uid, True
    except Exception as e:
        return uid, repr(e)[:100]

if __name__ == "__main__":
    train = open("results/train_uids.txt").read().split()[::-1]
    with Pool(20) as p:
        res = list(p.imap(work, train, chunksize=1))
    print("done:", sum(r[1] is True for r in res), "skipped:", sum(r[1] == "skip" for r in res),
          "failures:", [r for r in res if r[1] not in (True, "skip")][:5], flush=True)
