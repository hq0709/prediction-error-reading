"""World-model candidates for every LUNA16 scan outside the evaluation sets (the reader's training pool)."""
import glob, os
from multiprocessing import Pool
import pandas as pd
from perr.study import load_luna, pilot_cases, LUNA
from perr.worldmodels import ShapePrior, lung_domain

def all_uids():
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{LUNA}/subset[0-9]/*.mhd"))

def work(uid):
    try:
        s = load_luna(uid); lung_domain(s); ShapePrior().peaks(s); return uid, True
    except Exception as e:
        return uid, repr(e)[:100]

if __name__ == "__main__":
    held = set(pilot_cases(260))                     # pilot 0-49, dev 50-59, fresh 60-159, fresh2 160-259
    train = [u for u in all_uids() if u not in held]
    open("results/train_uids.txt", "w").write("\n".join(train))
    print("train scans:", len(train), flush=True)
    with Pool(20) as p:
        bad = [r for r in p.imap_unordered(work, train) if r[1] is not True]
    print("done; failures:", bad[:5], len(bad))
