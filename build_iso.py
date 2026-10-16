"""Build the 1 mm cache of every LUNA16 study used anywhere (training pool and all evaluation sets)."""
from multiprocessing import Pool

from perr.context_wm import build_iso
from perr.study import pilot_cases

if __name__ == "__main__":
    uids = open("results/train_uids.txt").read().split() + pilot_cases(260)
    with Pool(40) as p:
        for i, _ in enumerate(p.imap_unordered(build_iso, uids, chunksize=1)):
            if i % 100 == 0:
                print(i, flush=True)
    print("done", len(uids))
