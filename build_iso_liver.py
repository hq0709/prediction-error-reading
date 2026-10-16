"""Build the 1 mm cache of every MSD Task03 liver case (training pool and evaluation half)."""
from multiprocessing import Pool

from perr.context_wm import build_iso
from perr.liver import cases, load_liver


def work(c):
    load_liver(c)                 # makes sure the liver cache (volume, masks) exists
    build_iso(c, "liver")


if __name__ == "__main__":
    cs = cases()
    with Pool(24) as p:
        p.map(work, cs, chunksize=1)
    print("done", len(cs))
