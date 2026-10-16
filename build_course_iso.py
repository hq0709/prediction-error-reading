"""Build the 1.5 mm cache of every autoPET pair (training, validation and test patients)."""
from multiprocessing import Pool

from perr.autopet import pairs
from perr.course_wm import build_iso

if __name__ == "__main__":
    keys = pairs()
    with Pool(32) as p:
        p.map(build_iso, keys, chunksize=1)
    print("done", len(keys))
