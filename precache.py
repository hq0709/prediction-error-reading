"""Cache volumes, dilated lung masks and continuity-surprise maps for the pilot cases."""
import sys
from multiprocessing import Pool
from perr.study import load_luna, pilot_cases
from perr.worldmodels import Continuity, lung_mask_dilated

def work(uid):
    s = load_luna(uid)
    lung_mask_dilated(s)
    sur = Continuity().surprise(s)
    return uid, s.shape, len(s.nodules), len(sur["peaks"])

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    with Pool(16) as p:
        for r in p.imap_unordered(work, pilot_cases(n)):
            print(*r, flush=True)
