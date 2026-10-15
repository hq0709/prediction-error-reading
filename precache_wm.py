import sys
from multiprocessing import Pool
from perr.study import load_luna, pilot_cases
from perr.worldmodels import ShapePrior, lung_domain
from perr.score import score_peaks
def work(uid):
    s = load_luna(uid); lung_domain(s); wm = ShapePrior(); pk = wm.peaks(s)
    return [score_peaks(s, pk, k) for k in (1, 3, 5, 10)]
if __name__ == "__main__":
    import numpy as np
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    with Pool(25) as p: rs = p.map(work, pilot_cases(n))
    print("shape WM top1/3/5/10 recall:", np.round(np.mean(rs, 0), 2))
