from multiprocessing import Pool
from perr.study import load_luna, pilot_cases
from perr.worldmodels import ShapePrior, lung_domain
def work(uid):
    s = load_luna(uid); lung_domain(s); ShapePrior().peaks(s); return uid
if __name__ == "__main__":
    with Pool(20) as p: r = p.map(work, pilot_cases(160)[60:160])
    print("cached", len(r))
