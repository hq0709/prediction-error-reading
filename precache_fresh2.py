from multiprocessing import Pool
from perr.study import load_luna, pilot_cases
from perr.worldmodels import ShapePrior, lung_domain
def work(uid):
    s = load_luna(uid); lung_domain(s); ShapePrior().peaks(s); return uid
if __name__ == "__main__":
    with Pool(16) as p: r = p.map(work, pilot_cases(260)[160:260])
    print("cached", len(r))
