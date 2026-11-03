import numpy as np
from multiprocessing import Pool
from proto_wm import domain, residual, peaks, DIRS
from perr.study import load_luna, pilot_cases
from perr.score import score_peaks

class Clipped:
    def __init__(self, s, hi): self.s=s; self.hi=hi
    def __getattr__(self, k): return getattr(self.s, k)
    @property
    def vol(self): return np.clip(self.s.vol, -1000, self.hi)

VARIANTS = {
  "clip100_d13_6_12":     (100, DIRS, [6,12]),
  "clip100_d13_6_12_18":  (100, DIRS, [6,12,18]),
  "clip0_d13_6_12_18":    (0, DIRS, [6,12,18]),
  "clip100_z_6_12_18":    (100, [(1,0,0)], [6,12,18]),
}
def work(uid):
    s0 = load_luna(uid); dom = domain(s0); res={}
    for name,(hi,dirs,scales) in VARIANTS.items():
        s = Clipped(s0, hi)
        pk = peaks(residual(s, dirs, scales, dom), s0.spacing, 10)
        res[name] = [score_peaks(s0, pk, k) for k in (1,3,5,10)]
    return res
if __name__ == "__main__":
    with Pool(25) as p: rs = p.map(work, pilot_cases(50))
    for name in VARIANTS:
        print(name, "top1/3/5/10:", np.round(np.array([r[name] for r in rs]).mean(0),2))
