import numpy as np
from multiprocessing import Pool
from proto_wm import domain, residual, peaks, DIRS
from perr.study import load_luna, pilot_cases
from perr.score import score_peaks
def work(uid):
    s = load_luna(uid); dom = domain(s)
    R = residual(s, DIRS, [6,12,18], dom)
    pk = peaks(R, s.spacing, 100)
    return [score_peaks(s, pk, k) for k in (10,25,50,100)], [int(s.vol[int(z),int(y),int(x)]>300) for z,y,x,_ in pk[:10]]
if __name__=="__main__":
    with Pool(25) as p: rs = p.map(work, pilot_cases(50))
    print("recall@10/25/50/100:", np.round(np.array([r[0] for r in rs]).mean(0),2))
    print("fraction of top-10 peaks on >300 HU voxels:", np.round(np.mean([r[1] for r in rs]),2))
