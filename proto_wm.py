"""Prototype world-model variants; report top-k recall of >=6 mm nodules on the 50 pilot cases."""
import sys, numpy as np
from multiprocessing import Pool
from scipy import ndimage
from perr.study import load_luna, pilot_cases
from perr.score import score_peaks

DIRS = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,-1,0),(1,0,1),(1,0,-1),(0,1,1),(0,1,-1),(1,1,1),(1,1,-1),(1,-1,1),(1,-1,-1)]

def domain(s):
    m = s.lung > 0
    m = ndimage.binary_closing(m, structure=np.ones((3,7,7)), iterations=1)
    m = np.stack([ndimage.binary_fill_holes(sl) for sl in m])
    return m

def residual(s, dirs, scales, dom):
    zz, yy, xx = np.nonzero(dom.any(axis=(1,2)))[0], np.nonzero(dom.any(axis=(0,2)))[0], np.nonzero(dom.any(axis=(0,1)))[0]
    sp = np.array(s.spacing)
    pad = int(np.ceil(max(scales) / sp.min())) + 1
    z0,z1 = max(0, zz.min()-2), zz.max()+3; y0,y1 = max(0,yy.min()-2), yy.max()+3; x0,x1 = max(0,xx.min()-2), xx.max()+3
    V = s.vol[z0:z1, y0:y1, x0:x1].astype(np.float32)
    D = dom[z0:z1, y0:y1, x0:x1]
    P = np.pad(V, pad, mode="edge")
    best = np.full(V.shape, -1e9, np.float32)
    for d in scales:
        E = np.full(V.shape, -1e9, np.float32)
        for e in dirs:
            u = np.array(e, float) / sp; u = u / np.linalg.norm(np.array(e, float))  # direction in voxel units per mm... normalise physically below
            ph = np.array(e, float) * sp; ph = ph / np.linalg.norm(ph)            # unit vector in mm
            o = np.round(d * ph / sp).astype(int)
            a = P[pad-o[0]:pad-o[0]+V.shape[0], pad-o[1]:pad-o[1]+V.shape[1], pad-o[2]:pad-o[2]+V.shape[2]]
            b = P[pad+o[0]:pad+o[0]+V.shape[0], pad+o[1]:pad+o[1]+V.shape[1], pad+o[2]:pad+o[2]+V.shape[2]]
            np.maximum(E, (a + b) / 2, out=E)
        np.maximum(best, V - E, out=best)
    R = np.where(D, np.clip(best, 0, None), 0).astype(np.float32)
    full = np.zeros(s.vol.shape, np.float32); full[z0:z1, y0:y1, x0:x1] = R
    return full

def peaks(R, sp, k=10, nms=10.0):
    out=[]; W=R.copy(); r=[max(1,int(np.ceil(nms/x))) for x in sp]
    for _ in range(k):
        i=int(np.argmax(W)); z,y,x=np.unravel_index(i,W.shape); v=float(W[z,y,x])
        if v<=0: break
        out.append((z,y,x,v)); W[max(0,z-r[0]):z+r[0]+1, max(0,y-r[1]):y+r[1]+1, max(0,x-r[2]):x+r[2]+1]=0
    return out

VARIANTS = {
  "z6":      ([(1,0,0)], [6]),
  "d13_6":   (DIRS, [6]),
  "d13_6_12":(DIRS, [6,12]),
  "d13_6_12_18": (DIRS, [6,12,18]),
}
def work(uid):
    s = load_luna(uid); dom = domain(s)
    res = {}
    for name,(dirs,scales) in VARIANTS.items():
        pk = peaks(residual(s, dirs, scales, dom), s.spacing)
        res[name] = [score_peaks(s, pk, k) for k in (1,3,5,10)]
    return res

if __name__ == "__main__":
    with Pool(25) as p: rs = p.map(work, pilot_cases(50))
    for name in VARIANTS:
        a = np.array([r[name] for r in rs]).mean(0)
        print(name, "top1/3/5/10 recall:", np.round(a,2))
