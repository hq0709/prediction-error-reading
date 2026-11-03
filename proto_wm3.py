import numpy as np
from multiprocessing import Pool
from scipy import ndimage
from proto_wm import domain, peaks, DIRS
from perr.study import load_luna, pilot_cases
from perr.score import score_peaks

def crop(s, dom, margin):
    zz = np.nonzero(dom.any(axis=(1,2)))[0]; yy = np.nonzero(dom.any(axis=(0,2)))[0]; xx = np.nonzero(dom.any(axis=(0,1)))[0]
    return (slice(max(0,zz.min()-margin), zz.max()+margin+1), slice(max(0,yy.min()-margin), yy.max()+margin+1), slice(max(0,xx.min()-margin), xx.max()+margin+1))

def strict_continuity(s, dom, scales=(5,10,15), cap=100):
    sl = crop(s, dom, 2); V = np.clip(s.vol[sl].astype(np.float32), -1000, cap); D = dom[sl]
    sp = np.array(s.spacing); pad = int(np.ceil(max(scales)/sp.min()))+1
    P = np.pad(V, pad, mode="edge"); best = np.full(V.shape, -1e9, np.float32)
    for d in scales:
        E = np.full(V.shape, -1e9, np.float32)
        for e in DIRS:
            ph = np.array(e,float)*sp; ph/=np.linalg.norm(ph); o = np.round(d*ph/sp).astype(int)
            a = P[pad-o[0]:pad-o[0]+V.shape[0], pad-o[1]:pad-o[1]+V.shape[1], pad-o[2]:pad-o[2]+V.shape[2]]
            b = P[pad+o[0]:pad+o[0]+V.shape[0], pad+o[1]:pad+o[1]+V.shape[1], pad+o[2]:pad+o[2]+V.shape[2]]
            np.maximum(E, np.minimum(a, b), out=E)
        np.maximum(best, V - E, out=best)
    R = np.zeros(s.vol.shape, np.float32); R[sl] = np.where(D, np.clip(best,0,None), 0); return R

def hessian_dot(s, dom, sigmas_mm=(1.5, 2.5, 4.0), cap=200):
    sl = crop(s, dom, 8); V = np.clip(s.vol[sl].astype(np.float32), -1000, cap); D = dom[sl]
    sp = np.array(s.spacing); idx = np.nonzero(D); best = np.zeros(len(idx[0]), np.float32)
    for sm in sigmas_mm:
        sig = sm / sp
        H = {}
        for (i,j) in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]:
            order = [0,0,0]; order[i]+=1; order[j]+=1
            H[(i,j)] = ndimage.gaussian_filter(V, sig, order=order)[idx] * (sm**2) / (sp[i]*sp[j])
        M = np.empty((len(idx[0]),3,3), np.float32)
        for (i,j),v in H.items(): M[:,i,j]=v; M[:,j,i]=v
        lam = np.linalg.eigvalsh(M)            # ascending: lam[:,0] most negative
        l1, l2, l3 = lam[:,0], lam[:,1], lam[:,2]   # bright blob: all negative; |l1|>=|l2|>=|l3|
        dot = np.where((l1<0)&(l2<0)&(l3<0), (l3**2)/np.maximum(-l1,1e-3), 0)   # Li et al. 2003 dot enhancement
        np.maximum(best, dot.astype(np.float32), out=best)
    R = np.zeros(s.vol.shape, np.float32); tmp = np.zeros(D.shape, np.float32); tmp[idx]=best; R[sl]=tmp; return R

VARIANTS = {"strict_cont": strict_continuity, "hessian_dot": hessian_dot}
def work(uid):
    s = load_luna(uid); dom = domain(s); res={}
    for name, fn in VARIANTS.items():
        pk = peaks(fn(s, dom), s.spacing, 25)
        res[name] = [score_peaks(s, pk, k) for k in (1,3,5,10,25)]
    return res
if __name__ == "__main__":
    with Pool(25) as p: rs = p.map(work, pilot_cases(50))
    for name in VARIANTS: print(name, "top1/3/5/10/25:", np.round(np.array([r[name] for r in rs]).mean(0),2))
