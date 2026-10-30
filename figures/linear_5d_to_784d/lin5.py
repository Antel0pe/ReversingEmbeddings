import numpy as np
from grey_ones import render, RANGES, KNOBS

rng = np.random.default_rng(0)

def samp(n, ranges):
    return ranges[:,0] + rng.random((n,5))*(ranges[:,1]-ranges[:,0])

def fit_report(ranges, label, n_tr=20000, n_te=5000):
    Ptr = samp(n_tr, ranges); Pte = samp(n_te, ranges)
    Xtr = render(Ptr, dtype=np.float64).reshape(n_tr,-1)
    Xte = render(Pte, dtype=np.float64).reshape(n_te,-1)
    # affine: [P,1] @ A = X
    Atr = np.hstack([Ptr, np.ones((n_tr,1))])
    Ate = np.hstack([Pte, np.ones((n_te,1))])
    W,*_ = np.linalg.lstsq(Atr, Xtr, rcond=None)
    for nm, A, X in (("train",Atr,Xtr),("test",Ate,Xte)):
        R = A@W - X
        ss_res = (R**2).sum(); ss_tot = ((X - Xtr.mean(0))**2).sum()
        r2 = 1-ss_res/ss_tot
        relnorm = np.sqrt(ss_res/ (X**2).sum())
        print(f"{label:>14} {nm:5}  R2={r2:.6f}  ||res||/||X||={relnorm:.4f}  "
              f"max|pix err|={np.abs(R).max():.4f}  rms pix err={np.sqrt((R**2).mean()):.4f}")
    # ceiling: best rank-5 affine subspace at all (PCA), applies to ANY map from 5 numbers
    mu = Xtr.mean(0); Xc = Xtr-mu
    s = np.linalg.svd(Xc, compute_uv=False)
    tot = (s**2).sum()
    frac = np.cumsum(s**2)/tot
    print(f"{label:>14} PCA    variance kept by best k-dim affine subspace: " +
          ", ".join(f"k={k}:{frac[k-1]*100:.4f}%" for k in (1,2,3,4,5,6,8,10,15,20,30,50)))
    need = int(np.searchsorted(frac, 0.999999)+1)
    print(f"{label:>14} PCA    dims needed for 99.9999% of variance: {need}")
    return s, frac

print("=== full default range box ===")
fit_report(RANGES, "full box")

print()
print("=== shrunken boxes around root 1 (14,14,19.75,3.2,0) ===")
root = np.array([14.0,14.0,19.75,3.2,0.0])
half = (RANGES[:,1]-RANGES[:,0])/2
for f in (1.0, 0.1, 0.01, 0.001):
    r = np.stack([root-half*f, root+half*f],1)
    fit_report(r, f"scale {f:g}", 8000, 2000)
