import sys, pathlib
_HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE.parents[1]), str(_HERE)]
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.optimize import least_squares
from grey_ones import RANGES
from warp2 import render_at, warper, ROOT_P, IDENT, NAMES

N = 112
def fit_all(P, lam, schedule=(2.0,1.0,0.4,0.0)):
    warp = warper(N); sc = N/28.0
    root = render_at(ROOT_P[None], N)[0]; I = render_at(P, N)
    TH, RES = [], []
    for k in range(len(P)):
        th = IDENT.copy(); nrm = np.linalg.norm(I[k])
        for sg in schedule:
            sr = gaussian_filter(root, sg*sc) if sg else root
            st = gaussian_filter(I[k], sg*sc) if sg else I[k]
            f = lambda t: np.concatenate([(warp(sr,t)-st).ravel(), lam*nrm*(t-IDENT)])
            th = least_squares(f, th, method="lm", xtol=1e-13, ftol=1e-13, max_nfev=6000).x
        TH.append(th); RES.append(np.linalg.norm(warp(root,th)-I[k])/nrm)
    return np.array(TH), np.array(RES), root, I

n, lam = int(sys.argv[1]), float(sys.argv[2])
rng = np.random.default_rng(11)
P = RANGES[:,0] + rng.random((n,5))*(RANGES[:,1]-RANGES[:,0])
TH, RES, root, I = fit_all(P, lam)
warp = warper(N)
print(f"=== grid {N}, n={n}, ridge lam={lam} ===")
print(f"full 12-param warp fit residual: median {np.median(RES):.4f} max {RES.max():.4f}")
D = TH - IDENT
print("mean |coeff|: " + " ".join(f"x<-{nm}:{np.abs(D[:,i]).mean():.4f}" for i,nm in enumerate(NAMES)))
print("              " + " ".join(f"y<-{nm}:{np.abs(D[:,6+i]).mean():.4f}" for i,nm in enumerate(NAMES)))

tr, te = slice(0,n//2), slice(n//2,n)
# --- warp-space PCA ---
mw = TH[tr].mean(0); Aw = TH[tr]-mw
Uw,Sw,Vtw = np.linalg.svd(Aw, full_matrices=False)
print("warp-coeff singular values: " + " ".join(f"{v:.2e}" for v in Sw))
# --- pixel-space PCA ---
F = I.reshape(n,-1); mp = F[tr].mean(0); Ap = F[tr]-mp
Up,Sp,Vtp = np.linalg.svd(Ap, full_matrices=False)

nrm = np.linalg.norm(F[te]-0)
print(f"\n{'k':>3}  {'warp-space rank-k':>18}  {'pixel-space rank-k':>19}")
Dte = F[te]-mp
for k in (1,2,3,4,5,6,7,8):
    Tk = mw + (TH[te]-mw) @ Vtw[:k].T @ Vtw[:k]
    rec = np.stack([warp(root, t) for t in Tk]).reshape(len(Tk),-1)
    ew = np.linalg.norm(rec-F[te])/np.linalg.norm(F[te])
    recp = mp + Dte @ Vtp[:k].T @ Vtp[:k]
    ep = np.linalg.norm(recp-F[te])/np.linalg.norm(F[te])
    print(f"{k:>3}  {ew:>18.4f}  {ep:>19.4f}")
print(f"\nfloor: full 12-param warp (no truncation) = "
      f"{np.linalg.norm(np.stack([warp(root,t) for t in TH[te]]).reshape(len(TH[te]),-1)-F[te])/np.linalg.norm(F[te]):.4f}")
