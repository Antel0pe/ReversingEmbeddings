import sys, pathlib
_HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE.parents[1]), str(_HERE)]
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.optimize import least_squares
from grey_ones import RANGES
from warp2 import render_at, make_basis, ROOT_P, IDENT, NAMES

N=112; SC=N/28.0; B=make_basis(N)
def warp(src, th, order=1):
    x0=(14+14*np.einsum('k,kij->ij',th[:6],B))*SC-0.5
    y0=(14+14*np.einsum('k,kij->ij',th[6:],B))*SC-0.5
    return map_coordinates(src,[y0,x0],order=order,mode="constant",cval=0.0)

def exact_theta(P):
    """Backward-warp coefficients implied by the affine-orbit model, from true knobs."""
    cx,cy,h,w,lean = P.T; s=np.tan(np.radians(lean))
    rx, ry = 3.2/w, 19.75/h
    th=np.zeros((len(P),12))
    th[:,1]=rx; th[:,2]=rx*s; th[:,0]=-rx*((cx-14)/14)-rx*s*((cy-14)/14)
    th[:,8]=ry; th[:,6]=-ry*((cy-14)/14)
    return th

n=120; lam=0.05
rng=np.random.default_rng(11)
P=RANGES[:,0]+rng.random((n,5))*(RANGES[:,1]-RANGES[:,0])
root=render_at(ROOT_P[None],N)[0]; I=render_at(P,N); F=I.reshape(n,-1)
TH=[]
for k in range(n):
    th=IDENT.copy(); nrm=np.linalg.norm(I[k])
    for sg in (2.0,1.0,0.4,0.0):
        sr=gaussian_filter(root,sg*SC) if sg else root
        st=gaussian_filter(I[k],sg*SC) if sg else I[k]
        f=lambda t: np.concatenate([(warp(sr,t)-st).ravel(), lam*nrm*(t-IDENT)])
        th=least_squares(f,th,method="lm",xtol=1e-13,ftol=1e-13,max_nfev=6000).x
    TH.append(th)
TH=np.array(TH); TX=exact_theta(P)
np.savez(str(_HERE / "w4.npz"),TH=TH,TX=TX,P=P)

def relerr(TT, order=1):
    R=np.stack([warp(root,t,order) for t in TT]).reshape(len(TT),-1)
    return np.linalg.norm(R-F)/np.linalg.norm(F)

print("CONTROL 1 -- does the 5-coefficient affine warp actually reproduce the images?")
print(f"  exact affine warp of root, bilinear  rel err = {relerr(TX,1):.4f}")
print(f"  exact affine warp of root, cubic     rel err = {relerr(TX,3):.4f}")
print(f"  fitted 12-param warp,     bilinear   rel err = {relerr(TH,1):.4f}")

print("\nCONTROL 2 -- rank of the EXACT coefficient vectors (claim: exactly 5)")
sx=np.linalg.svd(TX-TX.mean(0),compute_uv=False)
print("  " + " ".join(f"{v:.2e}" for v in sx))

print("\nCONTROL 3 -- rank of the FITTED coefficient vectors")
sf=np.linalg.svd(TH-TH.mean(0),compute_uv=False)
print("  " + " ".join(f"{v:.2e}" for v in sf))

print("\nCONTROL 4 -- zero the 7 coefficients the model says must vanish, in the FIT")
keep=[0,1,2,6,8]
TP=np.zeros_like(TH); TP[:,keep]=TH[:,keep]
print(f"  fitted, 7 junk coeffs zeroed         rel err = {relerr(TP,1):.4f}")
print(f"  (fitted, untouched                   rel err = {relerr(TH,1):.4f})")
print("\n  recovered vs exact, mean |error| on the 5 real coefficients:")
for i in keep:
    nm = (f"x<-{NAMES[i]}" if i<6 else f"y<-{NAMES[i-6]}")
    print(f"    {nm:7s} exact mean {TX[:,i].mean():+.4f}   fit error {np.abs(TH[:,i]-TX[:,i]).mean():.5f}")
print("  mean |value| of the 7 that should be zero:")
for i in [3,4,5,7,9,10,11]:
    nm = (f"x<-{NAMES[i]}" if i<6 else f"y<-{NAMES[i-6]}")
    print(f"    {nm:7s} {np.abs(TH[:,i]).mean():.5f}")

print("\nCONTROL 5 -- can 5 numbers predict all 12 fitted coefficients? (cubic fit, held out)")
import itertools
Z=(TH-TH.mean(0))/TH.std(0).clip(1e-9)
U,S,Vt=np.linalg.svd(Z-Z.mean(0),full_matrices=False)
for k in (3,4,5,6):
    L=(Z@Vt[:k].T)
    cols=[np.ones(n)]+[np.prod(L[:,c],1) for d in (1,2,3) for c in itertools.combinations_with_replacement(range(k),d)]
    Bm=np.stack(cols,1); tr=slice(0,60); te=slice(60,120)
    co,*_=np.linalg.lstsq(Bm[tr],P[tr],rcond=None)
    e=np.abs(Bm[te]@co-P[te]).max(0)
    print(f"  {k} warp-PCA scores -> true knobs, held-out max err: " +
          " ".join(f"{a}={b:.3f}" for a,b in zip(["cx","cy","h","w","lean"],e)))
