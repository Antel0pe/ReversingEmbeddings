import sys, pathlib
_HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE.parents[1]), str(_HERE)]
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.optimize import least_squares
from grey_ones import RANGES

ROOT_P = np.array([14.0, 14.0, 19.75, 3.2, 0.0])
NAMES = ["1", "x", "y", "x2", "xy", "y2"]
IDENT = np.array([0.,1.,0.,0.,0.,0., 0.,0.,1.,0.,0.,0.])

def render_at(P, N, SUB=16, chunk=64):
    P = np.atleast_2d(np.asarray(P, np.float64)); d = 28.0/N
    out = np.empty((len(P), N, N))
    ys_lo = np.arange(N*SUB)*(d/SUB); ys_mid = ys_lo + 0.5*d/SUB
    cols = np.arange(N)*d
    for s in range(0, len(P), chunk):
        cx,cy,h,w,lean = [P[s:s+chunk,k][:,None] for k in range(5)]
        top,bot = cy-h/2, cy+h/2
        wy = np.clip(np.minimum(ys_lo+d/SUB,bot)-np.maximum(ys_lo,top),0,d/SUB)*(SUB/d)
        centre = cx - np.tan(np.radians(lean))*(ys_mid-cy)
        L,R = centre-w/2, centre+w/2
        ov = np.clip(np.minimum(R[:,:,None],cols+d)-np.maximum(L[:,:,None],cols),0,d)/d
        out[s:s+chunk] = (ov*wy[:,:,None]).reshape(len(L),N,SUB,N).mean(2)
    return out

def make_basis(N):
    c = (np.arange(N)+0.5)*(28.0/N)
    Xn = np.broadcast_to((c-14)/14,(N,N)); Yn = np.broadcast_to(((c-14)/14)[:,None],(N,N))
    return np.stack([np.ones_like(Xn),Xn,Yn,Xn*Xn,Xn*Yn,Yn*Yn])

def warper(N):
    B = make_basis(N); sc = N/28.0
    def warp(src, th):
        x0 = (14 + 14*np.einsum('k,kij->ij', th[:6], B))*sc - 0.5
        y0 = (14 + 14*np.einsum('k,kij->ij', th[6:], B))*sc - 0.5
        return map_coordinates(src, [y0,x0], order=1, mode="constant", cval=0.0)
    return warp

def fit_all(P, N, schedule=(2.0,1.0,0.4,0.0)):
    warp = warper(N); sc = N/28.0
    root = render_at(ROOT_P[None], N)[0]; I = render_at(P, N)
    TH, RES = [], []
    for k in range(len(P)):
        th = IDENT.copy()
        for sg in schedule:
            sr = gaussian_filter(root, sg*sc) if sg else root
            st = gaussian_filter(I[k], sg*sc) if sg else I[k]
            th = least_squares(lambda t: (warp(sr,t)-st).ravel(), th, method="lm",
                               xtol=1e-13, ftol=1e-13, max_nfev=6000).x
        TH.append(th); RES.append(np.linalg.norm(warp(root,th)-I[k])/np.linalg.norm(I[k]))
    return np.array(TH), np.array(RES), root, I

if __name__ == "__main__":
    n = int(sys.argv[1]); N = int(sys.argv[2])
    rng = np.random.default_rng(7)
    P = RANGES[:,0] + rng.random((n,5))*(RANGES[:,1]-RANGES[:,0])
    TH, RES, root, I = fit_all(P, N)
    np.savez(fstr(_HERE), TH=TH, RES=RES, P=P)
    print(f"grid {N}x{N}, n={n}")
    print(f"  warp-fit residual: median {np.median(RES):.4f}  max {RES.max():.4f}")
    D = TH - IDENT
    print("  mean |coeff|:  " + "  ".join(f"x<-{nm}:{np.abs(D[:,i]).mean():.4f}" for i,nm in enumerate(NAMES)))
    print("                 " + "  ".join(f"y<-{nm}:{np.abs(D[:,6+i]).mean():.4f}" for i,nm in enumerate(NAMES)))
    sv = np.linalg.svd(D - D.mean(0), compute_uv=False)
    print("  warp-coeff singular values (identity+mean removed):")
    print("    " + "  ".join(f"{v:.3e}" for v in sv))
    print("    energy kept by first k:  " + "  ".join(
        f"k={k}:{(sv[:k]**2).sum()/(sv**2).sum():.6f}" for k in (3,4,5,6,7)))
