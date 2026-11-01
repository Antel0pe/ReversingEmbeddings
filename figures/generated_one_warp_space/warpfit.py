import sys, pathlib
_HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE.parents[1]), str(_HERE)]
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.optimize import least_squares
from grey_ones import render, RANGES

ROOT_P = [14.0, 14.0, 19.75, 3.2, 0.0]
c = np.arange(28) + 0.5
Xn = np.broadcast_to((c - 14) / 14, (28, 28))
Yn = np.broadcast_to(((c - 14) / 14)[:, None], (28, 28))
# generic degree-2 polynomial warp basis: no affine assumption supplied
BASIS = np.stack([np.ones_like(Xn), Xn, Yn, Xn*Xn, Xn*Yn, Yn*Yn])
NAMES = ["1", "x", "y", "x2", "xy", "y2"]
IDENT = np.array([0., 1., 0., 0., 0., 0.,
                  0., 0., 1., 0., 0., 0.])

def warp(src, th):
    a, b = th[:6], th[6:]
    x0 = 14 + 14 * np.einsum('k,kij->ij', a, BASIS) - 0.5
    y0 = 14 + 14 * np.einsum('k,kij->ij', b, BASIS) - 0.5
    return map_coordinates(src, [y0, x0], order=1, mode="constant", cval=0.0)

def fit(target, root, schedule=(2.0, 1.0, 0.4, 0.0)):
    th = IDENT.copy()
    for sg in schedule:
        s_r = gaussian_filter(root, sg) if sg else root
        s_t = gaussian_filter(target, sg) if sg else target
        r = least_squares(lambda t: (warp(s_r, t) - s_t).ravel(), th,
                          method="lm", xtol=1e-12, ftol=1e-12, max_nfev=4000)
        th = r.x
    return th, np.linalg.norm(warp(root, th) - target) / np.linalg.norm(target)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    rng = np.random.default_rng(7)
    P = RANGES[:, 0] + rng.random((n, 5)) * (RANGES[:, 1] - RANGES[:, 0])
    root = render([ROOT_P], dtype=np.float64)[0]
    I = render(P, dtype=np.float64)
    TH, RES = [], []
    for k in range(n):
        th, res = fit(I[k], root); TH.append(th); RES.append(res)
    TH = np.array(TH); RES = np.array(RES)
    np.savez(str(_HERE / "warps.npz"), TH=TH, RES=RES, P=P)
    print(f"n={n}  warp-fit residual: median {np.median(RES):.4f}  "
          f"90th {np.percentile(RES,90):.4f}  max {RES.max():.4f}")
    print("\nmean |coefficient| over samples (identity subtracted):")
    D = TH - IDENT
    for i, nm in enumerate(NAMES):
        print(f"   x0 <- {nm:3s} {np.abs(D[:,i]).mean():.5f}      "
              f"y0 <- {nm:3s} {np.abs(D[:,6+i]).mean():.5f}")
