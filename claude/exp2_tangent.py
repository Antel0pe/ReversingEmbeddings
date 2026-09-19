"""E4: are the knobs the intrinsic directions?  At each real 1, the Jacobian of the generator
(one 784-vector per knob) spans a candidate tangent space. Measure how much of the displacement
to the real neighbours lies in that span, against controls with the same number of directions.
E5: what is the leftover made of, and where does it sit?"""
import numpy as np
from scipy.ndimage import binary_dilation
from common import *
X, y, ones, P, E = load()
R = np.load("claude/renders.npz")
n = len(ones); rng = np.random.default_rng(0)
D = pdist(ones, ones); np.fill_diagonal(D, np.inf)
K = 10
NN = np.argsort(D, 1)[:, :K]
mu = ones.mean(0); Z = ones - mu
w, Vg = np.linalg.eigh(Z.T @ Z); Vg = Vg[:, ::-1]           # global PCs

def captured(Delta, Q):
    return ((Delta @ Q) ** 2).sum() / (Delta ** 2).sum()

sample = rng.choice(n, 1200, replace=False)
print(f"E4  fraction of neighbour displacement (10 nearest real 1s) captured by k directions, median over {len(sample)} 1s\n")
print(f"{'rung':>9s} {'k':>3s} {'knob Jacobian':>14s} {'global PCs':>11s} {'random':>8s} {'local PCA(20nn)':>16s}  | render-to-render: Jacobian")
for r in RUNGS:
    k = len(ladder.KNOBS[r]); cj, cg, cr, cl, crr = [], [], [], [], []
    for i in sample:
        J = ladder.jacobian(P[r][i], r)
        Qj = np.linalg.qr(J)[0]
        Delta = ones[NN[i]] - ones[i]
        cj.append(captured(Delta, Qj))
        cg.append(captured(Delta, Vg[:, :k]))
        Qr = np.linalg.qr(rng.normal(size=(784, k)))[0]; cr.append(captured(Delta, Qr))
        nn20 = np.argsort(D[i])[:20]; Zl = ones[nn20] - ones[nn20].mean(0)
        Ql = np.linalg.svd(Zl, full_matrices=False)[2][:k].T; cl.append(captured(Delta, Ql))
        Drr = R[r][NN[i]] - R[r][i]; crr.append(captured(Drr, Qj))
    print(f"{r:>9s} {k:3d} {100*np.median(cj):13.1f}% {100*np.median(cg):10.1f}% {100*np.median(cr):7.1f}% {100*np.median(cl):15.1f}%  | {100*np.median(crr):.1f}%")

print("\n    (local PCA of the same neighbourhood is in-sample: it sees the neighbours it is scored on.)")

# which knob differs between a 1 and its nearest neighbour?
r = "foot"; names = ladder.KNOBS[r]; Pr = P[r]
dP = np.abs(Pr[NN[:, 0]] - Pr) / Pr.std(0)
print("\n    knob change between a 1 and its nearest neighbour, in units of that knob's population std (median):")
print("    " + "  ".join(f"{nm} {np.median(dP[:, j]):.2f}" for j, nm in enumerate(names)))

print("\nE5  the leftover: real minus best render, rung 'foot'")
res = ones - R["foot"]
rn = np.linalg.norm(res, axis=1)
edge = np.array([binary_dilation((im > 0.02) & (im < 0.98), iterations=1).ravel() for im in R["foot"].reshape(-1, 28, 28)])
frac_energy = np.array([(res[i][edge[i]] ** 2).sum() / max((res[i] ** 2).sum(), 1e-12) for i in range(n)])
print(f"    leftover norm median {np.median(rn):.2f}; nearest-neighbour spacing median {np.median(D.min(1)):.2f}")
print(f"    share of leftover energy on the stroke's EDGE pixels: median {100*np.median(frac_energy):.0f}%  "
      f"(edge pixels are {100*edge.mean():.0f}% of the canvas)")
# how many flat directions does the leftover need?
wr = np.linalg.eigh((res - res.mean(0)).T @ (res - res.mean(0)))[0][::-1]; cum = np.cumsum(wr) / wr.sum()
print(f"    PCs for 95% of leftover variance: {np.searchsorted(cum, .95)+1}   (real 1s: 71, straight 1s: 50 in your notebooks)")
# split of neighbour displacement into along-surface vs leftover
j = NN[:, 0]
along = np.linalg.norm(R["foot"][j] - R["foot"], axis=1); wob = np.linalg.norm(res[j] - res, axis=1)
print(f"    nearest-neighbour displacement: median {np.median(D.min(1)):.2f} = along the surface {np.median(along):.2f} + leftover change {np.median(wob):.2f} (medians, vectors add in quadrature)")
np.savez("claude/tangent.npz", NN=NN, sample=sample)
