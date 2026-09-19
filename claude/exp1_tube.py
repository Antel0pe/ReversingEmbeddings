"""E1: how thick is the tube around each rung, and does the surface itself sit inside the class?
E3: in knob coordinates, is the set of 1s (nearly) a product of independent ranges?"""
import numpy as np
from common import *
X, y, ones, P, E = load()
n = len(ones)
print(f"{n} training 1s.  THETA = {THETA} (mean nearest-neighbour distance)\n")

print("E1  tube radius per rung  (residual = ||real - best render||, same units as THETA)")
print(f"{'rung':>9s} {'knobs':>5s} {'median':>7s} {'90th':>6s} {'<1.5':>6s} {'<1.89':>6s} {'<2.0':>6s} {'<2.5':>6s}")
for r in RUNGS:
    e = E[r]
    print(f"{r:>9s} {len(ladder.KNOBS[r]):5d} {np.median(e):7.2f} {np.percentile(e,90):6.2f} "
          + " ".join(f"{100*(e<c).mean():5.1f}%" for c in (1.5, THETA, 2.0, 2.5)))

print("\nE2  is the surface inside the class?  render every fitted point and ask the data")
d_nn, _ = nearest(ones, ones, exclude_self=True)
print(f"  real 1s: nearest other real 1 median {np.median(d_nn):.2f};  kNN(10) among 60k digits: "
      f"{100*(knn_ones(ones, X, y) >= 9).mean():.1f}% have >=9 ones")
R = {}
for r in RUNGS:
    R[r] = renders(P[r], r)
    d, idx = nearest(R[r], ones)
    dother = nearest(R[r], ones, exclude_self=True)[0]     # nearest real 1 that is NOT the source
    k = knn_ones(R[r], X, y)
    print(f"  {r:>9s} renders: nearest real 1 median {np.median(d):.2f} (excluding source {np.median(dother):.2f}), "
          f"within THETA of a real 1 {100*(d<THETA).mean():.0f}%;  kNN>=9 ones {100*(k>=9).mean():.1f}%, "
          f"nearest digit is a 1 {100*(k>=1).mean():.1f}%")
np.savez("claude/renders.npz", **R)

print("\nE3  product test in knob space: shuffle each knob independently across images, render, ask the data")
rng = np.random.default_rng(0)
for r in ["straight", "foot"]:
    Pr = P[r]
    Ps = np.column_stack([rng.permutation(Pr[:, j]) for j in range(Pr.shape[1])])
    Rs = renders(Ps, r)
    d = nearest(Rs, ones)[0]; k = knn_ones(Rs, X, y)
    print(f"  {r:>9s} independent-knob renders: nearest real 1 median {np.median(d):.2f}, within THETA {100*(d<THETA).mean():.0f}%, "
          f"kNN>=9 ones {100*(k>=9).mean():.1f}%, nearest digit a 1 {100*(k>=1).mean():.1f}%")
    C = np.corrcoef(Pr.T); names = ladder.KNOBS[r]
    pairs = sorted([(abs(C[i,j]), names[i], names[j], C[i,j]) for i in range(len(names)) for j in range(i+1, len(names))], reverse=True)[:4]
    print("            strongest knob correlations: " + ", ".join(f"{a}~{b} {c:+.2f}" for _, a, b, c in pairs))

# per-pixel control from the notebook, for scale: independent PIXELS
idx = rng.integers(0, n, size=(2000, 784)); indep = ones[idx, np.arange(784)]
d = nearest(indep, ones)[0]; k = knn_ones(indep, X, y)
print(f"  (control) independent-PIXEL images: nearest real 1 median {np.median(d):.2f}, kNN>=9 ones {100*(k>=9).mean():.1f}%")
