"""Fit every MNIST training 1 on every rung (cascade: each rung starts from the one below)."""
import sys, time, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "claude")
import mnist, ladder
X, y = mnist.load("train"); X = X.reshape(len(X), -1) / 255.0
ones = X[y == 1]
f = np.load("data/straight_fit.npz")
fits = {"straight": (f["P"], f["E"])}
P0 = f["P"]
for rung in ["bent", "flag", "foot"]:
    t = time.time()
    P, E = ladder.fit_all(ones, rung, P0)
    print(f"{rung}: {time.time()-t:.0f}s  median resid {np.median(E):.3f}", flush=True)
    fits[rung] = (P, E); P0 = P
np.savez("claude/fits.npz", **{f"P_{k}": v[0] for k, v in fits.items()}, **{f"E_{k}": v[1] for k, v in fits.items()})
