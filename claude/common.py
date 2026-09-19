import sys, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "claude")
import mnist, ladder
THETA = 1.89
RUNGS = ["straight", "bent", "flag", "foot"]

def load():
    X, y = mnist.load("train"); X = X.reshape(len(X), -1) / 255.0
    ones = X[y == 1]
    f = np.load("claude/fits.npz")
    P = {r: f[f"P_{r}"] for r in RUNGS}; E = {r: f[f"E_{r}"] for r in RUNGS}
    return X, y, ones, P, E

def pdist(A, B):
    a = (A ** 2).sum(1)[:, None]; b = (B ** 2).sum(1)[None]
    return np.sqrt(np.maximum(a + b - 2 * A @ B.T, 0))

def nearest(Q, R, chunk=1000, exclude_self=False):
    d = np.empty(len(Q)); idx = np.empty(len(Q), int)
    for s in range(0, len(Q), chunk):
        D = pdist(Q[s:s+chunk], R)
        if exclude_self:
            D[np.arange(len(D)), np.arange(s, s + len(D))] = np.inf
        idx[s:s+chunk] = D.argmin(1); d[s:s+chunk] = D.min(1)
    return d, idx

def knn_ones(Q, X, y, k=10, chunk=500):
    out = np.empty(len(Q), int)
    for s in range(0, len(Q), chunk):
        D = pdist(Q[s:s+chunk], X)
        nn = np.argpartition(D, k, axis=1)[:, :k]
        out[s:s+chunk] = (y[nn] == 1).sum(1)
    return out

def renders(P, rung):
    return np.array([ladder.render(p, rung) for p in P])
