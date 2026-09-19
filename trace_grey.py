"""Tracing the dimension of a smooth grey family from images alone.

With coverage greys every knob is smooth, so a small enough neighbourhood of any
image is (nearly) flat: its members spread along exactly as many independent
directions as there are knobs. Local PCA counts them; the sign is an EIGEN-GAP,
a cliff in the variance spectrum right after the last real direction.

Two ways of asking the data:
  local_ball  -- "as much detail as needed": members of the complete set within
                 pixel distance r of an image (produced on request; the analysis
                 only ever sees the images)
  local_knn   -- a finite random sample, like a real dataset: the k nearest
                 members of the sample

The generator's knobs are used for one thing only: CHECKING whether the traced
directions are the true ones, never for finding them.
"""

import numpy as np
import grey_ones as G


def spectrum(Z, vectors=False):
    """Local PCA via the covariance's eigendecomposition. (np.linalg.svd is
    pathologically slow in this environment on 1000s x 784 matrices.)"""
    Z = (Z - Z.mean(0)).astype(np.float64)
    act = np.where(np.abs(Z).max(0) > 0)[0]        # pixels that never change add only zeros
    lam_a, U_a = np.linalg.eigh(Z[:, act].T @ Z[:, act] / max(len(Z) - 1, 1))
    lam_a, U_a = lam_a[::-1].clip(0), U_a[:, ::-1]
    lam = np.zeros(Z.shape[1]); lam[:len(act)] = lam_a
    if not vectors:
        return lam
    U = np.zeros((Z.shape[1], len(act))); U[act] = U_a
    return lam, U


def gap_dim(lam, max_d=20, floor=0.01):
    """Position of the biggest cliff in the spectrum, looking only at directions that
    carry at least `floor` of the variance -- below that, ratios between two
    near-zero eigenvalues are numerical noise, not structure."""
    share = lam / lam.sum()
    m = max(1, min(max_d, int((share >= floor).sum())))
    r = lam[:m] / np.maximum(lam[1:m + 1], 1e-300)
    return int(np.argmax(r) + 1), float(r.max())


def sensitivity(p):
    """Pixel change per unit of each knob -- only used to make the request isotropic."""
    base = G.render(p)[0]; out = []
    for k in range(5):
        q = p.copy(); e = 1e-3 * (G.RANGES[k, 1] - G.RANGES[k, 0]); q[k] += e
        out.append(np.linalg.norm(G.render(q)[0] - base) / e)
    return np.array(out)


def local_ball(p, r, n, rng):
    """Up to n members of the complete set within pixel distance r of render(p)."""
    base = G.render(p)[0].ravel(); sens = sensitivity(p)
    got = []
    tries = 0
    while sum(len(g) for g in got) < n and tries < 40:
        d = rng.normal(size=(4 * n, 5)) / sens * (r / np.sqrt(5)) * rng.uniform(0.2, 1.6, size=(4 * n, 1))
        Q = p + d
        inside = ((Q >= G.RANGES[:, 0]) & (Q <= G.RANGES[:, 1])).all(1)
        X = G.render(Q[inside]).reshape(-1, G.N_PIX ** 2)
        keep = np.linalg.norm(X - base, axis=1) <= r
        got.append(X[keep]); tries += 1
    return np.vstack(got)[:n], base


def jacobian(p):
    base = G.render(p)[0].ravel(); J = []
    for k in range(5):
        q = p.copy(); e = 1e-4 * (G.RANGES[k, 1] - G.RANGES[k, 0]); q[k] += e
        J.append((G.render(q)[0].ravel() - base) / e)
    return np.array(J).T                                    # 784 x 5


def principal_cosines(A, B):
    qa, _ = np.linalg.qr(A); qb, _ = np.linalg.qr(B)
    return np.linalg.svd(qa.T @ qb, compute_uv=False)


def varimax(Phi, iters=200, tol=1e-8):
    """Rotate a basis so each direction uses as few pixels as possible."""
    p, k = Phi.shape; R = np.eye(k); d = 0
    for _ in range(iters):
        L = Phi @ R
        u, s, vt = np.linalg.svd(Phi.T @ (L ** 3 - L @ np.diag((L ** 2).sum(0)) / p))
        R = u @ vt; d_old, d = d, s.sum()
        if d_old and d / d_old < 1 + tol: break
    return Phi @ R
