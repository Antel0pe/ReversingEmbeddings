"""Vanishing component analysis (Livni et al. 2013), simplified: find EQUATIONS the data obeys.

Instead of directions, look for polynomial equations that come out (nearly) zero
at every data point. A circle obeys x^2 + y^2 - 1 = 0: one equation, and it IS
the circle. In general a curve in k numbers needs k - 1 independent equations to
pin it down (two equations in 3-D cut out a curve, like two surfaces meeting).

The method, in plain steps (this is the plain-SVD version of the idea, not
Livni's degree-by-degree construction):
  1. Squash the pictures to k PCA numbers first (784 numbers is far too many
     for this: 784 numbers at degree 2 already make ~300,000 terms).
  2. List every term up to degree d: 1, x, y, x^2, xy, y^2, ...
  3. Find combinations of those terms that are ~0 on every training point.
     Those combinations are the equations.
  4. Check them on held-out points, and ask how many INDEPENDENT directions
     they block at each point (k minus that = the dimension of what is left:
     1 means "a curve").

Two prices to watch: step 1 throws away whatever k PCA numbers miss, and with
too few points for the number of terms, fake equations appear (anything can be
fitted). So this uses 5x finer sweeps than the other tests.

Run:  .venv/bin/python curved_components/vanishing_components.py
"""

import itertools

import numpy as np

from data import load, pca_misses, save

TOL = 1e-3          # an equation counts if it is this close to zero (data scaled to size ~1)


def exponents(k, d):
    return [e for n in range(d + 1) for e in itertools.product(range(n + 1), repeat=k) if sum(e) == n] \
        if k == 1 else [e for e in itertools.product(range(d + 1), repeat=k) if sum(e) <= d]


def terms(Z, E):
    return np.stack([np.prod(Z ** np.array(e), 1) for e in E], 1)


def term_grads(Z, E):
    """d(term)/d(z_j) for every term, point, coordinate -> (n, terms, k)."""
    E = np.array(E)
    G = np.zeros((len(Z), len(E), Z.shape[1]))
    for j in range(Z.shape[1]):
        Ej = E.copy()
        c = Ej[:, j].astype(float)
        Ej[:, j] = np.maximum(Ej[:, j] - 1, 0)
        G[:, :, j] = c * np.prod(Z[:, None, :] ** Ej[None], 2)
    return G


def find_equations(train, test, k, d):
    mu = train.mean(0)
    V = np.linalg.svd(train - mu, full_matrices=False)[2][:k]
    Ztr, Zte = (train - mu) @ V.T, (test - mu) @ V.T
    s = np.abs(Ztr).max()
    Ztr, Zte = Ztr / s, Zte / s
    E = exponents(k, d)
    M = terms(Ztr, E)
    if len(E) >= len(Ztr):
        return dict(terms=len(E), found=None)            # more terms than points: meaningless
    _, sv, Wt = np.linalg.svd(M, full_matrices=False)
    rms = sv / np.sqrt(len(Ztr))                         # typical size of each candidate on the data
    eqs = Wt[rms < TOL]
    out = dict(terms=len(E), found=len(eqs), best_rms=rms.min())
    if len(eqs):
        out["held_out_rms"] = np.sqrt((terms(Zte, E) @ eqs.T) ** 2).mean()
        G = np.einsum("nte,qt->nqe", term_grads(Zte, E), eqs)   # gradient of each equation at each point
        ranks = [np.linalg.matrix_rank(g, tol=1e-2 * np.abs(g).max()) for g in G]
        out["left_dim"] = k - int(np.median(ranks))
    return out


def main():
    res = {}
    for name in ["circle", "slide", "lean"]:
        d = load(name, fine=5) if name != "circle" else load(name)
        pca = pca_misses(d["train"], d["test"], 6)
        ks = [2] if name == "circle" else [2, 3, 4, 6]
        for k in ks:
            for deg in [2, 3, 4, 6]:
                r = find_equations(d["train"], d["test"], k, deg)
                res[f"{name}_k{k}_d{deg}"] = np.array([r.get("found") or 0, r.get("left_dim", k), r.get("best_rms", np.nan)])
                if r["found"] is None:
                    msg = f"{r['terms']} terms > {len(d['train'])} points, skipped"
                elif r["found"] == 0:
                    msg = f"no equation (closest candidate is {r['best_rms']:.4f} off zero)"
                else:
                    msg = (f"{r['found']:3d} equations, held-out {r['held_out_rms']:.1e} off zero, "
                           f"what they leave free: {r['left_dim']} number(s)")
                print(f"{name:6s} k={k} PCA numbers (PCA alone misses {pca[k]:5.1f}% of the picture), "
                      f"degree<={deg:d}: {msg}")
    save("vanishing_components", **res)
    return res


if __name__ == "__main__":
    main()
