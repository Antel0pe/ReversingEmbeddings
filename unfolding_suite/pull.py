"""Pull the paper apart, then lay it flat.

Strings join each dot to its 10 closest dots, each string as long as that pair's distance in 3-D.
Every dot is then pushed away from the middle while the strings hold their length (they may
stretch about 1%). The paper may pass through itself. When the pulled shape has gone flat, it is
dropped onto its best plane.

No attempt is made to spot strings that join two different layers of the paper. Each one acts like
a staple that holds those layers together, so the method only unfolds sheets whose 10 closest dots
are all on the dot's own layer. See README "Pulling the paper apart" for measurements.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from .methods import method


def spread_shares(X):
    """The three eigenvalues of the cloud's spread, as shares of the total (largest first)."""
    ev = np.linalg.eigvalsh(np.cov((X - X.mean(0)).T))[::-1]
    return (ev / ev.sum()).round(4).tolist()


def pull(points, links, push=1e-3, rounds=15, iters=400, verbose=False):
    """Keep link lengths, push everything away from the middle. Returns the pulled dots.

    Energy = sum over links of (squared length - original squared length)^2 / spacing^4
             - push * sum of squared distances from the middle / spacing^2.
    Run in rounds; each round starts from the last round's shape.
    Returns (pulled dots, spread shares after each round, median string stretch).
    """
    links = np.unique(np.sort(links, 1), axis=0)
    i, j = links.T
    d2 = ((points[i] - points[j]) ** 2).sum(1)
    h2 = np.median(d2)
    X = points - points.mean(0)
    n, d = X.shape                                            # any number of coordinates, not just 3
    m = len(links)
    incidence = csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)], np.r_[i, j])),
                           shape=(m, n)).T.tocsr()             # sums each string's pull onto its two ends

    def energy(flat_x):
        Y = flat_x.reshape(n, d)
        diff = Y[i] - Y[j]
        strain = ((diff ** 2).sum(1) - d2) / h2
        c = Y - Y.mean(0)
        e = (strain ** 2).sum() - push * (c ** 2).sum() / h2
        g_link = (4 * strain / h2)[:, None] * diff
        grad = incidence @ g_link - 2 * push * c / h2
        return e, grad.ravel()

    history = [spread_shares(X)]
    for _ in range(rounds):
        X = minimize(energy, X.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": iters}).x.reshape(n, d)
        history.append(spread_shares(X))
        if verbose:
            print(f"    round {len(history) - 1}: spread shares {history[-1][:6]}", flush=True)
    stretch = np.sqrt(((X[i] - X[j]) ** 2).sum(1) / d2)
    return X, history, float(np.median(np.abs(stretch - 1)))


def pull_flat(points, dim, rng, k=10, push=1e-3, rounds=15):
    """Strings to the k closest dots; push everything outward; drop onto the best plane."""
    _, near = cKDTree(points).query(points, k=k + 1)
    links = np.c_[np.repeat(np.arange(len(points)), k), near[:, 1:].ravel()]
    X, history, _ = pull(points, links, push, rounds)
    centred = X - X.mean(0)
    axes = np.linalg.svd(centred, full_matrices=False)[2][:dim].T
    return centred @ axes, {"edges": links, "plane": axes,
                            "captured": float(1 - history[-1][2])}


method("pull_k10", k=10)(pull_flat)
