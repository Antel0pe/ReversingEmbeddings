"""Hold short distances, squash the extra dimension: does the sheet come out flat AND unfolded?

The user's idea (2026-11-17): keep each dot's distances to its nearest dots fixed, and
move the dots so that as many directions as possible shrink to ~0.

Three versions, same strings and same solver as `pull.py`, differing only in the reward:
  none    strings only (nothing asks for flatness): a control, should stay as it started
  squash  the user's idea: penalise spread along the cloud's thinnest direction, which is
          re-measured from the current shape every round (that answers "which direction")
  push    `pull_k10`'s reward: push every dot away from the middle

Two string sets: the 10 nearest dots in 3-D (fair), and the paper's true grid
neighbours (cheat control, so the reward is judged without staple strings).

Run: OMP_NUM_THREADS=1 .venv/bin/python -m unfolding_suite.experiments.squash_thin_axis
"""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from unfolding_suite.pull import spread_shares
from unfolding_suite.run import failure_words
from unfolding_suite.scoring import score_layout, to_grid_order
from unfolding_suite.sheets import SHEETS, make_instance
from unfolding_suite.verify import verify

OUT = Path(__file__).resolve().parent.parent / "runs" / "squash_thin_axis"
SHEET_NAMES = ["gentle_arc", "s_curl", "roll", "hairpin_flap", "hard_crumple"]


def knn_links(points, k=10):
    _, near = cKDTree(points).query(points, k=k + 1)
    return np.c_[np.repeat(np.arange(len(points)), k), near[:, 1:].ravel()]


def true_links(inst):
    """Grid neighbours on the paper (8 around each dot), as indices into inst.points."""
    inv = np.argsort(inst.order)
    g = inst.grid
    r, c = np.divmod(np.arange(g * g), g)
    pairs = []
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        ok = (r + dr < g) & (c + dc >= 0) & (c + dc < g)
        a = np.flatnonzero(ok)
        pairs.append(np.c_[a, (r[ok] + dr) * g + c[ok] + dc])
    return inv[np.vstack(pairs)]


def relax(points, links, reward, strength, rounds=15, iters=400):
    links = np.unique(np.sort(links, 1), axis=0)
    i, j = links.T
    d2 = ((points[i] - points[j]) ** 2).sum(1)
    h2 = np.median(d2)
    X = points - points.mean(0)
    n, d = X.shape
    m = len(links)
    inc = csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)], np.r_[i, j])),
                     shape=(m, n)).T.tocsr()
    history = [spread_shares(X)]
    for _ in range(rounds):
        thin = np.linalg.eigh(np.cov((X - X.mean(0)).T))[1][:, 0]       # current thinnest direction

        def energy(x):
            Y = x.reshape(n, d)
            diff = Y[i] - Y[j]
            strain = ((diff ** 2).sum(1) - d2) / h2
            c = Y - Y.mean(0)
            g = inc @ ((4 * strain / h2)[:, None] * diff)
            e = (strain ** 2).sum()
            if reward == "push":
                e -= strength * (c ** 2).sum() / h2
                g = g - 2 * strength * c / h2
            elif reward == "squash":
                t = c @ thin
                e += strength * (t ** 2).sum() / h2
                g = g + 2 * strength * np.outer(t, thin) / h2
            return e, g.ravel()

        X = minimize(energy, X.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": iters}).x.reshape(n, d)
        history.append(spread_shares(X))
    stretch = np.sqrt(((X[i] - X[j]) ** 2).sum(1) / d2)
    return X, history, float(np.median(np.abs(stretch - 1))), float(np.percentile(np.abs(stretch - 1), 99))


def job(args):
    name, strings, reward, strength, seed = args
    inst = make_instance(SHEETS[name], 65, seed)
    links = knn_links(inst.points) if strings == "nearest10" else true_links(inst)
    X, hist, med, p99 = relax(inst.points, links, reward, strength)
    c = X - X.mean(0)
    flat = c @ np.linalg.svd(c, full_matrices=False)[2][:2].T
    r = score_layout(to_grid_order(flat, inst), inst.uv, inst.xyz, inst.grid, inst.sheet.family)
    v = verify(inst.points, flat)
    return {"sheet": name, "strings": strings, "reward": reward, "strength": strength, "seed": seed,
            "thin_share_before": hist[0][2], "thin_share_after": hist[-1][2],
            "string_stretch_median": med, "string_stretch_p99": p99,
            "passed": bool(r["passed"]), "words": failure_words(r),
            "folded": float(max(r["fold_share"], r["overlap_share"])),
            "layers": v["reading"]}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, st, rw, k, 100) for s in SHEET_NAMES for st in ("true", "nearest10")
            for rw, k in [("none", 0), ("squash", 1e-3), ("squash", 1e-2), ("push", 1e-3)]]
    with Pool() as p:
        rows = p.map(job, jobs)
    (OUT / "results.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(f"{r['sheet']:13s} {r['strings']:9s} {r['reward']:6s} {r['strength']:<6g} "
              f"thin {r['thin_share_before']:.4f}->{r['thin_share_after']:.4f}  "
              f"stretch {r['string_stretch_median']:.4f}/{r['string_stretch_p99']:.3f}  "
              f"{'PASS' if r['passed'] else 'fail'} folded {r['folded']:.3f} layers {r['layers']}  {r['words']}")
