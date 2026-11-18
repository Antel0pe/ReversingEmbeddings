"""Seed the direction count from EVERY 1 in the dense group, with two controls.

Question: if every start in the dense group gives the same curve / threshold,
is that the real number of ways to vary (extra directions = bending)?
A consistent number is only meaningful if a cloud with NO such structure does
not give the same consistent number, and a family with a KNOWN answer does
show it. So the identical pipeline runs on three datasets of 7,877 images:

  mnist      all MNIST 1s (train + test), scrambled (`scrambled_ones`)
  generated  generated 1s (`grey_ones`, five knobs, default box): known answer 5
  strokes12  drawn 1s with 12 knobs (`knob_strokes.make(n, 12)`): known answer 12
  shapeless  Gaussian cloud with the same mean and spread (covariance) as the
             MNIST 1s: no low-dimensional structure at all

Pipeline (same as `make_scrambled_one_dense_core.py`): most crowded 5% ->
mutual 10-nearest links -> drop 1s with < 3 links -> biggest connected group.
Then for EVERY member as start, add group neighbours nearest first and count the
fewest SVD directions reaching every arrow with <= 10% of its length left over.

Output: figures/scrambled_one_many_starts/many_starts.png (+ .json)
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

import knob_strokes
from grey_ones import render, sample
from scrambled_ones import load

OUT = Path(__file__).resolve().parent / "figures" / "scrambled_one_many_starts"
SHARE, K, MIN_LINKS, TOL = 0.05, 10, 3, 0.10
KMAX = 200


def dense_group(X):
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    rel = D / np.linalg.norm(X, axis=1)[:, None]
    kth = np.sort(rel, axis=1)[:, K - 1]
    dense = np.argsort(kth)[:int(SHARE * len(X))]
    nb = np.argsort(D[np.ix_(dense, dense)], axis=1)[:, :K]
    A = np.zeros((len(dense),) * 2, bool)
    A[np.repeat(np.arange(len(dense)), K), nb.ravel()] = True
    M = A & A.T
    keep = np.ones(len(dense), bool)
    while True:
        weak = keep & (M[:, keep].sum(1) < MIN_LINKS)
        if not weak.any():
            break
        keep &= ~weak
    _, lab = connected_components(csr_matrix(M[np.ix_(keep, keep)]), directed=False)
    sizes = np.bincount(lab)
    return dense[keep][lab == np.argmax(sizes)], rel, sorted(sizes.tolist(), reverse=True)


def count_directions(arrows):
    """Same rule as analyze_mnist_one_flat_patch_directions.count_directions,
    computed through the small k x k Gram matrix (identical answer, faster)."""
    G = arrows @ arrows.T
    w, U = np.linalg.eigh(G)
    w, U = np.maximum(w[::-1], 0), U[:, ::-1]
    coef2 = U ** 2 * w                       # arrow i's squared length along direction j
    left = 1 - np.cumsum(coef2, 1) / np.diag(G)[:, None]
    worst = np.sqrt(np.maximum(left.max(0), 0))
    ok = np.nonzero(worst <= TOL)[0]
    return int(ok[0]) + 1 if len(ok) else len(arrows)


def curves(X, group, rel, ks):
    out = np.full((len(group), len(ks)), np.nan)
    for s, c in enumerate(group):
        pool = group[group != c]
        order = pool[np.argsort(rel[c, pool])]
        arrows = X[order] - X[c]
        for j, k in enumerate(ks):
            if k <= len(arrows):
                out[s, j] = count_directions(arrows[:k])
    return out


def main():
    rng = np.random.default_rng(0)
    mn = np.vstack([load("train"), load("test")]).astype(np.float64)
    data = {
        "mnist": mn,
        "generated": render(sample(len(mn), rng)).reshape(len(mn), 784).astype(np.float64),
        "strokes12": knob_strokes.make(len(mn), 12).reshape(len(mn), 784).astype(np.float64),
        "shapeless": rng.multivariate_normal(mn.mean(0), np.cov(mn.T), size=len(mn)),
    }
    ks = np.unique(np.concatenate([np.arange(1, 61), np.arange(60, KMAX + 1, 5)]))
    summary = {}
    for name, X in data.items():
        t = time.perf_counter()
        group, rel, sizes = dense_group(X)
        C = curves(X, group, rel, ks)
        nn = np.array([np.sort(rel[c, group[group != c]])[0] for c in group])
        first_share = np.array([ks[np.argmax(row < ks)] if (row < ks).any() else np.nan for row in C])
        at = {int(k): C[:, list(ks).index(k)] for k in (25, 50, 100, KMAX)}
        summary[name] = {
            "group_sizes": sizes[:5], "group": len(group),
            "nearest_median": float(np.median(nn)),
            "first_share": np.nanpercentile(first_share, [10, 50, 90]).tolist(),
            "at": {k: np.nanpercentile(v, [10, 50, 90]).tolist() for k, v in at.items()},
            "ks": ks.tolist(), "curves": np.where(np.isnan(C), None, C).tolist(),
        }
        s = summary[name]
        print(f"{name:9s} group {len(group)} (pieces {sizes[:5]}), nearest median {s['nearest_median']:.3f}, "
              f"{time.perf_counter() - t:.0f} s")
        print(f"          first shared direction at k (10/50/90%): {s['first_share']}")
        for k, v in s["at"].items():
            print(f"          directions at k={k}: {v}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "many_starts.json").write_text(json.dumps(summary))

    style = {"mnist": ("#1f5fbf", "MNIST 1s"),
             "generated": ("#2a9d5c", "generated 1s, 5 knobs"),
             "strokes12": ("#c98a1b", "drawn 1s, 12 knobs"),
             "shapeless": ("#9a9a9a", "shapeless cloud, MNIST's spread")}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    C = np.array(summary["mnist"]["curves"], dtype=float)
    for row in C:
        a1.plot(ks, row, color="#1f5fbf", lw=0.5, alpha=0.15)
    a1.plot(ks, np.nanmedian(C, 0), color="#0b2f6b", lw=2.6, label="middle start (median)")
    lo, med, hi = summary["mnist"]["first_share"]
    a1.text(0.03, 0.97, f"{len(C)} starts = every 1 in the dense group (one faint line each)\n"
            f"first shared direction at k = {med:.0f}  (10-90% of starts: {lo:.0f}-{hi:.0f})",
            transform=a1.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
    a1.set_title("1. MNIST: does every start give the same curve?", fontsize=11)
    rows = [f"{'':32s}{'gap':>6s}{'first':>7s}{'k=200':>7s}"]
    for name, (colour, label) in style.items():
        s_ = summary[name]
        Cn = np.array(s_["curves"], dtype=float)
        a2.plot(ks, np.nanmedian(Cn, 0), color=colour, lw=2.6, label=label)
        rows.append(f"{label:32s}{100 * s_['nearest_median']:5.1f}%{s_['first_share'][1]:7.0f}"
                    f"{s_['at'][KMAX][1]:7.0f}")
    a2.text(0.03, 0.97, "\n".join(rows) + "\n\ngap = typical distance to nearest neighbour"
            "\nfirst = neighbours before two share a direction",
            transform=a2.transAxes, fontsize=9, va="top", family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
    a2.set_title("2. Middle start of each dataset, including controls with known answers", fontsize=11)
    a2.legend(loc="lower right", frameon=False, fontsize=9.5)
    for ax in (a1, a2):
        ax.plot(ks, ks, "k--", lw=1.1)
        ax.set_xlabel("number of neighbours (nearest first, from the dense group)")
        ax.set_xlim(0, KMAX); ax.set_ylim(0, KMAX)
        ax.grid(alpha=0.25)
    a1.set_ylabel("directions needed to reach every neighbour\n(dashed = one per neighbour)")
    a1.legend(loc="lower right", frameon=False, fontsize=9.5)
    fig.suptitle("Direction count seeded from every 1 in a dense group "
                 "(reached = at most 10% of each arrow's length left over)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "many_starts.png", dpi=130)
    print("saved figures/scrambled_one_many_starts/many_starts.png")


if __name__ == "__main__":
    main()
