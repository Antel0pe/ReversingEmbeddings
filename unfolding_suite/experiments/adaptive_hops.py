"""Hop length chosen by local flatness: link far where the sheet is flat, near where it curves.

For each point, keep adding neighbours while they still lie close to one flat plane. Stop
as soon as they do not. A neighbourhood that reaches a second layer or a sharp bend is not
flat, so the growth should stop before the link is made.

Rejected: it folds the S-curl worse than plain walking (see README, Results so far).
Run: .venv/bin/python -m unfolding_suite --plugin unfolding_suite/experiments/adaptive_hops.py \
         --methods adaptive_tol05 adaptive_tol10 adaptive_tol20 --name adaptive --against current:isomap_k10
"""
import numpy as np
from scipy.spatial import cKDTree

from unfolding_suite.methods import layout_from_links, method

K_MIN, K_MAX = 6, 40


def flat_neighbour_counts(points, tol):
    """How many neighbours each point may keep while they remain close to one plane."""
    _, near = cKDTree(points).query(points, k=K_MAX + 1)
    counts = np.full(len(points), K_MIN)
    still_growing = np.ones(len(points), bool)
    for k in range(K_MIN + 2, K_MAX + 1, 2):
        patch = points[near[:, :k + 1]]
        patch = patch - patch.mean(1, keepdims=True)
        sv = np.linalg.svd(patch, compute_uv=False)
        out_of_plane = sv[:, 2] / np.sqrt(sv[:, 0] ** 2 + sv[:, 1] ** 2)
        still_growing &= out_of_plane <= tol
        counts[still_growing] = k
    return near, counts


def adaptive(points, dim, rng, tol):
    near, counts = flat_neighbour_counts(points, tol)
    links = np.concatenate([np.c_[np.full(c, i), near[i, 1:c + 1]] for i, c in enumerate(counts)])
    flat, captured = layout_from_links(points, links, dim)
    return flat, {"edges": links, "captured": captured}


method("adaptive_tol05", tol=0.05)(adaptive)
method("adaptive_tol10", tol=0.10)(adaptive)
method("adaptive_tol20", tol=0.20)(adaptive)
