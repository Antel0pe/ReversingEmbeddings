"""Experiments on three ideas about why unfolding is hard (user's framing, 2026-11-09).

1. The dots only know where they are; nothing says "this is a sheet".
2. Nearest dots, directions and distances cannot be trusted naively.
3. To map the sheet you must also know where it ISN'T: negative evidence that pushes
   apart things that look close but are not joined by sheet.

Methods (all see only the 3-D dots):

  nearby_only      Uses only WHO is near whom (10 nearest), no distances or directions.
                   Laplacian eigenmap: each dot sits near the average of its neighbours,
                   with the two smoothest non-constant patterns as x and y.
  push_random      Negative points, the simple way (UMAP-style): pull linked dots together,
                   push random pairs apart. Starts from nearby_only.
  push_random_cold The same from a random start, so the pushes have to do all the work.
  push_targeted    Negative points, targeted: start from check_k10, then find pairs that land
                   close in the flat picture but are far apart walking along the sheet, push
                   them apart, while springs keep linked dots at their 3-D distance.
  air_probe        Negative points in 3-D: drop any link whose middle is in empty space
                   (no dot near its 1/4, 1/2, 3/4 points), then walk as usual.
  air_probe_check  air_probe's test on top of the existing link check.
  star_tutte       (Claude's idea, not the user's.) Build links that never cross: each dot
                   keeps only the neighbours that a flat triangulation of its own patch
                   connects it to, and both ends must agree. Then convex_edge's layout,
                   which provably cannot fold when the links form a flat triangulation.

Run: .venv/bin/python -m unfolding_suite --plugin unfolding_suite/experiments/negative_space.py \
        --methods nearby_only push_random push_random_cold push_targeted air_probe air_probe_check star_tutte \
        --name negative_space --against current:check_k10
"""
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import eigsh
from scipy.spatial import Delaunay, cKDTree

from unfolding_suite.link_check import (average_inside, checked_links, layout_edge, onto_convex_outline,
                                        outer_ring, reattach_stranded)
from unfolding_suite.methods import layout_from_links, method


def knn_links(points, k):
    _, near = cKDTree(points).query(points, k=k + 1)
    return np.c_[np.repeat(np.arange(len(points)), k), near[:, 1:].ravel()]


def unique_undirected(links):
    links = np.sort(links, axis=1)
    return np.unique(links[links[:, 0] != links[:, 1]], axis=0)


# ------------------------------------------------------------------ idea 2: nearness only
def spectral(points, links, dim=2):
    n = len(points)
    W = csr_matrix((np.ones(len(links)), (links[:, 0], links[:, 1])), shape=(n, n))
    W = W.maximum(W.T)
    d = np.asarray(W.sum(1)).ravel()
    S = diags(1 / np.sqrt(d)) @ W @ diags(1 / np.sqrt(d))
    start = np.random.default_rng(0).standard_normal(n)
    vals, vecs = eigsh(S, k=dim + 1, which="LA", v0=start)
    vecs = vecs[:, np.argsort(-vals)][:, 1:dim + 1] / np.sqrt(d)[:, None]
    return vecs


@method("nearby_only")
def nearby_only(points, dim, rng, k=10):
    """Only who-is-near-whom (10 nearest), no lengths: the two smoothest patterns on the link graph."""
    links = knn_links(points, k)
    return spectral(points, links, dim), {"edges": links}


# ------------------------------------------------------------ idea 3a: random push-apart
UMAP_A, UMAP_B = 1.577, 0.895          # UMAP's curve for min_dist = 0.1


def pull_push(links, Y, rng, epochs=400, negatives=5):
    """Attract along links, repel random pairs. Batched version of UMAP's update."""
    n = len(Y)
    links = np.r_[links, links[:, ::-1]]
    for epoch in range(epochs):
        lr = 1.0 - epoch / epochs
        i, j = links[:, 0], links[:, 1]
        diff = Y[i] - Y[j]
        d2 = (diff ** 2).sum(1)
        coef = -2 * UMAP_A * UMAP_B * d2 ** (UMAP_B - 1) / (1 + UMAP_A * d2 ** UMAP_B)
        coef[d2 == 0] = 0
        move = np.zeros_like(Y)
        np.add.at(move, i, np.clip(coef[:, None] * diff, -4, 4))
        ii = np.repeat(i, negatives)
        kk = rng.integers(n, size=len(ii))
        diff = Y[ii] - Y[kk]
        d2 = (diff ** 2).sum(1)
        coef = 2 * UMAP_B / ((0.001 + d2) * (1 + UMAP_A * d2 ** UMAP_B))
        coef[ii == kk] = 0
        np.add.at(move, ii, np.clip(coef[:, None] * diff, -4, 4))
        deg = np.bincount(i, minlength=n)[:, None]
        Y = Y + lr * move / np.maximum(deg, 1) * 0.5      # batched, so average per dot
    return Y


def scaled(Y, size=10.0):
    Y = Y - Y.mean(0)
    return Y * (size / np.abs(Y).max())


@method("push_random")
def push_random(points, dim, rng, k=10):
    """Pull 10-nearest links together, push random pairs apart (UMAP-style), from nearby_only's start."""
    links = knn_links(points, k)
    return pull_push(links, scaled(spectral(points, links, dim)), rng), {"edges": links}


@method("push_random_cold")
def push_random_cold(points, dim, rng, k=10):
    """Pull links together, push random pairs apart, from a random start."""
    links = knn_links(points, k)
    return pull_push(links, rng.uniform(-10, 10, (len(points), dim)), rng), {"edges": links}


# --------------------------------------------------------- idea 3b: targeted push-apart
@method("push_targeted")
def push_targeted(points, dim, rng, k=10, rounds=300):
    """check_k10, then push apart pairs that are close in the picture but far along the sheet."""
    links = unique_undirected(checked_links(points, k, 12, 0.5))
    n = len(points)
    length = np.linalg.norm(points[links[:, 0]] - points[links[:, 1]], axis=1)
    graph = csr_matrix((length, (links[:, 0], links[:, 1])), shape=(n, n))
    walk = shortest_path(graph, method="D", directed=False)
    flat, _ = layout_from_links(points, links, dim)
    Y = flat.copy()
    step = np.median(length)
    keep_apart = 3 * step                       # negatives should end at least this far apart
    for r in range(rounds):
        if r % 10 == 0:                         # refresh the negative pairs
            pairs = cKDTree(Y).query_pairs(keep_apart, output_type="ndarray")
            if len(pairs):
                gap = np.linalg.norm(Y[pairs[:, 0]] - Y[pairs[:, 1]], axis=1)
                far = walk[pairs[:, 0], pairs[:, 1]] > np.maximum(4 * gap, 2 * keep_apart)
                negative = pairs[far]
        move = np.zeros_like(Y)
        diff = Y[links[:, 0]] - Y[links[:, 1]]
        d = np.linalg.norm(diff, axis=1) + 1e-12
        f = ((d - length) / d)[:, None] * diff
        np.add.at(move, links[:, 0], -f)
        np.add.at(move, links[:, 1], f)
        if len(negative):
            diff = Y[negative[:, 0]] - Y[negative[:, 1]]
            d = np.linalg.norm(diff, axis=1) + 1e-12
            f = (np.maximum(keep_apart - d, 0) / d)[:, None] * diff
            np.add.at(move, negative[:, 0], f)
            np.add.at(move, negative[:, 1], -f)
        Y += 0.1 * move
    return Y, {"edges": links, "negatives": len(negative)}


# --------------------------------------------------- idea 3c: negative probes in 3-D
def air_flags(points, links, spacing_multiple=0.75):
    """True for links whose 1/4, 1/2 or 3/4 point has no dot near it (the link crosses air)."""
    tree = cKDTree(points)
    spacing = np.median(tree.query(points, k=2)[0][:, 1])
    worst = np.zeros(len(links))
    for t in (0.25, 0.5, 0.75):
        probe = (1 - t) * points[links[:, 0]] + t * points[links[:, 1]]
        worst = np.maximum(worst, tree.query(probe)[0])
    return worst > spacing_multiple * spacing


@method("air_probe")
def air_probe(points, dim, rng, k=10):
    """Drop 10-nearest links whose middle has no dot nearby (they cross air), then walk."""
    links = knn_links(points, k)
    links = reattach_stranded(points, links[~air_flags(points, links)])
    flat, captured = layout_from_links(points, links, dim)
    return flat, {"edges": links, "captured": captured}


@method("air_probe_check")
def air_probe_check(points, dim, rng, k=10):
    """The existing link check, plus dropping links whose middle crosses air, then walk."""
    links = checked_links(points, k, 12, 0.5)
    links = reattach_stranded(points, links[~air_flags(points, links)])
    flat, captured = layout_from_links(points, links, dim)
    return flat, {"edges": links, "captured": captured}


# ------------------------------------------------- Claude's idea: links that never cross
def star_links(points, k=12, plane_k=12):
    """Each dot's neighbours in a flat triangulation of its own patch; keep links both ends agree on."""
    candidates = checked_links(points, k, plane_k, 0.5)
    n = len(points)
    order = np.argsort(candidates[:, 0], kind="stable")
    candidates = candidates[order]
    starts = np.searchsorted(candidates[:, 0], np.arange(n + 1))
    chosen = []
    for i in range(n):
        nb = candidates[starts[i]:starts[i + 1], 1]
        if len(nb) < 3:
            chosen += [(i, j) for j in nb]
            continue
        patch = points[np.r_[i, nb]]
        centred = patch - patch.mean(0)
        axes = np.linalg.svd(centred, full_matrices=False)[2][:2].T
        flat = centred @ axes
        try:
            tri = Delaunay(flat)
        except Exception:                                   # noqa: BLE001 - degenerate patch
            chosen += [(i, j) for j in nb]
            continue
        ptr, idx = tri.vertex_neighbor_vertices
        chosen += [(i, nb[m - 1]) for m in idx[ptr[0]:ptr[1]]]
    chosen = np.array(chosen)
    key = chosen[:, 0] * n + chosen[:, 1]
    mutual = np.isin(chosen[:, 1] * n + chosen[:, 0], key)
    return reattach_stranded(points, chosen[mutual])


@method("star_tutte")
def star_tutte(points, dim, rng):
    """(Claude's idea) Links from each dot's own flat triangulation, both ends agreeing; convex_edge layout."""
    links = star_links(points)
    edge = outer_ring(points)
    flat = np.zeros((len(points), dim))
    flat[edge] = onto_convex_outline(layout_edge(points, links, edge, dim))
    return average_inside(points, links, flat, edge, by_distance=True), {"edges": links}


# ------------------------------------- Claude's idea, round 2: get the outline's ORDER right
def ring_order(points, links, edge):
    """Order the edge dots around the outline: the two smoothest patterns of the edge-only
    link graph are a cosine and a sine of position around a ring, so their angle is the order."""
    which = np.where(edge)[0]
    on_ring = edge[links[:, 0]] & edge[links[:, 1]]
    remap = -np.ones(len(points), int)
    remap[which] = np.arange(len(which))
    ring = remap[links[on_ring]]
    xy = spectral(points[which], ring, 2)
    return which[np.argsort(np.arctan2(xy[:, 1], xy[:, 0]))]


def ring_on_circle(points, links, edge, dim=2):
    loop = ring_order(points, checked_links(points, 20, 12, 0.5), edge)    # same wide links outer_ring uses
    seg = np.linalg.norm(points[np.roll(loop, -1)] - points[loop], axis=1)
    angle = 2 * np.pi * np.r_[0, np.cumsum(seg)[:-1]] / seg.sum()
    flat = np.zeros((len(points), dim))
    flat[loop] = np.c_[np.cos(angle), np.sin(angle)]
    return average_inside(points, links, flat, edge, by_distance=True)


@method("ring_circle")
def ring_circle(points, dim, rng, k=10):
    """(Claude's idea) Checked links; edge dots ordered as a ring and put on a circle; average the inside."""
    links = checked_links(points, k, 12, 0.5)
    return ring_on_circle(points, links, outer_ring(points), dim), {"edges": links}


@method("ring_circle_star")
def ring_circle_star(points, dim, rng):
    """(Claude's idea) Non-crossing star links; edge dots ordered as a ring on a circle; average the inside."""
    links = star_links(points)
    return ring_on_circle(points, links, outer_ring(points), dim), {"edges": links}
