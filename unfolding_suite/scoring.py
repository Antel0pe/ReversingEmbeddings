"""Scoring: is a flattening really one-to-one, and is it the right shape?

The method returns one flat position per point. The suite knows which points were
neighbours on the paper, so it joins them into triangles and fills the map in between.
That turns "one-to-one" into something that can be certified for EVERY point of the
sheet rather than only the sampled ones:

    If every triangle keeps the same orientation, and the outer edge does not cross
    itself, then the map filled in across the triangles is one-to-one.

Both conditions are needed. A sheet can wrap around and lie over itself with every
triangle the right way up, and then only the outer edge reveals it.

The same check runs on meshes using every 2nd and 4th grid node. Those triangles span
several spacings, so wobble the size of one spacing cannot flip them but a real fold
still does. That separates wobble from folds.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

STRIDES = (1, 2, 4)
WOBBLE_LIMIT = 0.01        # flips at the finest mesh only count as wobble below this share
TEAR = 5.0                 # an edge this many times longer than those around it is a tear
CRUSH = 0.2                # an edge this many times shorter than those around it is crushed
SQUARE_MEDIAN = 3.0        # % of the side: typical point off the true square
SQUARE_WORST = 10.0        # % of the side: worst point off the true square
ROUGH_LIMIT = 0.5          # neighbouring triangles' stretch may differ by this share (95th pct)
TEAR_TOLERANCE = 5e-4      # a share of edges this small is an isolated outlier (about 4 edges), not a tear
MAX_RASTER = 3e7           # bound on raster work, so a wrecked layout cannot stall the suite


def triangles(grid, stride=1):
    """Lower and upper triangle of each cell of the grid mesh, as node indices."""
    nodes = np.arange(0, grid, stride)
    j, i = np.meshgrid(nodes[:-1], nodes[:-1], indexing="ij")
    a, b = (j * grid + i).ravel(), (j * grid + i + stride).ravel()
    c, d = ((j + stride) * grid + i).ravel(), ((j + stride) * grid + i + stride).ravel()
    return np.r_[np.c_[a, b, c], np.c_[b, d, c]]


def signed_area(P, T):
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def outer_edge(grid, stride=1):
    """The paper's boundary as a closed loop of node indices."""
    k = np.arange(0, grid, stride)
    last = grid - 1
    return np.r_[k[:-1], last + k[:-1] * grid, last * grid + k[::-1][:-1], k[::-1][:-1] * grid]


def outer_edge_crossings(flat, loop):
    """How many pairs of outer-edge segments cross. Zero for a simple outline."""
    A, B = flat[loop], flat[np.roll(loop, -1)]
    D = B - A
    m = len(loop)
    crossings = 0
    for i in range(m):
        side1 = D[i, 0] * (A[:, 1] - A[i, 1]) - D[i, 1] * (A[:, 0] - A[i, 0])
        side2 = D[i, 0] * (B[:, 1] - A[i, 1]) - D[i, 1] * (B[:, 0] - A[i, 0])
        side3 = D[:, 0] * (A[i, 1] - A[:, 1]) - D[:, 1] * (A[i, 0] - A[:, 0])
        side4 = D[:, 0] * (B[i, 1] - A[:, 1]) - D[:, 1] * (B[i, 0] - A[:, 0])
        hit = (side1 * side2 < 0) & (side3 * side4 < 0)
        hit[[i, (i + 1) % m, (i - 1) % m]] = False
        crossings += int(hit.sum())
    return crossings // 2


def fold_check(flat, uv, grid, stride):
    """Share of the paper lying on the minority orientation, flips and collapsed triangles."""
    T = triangles(grid, stride)
    paper = signed_area(uv, T)
    if (paper <= 0).any():
        raise AssertionError("the paper mesh itself has a bad triangle")
    area = signed_area(flat, T)
    collapsed = np.abs(area) <= 1e-12 * max(np.abs(area).max(), 1e-300)
    pos, neg = paper[(area > 0) & ~collapsed].sum(), paper[(area < 0) & ~collapsed].sum()
    minority = (area < 0) if pos >= neg else (area > 0)
    return {"fold_share": float(min(pos, neg) / paper.sum()),
            "flipped_triangles": int((minority & ~collapsed).sum()),
            "collapsed_triangles": int(collapsed.sum()),
            "outer_edge_crossings": outer_edge_crossings(flat, outer_edge(grid, stride))}


def overlap_share(flat, grid):
    """Of the flat area the sheet covers, the share covered more than once.

    Counts how many mesh triangles cover each cell of a fine raster, so it sees overlaps
    between parts of the sheet that are far apart as well as folds next to each other.
    """
    T = triangles(grid)
    pts = flat[T]
    low, high = pts.min(1), pts.max(1)
    extent = flat.max(0) - flat.min(0)
    edges = np.linalg.norm(flat[T[:, 1]] - flat[T[:, 0]], axis=1)
    cell = max(np.median(edges) / 4, extent.max() / 2048, 1e-300)
    cell = max(cell, np.sqrt(np.prod(high - low, axis=1).sum() / MAX_RASTER))
    origin = flat.min(0)
    shape = np.ceil(extent / cell).astype(int) + 2
    count = np.zeros(shape[::-1], np.int32)
    first = np.floor((low - origin) / cell).astype(int)
    last = np.ceil((high - origin) / cell).astype(int)
    for k in range(len(T)):
        x0, y0, x1, y1 = first[k, 0], first[k, 1], last[k, 0], last[k, 1]
        xs = origin[0] + (np.arange(x0, x1 + 1) + 0.5) * cell
        ys = origin[1] + (np.arange(y0, y1 + 1) + 0.5) * cell
        X, Y = np.meshgrid(xs, ys)
        (ax, ay), (bx, by), (cx, cy) = pts[k]
        w0 = (bx - ax) * (Y - ay) - (by - ay) * (X - ax)
        w1 = (cx - bx) * (Y - by) - (cy - by) * (X - bx)
        w2 = (ax - cx) * (Y - cy) - (ay - cy) * (X - cx)
        inside = ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))
        count[y0:y1 + 1, x0:x1 + 1] += inside
    covered = (count >= 1).sum()
    return float((count >= 2).sum() / max(covered, 1))


def align_to_square(flat, uv):
    """The layout after the best turn, mirror, shift and uniform resize onto the true square."""
    F, U = flat - flat.mean(0), uv - uv.mean(0)
    a, s, b = np.linalg.svd(F.T @ U)
    scale = s.sum() / max((F ** 2).sum(), 1e-300)
    return scale * F @ (a @ b) + uv.mean(0), scale


def square_fit(flat, uv):
    """How far the layout is from the true square, ignoring orientation and size."""
    aligned, scale = align_to_square(flat, uv)
    miss = np.linalg.norm(aligned - uv, axis=1) / 2 * 100                # % of the side
    return {"square_error_median": float(np.median(miss)), "square_error_worst": float(miss.max()),
            "scale_to_paper": float(scale)}


def smoothness(flat, uv, xyz, grid):
    """Tears, crushing, local shape change, and how abruptly stretching changes."""
    # An edge is compared with the edges around it, not with the whole sheet: a stretched
    # sheet legitimately varies from place to place, but it does not jump between neighbours.
    node = np.arange(grid * grid).reshape(grid, grid)
    ratios = []
    for a, b in [(node[:, :-1], node[:, 1:]), (node[:-1], node[1:])]:
        length_flat = np.linalg.norm(flat[a] - flat[b], axis=2)
        ratios.append(length_flat / np.linalg.norm(xyz[a] - xyz[b], axis=2))
    local = [median_filter(r, size=9, mode="nearest") for r in ratios]
    relative = np.concatenate([(r / np.maximum(m, 1e-300)).ravel() for r, m in zip(ratios, local)])
    ratio = np.concatenate([r.ravel() for r in ratios])
    typical = max(np.median(ratio), 1e-300)
    out = {"tear_share": float((relative > TEAR).mean()),
           "crush_share": float((relative < CRUSH).mean()),
           "length_ratio_p5": float(np.percentile(ratio / typical, 5)),
           "length_ratio_p95": float(np.percentile(ratio / typical, 95))}
    s = 2
    n = (grid - 1) // s
    T2 = triangles(grid, s)
    P = np.linalg.inv(np.stack([uv[T2[:, 1]] - uv[T2[:, 0]], uv[T2[:, 2]] - uv[T2[:, 0]]], axis=2))
    F = np.stack([flat[T2[:, 1]] - flat[T2[:, 0]], flat[T2[:, 2]] - flat[T2[:, 0]]], axis=2)
    J = F @ P                                     # stretch of each triangle, paper -> flat
    lower, upper = J[:n * n].reshape(n, n, 2, 2), J[n * n:].reshape(n, n, 2, 2)
    pairs = [(lower, upper), (upper[:, :-1], lower[:, 1:]), (upper[:-1], lower[1:])]
    size = np.median(np.linalg.norm(J, axis=(1, 2)))
    change = np.concatenate([np.linalg.norm((a - b).reshape(-1, 4), axis=1) for a, b in pairs]) / max(size, 1e-300)
    out["roughness"] = float(np.percentile(change, 95))
    return out


def verdict(checks):
    """certified / wobble only / folded, from the fold checks at every mesh scale."""
    fine, coarse = checks[0], checks[1:]
    bad = lambda c: c["flipped_triangles"] + c["collapsed_triangles"] + c["outer_edge_crossings"]
    if all(bad(c) == 0 for c in checks):
        return "certified"
    if all(bad(c) == 0 for c in coarse) and fine["fold_share"] < WOBBLE_LIMIT:
        return "wobble only"
    return "folded"


def to_grid_order(flat, instance):
    """Check a method's answer and put it back in paper order."""
    flat = np.asarray(flat, np.float64)
    if flat.shape != (len(instance.uv), 2):
        raise ValueError(f"expected shape {(len(instance.uv), 2)}, got {flat.shape}")
    if not np.isfinite(flat).all():
        raise ValueError("the answer contains NaN or infinity")
    in_grid_order = np.empty_like(flat)
    in_grid_order[instance.order] = flat
    return in_grid_order


def score(flat, instance):
    """Everything the suite reports about one flattening of one instance."""
    return score_layout(to_grid_order(flat, instance), instance.uv, instance.xyz, instance.grid,
                        instance.sheet.family)


def score_layout(flat, uv, xyz, grid, family):
    checks = [fold_check(flat, uv, grid, s) for s in STRIDES]
    overlap = overlap_share(flat, grid)
    out = {"verdict": verdict(checks), "overlap_share": overlap,
           "fold_share": max(c["fold_share"] for c in checks[1:]),
           "fine_fold_share": checks[0]["fold_share"],
           "outer_edge_crossings": max(c["outer_edge_crossings"] for c in checks),
           **square_fit(flat, uv), **smoothness(flat, uv, xyz, grid)}
    out["square"] = bool(out["square_error_median"] <= SQUARE_MEDIAN and out["square_error_worst"] <= SQUARE_WORST) \
        if family == "bent" else None
    out["smooth"] = bool(max(out["tear_share"], out["crush_share"]) <= TEAR_TOLERANCE and out["roughness"] <= ROUGH_LIMIT)
    out["passed"] = bool(out["verdict"] != "folded" and out["square"] is not False and out["smooth"])
    return out
