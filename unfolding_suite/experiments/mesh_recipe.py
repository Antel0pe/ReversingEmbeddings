"""The graphics recipe for flattening a sheet from dots alone (2026-11-14).

  1. Local tabletop: fit a plane to each dot's nearest dots (as link_check does) and keep only
     neighbours that lie along that plane at both ends (the existing link check).
  2. Weave triangles: at each dot, lay its checked neighbours onto its own tabletop and connect
     them into triangles there (2-D Delaunay); keep a triangle only if all three of its dots
     propose it (the "tangential Delaunay" idea). Then clean up until it is one sheet: no edge in
     more than two triangles, the triangles around each dot form one fan, consistent facing,
     one connected piece. Interior holes are closed with a fan around a temporary extra dot.
  3. Tutte: put the sheet's outer edge (read off the mesh, in order) on a circle, spaced by its
     3-D lengths, and every other dot at the average of its mesh neighbours. If the mesh is a
     proper disc, this cannot fold (Tutte's theorem).
  4. Relax without folding: unpin the edge and minimise the symmetric Dirichlet energy (each
     triangle's stretch plus its inverse stretch; it goes to infinity as a triangle is squashed
     flat), with a step-length rule that never lets any triangle reach zero area.
     The paper's edge is NOT stopped from crossing itself (the scorer checks that).

Methods: mesh_tutte (steps 1-3), mesh_full (steps 1-4).
Dots left out of the mesh get the average flat position of their nearest meshed dots.

Run on the suite:
  .venv/bin/python -m unfolding_suite --plugin unfolding_suite/experiments/mesh_recipe.py \
      --methods mesh_tutte mesh_full --sheets all --first-seed 100 --seeds 3 --name mesh_recipe
"""
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from scipy.spatial import Delaunay, cKDTree

from unfolding_suite.methods import method


# ------------------------------------------------------------------ steps 1 and 2: weave triangles
def tabletops(points, plane_k):
    _, near = cKDTree(points).query(points, k=plane_k + 1)
    patch = points[near]
    return np.linalg.svd(patch - patch.mean(1, keepdims=True), full_matrices=False)[2]   # (n, 3, 3) rows = axes


def propose_triangles(points, k=16, plane_k=12, limit=0.5, long_edge=np.inf):
    """Each dot proposes the triangles around it, made on its own tabletop. Returns {triangle: votes}."""
    n = len(points)
    axes = tabletops(points, plane_k)
    normals = axes[:, 2]
    dist, near = cKDTree(points).query(points, k=k + 1)
    spacing = np.median(dist[:, 1])
    votes = defaultdict(int)
    for i in range(n):
        nb = near[i, 1:]
        step = points[nb] - points[i]
        step /= np.linalg.norm(step, axis=1, keepdims=True)
        tilt = np.maximum(np.abs(step @ normals[i]), np.abs((step * normals[nb]).sum(1)))
        nb = nb[tilt <= limit]
        if len(nb) < 2:
            continue
        local = np.r_[[i], nb]
        xy = (points[local] - points[i]) @ axes[i, :2].T
        try:
            tri = Delaunay(xy).simplices
        except Exception:                                            # noqa: BLE001  (all in a line)
            continue
        for t in tri[(tri == 0).any(1)]:
            g = local[t]
            sides = np.linalg.norm(points[g] - points[np.roll(g, 1)], axis=1)
            if sides.max() > long_edge * spacing:
                continue
            votes[tuple(sorted(g))] += 1
    return votes


def edge_map(tris):
    em = defaultdict(list)
    for t, (a, b, c) in enumerate(tris):
        for e in ((a, b), (b, c), (c, a)):
            em[(min(e), max(e))].append(t)
    return em


def clean_to_sheet(points, tris, score):
    """Remove triangles until the mesh is one orientable sheet with manifold edges and vertices."""
    tris, score = np.asarray(tris), np.asarray(score, float)
    for _ in range(20):
        changed = False
        em = edge_map(tris)                                          # edges in > 2 triangles: drop worst
        drop = set()
        for ts in em.values():
            if len(ts) > 2:
                ranked = sorted(ts, key=lambda t: -score[t])
                drop.update(ranked[2:])
        if drop:
            keep = np.array([t not in drop for t in range(len(tris))])
            tris, score, changed = tris[keep], score[keep], True
        vt = defaultdict(list)                                       # vertices: keep the biggest fan
        for t, tri in enumerate(tris):
            for v in tri:
                vt[v].append(t)
        drop = set()
        for v, ts in vt.items():
            if len(ts) < 2:
                continue
            parent = {t: t for t in ts}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            by_edge = defaultdict(list)
            for t in ts:
                for u in tris[t]:
                    if u != v:
                        by_edge[u].append(t)
            for group in by_edge.values():
                for t in group[1:]:
                    parent[find(t)] = find(group[0])
            fans = defaultdict(list)
            for t in ts:
                fans[find(t)].append(t)
            if len(fans) > 1:
                best = max(fans.values(), key=lambda f: (len(f), score[f].sum()))
                drop.update(t for f in fans.values() if f is not best for t in f)
        if drop:
            keep = np.array([t not in drop for t in range(len(tris))])
            tris, score, changed = tris[keep], score[keep], True
        if not changed:
            break
    tris = orient(tris)
    em = edge_map(tris)                                              # largest connected piece
    rows, cols = [], []
    for ts in em.values():
        if len(ts) == 2:
            rows.append(ts[0]); cols.append(ts[1])
    lab = connected_components(csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(tris),) * 2),
                               directed=False)[1]
    return tris[lab == np.bincount(lab).argmax()]


def orient(tris):
    """Give every triangle the same facing (shared edges run in opposite directions); drop conflicts."""
    tris = tris.copy()
    em = edge_map(tris)
    state = np.zeros(len(tris), int)                                 # 0 unseen, 1 placed, -1 dropped
    for start in range(len(tris)):
        if state[start]:
            continue
        state[start] = 1
        stack = [start]
        while stack:
            t = stack.pop()
            a, b, c = tris[t]
            for u, v in ((a, b), (b, c), (c, a)):
                for s in em[(min(u, v), max(u, v))]:
                    if s == t or state[s] == -1:
                        continue
                    x, y, z = tris[s]
                    same_way = (x, y) == (u, v) or (y, z) == (u, v) or (z, x) == (u, v)
                    if state[s] == 0:
                        if same_way:
                            tris[s] = tris[s][::-1]
                        state[s] = 1
                        stack.append(s)
                    elif same_way:                                   # placed already, and disagrees
                        state[s] = -1
    return tris[state == 1]


def boundary_loops(tris):
    """Boundary edges in the direction the triangles run them, chained into loops."""
    count = defaultdict(int)
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            count[(min(u, v), max(u, v))] += 1
    nxt = {}
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            if count[(min(u, v), max(u, v))] == 1:
                nxt[u] = v
    loops, seen = [], set()
    for s in nxt:
        if s in seen:
            continue
        loop, v = [], s
        while v not in seen and v in nxt:
            seen.add(v)
            loop.append(v)
            v = nxt[v]
        if v == s and len(loop) >= 3:
            loops.append(loop)
    return loops


def build_mesh(points, k=16, plane_k=12, limit=0.5, min_votes=3):
    votes = propose_triangles(points, k, plane_k, limit)
    tris = [t for t, v in votes.items() if v >= min_votes]
    score = [votes[t] for t in tris]
    return clean_to_sheet(points, tris, score)


def close_holes(points, tris):
    """Outer edge = longest boundary loop; every other loop gets a fan around a new temporary dot."""
    loops = boundary_loops(tris)
    length = [np.linalg.norm(points[l] - points[np.roll(l, -1)], axis=1).sum() for l in loops]
    outer = loops[int(np.argmax(length))]
    extra, new_tris, P = [], [list(t) for t in tris], [points]
    nid = len(points)
    for l, loop in enumerate(loops):
        if loop is outer:
            continue
        P.append(points[loop].mean(0, keepdims=True))
        for u, v in zip(loop, np.roll(loop, -1)):
            new_tris.append([v, u, nid])                             # boundary edge runs u->v in its triangle
        extra.append(nid)
        nid += 1
    return np.vstack(P), np.array(new_tris), outer, len(loops) - 1


# ------------------------------------------------------------------ step 3: Tutte
def tutte(points, tris, outer):
    n = len(points)
    e = np.r_[tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]
    A = csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    A = ((A + A.T) > 0).astype(float)
    L = diags(np.asarray(A.sum(1)).ravel()) - A
    seg = np.linalg.norm(points[outer] - points[np.roll(outer, -1)], axis=1)
    angle = 2 * np.pi * np.r_[0, np.cumsum(seg)[:-1]] / seg.sum()
    flat = np.zeros((n, 2))
    flat[outer] = np.c_[np.cos(angle), np.sin(angle)] * seg.sum() / (2 * np.pi)
    used = np.zeros(n, bool)
    used[tris.ravel()] = True
    inside = np.where(used & ~np.isin(np.arange(n), outer))[0]
    rhs = -L[inside][:, outer] @ flat[outer]
    solve = splu(L[inside][:, inside].tocsc()).solve
    flat[inside] = np.c_[solve(rhs[:, 0]), solve(rhs[:, 1])]
    if signed_areas(flat, tris).sum() < 0:                           # face the same way as the triangles
        flat[:, 0] *= -1
    return flat


def signed_areas(X, tris):
    a, b, c = X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]
    return 0.5 * ((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])


# ------------------------------------------------------------------ step 4: relax without folding
def rest_shapes(points, tris):
    p0, p1, p2 = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    l1 = np.linalg.norm(e1, axis=1)
    u = e1 / l1[:, None]
    along = (e2 * u).sum(1)
    across = np.linalg.norm(e2 - along[:, None] * u, axis=1)
    R = np.zeros((len(tris), 2, 2))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 1] = l1, along, across          # columns = the two edges, flat
    return np.linalg.inv(R), 0.5 * l1 * across


def dirichlet(X, tris, Rinv, area, weight):
    D = np.stack([X[tris[:, 1]] - X[tris[:, 0]], X[tris[:, 2]] - X[tris[:, 0]]], axis=2)   # (m, 2, 2)
    J = D @ Rinv
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    if (det <= 0).any():
        return np.inf, None
    fro = (J ** 2).sum((1, 2))
    w = weight * area
    e = (w * fro * (1 + 1 / det ** 2)).sum() / w.sum()
    cof = np.stack([np.stack([J[:, 1, 1], -J[:, 1, 0]], 1), np.stack([-J[:, 0, 1], J[:, 0, 0]], 1)], 1)
    dJ = (w / w.sum())[:, None, None] * (2 * J * (1 + 1 / det ** 2)[:, None, None]
                                         - (2 * fro / det ** 3)[:, None, None] * cof)
    dD = dJ @ np.transpose(Rinv, (0, 2, 1))
    g = np.zeros_like(X)
    np.add.at(g, tris[:, 1], dD[:, :, 0])
    np.add.at(g, tris[:, 2], dD[:, :, 1])
    np.add.at(g, tris[:, 0], -dD[:, :, 0] - dD[:, :, 1])
    return e, g


def max_safe_step(X, d, tris):
    """Largest step along d before any triangle's area reaches zero."""
    a, b, c = X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]
    da, db, dc = d[tris[:, 0]], d[tris[:, 1]], d[tris[:, 2]]
    u, v, du, dv = b - a, c - a, db - da, dc - da
    cross = lambda p, q: p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]
    A2, B, C = cross(du, dv), cross(u, dv) + cross(du, v), cross(u, v)
    t = np.full(len(tris), np.inf)
    lin = np.abs(A2) < 1e-14
    with np.errstate(divide="ignore", invalid="ignore"):
        r = -C[lin] / B[lin]
        t[lin] = np.where(r > 0, r, np.inf)
        disc = B[~lin] ** 2 - 4 * A2[~lin] * C[~lin]
        sq = np.sqrt(np.where(disc >= 0, disc, np.nan))
        r1 = (-B[~lin] - sq) / (2 * A2[~lin])
        r2 = (-B[~lin] + sq) / (2 * A2[~lin])
        r1 = np.where(r1 > 0, r1, np.inf)
        r2 = np.where(r2 > 0, r2, np.inf)
        t[~lin] = np.where(np.isnan(sq), np.inf, np.minimum(r1, r2))
    return t.min()


def edge_crosses(X, outer):
    """Does the closed outline through X[outer] cross itself? (non-neighbouring sides only)"""
    a = X[outer]
    b = np.roll(a, -1, 0)
    m = len(outer)
    i, j = np.triu_indices(m, 2)
    keep = ~((i == 0) & (j == m - 1))
    i, j = i[keep], j[keep]
    cross = lambda o, p, q: (p[:, 0] - o[:, 0]) * (q[:, 1] - o[:, 1]) - (p[:, 1] - o[:, 1]) * (q[:, 0] - o[:, 0])
    d1, d2 = cross(a[i], b[i], a[j]), cross(a[i], b[i], b[j])
    d3, d4 = cross(a[j], b[j], a[i]), cross(a[j], b[j], b[i])
    return bool(((d1 * d2 < 0) & (d3 * d4 < 0)).any())


def relax(X, tris, Rinv, area, weight, iters=2000, memory=10, tol=1e-9, outer=None):
    """L-BFGS on the symmetric Dirichlet energy, never stepping past a flip.
    With `outer`, also never taking a step after which the paper's edge crosses itself."""
    e, g = dirichlet(X, tris, Rinv, area, weight)
    S, Y = [], []
    history = [e]
    for _ in range(iters):
        q = g.ravel().copy()
        alphas = []
        for s, y in reversed(list(zip(S, Y))):
            a_ = (s @ q) / (y @ s)
            alphas.append(a_)
            q -= a_ * y
        if S:
            q *= (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1])
        for (s, y), a_ in zip(zip(S, Y), reversed(alphas)):
            q += s * (a_ - (y @ q) / (y @ s))
        d = -q.reshape(X.shape)
        if (d.ravel() @ g.ravel()) >= 0:
            d, S, Y = -g, [], []
        step = min(1.0, 0.9 * max_safe_step(X, d, tris))
        while True:
            Xn = X + step * d
            en, gn = dirichlet(Xn, tris, Rinv, area, weight)
            if en <= e + 1e-4 * step * (d.ravel() @ g.ravel()) and (outer is None or not edge_crosses(Xn, outer)):
                break
            step *= 0.5
            if step < 1e-14:
                return X, history
        S.append((Xn - X).ravel()); Y.append((gn - g).ravel())
        if len(S) > memory:
            S.pop(0); Y.pop(0)
        if e - en < tol * abs(e):
            X, e = Xn, en
            break
        X, e, g = Xn, en, gn
        history.append(e)
    return X, history


# ------------------------------------------------------------------ the methods
def flatten_mesh(P, tris, outer, n_real, full, extra_weight=0.01, guard_edge=True):
    flat = tutte(P, tris, outer)
    if full:
        Rinv, area = rest_shapes(P, tris)
        weight = np.where((tris >= n_real).any(1), extra_weight, 1.0)   # patched holes count for little
        good = area > 1e-12 * area.max()
        start = flat
        flat, _ = relax(start, tris[good], Rinv[good], area[good], weight[good])
        if guard_edge and edge_crosses(flat, np.asarray(outer)):     # only then: redo, refusing crossings
            flat, _ = relax(start, tris[good], Rinv[good], area[good], weight[good], outer=np.asarray(outer))
    return flat


def fill_unmeshed(points, flat, meshed):
    out = flat[:len(points)].copy()
    missing = ~meshed
    if missing.any():
        _, near = cKDTree(points[meshed]).query(points[missing], k=4)
        out[missing] = flat[np.where(meshed)[0][near]].mean(1)
    return out


def mesh_method(points, dim, rng, full=False, k=16):
    """The graphics recipe: woven triangles, Tutte, and (full) a relaxation that cannot fold."""
    tris = build_mesh(points, k=k)
    P, all_tris, outer, holes = close_holes(points, tris)
    flat = flatten_mesh(P, all_tris, outer, len(points), full)
    meshed = np.zeros(len(points), bool)
    meshed[tris.ravel()] = True
    edges = np.unique(np.sort(np.r_[tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], 1), axis=0)
    V, E, F = len(np.unique(all_tris)), len(np.unique(np.sort(np.r_[all_tris[:, [0, 1]], all_tris[:, [1, 2]],
                                                                    all_tris[:, [2, 0]]], 1), axis=0)), len(all_tris)
    return fill_unmeshed(points, flat, meshed), {"edges": edges, "holes": holes, "euler": int(V - E + F),
                                                  "unmeshed": int((~meshed).sum())}


method("mesh_tutte")(lambda points, dim, rng: mesh_method(points, dim, rng, full=False))
method("mesh_full")(lambda points, dim, rng: mesh_method(points, dim, rng, full=True))
