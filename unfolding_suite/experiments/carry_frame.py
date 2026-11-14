"""Carry two directions along the paper, step by step, and place each dot by its step.

The user's idea (2026-11-13): at one starting dot, take the two perpendicular directions of the
sheet (a plane fitted to its nearest dots). That is the flat page. Then walk outward. At every new
dot, turn the carried pair of directions as little as possible so it lies in the new dot's own
plane, write the step from the previous dot in those directions, and place the new dot on the page.
"Up" is chosen and the walking order is built exactly as in `up_loops.py`.

Three versions:

  A  `tree`     one walk outward along a tree, rotation only, each dot placed once from its parent.
  B  `relax`    same carried directions, but every link (not just the tree's) votes on where its two
               ends go; the placements that agree best with all links are found together.
  C  `stretch`  like B, but each dot also gets its own stretch and small turn, chosen so that links
               agree; the stretch and turn may only change slowly from dot to dot.

Loop miss: for every link the tree did not use, the gap between where version A put its two ends and
the step that link itself says. On paper that was only bent, it should be near zero; it measures
stretching (and errors).

Run: OMP_NUM_THREADS=1 .venv/bin/python -m unfolding_suite.experiments.carry_frame [seeds...]
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix, identity
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree, shortest_path
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from unfolding_suite.diagnose import FAR
from unfolding_suite.experiments.up_loops import closest_links, facing
from unfolding_suite.link_check import reattach_stranded
from unfolding_suite.methods import method
from unfolding_suite.run import failure_words
from unfolding_suite.scoring import score_layout, to_grid_order
from unfolding_suite.sheets import SHEETS, make_instance
from unfolding_suite.verify import verify
import unfolding_suite.holdout  # noqa: F401  registers the h_ sheets

OUT = Path(__file__).resolve().parent.parent / "runs" / "carry_frame"
DESIGN = ["flat", "gentle_arc", "wave", "s_curl", "roll", "tight_roll", "diagonal_roll", "accordion", "cone",
          "bowl", "uneven_roll", "dents", "crumple", "hard_crumple"]
HOLDOUT = ["h_roll5", "h_scurl65", "h_three_lobes", "h_fan3", "h_cone30", "h_diag30", "h_arc200",
           "h_dents23", "h_crumple17", "h_crumple29"]
ANCHOR = 0.1          # version C: pull toward no stretch (chosen on design sheets, `smooth` part)


# ------------------------------------------------------------------ the method
def local_planes(points, plane_k=12):
    """Each dot's two in-sheet directions and its perpendicular, from a plane through its nearest dots."""
    _, near = cKDTree(points).query(points, k=plane_k + 1)
    patch = points[near]
    return np.linalg.svd(patch - patch.mean(1, keepdims=True), full_matrices=False)[2]   # (n, 3, 3) rows


def tree_order(normals, links, n):
    """up_loops' tree: prefer steps between dots whose perpendiculars are most parallel."""
    w = np.abs((normals[links[:, 0]] * normals[links[:, 1]]).sum(1))
    tree = minimum_spanning_tree(csr_matrix((1.0 - w + 1e-6, (links[:, 0], links[:, 1])), shape=(n, n)))
    return tree + tree.T


def central_dot(tree):
    """A dot near the middle of the sheet (in steps): halfway along the longest walk found by two sweeps."""
    d0 = shortest_path(tree, indices=0, unweighted=True, directed=False)
    a = int(np.argmax(d0))
    da = shortest_path(tree, indices=a, unweighted=True, directed=False)
    b = int(np.argmax(da))
    db = shortest_path(tree, indices=b, unweighted=True, directed=False)
    return int(np.argmin(np.maximum(da, db)))


def turn_into(T, E):
    """Turn the carried pair E (3x2) as little as possible so it lies in the plane spanned by T (3x2)."""
    u, _, vt = np.linalg.svd(T.T @ E)
    return T @ (u @ vt)


def step2d(d, Ei, Ej):
    """A 3-D step written in the carried directions at both ends, kept at its 3-D length."""
    s = 0.5 * (d @ Ei + d @ Ej)
    return s * (np.linalg.norm(d) / max(np.linalg.norm(s), 1e-12))


def carry(points, links, plane_k=12):
    """Version A plus everything B and C need. links: (m, 2) unique undirected links."""
    n = len(points)
    axes = local_planes(points, plane_k)
    normals = axes[:, 2]
    tree = tree_order(normals, links, n)
    root = central_dot(tree)
    order, parent = breadth_first_order(tree, root, directed=False)
    if len(order) < n:
        raise RuntimeError("the links do not reach every dot")

    up = normals.copy()                                   # consistent up, as in up_loops, but from the root
    tangent = axes[:, :2].transpose(0, 2, 1).copy()       # (n, 3, 2)
    E = np.zeros((n, 3, 2))
    flat = np.zeros((n, 2))
    T0 = tangent[root]
    if np.linalg.det(np.c_[T0, up[root]]) < 0:
        T0[:, 1] *= -1
    E[root] = T0                                          # the starting page: the two directions at the start
    for c in order[1:]:
        p = parent[c]
        if up[c] @ up[p] < 0:
            up[c] = -up[c]
        T = tangent[c]
        if np.linalg.det(np.c_[T, up[c]]) < 0:            # keep the pair the same way round as "up" says
            T[:, 1] *= -1
        E[c] = turn_into(T, E[p])
        flat[c] = flat[p] + step2d(points[c] - points[p], E[p], E[c])

    i, j = links.T
    d = points[j] - points[i]
    steps = 0.5 * (np.einsum("mk,mkc->mc", d, E[i]) + np.einsum("mk,mkc->mc", d, E[j]))
    steps *= (np.linalg.norm(d, axis=1) / np.maximum(np.linalg.norm(steps, axis=1), 1e-12))[:, None]
    in_tree = np.asarray(tree[i, j]).ravel() != 0
    length = np.linalg.norm(points[j] - points[i], axis=1)
    miss = np.linalg.norm((flat[j] - flat[i]) - steps, axis=1) / length

    # how much the carried directions disagree across links the tree did not use (degrees)
    Tb = tangent[j]
    u, _, vt = np.linalg.svd(np.einsum("mkc,mkd->mcd", Tb, E[i]))
    Q = u @ vt                                            # i's pair carried across the link, in j's plane
    R = np.einsum("mdc,mde->mce", Q, np.einsum("mkc,mkd->mcd", Tb, E[j]))   # from that to j's own carried pair
    turn = np.degrees(np.arctan2(R[:, 1, 0], R[:, 0, 0]))
    return {"flat": flat, "steps": steps, "root": root, "miss": miss, "in_tree": in_tree,
            "turn": np.abs(turn), "links": links, "length": length}


def incidence(links, n):
    m = len(links)
    r = np.r_[np.arange(m), np.arange(m)]
    return csr_matrix((np.r_[-np.ones(m), np.ones(m)], (r, np.r_[links[:, 0], links[:, 1]])), shape=(m, n))


def relax(carried, n):
    """Version B: positions that best agree with every link's carried step, at once (root held at 0)."""
    D = incidence(carried["links"], n)
    keep = np.ones(n, bool)
    keep[carried["root"]] = False
    A = D[:, keep]
    L = (A.T @ A).tocsc()
    flat = np.zeros((n, 2))
    for k in range(2):
        flat[keep, k] = spsolve(L, A.T @ carried["steps"][:, k])
    return flat


def stretch(carried, n, anchor=ANCHOR, smooth=1.0):
    """Version C: each dot also gets a stretch-and-turn (a, b) = s*(cos t, sin t), slowly varying.

    Each link (i, j) asks  P_j - P_i = M step,  with M the average of the two dots' stretch-and-turns.
    Linked dots' (a, b) should agree (weight `smooth` x typical link length), and every dot is pulled
    gently toward no stretch, no turn (weight `anchor` x typical link length). Without that pull,
    shrinking the whole page is free (every disagreement shrinks with it); measured: a crumple shrank
    to 3-30% of its size. The root dot is held at position 0.
    """
    links, steps = carried["links"], carried["steps"]
    m = len(links)
    i, j = links.T
    h = np.median(carried["length"])
    D = incidence(links, n)
    sx, sy = steps[:, 0], steps[:, 1]
    rows = np.r_[np.arange(m), np.arange(m)]
    cols = np.r_[i, j]
    half = lambda v: coo_matrix((np.r_[v, v] * 0.5, (rows, cols)), shape=(m, n))
    # unknowns: [Px, Py, a, b];   P_j - P_i - 0.5(M_i + M_j) s = 0,   M = [[a, -b], [b, a]]
    Z = csr_matrix((m, n))
    eq_x = [D, Z, -half(sx), half(sy)]
    eq_y = [Z, D, -half(sy), -half(sx)]
    S = D * (smooth * h)
    Zn = csr_matrix((m, n))
    sm_a = [Zn, Zn, S, Zn]
    sm_b = [Zn, Zn, Zn, S]
    I = identity(n, format="csr") * (anchor * h)
    Znn = csr_matrix((n, n))
    pull_a = [Znn, Znn, I, Znn]
    pull_b = [Znn, Znn, Znn, I]
    A = bmat([eq_x, eq_y, sm_a, sm_b, pull_a, pull_b]).tocsc()
    target = np.r_[np.zeros(4 * m), np.full(n, anchor * h), np.zeros(n)]
    r = carried["root"]
    fixed = np.array([r, n + r])
    free = np.setdiff1d(np.arange(4 * n), fixed)
    Af = A[:, free]
    x = np.zeros(4 * n)
    x[free] = spsolve((Af.T @ Af).tocsc(), Af.T @ target)
    flat = np.c_[x[:n], x[n:2 * n]]
    scale = np.hypot(x[2 * n:3 * n], x[3 * n:])
    return flat, scale


def up_checked_links(points, k=10, limit=0.0):
    """up_loops' fair links: k closest, minus those whose ends' ups disagree (ups carried from the start dot)."""
    links = closest_links(points, k)
    axes = local_planes(points)
    normals = axes[:, 2]
    tree = tree_order(normals, links, len(points))
    order, parent = breadth_first_order(tree, 0, directed=False)
    up = normals.copy()
    for c in order[1:]:
        if up[c] @ up[parent[c]] < 0:
            up[c] = -up[c]
    kept = reattach_stranded(points, links[facing(up, links) >= limit])
    return np.unique(np.sort(kept, 1), axis=0)


def _suite(version):
    def fn(points, dim, rng):
        links = up_checked_links(points)
        c = carry(points, links)
        flat = {"tree": lambda: c["flat"], "relax": lambda: relax(c, len(points)),
                "stretch": lambda: stretch(c, len(points))[0]}[version]()
        return flat, {"edges": links}
    fn.__doc__ = {"tree": "Carry two directions outward from one dot along up_loops' tree; place each dot by its step.",
                  "relax": "Carried directions; every link votes on placement, solved together.",
                  "stretch": "Carried directions plus a slowly varying stretch-and-turn per dot, solved together."}[version]
    return fn


for _v in ("tree", "relax", "stretch"):
    method(f"carry_{_v}")(_suite(_v))


# ------------------------------------------------------------------ experiment
def true_links(inst, k=10):
    """Cheat control: the paper's own 10 nearest dots."""
    _, near = cKDTree(inst.uv).query(inst.uv, k=k + 1)
    g = np.c_[np.repeat(np.arange(len(inst.uv)), k), near[:, 1:].ravel()]
    where = np.empty(len(inst.order), int)
    where[inst.order] = np.arange(len(inst.order))
    return np.unique(np.sort(where[g], 1), axis=0)


def run_one(job):
    name, seed, which, anchor = job
    inst = make_instance(SHEETS[name], 65, seed)
    pts = inst.points
    links = true_links(inst) if which == "true" else up_checked_links(pts)
    g = inst.order[links]
    jumps = int((np.linalg.norm(inst.uv[g[:, 0]] - inst.uv[g[:, 1]], axis=1) > FAR).sum())
    t = time.time()
    c = carry(pts, links)
    flats = {"A_tree": c["flat"], "B_relax": relax(c, len(pts))}
    flats["C_stretch"], scale = stretch(c, len(pts), anchor)
    out = {"sheet": name, "family": inst.sheet.family, "seed": seed, "links": which, "anchor": anchor,
           "jumps": jumps, "seconds": round(time.time() - t, 1),
           "miss_median": float(np.median(c["miss"][~c["in_tree"]])),
           "miss_p95": float(np.quantile(c["miss"][~c["in_tree"]], 0.95)),
           "turn_median_deg": float(np.median(c["turn"][~c["in_tree"]])),
           "turn_p95_deg": float(np.quantile(c["turn"][~c["in_tree"]], 0.95)),
           "scale_p05_p95": [float(np.quantile(scale, 0.05)), float(np.quantile(scale, 0.95))]}
    for v, flat in flats.items():
        r = score_layout(to_grid_order(flat, inst), inst.uv, inst.xyz, inst.grid, inst.sheet.family)
        out[v] = {"passed": bool(r["passed"]), "words": failure_words(r),
                  "fold": float(max(r["fold_share"], r["overlap_share"])),
                  "square_median": r.get("square_error_median"), "layers": verify(pts, flat)["reading"]}
    if seed == 100:
        grid_miss = np.zeros(len(pts))
        cnt = np.zeros(len(pts))
        nt = ~c["in_tree"]
        for col in (0, 1):
            np.add.at(grid_miss, links[nt, col], c["miss"][nt])
            np.add.at(cnt, links[nt, col], 1)
        out["dots"] = {"miss": (grid_miss / np.maximum(cnt, 1)).tolist(), "root": int(c["root"]),
                       "uv": inst.uv[inst.order].tolist(), "xyz": inst.xyz[inst.order].tolist(),
                       **{v: f.tolist() for v, f in flats.items()}}
    return out


def show(r):
    v = " | ".join(f"{k[0]} {'PASS' if r[k]['passed'] else 'fail'} {r[k]['words']:12s} {r[k]['layers']:7s}"
                   for k in ("A_tree", "B_relax", "C_stretch"))
    print(f"{r['sheet']:13s} {r['links']:4s} s{r['seed']} pull {r['anchor']:<4} stretch {r['scale_p05_p95'][0]:.2f}-{r['scale_p05_p95'][1]:.2f} jumps {r['jumps']:4d} | miss med {r['miss_median']:.3f} "
          f"p95 {r['miss_p95']:.3f} turn med {r['turn_median_deg']:5.2f} | {v}", flush=True)


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "main"
    OUT.mkdir(parents=True, exist_ok=True)
    if part == "smooth":                                  # choose version C's pull toward no stretch, design sheets only
        jobs = [(s, 100, w, sm) for s in DESIGN for w in ("true", "up") for sm in (0.01, 0.1, 1.0)]
    else:
        seeds = [int(s) for s in sys.argv[2:]] or [100, 101, 102]
        jobs = [(s, seed, w, ANCHOR) for s in DESIGN + HOLDOUT for seed in seeds for w in ("true", "up")]
    rows = []
    with Pool(4) as pool:
        for r in pool.imap_unordered(run_one, jobs):
            rows.append(r)
            show(r)
    (OUT / f"{part}.json").write_text(json.dumps(rows))
