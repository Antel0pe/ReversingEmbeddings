"""Idea B: carry "up" along the paper; a connection to a layer whose up points the other way is wrong.

Each dot gets a direction perpendicular to the paper at that dot (a plane fitted to its 12
closest dots). That direction has no sign: the plane fit cannot tell up from down. Choosing a
consistent up for every dot is done for the whole sheet at once: build a table that says, for
each connected pair, how parallel their perpendiculars are (+1 same, -1 opposite), and take its
top eigenvector. Its signs give every dot an "up" such that connected dots agree as much as
possible. This is the "carry up around loops" idea solved for all loops together.

Then a connection is thrown out if the two dots' ups point in opposite directions
(dot product below `limit`). Real paper turns gradually, so true neighbours agree.

Run:
  .venv/bin/python -m unfolding_suite.experiments.up_loops ceiling   # with the TRUE up, what share of wrong connections face opposite ways?
  .venv/bin/python -m unfolding_suite.experiments.up_loops filter    # the fair method: how many wrong / right connections it removes
  OMP_NUM_THREADS=1 .venv/bin/python -m unfolding_suite.experiments.up_loops score
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree
from scipy.spatial import cKDTree

from unfolding_suite.diagnose import FAR
from unfolding_suite.experiments.height_split import one as score_one
from unfolding_suite.experiments.height_split import tilt_ok
from unfolding_suite.link_check import reattach_stranded
from unfolding_suite.methods import layout_from_links, method
from unfolding_suite.sheets import SHEETS, make_instance

OUT = Path(__file__).resolve().parent.parent / "runs" / "up_loops"
FOCUS = ["s_curl", "roll", "h_roll5", "tight_roll", "accordion", "h_fan3", "h_arc200"]
SEEDS = (100, 101, 102)


def closest_links(points, k=10):
    _, near = cKDTree(points).query(points, k=k + 1)
    links = np.c_[np.repeat(np.arange(len(points)), k), near[:, 1:].ravel()]
    return np.unique(np.sort(links, 1), axis=0)


def perpendiculars(points, plane_k=12):
    _, near = cKDTree(points).query(points, k=plane_k + 1)
    patch = points[near]
    return np.linalg.svd(patch - patch.mean(1, keepdims=True), full_matrices=False)[2][:, 2]


def consistent_up(normals, links):
    """Carry "up" outward from one dot, always stepping across the most-parallel connection first.

    The connections form a tree that prefers steps where the two perpendiculars are nearly
    parallel (a maximum spanning tree on |agreement|); walking it from one dot, each dot's up is
    flipped to agree with the dot it was reached from.
    """
    n = len(normals)
    w = np.abs((normals[links[:, 0]] * normals[links[:, 1]]).sum(1))
    cost = csr_matrix((1.0 - w + 1e-6, (links[:, 0], links[:, 1])), shape=(n, n))
    tree = minimum_spanning_tree(cost)
    tree = tree + tree.T
    order, parent = breadth_first_order(tree, 0, directed=False)
    up = normals.copy()
    for i in order[1:]:
        if (up[i] * up[parent[i]]).sum() < 0:
            up[i] = -up[i]
    return up


def facing(up, links):
    return (up[links[:, 0]] * up[links[:, 1]]).sum(1)


def up_links(points, k=10, limit=0.0):
    links = closest_links(points, k)
    up = consistent_up(perpendiculars(points), links)
    return links[facing(up, links) >= limit], links


def make(limit, also_surface):
    def fn(points, dim, rng):
        kept, _ = up_links(points, 10, limit)
        if also_surface:
            kept = kept[tilt_ok(points, kept)]
        kept = reattach_stranded(points, kept)
        flat, captured = layout_from_links(points, kept, dim)
        return flat, {"edges": kept, "captured": captured}
    return fn


for lim in (0.0, 0.5):
    method(f"up_{lim}")(make(lim, False))
    method(f"up_{lim}_surface")(make(lim, True))


# ------------------------------------------------------------------ truth-based checks
def true_up(inst):
    """Perpendicular from the paper's own grid, consistent everywhere (the cheat)."""
    G = inst.grid
    X = inst.xyz.reshape(G, G, 3)
    n = np.cross(np.gradient(X, axis=1), np.gradient(X, axis=0)).reshape(-1, 3)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return n                                                  # indexed by grid node


def label(inst, links):
    g = inst.order[links]
    return np.linalg.norm(inst.uv[g[:, 0]] - inst.uv[g[:, 1]], axis=1) > FAR, g


def ceiling(name):
    inst = make_instance(SHEETS[name], 65, 100)
    links = closest_links(inst.points)
    wrong, g = label(inst, links)
    f = facing(true_up(inst), g)
    return {"sheet": name, "wrong": int(wrong.sum()),
            "wrong_facing_opposite": float((f[wrong] < 0).mean()) if wrong.any() else None,
            "wrong_facing_median": float(np.median(f[wrong])) if wrong.any() else None,
            "right_facing_opposite": float((f[~wrong] < 0).mean()),
            "right_facing_min": float(f[~wrong].min())}


def filter_check(name):
    inst = make_instance(SHEETS[name], 65, 100)
    links = closest_links(inst.points)
    wrong, g = label(inst, links)
    up = consistent_up(perpendiculars(inst.points), links)
    f = facing(up, links)
    t = facing(true_up(inst), g)
    # did the fair "up" match the true one (up to one overall flip)?
    est = up[np.argsort(inst.order)]
    A = inst.xyz - inst.xyz.mean(0)                              # the method sees the sheet turned; turn the truth to match
    B = inst.points[np.argsort(inst.order)] - inst.points.mean(0)
    u, _, vt = np.linalg.svd(A.T @ B)
    agree = (est * (true_up(inst) @ (u @ vt))).sum(1)
    agree = max((agree > 0).mean(), (agree < 0).mean())
    rows = {"sheet": name, "wrong": int(wrong.sum()), "up_matches_truth": float(agree)}
    for lim in (0.0, 0.5):
        cut = f < lim
        rows[f"limit {lim}"] = {"wrong_removed": int((cut & wrong).sum()), "right_removed": int((cut & ~wrong).sum()),
                                "right_total": int((~wrong).sum())}
    surf = ~tilt_ok(inst.points, links)
    rows["surface filter"] = {"wrong_removed": int((surf & wrong).sum()), "right_removed": int((surf & ~wrong).sum())}
    rows["both (limit 0)"] = {"wrong_removed": int(((f < 0) | surf)[wrong].sum()),
                              "right_removed": int(((f < 0) | surf)[~wrong].sum())}
    return rows


if __name__ == "__main__":
    part = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    if part == "ceiling":
        rows = [ceiling(s) for s in FOCUS]
        for r in rows:
            print(r)
    elif part == "filter":
        rows = [filter_check(s) for s in FOCUS]
        for r in rows:
            print(r)
    else:
        methods = sys.argv[2:] or ["up_0.0", "up_0.5", "up_0.0_surface", "up_0.5_surface"]
        jobs = [(s, m, seed) for s in SHEETS for m in methods for seed in SEEDS]
        rows = {}
        with Pool(4) as pool:
            for (s, m, seed), r in pool.imap_unordered(score_one, jobs):
                rows.setdefault(f"{s} / {m}", {})[str(seed)] = r
                print(f"{'PASS' if r['passed'] else 'FAIL'} {s:15s} {m:16s} {seed} jumps {r['jumps']} {r['words']}",
                      flush=True)
    (OUT / f"{part}.json").write_text(json.dumps(rows, indent=1))
