"""How many dots does each sheet need before the dots themselves say which layer is which?

Truth only (this is the experiment's setup, never shown to a method). For each sheet and dot
count, find the closest 3-D approach between two dots that are on different layers, measured
in local dot spacings (the larger nearest-dot distance of the two ends). "Different layers" =
walking between them along the real sheet (the true paper mesh, 8 directions, 3-D lengths) is
more than WALK times the straight distance. This works for stretched sheets too, where paper
distance does not say how far apart two dots are along the sheet.
"Enough dots" = that closest approach is at least ENOUGH spacings: there is clearly more
empty space between layers than between neighbouring dots on one layer.

Run: .venv/bin/python -m unfolding_suite.experiments.enough_dots
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from unfolding_suite.diagnose import FAR
from unfolding_suite.sheets import SHEETS, make_instance

OUT = Path(__file__).resolve().parent.parent / "runs" / "enough_dots"
GRIDS = (65, 129, 257)
ENOUGH = 2.0
WALK = 3.0


def mesh_walk(inst, pairs, limit, chunk=400):
    """Walking distance along the true paper mesh (8 neighbours, 3-D lengths) for each pair."""
    g, xyz = inst.grid, inst.xyz
    idx = np.arange(g * g).reshape(g, g)
    edges = []
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        r1, c0, c1 = g - dr, max(0, -dc), g - max(0, dc)
        edges.append(np.c_[idx[:r1, c0:c1].ravel(), idx[dr:, c0 + dc:c1 + dc].ravel()])
    e = np.concatenate(edges)
    w = np.linalg.norm(xyz[e[:, 0]] - xyz[e[:, 1]], axis=1)
    graph = csr_matrix((w, (e[:, 0], e[:, 1])), shape=(g * g, g * g))
    out = np.full(len(pairs), np.inf)
    src = np.unique(pairs[:, 0])
    order = np.argsort(pairs[:, 0], kind="stable")
    starts = np.searchsorted(pairs[order, 0], src)
    groups = dict(zip(src, np.split(order, starts[1:])))
    for i in range(0, len(src), chunk):
        block = src[i:i + chunk]
        d = dijkstra(graph, directed=False, indices=block, limit=limit)
        for row, s_ in enumerate(block):
            sel = groups[s_]
            out[sel] = d[row, pairs[sel, 1]]
    return out


def closest_far_approach(inst):
    pts, uv = inst.xyz, inst.uv
    tree = cKDTree(pts)
    local = tree.query(pts, k=2)[0][:, 1]
    reach = 6 * np.median(local)
    pairs = tree.query_pairs(reach, output_type="ndarray")
    if not len(pairs):
        return None, 0
    straight = np.linalg.norm(pts[pairs[:, 0]] - pts[pairs[:, 1]], axis=1)
    walk = mesh_walk(inst, pairs, WALK * reach)
    far = walk > WALK * straight
    if not far.any():
        return None, 0
    p = pairs[far]
    ratio = np.linalg.norm(pts[p[:, 0]] - pts[p[:, 1]], axis=1) / np.maximum(local[p[:, 0]], local[p[:, 1]])
    return float(ratio.min()), int(len(np.unique(p[ratio < ENOUGH].ravel())))


if __name__ == "__main__":
    names = sys.argv[1:] or list(SHEETS)
    out = {}
    for name in names:
        row = {}
        for g in GRIDS:
            gap, tight = closest_far_approach(make_instance(SHEETS[name], g, 100))
            row[g] = {"gap_spacings": gap, "dots_too_close": tight}
        need = next((g for g in GRIDS if row[g]["gap_spacings"] is None or row[g]["gap_spacings"] >= ENOUGH), None)
        out[name] = {"family": SHEETS[name].family, "group": SHEETS[name].group, "by_grid": row, "enough_grid": need}
        print(name, {g: (None if r["gap_spacings"] is None else round(r["gap_spacings"], 2), r["dots_too_close"])
                     for g, r in row.items()}, "->", need, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "density.json").write_text(json.dumps(out, indent=1))
