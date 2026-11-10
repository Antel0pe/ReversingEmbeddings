"""Why the negative_space methods fail. Uses the truth, so it is a diagnosis, not a method.

  1. Which bad links each filter removes, and how many good links it wrongly cuts.
  2. How many links cross each other on the true paper (a flat triangulation has none).
  3. Ceilings: each layout step given the TRUE links (and, for Tutte, the true edge).

Run: .venv/bin/python -m unfolding_suite.experiments.negative_space_why
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from unfolding_suite.diagnose import FAR
from unfolding_suite.experiments.negative_space import (air_flags, knn_links, spectral, star_links,
                                                        unique_undirected)
from unfolding_suite.link_check import average_inside, checked_links, layout_edge, onto_convex_outline
from unfolding_suite.methods import layout_from_links
from unfolding_suite.scoring import outer_edge, score_layout, triangles
from unfolding_suite.sheets import SHEETS, make_instance

SHEET_NAMES = ["s_curl", "roll", "tight_roll", "accordion", "uneven_roll", "crumple", "hard_crumple",
               "h_roll5", "h_crumple17", "h_crumple29"]


def to_grid(instance, links):
    return instance.order[links]


def crossings(uv, links):
    """Pairs of links that cross each other on the paper (sharing an end does not count)."""
    links = unique_undirected(links)
    a, b = uv[links[:, 0]], uv[links[:, 1]]
    mid, half = (a + b) / 2, np.linalg.norm(b - a, axis=1).max() / 2
    links = links[np.linalg.norm(b - a, axis=1) < FAR]             # long jumps are counted separately
    a, b, mid = uv[links[:, 0]], uv[links[:, 1]], (uv[links[:, 0]] + uv[links[:, 1]]) / 2
    half = np.linalg.norm(b - a, axis=1).max()
    pairs = cKDTree(mid).query_pairs(half, output_type="ndarray")
    p, q = links[pairs[:, 0]], links[pairs[:, 1]]
    shared = (p[:, :1] == q).any(1) | (p[:, 1:] == q).any(1)
    pairs, p, q = pairs[~shared], p[~shared], q[~shared]

    def side(o, d, x):
        return np.sign((d[:, 0] - o[:, 0]) * (x[:, 1] - o[:, 1]) - (d[:, 1] - o[:, 1]) * (x[:, 0] - o[:, 0]))

    A, B, C, D = uv[p[:, 0]], uv[p[:, 1]], uv[q[:, 0]], uv[q[:, 1]]
    cross = (side(A, B, C) * side(A, B, D) < 0) & (side(C, D, A) * side(C, D, B) < 0)
    return int(cross.sum()), len(links)


def tutte(xyz, links, edge_loop):
    """Hold the edge on a convex outline (from walking distances), average the inside."""
    n = len(xyz)
    edge = np.zeros(n, bool)
    edge[edge_loop] = True
    flat = np.zeros((n, 2))
    flat[edge] = onto_convex_outline(layout_edge(xyz, links, edge))
    return average_inside(xyz, links, flat, edge, by_distance=True)


def tutte_circle(xyz, links, edge_loop):
    """Hold the edge on a circle in its true order, spaced by 3-D length; average the inside."""
    n = len(xyz)
    edge = np.zeros(n, bool)
    edge[edge_loop] = True
    seg = np.linalg.norm(xyz[np.roll(edge_loop, -1)] - xyz[edge_loop], axis=1)
    angle = 2 * np.pi * np.r_[0, np.cumsum(seg)[:-1]] / seg.sum()
    flat = np.zeros((n, 2))
    flat[edge_loop] = np.c_[np.cos(angle), np.sin(angle)]
    return average_inside(xyz, links, flat, edge, by_distance=True)


def edge_order_scrambled(xyz, links, edge_loop):
    """Share of edge dots that land out of order when convex_edge places the edge."""
    n = len(xyz)
    edge = np.zeros(n, bool)
    edge[edge_loop] = True
    flat = np.zeros((n, 2))
    flat[edge] = onto_convex_outline(layout_edge(xyz, links, edge))
    p = flat[edge_loop] - flat[edge_loop].mean(0)
    turn = np.diff(np.unwrap(np.arctan2(p[:, 1], p[:, 0])))
    return round(100 * float(np.mean(np.sign(turn) != np.sign(np.median(turn)))), 2)


def grid_links(grid):
    T = triangles(grid)
    return unique_undirected(np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [0, 2]]])


def brief(r):
    return {"verdict": r["verdict"], "folded_%": round(100 * max(r["fold_share"], r["overlap_share"]), 2),
            "square_err_%": round(r["square_error_median"], 2), "roughness": round(r["roughness"], 3),
            "torn_%": round(100 * max(r["tear_share"], r["crush_share"]), 3), "passed": r["passed"]}


def main(names=SHEET_NAMES, seed=0):
    out = {}
    for name in names:
        inst = make_instance(SHEETS[name], 65, seed)
        P, uv, xyz, fam = inst.points, inst.uv, inst.xyz, inst.sheet.family
        row = {}
        # 1. filters
        raw = knn_links(P, 10)
        filters = {"10 nearest": raw,
                   "link check": checked_links(P, 10, 12, 0.5),
                   "air probe": raw[~air_flags(P, raw)],
                   "check + air": (lambda c: c[~air_flags(P, c)])(checked_links(P, 10, 12, 0.5)),
                   "star": star_links(P)}
        for label, links in filters.items():
            g = to_grid(inst, unique_undirected(links))
            far = np.linalg.norm(uv[g[:, 0]] - uv[g[:, 1]], axis=1) > FAR
            row[label] = {"links": len(g), "jump_links": int(far.sum())}
        good_raw = unique_undirected(to_grid(inst, raw))
        good_raw = good_raw[np.linalg.norm(uv[good_raw[:, 0]] - uv[good_raw[:, 1]], axis=1) <= FAR]
        for label in ("link check", "air probe", "check + air", "star"):
            kept = unique_undirected(to_grid(inst, filters[label]))
            kept_set = set(map(tuple, kept))
            row[label]["good_links_cut_%"] = round(100 * np.mean([tuple(e) not in kept_set for e in good_raw]), 1)
        for label in ("10 nearest", "link check", "star"):
            c, m = crossings(uv, to_grid(inst, filters[label]))
            row[label]["crossing_pairs"] = c
        # where does the air probe cut? stretch near cut links vs kept links
        g = unique_undirected(to_grid(inst, raw))
        g = g[np.linalg.norm(uv[g[:, 0]] - uv[g[:, 1]], axis=1) <= FAR]
        ratio = np.linalg.norm(xyz[g[:, 0]] - xyz[g[:, 1]], axis=1) / np.linalg.norm(uv[g[:, 0]] - uv[g[:, 1]], axis=1)
        inv = np.argsort(inst.order)
        flagged = air_flags(P, inv[g])
        if flagged.any():
            row["air probe"]["stretch_of_cut_links"] = round(float(np.median(ratio[flagged])), 2)
        row["air probe"]["stretch_of_kept_links"] = round(float(np.median(ratio[~flagged])), 2)
        # 3. ceilings with true links
        true_k10 = np.c_[np.repeat(np.arange(len(uv)), 10), cKDTree(uv).query(uv, k=11)[1][:, 1:].ravel()]
        row["ceiling: nearby_only, true links"] = brief(score_layout(spectral(xyz, true_k10), uv, xyz, 65, fam))
        row["ceiling: tutte, true mesh + true edge"] = brief(score_layout(
            tutte(xyz, grid_links(65), outer_edge(65)), uv, xyz, 65, fam))
        row["ceiling: tutte, true 10-nearest + true edge"] = brief(score_layout(
            tutte(xyz, true_k10, outer_edge(65)), uv, xyz, 65, fam))
        row["ceiling: tutte, true mesh + true edge on a circle"] = brief(score_layout(
            tutte_circle(xyz, grid_links(65), outer_edge(65)), uv, xyz, 65, fam))
        row["edge dots out of order after convex_edge's edge step, %"] = edge_order_scrambled(
            xyz, grid_links(65), outer_edge(65))
        star_g = to_grid(inst, star_links(P))
        row["star links + true edge"] = brief(score_layout(tutte(xyz, star_g, outer_edge(65)), uv, xyz, 65, fam))
        out[name] = row
        print(name, json.dumps(row), flush=True)
    path = Path(__file__).resolve().parent.parent / "runs" / "negative_space" / "why.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main(sys.argv[1:] or SHEET_NAMES)
