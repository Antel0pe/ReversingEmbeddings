"""Follow-up to analyze_crumpled_sheet_unrolling.py: why don't unrolled sheets come back square?

Three checks, all on the same 4,000 dots per sheet:

  stretch  Join the dots into small triangles using their true paper positions (u, v).
           Compare each triangle's area in 3-D with its area on the flat square. Real
           paper keeps the ratio at exactly 1. Anything else means making the sheet
           stretched or shrank the paper, so no flat layout can be both square and
           faithful to the sheet.
  flips    In the unrolled layout, does any triangle come out mirror-flipped? A flipped
           triangle is a fold, even one too thin to catch two dots on top of each other.
           If no triangle flips and the outer edge does not cross itself, the layout,
           filled in linearly across each triangle, is one-to-one for EVERY point of the
           paper, sampled or not.
  square   Two new sheets bent without any stretching. Their true flat shape is the
           original square, so the unrolled layout can be compared with it directly,
           allowing only a turn or mirror (no resizing).

Run from the repo root: .venv/bin/python analyze_crumpled_sheet_followup.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

from analyze_crumpled_sheet_unrolling import SHEETS, one_to_one, squash, unroll, layer_gap

OUT = Path("figures/crumpled_sheet_unrolling")
data = np.load(OUT / "sheets.npz")
UV = data["UV"]


def bent_without_stretch(kappa, n=20001):
    """A curve of length 2 bending at rate kappa(s), swept 2 units sideways: real paper."""
    s = np.linspace(0, 2, n)
    k = kappa(s)
    th = np.r_[0, np.cumsum((k[1:] + k[:-1]) / 2 * np.diff(s))]
    x = np.r_[0, np.cumsum((np.cos(th[1:]) + np.cos(th[:-1])) / 2 * np.diff(s))]
    z = np.r_[0, np.cumsum((np.sin(th[1:]) + np.sin(th[:-1])) / 2 * np.diff(s))]
    return lambda u, v: np.c_[np.interp(u + 1, s, x), v, np.interp(u + 1, s, z)]


NEW = {"rolled, no stretch": bent_without_stretch(lambda s: 3.5 * s),
       "S-curled, no stretch": bent_without_stretch(lambda s: 7 * np.sin(np.pi * s))}

hull_edges = Delaunay(UV).convex_hull


def signed_areas(P2, T):
    a, b, c = P2[T[:, 0]], P2[T[:, 1]], P2[T[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def areas_3d(X, T):
    a, b, c = X[T[:, 0]], X[T[:, 1]], X[T[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def cross2(p, q):
    return p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0]


def edge_crossings(P2, edges):
    """Count pairs of outer-edge segments that cross in the flat layout."""
    n = 0
    A, B = P2[edges[:, 0]], P2[edges[:, 1]]
    for i in range(len(edges)):
        share = np.isin(edges, edges[i]).any(1)
        d1 = cross2(B[i] - A[i], A - A[i])
        d2 = cross2(B[i] - A[i], B - A[i])
        d3 = cross2(B - A, A[i] - A)
        d4 = cross2(B - A, B[i] - A)
        hit = (d1 * d2 < 0) & (d3 * d4 < 0) & ~share
        n += int(hit.sum())
    return n // 2


def fit_to_square(Y):
    """Best turn/mirror (no resizing) of the layout onto the true (u, v) positions."""
    Yc, Uc = Y - Y.mean(0), UV - UV.mean(0)
    u, _, vt = np.linalg.svd(Yc.T @ Uc)
    return Yc @ (u @ vt) + UV.mean(0)


def good_triangles(P, min_angle=20.0):
    """Triangles of neighbouring dots on the flat paper, skipping thin slivers and any
    triangle touching the paper's edge. Random dots make slivers, and a sliver flips at
    the slightest wobble, which would be counted as a fold that isn't there."""
    D = Delaunay(P)
    T = D.simplices
    edge = np.zeros(len(P), bool)
    edge[D.convex_hull.ravel()] = True
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    ang = []
    for p, q, r in [(a, b, c), (b, c, a), (c, a, b)]:
        u, v = q - p, r - p
        ang.append(np.degrees(np.arccos(np.clip((u * v).sum(1) / np.linalg.norm(u, axis=1)
                                                / np.linalg.norm(v, axis=1), -1, 1))))
    keep = (np.min(ang, axis=0) >= min_angle) & ~edge[T].any(1)
    return T[keep]


def folded_share(Y, P, T):
    """Share of the paper (by true area) covered by mirror-flipped triangles."""
    a = signed_areas(Y, T)
    flipped = np.sign(np.median(a)) * a < 0
    area = np.abs(signed_areas(P, T))
    return float(area[flipped].sum() / area.sum()), flipped


def main():
    import analyze_crumpled_sheet_unrolling as base

    tri = good_triangles(UV)
    uv_area = np.abs(signed_areas(UV, tri))
    results = {}
    layouts = {name: (data[f"{name}|sheet"], data[f"{name}|unroll"]) for name in SHEETS}
    dirs = np.random.default_rng(0).normal(size=(1000, 3))
    for name, f in NEW.items():
        X = f(UV[:, 0], UV[:, 1])
        Y, _ = unroll(X, 10)
        layouts[name] = (X, Y)
        spacing, closest = layer_gap(X, UV)
        results[name] = {"squash_best_of_1000_tilts_overlap": float(min(one_to_one(squash(X, d), UV)["overlap"]
                                                                       for d in dirs)),
                         "unroll": one_to_one(Y, UV), "layer_room": closest / spacing}

    print(f"{len(tri)} well-shaped interior triangles used", flush=True)
    for name, (X, Y) in layouts.items():
        r = results.setdefault(name, {})
        made = areas_3d(X, tri) / uv_area
        unrolled = np.abs(signed_areas(Y, tri)) / areas_3d(X, tri)
        r["stretch_when_made_p5_p95"] = np.percentile(made, [5, 95]).tolist()
        r["unrolled_vs_sheet_area_p5_p95"] = np.percentile(unrolled, [5, 95]).tolist()
        r["outer_edge_crossings"] = edge_crossings(Y, hull_edges)
        # Bigger triangles (every 4th, 16th dot) span several dot spacings: wobble at the
        # scale of one spacing cannot flip them, a real fold still does.
        r["folded_share_by_dots_used"] = {}
        for step in [1, 4, 16]:
            k = np.arange(0, len(UV), step)
            r["folded_share_by_dots_used"][len(k)] = folded_share(Y[k], UV[k], good_triangles(UV[k]))[0]
        if name in NEW:
            miss = np.linalg.norm(fit_to_square(Y) - UV, axis=1)
            r["distance_from_true_square_median"] = float(np.median(miss))
            r["distance_from_true_square_worst"] = float(miss.max())
        print(f"{name:24s} made-stretch {np.round(r['stretch_when_made_p5_p95'], 2)}"
              f"  unrolled/sheet {np.round(r['unrolled_vs_sheet_area_p5_p95'], 2)}"
              f"  folded {[f'{100 * v:.2f}%' for v in r['folded_share_by_dots_used'].values()]}"
              f"  edge crossings {r['outer_edge_crossings']}"
              + (f"  off-square median {r['distance_from_true_square_median']:.4f}"
                 f" worst {r['distance_from_true_square_worst']:.4f}"
                 f"  overlap {r['unroll']['overlap']:.3f}  squash best {r['squash_best_of_1000_tilts_overlap']:.3f}"
                 if name in NEW else ""), flush=True)

    # The S-curl failed even without stretching. Is it hops jumping layers, and do more dots fix it?
    print("--- S-curl: hops between layers, and more dots", flush=True)
    for n in [4000, 8000, 12000]:
        UVn = UV if n == 4000 else np.random.default_rng(n).uniform(-1, 1, (n, 2))
        Xn = NEW["S-curled, no stretch"](UVn[:, 0], UVn[:, 1])
        Yn, _ = unroll(Xn, 10, UVn)
        sp, cl = layer_gap(Xn, UVn)
        res = {**one_to_one(Yn, UVn), "share_of_steps_that_jump_layers": base.SHORTCUTS[10], "layer_room": cl / sp,
               "folded_share": folded_share(Yn, UVn, good_triangles(UVn))[0]}
        results["S-curled, no stretch"].setdefault("more_points", {})[n] = res
        print(n, {k: round(v, 4) for k, v in res.items()}, flush=True)

    np.savez_compressed(OUT / "followup_layouts.npz",
                        **{f"{n}|{p}": a for n, (X, Y) in layouts.items() for p, a in [("sheet", X), ("unroll", Y)]})
    (OUT / "followup_results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
