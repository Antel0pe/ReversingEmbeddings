"""Why did a flattening fail? Turns a failed test into a concrete, checkable explanation.

Uses the truth the method never saw. Finds where on the paper things went wrong, picks
the clearest example of two far-apart parts of the paper landing together, and tests the
known causes against evidence. A cause is only named when its evidence is present.
"""
import numpy as np
from scipy.spatial import cKDTree

from .methods import layout_from_links
from .scoring import score_layout, signed_area, triangles

FAR = 0.3                  # paper distance that counts as "a different part of the sheet"
LOPSIDED = 0.15            # a dot's links are lopsided if they spread this little across their main direction
LOPSIDED_SHARE = 0.01      # lopsided links count as a cause when at least this share of dots has them
_perfect_hops_cache = {}


def with_perfect_hops(instance, k=10):
    """The same layout step, given ideal neighbours taken from the true paper positions.

    If this passes, the method's hops were the problem. If it still fails, the layout step
    was: even correct local information could not be turned into a good flat sheet.
    """
    key = (instance.sheet.name, instance.grid, instance.points.tobytes()[:64])
    if key not in _perfect_hops_cache:
        _, near = cKDTree(instance.uv).query(instance.uv, k=k + 1)
        links = np.c_[np.repeat(np.arange(len(instance.uv)), k), near[:, 1:].ravel()]
        flat, _ = layout_from_links(instance.xyz, links, 2)
        _perfect_hops_cache[key] = score_layout(flat, instance.uv, instance.xyz, instance.grid, instance.sheet.family)
    return _perfect_hops_cache[key]


def lopsided_share(uv, edges):
    """Share of dots whose links, on the true paper, lie nearly along one line."""
    order = np.argsort(edges[:, 0], kind="stable")
    sources, starts = np.unique(edges[order, 0], return_index=True)
    offsets = np.split(uv[edges[order, 1]] - uv[edges[order, 0]], starts[1:])
    lopsided = 0
    for off in offsets:
        if len(off) >= 2:
            sv = np.linalg.svd(off, compute_uv=False)
            lopsided += sv[1] < LOPSIDED * sv[0]
    return lopsided / len(sources)


def where_it_fails(flat, uv, grid):
    """Per-triangle failure on the paper, and the clearest example of a collision."""
    T = triangles(grid)
    paper = signed_area(uv, T)
    area = signed_area(flat, T)
    majority = np.sign(paper[area > 0].sum() - paper[area < 0].sum()) or 1.0
    folded = majority * area < 0
    centre_flat, centre_paper = flat[T].mean(1), uv[T].mean(1)
    edges = np.linalg.norm(flat[T[:, 1]] - flat[T[:, 0]], axis=1)
    near = cKDTree(centre_flat).query_pairs(max(np.median(edges), 1e-12), output_type="ndarray")
    collides = np.zeros(len(T), bool)
    example = None
    if len(near):
        gap = np.linalg.norm(centre_paper[near[:, 0]] - centre_paper[near[:, 1]], axis=1)
        hits = near[gap > FAR]
        collides[hits.ravel()] = True
        if len(hits):
            a, b = near[np.argmax(gap)]
            example = {"paper_a": centre_paper[a].tolist(), "paper_b": centre_paper[b].tolist(),
                       "flat_a": centre_flat[a].tolist(), "flat_b": centre_flat[b].tolist(),
                       "paper_distance": float(gap.max()), "triangle_a": int(a), "triangle_b": int(b)}
    return folded, collides, example


def diagnose(flat, instance, result, info):
    """Return where it failed and a list of named causes with their evidence."""
    uv, xyz, grid = instance.uv, instance.xyz, instance.grid
    T = triangles(grid)
    folded, collides, example = where_it_fails(flat, uv, grid)
    weight = signed_area(uv, T)
    out = {"folded_paper": float(weight[folded].sum() / weight.sum()),
           "colliding_paper": float(weight[collides].sum() / weight.sum()), "causes": []}
    if example:
        p3d = xyz[T].mean(1)
        example["distance_3d"] = float(np.linalg.norm(p3d[example["triangle_a"]] - p3d[example["triangle_b"]]))
        out["example"] = example

    if info.get("edges") is not None and not result["passed"]:
        jump = np.linalg.norm(uv[info["edges"][:, 0]] - uv[info["edges"][:, 1]], axis=1) > FAR
        out["hops_total"], out["hops_jumping"] = len(jump), int(jump.sum())
        out["lopsided_dots"] = float(lopsided_share(uv, info["edges"]))
        perfect = with_perfect_hops(instance)
        out["with_perfect_hops_passes"] = perfect["passed"]
        out["with_perfect_hops_fold"] = perfect["fold_share"]
        if perfect["passed"]:
            verdict_text = "The same layout step passes when given perfect neighbours, so the hops are what went wrong."
            if jump.any():
                out["causes"].append({"cause": "hops jumped between layers", "plain":
                                      f"{int(jump.sum())} of {len(jump)} neighbour links join parts of the sheet that are "
                                      "far apart on the paper but close in 3D. Every walk that uses one takes a shortcut "
                                      f"through the air, so one bad link corrupts many distances. {verdict_text}"})
            elif out["lopsided_dots"] >= LOPSIDED_SHARE:
                out["causes"].append({"cause": "hops were lopsided", "plain":
                                      f"{100 * out['lopsided_dots']:.0f}% of dots link to neighbours lying nearly along one "
                                      "line on the paper, because the dots are spaced unevenly, so walks across the "
                                      f"other direction take long detours. {verdict_text}"})
            else:
                out["causes"].append({"cause": "hops: unknown reason", "plain":
                                      f"{verdict_text} No link jumped layers and links are not lopsided, so "
                                      "this is a new way for hops to fail."})
        else:
            out["causes"].append({"cause": "the layout step fails even with perfect hops", "plain":
                                  "Given ideal neighbours from the true paper, the same layout still folds "
                                  f"{100 * perfect['fold_share']:.1f}% of the paper or misses another check. So the hops "
                                  "are not the whole story: the walking distances themselves cannot all be kept on a "
                                  "flat surface."})
            if jump.any():
                out["causes"].append({"cause": "hops also jumped between layers", "plain":
                                      f"{int(jump.sum())} of {len(jump)} links jump between layers as well."})
    if info.get("plane") is not None:
        hidden = np.empty_like(xyz)
        hidden[instance.order] = instance.points
        a, b, c = hidden[T[:, 0]], hidden[T[:, 1]], hidden[T[:, 2]]
        normals = np.cross(b - a, c - a)
        facing = normals @ np.cross(info["plane"][:, 0], info["plane"][:, 1])
        away = (facing < 0) if (facing > 0).sum() >= (facing < 0).sum() else (facing > 0)
        out["paper_facing_away"] = float(weight[away].sum() / weight.sum())
        if away.any():
            out["causes"].append({"cause": "the sheet turned past edge-on to the plane", "plain":
                                  f"{100 * out['paper_facing_away']:.0f}% of the paper faces away from the plane. Where "
                                  "the sheet turns past edge-on, the part beyond is dropped back over the part before "
                                  "it, so two layers land on the same spots."})
        elif result["square"] is False:
            tilt = np.abs(facing) / np.maximum(np.linalg.norm(normals, axis=1), 1e-300)
            out["causes"].append({"cause": "the plane shortens tilted parts of the sheet", "plain":
                                  "Nothing folds, but dropping points straight onto a plane shortens any part of the "
                                  f"sheet tilted away from it. Typical tilted part keeps {100 * np.median(tilt):.0f}% of "
                                  f"its area and the most tilted keeps {100 * tilt.min():.0f}%, so the result cannot be "
                                  "the original square."})
    if result["passed"]:
        out["causes"] = []
    elif not out["causes"] and (result["verdict"] == "folded" or not result["smooth"] or result["square"] is False):
        out["causes"].append({"cause": "unexplained", "plain":
                              "None of the known causes has evidence here. This is a new way to fail: "
                              "look at the pictures, then add a check for it."})
    return out
