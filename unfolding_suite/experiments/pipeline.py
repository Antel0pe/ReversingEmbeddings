"""The best parts put together, with every decision traced against the truth (2026-11-14).

Pipeline (uses only the dots; the truth is used afterwards, to say which decision leaked):
  1. Neighbours: each dot's 10 closest dots, minus connections that tilt more than 30 degrees out
     of the local surface at either end (link_check), minus connections whose two ends face
     opposite ways after "up" is carried across the sheet (up_loops). Stranded dots re-attached.
  2. Initial unfold, one of:
       walk  shortest walks along the connections, then the best flat layout (Isomap)
       edge  find the paper's edge dots, lay them out from walking distances, push them onto a
             convex outline, then put every inside dot at the average of its neighbours
  3. Check: the double-layer check (verify.py) and, per dot, face up or face down
     (press_unfold.area_change).
  4. Flip: if any face-down patch was found, mirror it (and everything beyond it) across its
     crease line, then let the strings settle in the flat plane (press_unfold / patch_stitch).
  5. Check again, and score.
  Guarded variant (also truth-free): keep whichever of the unfold / flip / settle layouts has the
  least face-down paper, so a flip or a settle that makes things worse is undone.

Controls, to trace a failure to one decision:
  true_links       step 1 replaced by the paper's true 10 nearest neighbours
  true_links_edge  ...and, for `edge`, the true edge dots too
  true_mesh_edge   for `edge`: the paper's own triangle mesh (connections that never cross) and
                   the true edge dots
  true_mesh_circle the true mesh, and the true edge in its true order on a circle (Tutte's
                   theorem: cannot fold). A circle is not the square, so bent sheets fail squareness
If the pipeline fails but true_links passes, the neighbour decision leaked. If true_links fails
but true_links_edge passes, the edge decision leaked. If even that fails, the unfold / flip /
settle steps are at fault, and the per-stage fold shares say which.

Dot counts: each sheet runs at the count from runs/enough_dots/density.json (65 x 65 for most,
129 x 129 for h_roll5, 257 x 257 for tight_roll); crumples at 65 and 129.

Run: OMP_NUM_THREADS=1 .venv/bin/python -m unfolding_suite.experiments.pipeline [main|dense]
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from unfolding_suite.diagnose import FAR
from unfolding_suite.experiments.patch_stitch import settle_piece
from unfolding_suite.experiments.press_flat import true_links
from unfolding_suite.experiments.press_unfold import adjacency, area_change, groups, truth_flipped, unfold
from unfolding_suite.experiments.up_loops import consistent_up, facing, perpendiculars
from unfolding_suite.link_check import (average_inside, checked_links, layout_edge, onto_convex_outline,
                                        outer_ring, reattach_stranded)
from unfolding_suite.methods import layout_from_links
from unfolding_suite.run import failure_words
from unfolding_suite.scoring import outer_edge, score_layout, to_grid_order, triangles
from unfolding_suite.sheets import SHEETS, make_instance
from unfolding_suite.verify import verify

OUT = Path(__file__).resolve().parent.parent / "runs" / "pipeline"
SEEDS = (100, 101, 102)
CRUMPLES = ["crumple", "hard_crumple", "h_crumple17", "h_crumple29"]
SETTLE_ITERS = 1500


# ------------------------------------------------------------------ the pipeline's own decisions
def own_links(points, k=10):
    """Tilt check, then the up-side cut."""
    links = np.unique(np.sort(checked_links(points, k, 12, 0.5), 1), axis=0)
    up = consistent_up(perpendiculars(points), links)
    kept = links[facing(up, links) >= 0.0]
    return np.unique(np.sort(reattach_stranded(points, kept), 1), axis=0)


def unfold_walk(points, links):
    return layout_from_links(points, links, 2)[0]


def unfold_edge(points, links, edge):
    flat = np.zeros((len(points), 2))
    flat[edge] = onto_convex_outline(layout_edge(points, links, edge))
    return average_inside(points, links, flat, edge, by_distance=True)


# ------------------------------------------------------------------ truth, for tracing only
def is_wrong(inst, links):
    g = inst.order[links]
    return np.linalg.norm(inst.uv[g[:, 0]] - inst.uv[g[:, 1]], axis=1) > FAR


def link_truth(inst, links):
    """Wrong connections kept (far apart on the paper), and the share of the good connections among
    each dot's 10 closest dots in 3-D that the pipeline threw away."""
    _, near = cKDTree(inst.points).query(inst.points, k=11)
    cand = np.unique(np.sort(np.c_[np.repeat(np.arange(len(near)), 10), near[:, 1:].ravel()], 1), axis=0)
    good = cand[~is_wrong(inst, cand)]
    have = {tuple(p) for p in links}
    cut = sum(tuple(p) not in have for p in good)
    return int(is_wrong(inst, links).sum()), float(cut / max(len(good), 1))


def true_mesh(inst):
    """The paper's own triangle mesh edges (no crossings), in the order the method sees the dots."""
    T = triangles(inst.grid, 1)
    e = np.unique(np.sort(np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [0, 2]]], 1), axis=0)
    where = np.empty(len(inst.order), int)
    where[inst.order] = np.arange(len(inst.order))
    return np.unique(np.sort(where[e], 1), axis=0)


def true_edge(inst):
    edge = np.zeros(len(inst.order), bool)
    loop = outer_edge(inst.grid)
    where = np.empty(len(inst.order), int)
    where[inst.order] = np.arange(len(inst.order))
    edge[where[loop]] = True
    return edge


def true_circle(inst, links):
    """Tutte with perfect information: the paper's true edge, in its true order, spaced by its 3-D
    lengths around a circle; every inside dot at the average of its mesh neighbours."""
    where = np.empty(len(inst.order), int)
    where[inst.order] = np.arange(len(inst.order))
    loop = where[outer_edge(inst.grid)]
    P = inst.points[loop]
    step = np.linalg.norm(P - np.roll(P, -1, 0), axis=1)
    angle = 2 * np.pi * np.r_[0, np.cumsum(step)[:-1]] / step.sum()
    flat = np.zeros((len(inst.points), 2))
    flat[loop] = np.c_[np.cos(angle), np.sin(angle)]
    edge = np.zeros(len(inst.points), bool)
    edge[loop] = True
    return average_inside(inst.points, links, flat, edge, by_distance=True)


def edge_truth(inst, edge):
    t = true_edge(inst)
    return {"edge_found": int(edge.sum()), "edge_true": int(t.sum()),
            "edge_false_share": float((edge & ~t).sum() / max(edge.sum(), 1)),
            "edge_missed_share": float((t & ~edge).sum() / t.sum())}


def stage(inst, flat):
    r = score_layout(to_grid_order(flat, inst), inst.uv, inst.xyz, inst.grid, inst.sheet.family)
    v = verify(inst.points, flat)
    return {"passed": bool(r["passed"]), "words": failure_words(r),
            "fold": float(max(r["fold_share"], r["overlap_share"])),
            "square_median": r.get("square_error_median"), "smooth": bool(r.get("smooth", True)),
            "check": v["reading"], "check_stacked_share": v["stacked_share"]}


# ------------------------------------------------------------------ one run
def run(job):
    name, grid, seed, unfold_how, control = job
    inst = make_instance(SHEETS[name], grid, seed)
    pts = inst.points
    out = {"sheet": name, "grid": grid, "seed": seed, "unfold": unfold_how, "control": control,
           "family": inst.sheet.family, "dots": len(pts)}
    t0 = time.time()
    if control == "own":
        links = own_links(pts)
    elif control in ("true_mesh_edge", "true_mesh_circle"):
        links = true_mesh(inst)
    else:
        links = np.unique(np.sort(true_links(inst), 1), axis=0)
    out["wrong_kept"], out["good_cut"] = link_truth(inst, links)
    out["links"] = int(len(links))
    try:
        if unfold_how == "walk":
            flat = unfold_walk(pts, links)
        elif control == "true_mesh_circle":
            flat = true_circle(inst, links)
        else:
            edge = true_edge(inst) if control in ("true_links_edge", "true_mesh_edge") else outer_ring(pts)
            out.update(edge_truth(inst, edge))
            flat = unfold_edge(pts, links, edge)
    except Exception as e:                                    # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        return out, None
    out["t_unfold"] = time.time() - t0
    out["after_unfold"] = stage(inst, flat)

    A, _ = adjacency(len(pts), links)
    det = area_change(pts, flat, A)
    positive, comp, k = groups(A, det > 0)
    truth = truth_flipped(inst, flat)
    found = ~positive
    out["face_down_found"] = float(found.mean())
    out["face_down_true"] = float(truth.mean())
    out["face_agreement"] = float((found == truth).mean())
    flipped, settled = flat, flat
    out["flipped"] = bool(found.any())
    if found.any():
        flipped, creases = unfold(flat, A, positive, comp, k)
        out["creases"] = len(creases)
        out["crease_dots"] = [int(c[2]) for c in creases]
        out["after_flip"] = stage(inst, flipped)
        settled = settle_piece(pts, links, np.arange(len(pts)), flipped, SETTLE_ITERS)
        out["after_settle"] = stage(inst, settled)
        # crease straightness: how far the crease border dots sit from their fitted line, in dot spacings
        spacing = float(np.median(cKDTree(pts).query(pts, k=2)[0][:, 1]))
        out["crease_bend_spacings"] = crease_bend(flat, A, comp, spacing, len(pts))
    final = settled
    out["final"] = stage(inst, final)
    # guarded version (truth-free): keep whichever of unfold / flip / settle leaves the least
    # face-down paper, ties broken by the double-layer check's stacked share
    best, guarded = ("unfold", -found.mean(), -out["after_unfold"]["check_stacked_share"]), flat
    if found.any():
        for label, cand in (("flip", flipped), ("settle", settled)):
            down = ~groups(A, area_change(pts, cand, A) > 0)[0]
            key = (label, -down.mean(), -verify(pts, cand)["stacked_share"])
            if key[1:] > best[1:]:
                best, guarded = key, cand
    out["guarded_choice"] = best[0]
    out["guarded"] = stage(inst, guarded)
    out["seconds"] = time.time() - t0
    keep = {"flat_unfold": flat.astype(np.float32), "flat_final": np.asarray(final, np.float32),
            "flat_guarded": np.asarray(guarded, np.float32),
            "found": found, "truth": truth, "links": links.astype(np.int32)}
    return out, keep


def crease_bend(flat, A, comp, spacing, n):
    """Per crease: the worst distance of its border dots (in the unfolded layout, scaled to 3-D
    spacing) from the straight line fitted through them. Straight creases read near 0."""
    i, j = A.nonzero()
    cross = comp[i] != comp[j]
    scale = 1.0
    out = []
    pairs = {}
    for a, b in zip(i[cross], j[cross]):
        pairs.setdefault(tuple(sorted((comp[a], comp[b]))), set()).update((a, b))
    lengths = np.linalg.norm(flat[i] - flat[j], axis=1)
    scale = spacing / max(np.median(lengths), 1e-12)
    for dots in pairs.values():
        P = flat[np.array(sorted(dots))] * scale
        if len(P) < 4:
            continue
        c = P.mean(0)
        d = np.linalg.svd(P - c, full_matrices=False)[2][0]
        off = (P - c) - np.outer((P - c) @ d, d)
        out.append(float(np.linalg.norm(off, axis=1).max() / spacing))
    return out


def jobs_main():
    density = json.loads((OUT.parent / "enough_dots" / "density.json").read_text())
    jobs = []
    for name in SHEETS:
        grid = 65 if name in CRUMPLES else density[name]["enough_grid"] or 257
        if grid != 65:
            continue                                          # dense sheets run separately
        for seed in SEEDS:
            for how, controls in (("walk", ("own", "true_links")), ("edge", ("own", "true_links", "true_links_edge", "true_mesh_edge", "true_mesh_circle"))):
                jobs += [(name, grid, seed, how, c) for c in controls]
    return jobs


def jobs_dense():
    jobs = []
    for name, grid in (("h_roll5", 129), ("tight_roll", 257)) + tuple((c, 129) for c in CRUMPLES):
        for c in ("own", "true_links", "true_links_edge", "true_mesh_edge", "true_mesh_circle"):
            jobs.append((name, grid, 100, "edge", c))
        if grid == 129 and name == "h_roll5":
            jobs += [(name, grid, 100, "walk", c) for c in ("own", "true_links")]
    return jobs


def save(part, rows, arrays):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{part}.json").write_text(json.dumps(rows, indent=1))
    np.savez_compressed(OUT / f"{part}_layouts.npz", **arrays)


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "main"
    jobs = jobs_main() if part == "main" else jobs_dense()
    rows, arrays = [], {}
    procs = 4 if part == "main" else 1
    with Pool(procs) as pool:
        for out, keep in pool.imap_unordered(run, jobs):
            rows.append(out)
            tag = f"{out['sheet']}|{out['grid']}|{out['seed']}|{out['unfold']}|{out['control']}"
            if keep is not None and out["seed"] == 100:
                for k_, v in keep.items():
                    arrays[f"{tag}|{k_}"] = v
            f = out.get("final", {})
            print(f"{tag:42s} wrong {out['wrong_kept']:5d} cut {out['good_cut']:.3f} | unfold "
                  f"{out.get('after_unfold', {}).get('words', out.get('error', '?')):14s} "
                  f"{100 * out.get('after_unfold', {}).get('fold', 0):5.1f}% check {out.get('after_unfold', {}).get('check', '-'):8s}"
                  f"| down found {out.get('face_down_found', 0):.3f} true {out.get('face_down_true', 0):.3f} "
                  f"| final {f.get('words', '-'):14s} {100 * f.get('fold', 0):5.1f}% {f.get('check', '-')} "
                  f"| guarded {out.get('guarded', {}).get('words', '-')} ({out.get('guarded_choice', '-')}) "
                  f"{out.get('seconds', 0):.0f}s", flush=True)
    save(part, rows, arrays)
    print("wrote", OUT)
