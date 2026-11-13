"""Pulling the paper apart: does it make the hard sheets easier?

Two ways to "pull", both using only the dots (the truth is used only to judge):

  1. One-direction stretch. Find the cloud's three main axes (eigenvectors of its spread),
     multiply every dot's position along ONE axis by a factor. Re-pick each dot's 10
     nearest dots in the stretched cloud; measure link lengths in the original cloud;
     usual walking layout.
  2. Physical pull. Keep a set of short links at their original 3-D length (like
     strings) while pushing every dot away from the middle (maximising spread). The
     paper is allowed to pass through itself. Then either drop the pulled shape onto
     its best plane, or re-pick nearest dots in the pulled shape and lay out as usual.

Run: OMP_NUM_THREADS=1 .venv/bin/python -m unfolding_suite.experiments.pull_apart [part]
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from unfolding_suite.diagnose import FAR
from unfolding_suite.link_check import checked_links
from unfolding_suite.methods import layout_from_links
from unfolding_suite.pull import pull, spread_shares  # the pull now lives in the suite as method pull_k10
from unfolding_suite.run import failure_words
from unfolding_suite.scoring import score_layout, to_grid_order
from unfolding_suite.sheets import SHEETS, make_instance

OUT = Path(__file__).resolve().parent.parent / "runs" / "pull_apart"
HARD = ["flat", "wave", "s_curl", "roll", "tight_roll", "h_roll5", "crumple", "hard_crumple",
        "h_crumple17", "h_crumple29"]


def knn(points, k=10):
    _, near = cKDTree(points).query(points, k=k + 1)
    return np.c_[np.repeat(np.arange(len(points)), k), near[:, 1:].ravel()]


def jump_count(inst, links):
    """Links joining dots more than 0.3 apart on the paper (truth, for judging only)."""
    g = inst.order[links]
    return int((np.linalg.norm(inst.uv[g[:, 0]] - inst.uv[g[:, 1]], axis=1) > FAR).sum())


def score(inst, flat):
    r = score_layout(to_grid_order(flat, inst), inst.uv, inst.xyz, inst.grid, inst.sheet.family)
    return {"passed": r["passed"], "words": failure_words(r), "fold": max(r["fold_share"], r["overlap_share"])}


# ------------------------------------------------------------------ 1. one-direction stretch
def stretch_along(points, axis_rank, factor):
    centred = points - points.mean(0)
    _, vecs = np.linalg.eigh(np.cov(centred.T))
    d = vecs[:, ::-1][:, axis_rank]                      # 0 = most spread, 2 = least spread
    return centred + (factor - 1) * np.outer(centred @ d, d)


def part1(job):
    name, seed = job
    inst = make_instance(SHEETS[name], 65, seed)
    rows = []
    for axis_rank, factor in [(0, 1)] + [(a, f) for a in (0, 1, 2) for f in (2, 4)]:
        links = knn(stretch_along(inst.points, axis_rank, factor))
        try:
            flat, _ = layout_from_links(inst.points, links, 2)
            s = score(inst, flat)
        except RuntimeError as e:
            s = {"passed": False, "words": "broke apart", "fold": 1.0}
        rows.append({"sheet": name, "seed": seed, "axis": axis_rank, "factor": factor,
                     "jumps": jump_count(inst, links), **s})
    return rows


# ------------------------------------------------------------------ 2. physical pull
def part2(job):
    name, seed, which = job
    inst = make_instance(SHEETS[name], 65, seed)
    if which == "nearest":
        links = knn(inst.points)
    elif which == "checked":
        links = checked_links(inst.points, 10, 12, 0.5)
    else:                                                # "true": the paper's own neighbours (cheat control)
        _, near = cKDTree(inst.uv).query(inst.uv, k=11)
        g = np.c_[np.repeat(np.arange(len(inst.uv)), 10), near[:, 1:].ravel()]
        where = np.empty(len(inst.order), int)
        where[inst.order] = np.arange(len(inst.order))
        links = where[g]
    X, history, strain = pull(inst.points, links)
    centred = X - X.mean(0)
    plane = np.linalg.svd(centred, full_matrices=False)[2][:2].T
    s_plane = score(inst, centred @ plane)
    relinks = knn(X)
    try:
        s_relink = score(inst, layout_from_links(inst.points, relinks, 2)[0])
    except RuntimeError:
        s_relink = {"passed": False, "words": "broke apart", "fold": 1.0}
    return {"sheet": name, "seed": seed, "links": which, "jumps_held": jump_count(inst, links),
            "jumps_after_relink": jump_count(inst, relinks), "jumps_before": jump_count(inst, knn(inst.points)),
            "spread_before": history[0], "spread_after": history[-1], "history": history,
            "link_strain_median": strain, "plane": s_plane, "relink": s_relink}


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "1"
    seeds = [int(s) for s in sys.argv[2:]] or [100]
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(4) as pool:
        if part == "1":
            rows = [r for rs in pool.map(part1, [(s, seed) for s in HARD for seed in seeds]) for r in rs]
            for r in rows:
                print(f"{r['sheet']:13s} seed {r['seed']} axis {r['axis']} x{r['factor']}: jumps {r['jumps']:5d}  {r['words']}")
        else:
            jobs = [(s, seed, w) for s in HARD for seed in seeds for w in ("nearest", "checked", "true")]
            rows = []
            for r in pool.imap(part2, jobs):
                rows.append(r)
                print(f"{r['sheet']:13s} {r['links']:8s} held jumps {r['jumps_held']:4d} | 3rd axis share "
                      f"{r['spread_before'][2]:.3f} -> {r['spread_after'][2]:.4f} | strain {r['link_strain_median']:.3f}"
                      f" | plane: {r['plane']['words']:12s} | relink jumps {r['jumps_before']}->{r['jumps_after_relink']}"
                      f" {r['relink']['words']}", flush=True)
    (OUT / f"part{part}_seeds{'_'.join(map(str, seeds))}.json").write_text(json.dumps(rows, indent=1))
