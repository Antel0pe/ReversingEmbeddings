"""Pressing the paper flat: two plates close on the sheet while strings keep dot-to-dot lengths.

The user's idea (2026-11-13): put the crumpled paper between a table and a plate and press.
Because the paper cannot stretch, pressing pushes every point down AND out. Digitally:

  - strings join each dot to some neighbours, each as long as that pair's 3-D distance
    (the same strings as `pull.py`);
  - two parallel plates sit on either side of the cloud's best-fit plane (the plane the cloud
    is spread along most, from its spread matrix). A dot feels nothing while it is between the
    plates; a dot outside them is pushed back in, hard. In more than 3 dimensions the "plates"
    are a slab around a d-dimensional flat: every direction off the flat is squeezed.
  - the gap between the plates shrinks step by step to zero; after every step the dots settle
    (L-BFGS on strings + plates). The final flat answer is each dot's position in the plane.

Nothing stops the paper passing through itself, just as in `pull.py`. `--collide` adds a
soft "paper has thickness" push between dots that are close in space but not joined by a string.

Variants of strings: "nearest" (10 closest dots, the fair version), "checked" (link check
from link_check.py), "true" (the paper's own 10 nearest neighbours: a cheat control that asks
"does pressing work at all if the strings are perfect?").

Run: OMP_NUM_THREADS=1 .venv/bin/python -m unfolding_suite.experiments.press_flat [part] [seeds...]
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from unfolding_suite.diagnose import FAR
from unfolding_suite.link_check import checked_links
from unfolding_suite.run import failure_words
from unfolding_suite.scoring import score_layout, to_grid_order
from unfolding_suite.sheets import SHEETS, make_instance
from unfolding_suite.verify import verify
import unfolding_suite.holdout  # noqa: F401  registers the h_ sheets

OUT = Path(__file__).resolve().parent.parent / "runs" / "press_flat"
SHEET_LIST = ["flat", "wave", "accordion", "cone", "s_curl", "roll", "tight_roll", "h_roll5",
              "bowl", "dents", "crumple", "hard_crumple", "h_crumple17", "h_crumple29"]


def knn(points, k=10):
    _, near = cKDTree(points).query(points, k=k + 1)
    return np.c_[np.repeat(np.arange(len(points)), k), near[:, 1:].ravel()]


def true_links(inst, k=10):
    _, near = cKDTree(inst.uv).query(inst.uv, k=k + 1)
    g = np.c_[np.repeat(np.arange(len(inst.uv)), k), near[:, 1:].ravel()]
    where = np.empty(len(inst.order), int)
    where[inst.order] = np.arange(len(inst.order))
    return where[g]


def press(points, links, dim=2, steps=60, iters=900, plate=30.0, collide=0.0, record=None, shrink=0.9, push=0.0):
    """Close two plates on the sheet. Returns (pressed dots, in-plane coordinates, info)."""
    links = np.unique(np.sort(links, 1), axis=0)
    i, j = links.T
    d2 = ((points[i] - points[j]) ** 2).sum(1)
    h2 = np.median(d2)
    h = np.sqrt(h2)
    X = points - points.mean(0)
    n, D = X.shape
    axes = np.linalg.svd(X, full_matrices=False)[2]
    inplane, offplane = axes[:dim].T, axes[dim:].T          # (D, dim), (D, D-dim)
    m = len(links)
    incidence = csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)], np.r_[i, j])),
                           shape=(m, n)).T.tocsr()
    linked = set(map(tuple, links))

    def energy(flat_x, gap, pairs):
        Y = flat_x.reshape(n, D)
        diff = Y[i] - Y[j]
        strain = ((diff ** 2).sum(1) - d2) / h2
        e = (strain ** 2).sum()
        grad = incidence @ ((4 * strain / h2)[:, None] * diff)
        off = Y @ offplane                                  # (n, D-dim) height off the flat
        r = np.linalg.norm(off, axis=1)
        out = np.maximum(r - gap / 2, 0)                    # how far outside the slab
        e += plate * (out ** 2).sum() / h2
        if push:                                            # optional: also pull outward within the plane
            c = (Y - Y.mean(0)) @ inplane
            e -= push * (c ** 2).sum() / h2
            grad -= 2 * push / h2 * (c @ inplane.T)
        safe = np.where(r > 0, r, 1)
        grad += (2 * plate * out / h2 / safe)[:, None] * off @ offplane.T
        if collide and len(pairs):
            a, b = pairs.T
            dv = Y[a] - Y[b]
            dist = np.linalg.norm(dv, axis=1)
            short = np.maximum(0.6 * h - dist, 0)           # paper thickness ~0.6 dot spacings
            e += collide * (short ** 2).sum() / h2
            g = (-2 * collide * short / h2 / np.maximum(dist, 1e-12))[:, None] * dv
            np.add.at(grad, a, g)
            np.add.at(grad, b, -g)
        return e, grad.ravel()

    start = 2 * np.abs(X @ offplane).max() if D - dim == 1 else 2 * np.linalg.norm(X @ offplane, axis=1).max()
    gaps = np.r_[start * (shrink ** np.arange(steps)), 0.0]
    pairs = np.zeros((0, 2), int)
    for gap in gaps:
        if collide:
            cand = np.array(sorted(cKDTree(X).query_pairs(0.8 * h)), int).reshape(-1, 2)
            keep = [k for k, p in enumerate(map(tuple, cand)) if p not in linked]
            pairs = cand[keep]
        X = minimize(energy, X.ravel(), args=(gap, pairs), jac=True, method="L-BFGS-B",
                     options={"maxiter": iters}).x.reshape(n, D)
        if record is not None:
            record.append(X.copy())
    stretch = np.sqrt(((X[i] - X[j]) ** 2).sum(1) / d2)
    thick = np.linalg.norm(X @ offplane, axis=1)
    return X, X @ inplane, {"strain_median": float(np.median(np.abs(stretch - 1))),
                            "strain_p99": float(np.quantile(np.abs(stretch - 1), 0.99)),
                            "thickness_p99_spacings": float(np.quantile(thick, 0.99) / h)}


def jump_count(inst, links):
    g = inst.order[links]
    return int((np.linalg.norm(inst.uv[g[:, 0]] - inst.uv[g[:, 1]], axis=1) > FAR).sum())


def run_one(job):
    name, seed, which, collide, push = job
    inst = make_instance(SHEETS[name], 65, seed)
    links = {"nearest": lambda: knn(inst.points),
             "checked": lambda: checked_links(inst.points, 10, 12, 0.5),
             "true": lambda: true_links(inst)}[which]()
    t = time.time()
    X, flat, info = press(inst.points, links, collide=collide, push=push)
    r = score_layout(to_grid_order(flat, inst), inst.uv, inst.xyz, inst.grid, inst.sheet.family)
    v = verify(inst.points, flat)
    return {"sheet": name, "family": inst.sheet.family, "seed": seed, "links": which, "collide": collide, "push": push,
            "jumps": jump_count(inst, links), "passed": bool(r["passed"]), "words": failure_words(r),
            "fold": float(max(r["fold_share"], r["overlap_share"])),
            "square_median": r.get("square_error_median"), "layers": v["reading"],
            "seconds": round(time.time() - t, 1), **info,
            "flat": flat.tolist() if seed == 100 else None}


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "main"
    seeds = [int(s) for s in sys.argv[2:]] or [100]
    OUT.mkdir(parents=True, exist_ok=True)
    if part == "main":
        jobs = [(s, seed, w, 0.0, 0.0) for s in SHEET_LIST for seed in seeds for w in ("true", "nearest", "checked")]
        jobs += [(s, seed, w, 0.0, 1e-3) for s in SHEET_LIST for seed in seeds for w in ("true", "checked")]
    else:                                                    # "collide": paper with thickness
        jobs = [(s, seed, w, 30.0, 0.0) for s in ["wave", "s_curl", "roll", "h_roll5", "crumple"]
                for seed in seeds for w in ("true", "nearest")]
    rows = []
    with Pool(4) as pool:
        for r in pool.imap_unordered(run_one, jobs):
            rows.append(r)
            print(f"{r['sheet']:13s} {r['links']:8s} {'press+pull' if r['push'] else 'press':10s} seed {r['seed']} jumps {r['jumps']:5d} | "
                  f"{'PASS' if r['passed'] else 'fail'} {r['words']:16s} | layers {r['layers']:8s} | "
                  f"strain med {r['strain_median']:.3f} p99 {r['strain_p99']:.3f} | {r['seconds']}s", flush=True)
    (OUT / f"{part}_seeds{'_'.join(map(str, seeds))}.json").write_text(json.dumps(rows))
