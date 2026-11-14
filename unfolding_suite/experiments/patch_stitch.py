"""Progressive flattening: grow patches from random seeds, flatten each, stitch neighbours pairwise, recurse.

The user's idea (2026-11-13): "choose a few random points and extend out until they reach other
flattened areas and then flatten those sub areas together recursively up".

  1. Strings: each dot joined to some neighbours (same three kinds as press_flat: nearest 10,
     link-checked, or the paper's true neighbours as a cheat control).
  2. Grow: pick S random seed dots. Every dot joins the seed it can reach by the shortest walk
     along strings (all seeds grow at the same speed until their regions meet).
  3. Flatten each region on its own: shortest walks inside the region + the usual flat layout.
     Each region also takes a 2-hop rim of its neighbours' dots, so neighbouring flat pieces share dots.
  4. Stitch, recursively: pair each piece with the neighbour it shares the most dots with. Turn,
     shift and (if needed) mirror one piece so the shared dots match best (no stretching),
     average the shared dots, and treat the pair as one piece. Repeat until one piece is left.
     `settle=True` also lets the strings of each newly joined piece relax within the flat plane
     (a few hundred steps), i.e. "flatten the sub areas together" rather than only lay them side by side.

Run: OMP_NUM_THREADS=1 .venv/bin/python -m unfolding_suite.experiments.patch_stitch [seeds...]
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import eigsh

from unfolding_suite.experiments.press_flat import SHEET_LIST, jump_count, knn, true_links
from unfolding_suite.link_check import checked_links
from unfolding_suite.run import failure_words
from unfolding_suite.scoring import score_layout, to_grid_order
from unfolding_suite.sheets import SHEETS, make_instance
from unfolding_suite.verify import verify

OUT = Path(__file__).resolve().parent.parent / "runs" / "patch_stitch"


def flatten_piece(graph, idx):
    """Shortest walks inside the piece, then the best flat layout (classical scaling)."""
    sub = graph[idx][:, idx]
    d = dijkstra(sub, directed=False)
    if np.isinf(d).any():                                   # piece split: keep the part holding most dots
        _, lab = connected_components(sub, directed=False)
        keep = lab == np.bincount(lab).argmax()
        idx, d = idx[keep], d[np.ix_(keep, keep)]
    d2 = d ** 2
    c = -0.5 * (d2 - d2.mean(0) - d2.mean(1)[:, None] + d2.mean())
    vals, vecs = np.linalg.eigh(c)
    return idx, vecs[:, -2:] * np.sqrt(np.maximum(vals[-2:], 0))


def fit_onto(src, dst):
    """Turn (mirror allowed) + shift that best moves src onto dst. Returns a function."""
    ms, md = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - ms).T @ (dst - md))
    R = u @ vt
    return lambda P: (P - ms) @ R + md


def settle_piece(points, links_all, idx, coords, iters):
    """Let the strings inside a piece relax in the flat plane."""
    pos = -np.ones(len(points), int)
    pos[idx] = np.arange(len(idx))
    inside = (pos[links_all[:, 0]] >= 0) & (pos[links_all[:, 1]] >= 0)
    a, b = pos[links_all[inside, 0]], pos[links_all[inside, 1]]
    if len(a) == 0:
        return coords
    d2 = ((points[links_all[inside, 0]] - points[links_all[inside, 1]]) ** 2).sum(1)
    h2 = np.median(d2)
    m, n = len(a), len(idx)
    inc = csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)], np.r_[a, b])),
                     shape=(m, n)).T.tocsr()

    def energy(x):
        Y = x.reshape(n, 2)
        diff = Y[a] - Y[b]
        s = ((diff ** 2).sum(1) - d2) / h2
        return (s ** 2).sum(), (inc @ ((4 * s / h2)[:, None] * diff)).ravel()

    return minimize(energy, coords.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": iters}).x.reshape(n, 2)


def patch_stitch(points, links, n_seeds=20, rim=2, settle=False, rng=None, history=None):
    rng = rng or np.random.default_rng(0)
    n = len(points)
    links = np.unique(np.sort(links, 1), axis=0)
    w = np.linalg.norm(points[links[:, 0]] - points[links[:, 1]], axis=1)
    graph = csr_matrix((np.r_[w, w], (np.r_[links[:, 0], links[:, 1]], np.r_[links[:, 1], links[:, 0]])), shape=(n, n))
    hops = csr_matrix((np.ones(2 * len(links)), (np.r_[links[:, 0], links[:, 1]], np.r_[links[:, 1], links[:, 0]])),
                      shape=(n, n))
    seeds = rng.choice(n, n_seeds, replace=False)
    owner = dijkstra(graph, directed=False, indices=seeds).argmin(0)        # 2. grow
    pieces = []
    for s in range(n_seeds):                                                 # 3. flatten each region (+ rim)
        member = owner == s
        for _ in range(rim):
            member = member | (hops @ member.astype(float) > 0)
        idx, xy = flatten_piece(graph, np.flatnonzero(member))
        pieces.append((idx, xy))
    if history is not None:
        history.append(("owner", owner))
    rounds = 0
    while len(pieces) > 1:                                                   # 4. stitch pairwise, recurse
        rounds += 1
        sets = [set(p[0].tolist()) for p in pieces]
        pairs = sorted(((len(sets[i] & sets[j]), i, j) for i in range(len(pieces)) for j in range(i + 1, len(pieces))
                        if len(sets[i] & sets[j]) >= 3), reverse=True)
        if not pairs:
            raise RuntimeError("pieces no longer share dots")
        used, merged = set(), []
        for _, i, j in pairs:
            if i in used or j in used:
                continue
            used |= {i, j}
            (ia, xa), (ib, xb) = pieces[i], pieces[j]
            shared = np.intersect1d(ia, ib)
            pa = dict(zip(ia.tolist(), range(len(ia))))
            pb = dict(zip(ib.tolist(), range(len(ib))))
            sa = np.array([pa[k] for k in shared]); sb = np.array([pb[k] for k in shared])
            xb = fit_onto(xb[sb], xa[sa])(xb)
            idx = np.union1d(ia, ib)
            pos = {k: t for t, k in enumerate(idx.tolist())}
            acc, cnt = np.zeros((len(idx), 2)), np.zeros(len(idx))
            for ii, xx in ((ia, xa), (ib, xb)):
                at = np.array([pos[k] for k in ii.tolist()])
                acc[at] += xx; cnt[at] += 1
            xy = acc / cnt[:, None]
            if settle:
                xy = settle_piece(points, links, idx, xy, 300)
            merged.append((idx, xy))
        merged += [p for k, p in enumerate(pieces) if k not in used]
        pieces = merged
    idx, xy = pieces[0]
    flat = np.full((n, 2), np.nan)
    flat[idx] = xy
    if np.isnan(flat).any():                                                 # dots dropped from split pieces
        missing = np.isnan(flat[:, 0])
        flat[missing] = xy.mean(0)
    return flat, rounds


def run_one(job):
    name, seed, which, n_seeds, settle = job
    inst = make_instance(SHEETS[name], 65, seed)
    links = {"nearest": lambda: knn(inst.points),
             "checked": lambda: checked_links(inst.points, 10, 12, 0.5),
             "true": lambda: true_links(inst)}[which]()
    t = time.time()
    try:
        flat, rounds = patch_stitch(inst.points, links, n_seeds, settle=settle, rng=np.random.default_rng(seed))
        r = score_layout(to_grid_order(flat, inst), inst.uv, inst.xyz, inst.grid, inst.sheet.family)
        words, passed, fold = failure_words(r), bool(r["passed"]), float(max(r["fold_share"], r["overlap_share"]))
        layers = verify(inst.points, flat)["reading"]
    except Exception as e:                                                   # noqa: BLE001
        words, passed, fold, layers, rounds, flat = f"crashed: {e}"[:40], False, 1.0, "-", 0, None
    return {"sheet": name, "family": inst.sheet.family, "seed": seed, "links": which, "n_seeds": n_seeds,
            "settle": settle, "jumps": jump_count(inst, links), "passed": passed, "words": words, "fold": fold,
            "layers": layers, "rounds": rounds, "seconds": round(time.time() - t, 1),
            "flat": flat.tolist() if (flat is not None and seed == 100) else None}


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [100]
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, seed, w, ns, st) for s in SHEET_LIST for seed in seeds for w in ("true", "nearest", "checked")
            for ns in (8, 40) for st in (False, True)]
    rows = []
    with Pool(2) as pool:
        for r in pool.imap_unordered(run_one, jobs):
            rows.append(r)
            print(f"{r['sheet']:13s} {r['links']:8s} seeds {r['n_seeds']:3d} {'settle' if r['settle'] else 'rigid ':6s} "
                  f"jumps {r['jumps']:5d} | {'PASS' if r['passed'] else 'fail'} {r['words']:16s} | layers {r['layers']:8s}"
                  f" | {r['seconds']}s", flush=True)
    (OUT / f"seeds{'_'.join(map(str, seeds))}.json").write_text(json.dumps(rows))
