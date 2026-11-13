"""Pull the 1s apart: the unfolding suite's pull method on generated 1s and on real MNIST 1s.

Each image is a point in 784-D pixel space. Strings join each image to its 10 closest images
(each string as long as that pair's pixel distance). Every image is pushed away from the middle
while the strings hold their length. Afterwards, the pulled cloud's most-spread directions are
its new coordinates (same as the suite method `pull_k10`, but keeping more than 2).

Checks:
  generated 1s  the five true knobs are known: can the first 5 pulled coordinates tell them apart?
  MNIST 1s      no true knobs: compare against five properties read off each image with ink
                moments (centre x, centre y, height, width, slant; the closed form of
                analyze_generated_one_moment_chart.py, which is exact for generated 1s)
Baseline before any pulling: the same checks on the unpulled images' own most-spread directions.

Run: OPENBLAS_NUM_THREADS=2 .venv/bin/python analyze_pull_ones.py tune|run
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import mnist
from grey_ones import KNOBS, RANGES, render
from unfolding_suite.methods import layout_from_links
from unfolding_suite.pull import pull

OUT = Path(__file__).resolve().parent / "figures" / "pull_ones"
N = 1000
PROPS = ["centre x", "centre y", "height", "width", "slant (deg)"]


def generated(n=N, seed=0):
    rng = np.random.default_rng(seed)
    knobs = RANGES[:, 0] + rng.random((n, 5)) * np.diff(RANGES).ravel()
    return render(knobs, dtype=np.float64), knobs


def mnist_ones(n=N, seed=0):
    X, y = mnist.load("train")
    ones = X[y == 1].astype(np.float64) / 255.0
    return ones[np.random.default_rng(seed).choice(len(ones), n, replace=False)]


def moment_properties(images):
    """Centre, height, width and slant read off the ink (exact for generated 1s)."""
    c = np.arange(28) + 0.5
    x, y = np.broadcast_to(c, (28, 28)), np.broadcast_to(c[:, None], (28, 28))
    ink = images.sum((1, 2))
    mx, my = (images * x).sum((1, 2)) / ink, (images * y).sum((1, 2)) / ink
    vx = (images * x * x).sum((1, 2)) / ink - mx ** 2 - 1 / 12
    vy = (images * y * y).sum((1, 2)) / ink - my ** 2 - 1 / 12
    cxy = (images * x * y).sum((1, 2)) / ink - mx * my
    shear = -cxy / vy
    return np.stack([mx, my, np.sqrt(12 * vy), np.sqrt(np.clip(12 * (vx - shear ** 2 * vy), 0, None)),
                     np.degrees(np.arctan(shear))], 1)


def closest_links(P, k=10):
    """k closest images; if that leaves separate groups, join each to the main one by its closest pair."""
    _, near = cKDTree(P).query(P, k=k + 1)
    links = np.c_[np.repeat(np.arange(len(P)), k), near[:, 1:].ravel()]
    while True:
        g = csr_matrix((np.ones(len(links)), (links[:, 0], links[:, 1])), shape=(len(P), len(P)))
        count, lab = connected_components(g, directed=False)
        if count == 1:
            return links
        main = lab == np.bincount(lab).argmax()
        d, j = cKDTree(P[main]).query(P[~main])
        a = np.where(~main)[0][d.argmin()]
        links = np.r_[links, [[a, np.where(main)[0][j[d.argmin()]]]]]


def top_coords(X, m=10):
    c = X - X.mean(0)
    _, s, vt = np.linalg.svd(c, full_matrices=False)
    return c @ vt[:m].T, (s ** 2 / (s ** 2).sum())


def knn_r2(C, T, k=10, folds=5, seed=0):
    """How well do these coordinates pin down each target? Held-out nearest-neighbour prediction, R^2."""
    idx = np.random.default_rng(seed).permutation(len(C))
    pred = np.zeros_like(T)
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        _, nb = cKDTree(C[train]).query(C[test], k=k)
        pred[test] = T[train][nb].mean(1)
    return 1 - ((pred - T) ** 2).sum(0) / ((T - T.mean(0)) ** 2).sum(0)


def run_one(name, images, targets, push, rounds=10, iters=200):
    P = images.reshape(len(images), -1)
    links = closest_links(P)
    t = time.time()
    X, history, strain = pull(P, links, push=push, rounds=rounds, iters=iters, verbose=True)
    raw_c, raw_s = top_coords(P)
    pul_c, pul_s = top_coords(X)
    walk_c, _ = layout_from_links(P, links, 10)                 # the suite's walking method, same strings
    walk_c = walk_c[:, ::-1]                                    # most-spread first
    out = {"push": push, "seconds": time.time() - t, "string_stretch_median": strain,
           "spread_raw_top10": raw_s[:10].round(4).tolist(), "spread_pulled_top10": pul_s[:10].round(4).tolist(),
           "spread_history_top6": [h[:6] for h in history]}
    for label, C in (("raw", raw_c), ("pulled", pul_c), ("walking", walk_c)):
        for d in (2, 5, 10):
            out[f"r2_{label}_{d}"] = {tn: dict(zip(cols, knn_r2(C[:, :d], T).round(3).tolist()))
                                      for tn, (T, cols) in targets.items()}
    np.savez_compressed(OUT / f"{name}_push{push:g}.npz", pulled=pul_c, raw=raw_c, walking=walk_c, links=links)
    return out


if __name__ == "__main__":
    part = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    gen_images, gen_knobs = generated()
    gen_targets = {"true knobs": (gen_knobs, KNOBS), "moment properties": (moment_properties(gen_images), PROPS)}
    if part == "tune":
        for push in [float(a) for a in sys.argv[2:]] or [1e-3]:
            r = run_one("generated", gen_images, gen_targets, push)
            print(json.dumps({k: r[k] for k in ("push", "seconds", "string_stretch_median", "spread_raw_top10",
                                                "spread_pulled_top10")}), flush=True)
            for key in ("r2_raw_5", "r2_walking_5", "r2_pulled_5"):
                print(f"  {key}", r[key]["true knobs"], flush=True)
    else:
        push = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
        results = {}
        mn = mnist_ones()
        which = sys.argv[3].split(",") if len(sys.argv) > 3 else ["generated", "mnist"]
        for name, images, targets in (("generated", gen_images, gen_targets),
                                      ("mnist", mn, {"moment properties": (moment_properties(mn), PROPS)})):
            if name not in which:
                continue
            results[name] = run_one(name, images, targets, push)
            r = results[name]
            print(name, f"{r['seconds']:.0f}s stretch {r['string_stretch_median']:.4f}")
            print("  spread raw   ", r["spread_raw_top10"])
            print("  spread pulled", r["spread_pulled_top10"])
            for key in ("r2_raw_5", "r2_walking_5", "r2_pulled_5", "r2_raw_10", "r2_walking_10", "r2_pulled_10"):
                print(f"  {key}", r[key], flush=True)
        path = OUT / "results.json"
        old = json.loads(path.read_text()) if path.exists() else {}
        old.update(results)
        path.write_text(json.dumps(old, indent=1))
