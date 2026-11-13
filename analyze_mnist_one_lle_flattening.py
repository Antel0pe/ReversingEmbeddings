"""Flatten the MNIST 1s with the unfolding suite's locally linear method, and see how close it gets.

Run from the repo root:
    OPENBLAS_NUM_THREADS=4 .venv/bin/python analyze_mnist_one_lle_flattening.py

The method is `lle_k10` from unfolding_suite/experiments/fold_flatten.py, used unchanged
(only k and the number of flat numbers vary). Each dot is written as a weighted mix of its
k nearest dots (the piece around it, treated as flat). The flat layout is the one that keeps
every dot's mix as well as possible all at once.

There is no true flat answer for MNIST, so every check uses only the images and the layout:

  KEPT       share of each image's 10 most similar images that are also among its 10
             nearest in the flat layout (1 = every neighbourhood survives)
  STRANGERS  share of flat-layout neighbours that are not even among the image's 100 most
             similar images (of 6,742): different-looking 1s laid on top of each other
  REBUILD    rebuild each image as the average of the images of its 10 flat neighbours;
             median miss as % of the image's own size. Ceilings: its 10 most similar images
             (best possible) and 10 random images (no information)
  NEW        how many layout numbers are not just a bent copy of the earlier ones
             (number j counts as new if the earlier numbers predict it with R^2 < 0.5)

Control: generated 1s (five knobs, uniform in the default box, same count), where the true
answer is five numbers. There we also ask how well the layout predicts each knob.
Baselines: the best flat plane (PCA) and the suite's walking method (isomap_k10).
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_predict

import mnist
from grey_ones import KNOBS, RANGES, render
from unfolding_suite.experiments.fold_flatten import lle
from unfolding_suite.methods import isomap

OUT = Path(__file__).resolve().parent / "figures/mnist_one_lle_flattening"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)
NN, FAR_RANK = 10, 100


def mnist_ones():
    X, y = mnist.load("train")
    return X[y == 1].reshape(-1, 784).astype(np.float64) / 255.0


def generated(n):
    p = RANGES[:, 0] + RNG.random((n, 5)) * (RANGES[:, 1] - RANGES[:, 0])
    return render(p, dtype=np.float64).reshape(n, -1), p


class Pixels:
    """Everything about the images that the checks need, computed once."""

    def __init__(self, X):
        self.X = X
        sq = (X ** 2).sum(1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * X @ X.T, 0))
        np.fill_diagonal(D, np.inf)
        self.D = D
        self.order = np.argsort(D, 1)
        self.near = self.order[:, :NN]
        self.cutoff = D[np.arange(len(X)), self.order[:, FAR_RANK - 1]]   # distance of the 100th most similar
        self.size = np.linalg.norm(X, axis=1)

    def rebuild_miss(self, neighbours):
        pred = self.X[neighbours].mean(1)
        return float(np.median(np.linalg.norm(pred - self.X, axis=1) / self.size) * 100)


def flat_neighbours(F, k=NN):
    sq = (F ** 2).sum(1)
    D = sq[:, None] + sq[None] - 2 * F @ F.T
    np.fill_diagonal(D, np.inf)
    return np.argpartition(D, k, 1)[:, :k]


def r2(target, source, k=NN):
    pred = cross_val_predict(KNeighborsRegressor(k), source, target, cv=5)
    return float(1 - ((pred - target) ** 2).sum() / ((target - target.mean()) ** 2).sum())


def new_directions(F):
    """Count layout numbers that the earlier numbers do not already predict."""
    shares = [r2(F[:, j], F[:, :j]) for j in range(1, F.shape[1])]
    return 1 + sum(s < 0.5 for s in shares), shares


def check(px, F, knobs=None):
    n = len(F)
    nb = flat_neighbours(F)
    rows = np.arange(n)[:, None]
    kept = np.mean([len(np.intersect1d(nb[i], px.near[i])) for i in range(n)]) / NN
    strangers = float((px.D[rows, nb] > px.cutoff[:, None]).mean())
    new, shares = new_directions(F)
    out = {"kept": float(kept), "strangers": strangers, "rebuild": px.rebuild_miss(nb),
           "new": new, "copy_r2": shares}
    if knobs is not None:
        out["knob_r2"] = {name: r2(knobs[:, j], F) for j, name in enumerate(KNOBS)}
    return out


def pca(X, d):
    Xc = X - X.mean(0)
    return Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:d].T


def run(name, X, knobs, dims, ks):
    t0 = time.time()
    px = Pixels(X)
    res = {"n": len(X),
           "ceiling_similar": px.rebuild_miss(px.near),
           "ceiling_random": px.rebuild_miss(RNG.integers(len(X), size=(len(X), NN))),
           "mean_image": float(np.median(np.linalg.norm(X.mean(0) - X, axis=1) / px.size) * 100)}
    if knobs is not None:
        res["knob_r2_from_pixels"] = {nm: r2(knobs[:, j], X) for j, nm in enumerate(KNOBS)}
    print(f"\n=== {name}: {len(X)} images   rebuild from 10 most similar {res['ceiling_similar']:.1f}%, "
          f"from 10 random {res['ceiling_random']:.1f}%   ({time.time() - t0:.0f}s)")
    layouts = {}
    for d in dims:
        layouts[f"plane d={d}"] = lambda d=d: pca(X, d)
        for k in ks:
            if k > d:
                layouts[f"lle k={k} d={d}"] = lambda d=d, k=k: lle(X, d, None, k=k)[0]
        layouts[f"walk k=10 d={d}"] = lambda d=d: isomap(X, d, None, k=10)[0]
    res["layouts"] = {}
    for label, make in layouts.items():
        t = time.time()
        try:
            F = make()
            r = check(px, F, knobs)
        except Exception as e:                        # the walking graph can break into pieces
            r = {"error": f"{type(e).__name__}: {e}"}
            print(f"  {label:18s} FAILED: {r['error']}")
            res["layouts"][label] = r
            continue
        r["seconds"] = time.time() - t
        res["layouts"][label] = r
        np.save(OUT / f"{name}__{label.replace(' ', '_').replace('=', '')}.npy", F.astype(np.float32))
        line = (f"  {label:18s} kept {r['kept']:.2f}  strangers {100 * r['strangers']:5.1f}%  "
                f"rebuild {r['rebuild']:5.1f}%  new {r['new']}/{F.shape[1]}")
        if knobs is not None:
            line += "  knobs R2 " + " ".join(f"{v:.2f}" for v in r["knob_r2"].values())
        print(line + f"   ({r['seconds']:.0f}s)", flush=True)
    return res


def main():
    results = {}
    Xg, pg = generated(6742)
    results["generated"] = run("generated", Xg, pg, dims=(2, 5, 10), ks=(10, 20, 40))
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    results["mnist"] = run("mnist", mnist_ones(), None, dims=(2, 5, 10, 15), ks=(10, 20, 40))
    (OUT / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
