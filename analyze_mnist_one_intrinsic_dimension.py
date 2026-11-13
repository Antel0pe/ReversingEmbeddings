"""How many dimensions do MNIST 1s have? Several analytic counts, each calibrated.

Run from the repo root (parts can run separately):
    OPENBLAS_NUM_THREADS=4 .venv/bin/python analyze_mnist_one_intrinsic_dimension.py [part ...]
Parts: size  blursize  corr  local  blur  (default: all); add only=name,name to restrict
datasets (results merge into each part's json).

No learned model anywhere. Every count is a closed-form statistic of distances
or of local covariance matrices:

  TWONN    Facco et al. 2017. For each image, ratio of its 2nd to 1st nearest
           neighbour distance. On a d-dimensional set this ratio mu obeys
           P(mu > m) = m^-d, so d = n / sum(log mu).
  MLE      Levina & Bickel 2004, MacKay-Ghahramani averaging. Uses the 1st..k-th
           neighbour distances: d = 1 / mean_x[(1/(k-1)) sum_j log(T_k / T_j)].
  NN-SHRINK  How fast the nearest-neighbour distance shrinks as images are
           added. In d dimensions doubling the data shrinks it by 2^(-1/d), so
           d = -delta(log n) / delta(mean log r1).
  CORR     Grassberger-Procaccia correlation dimension. C(r) = fraction of pairs
           closer than r grows like r^d; the local slope of log C against log r
           is read at every scale.
  LOCAL    Held-out local planes. Remove an image, take its 50 nearest
           neighbours, fit the flat j-dimensional sheet through them (their mean
           plus top j covariance directions), and measure how much of the image's
           offset from the neighbourhood mean is still unexplained. On a smooth
           d-dimensional set the unexplained part collapses at j = d.

Data:
  MNIST 1s       MNIST train 1s plus QMNIST test 1s (QMNIST rebuilt the 60k
                 test set; its first 10k match MNIST test to 0.4% L2). 13,533.
  stroke families  `knob_strokes.make(n, K)`: drawn 1s with K = 5, 8, 12, 16
                 knobs, so the true answer is known. Every count is read against
                 these at the SAME number of images; an estimator that says 9 on
                 a 12-knob family at n = 13,533 is calibrated by that.
  generated 1s   `grey_ones.render` over its default box (the repo's 5-knob family).
  wobbly families  `knob_strokes.make_wobbly`: 8 or 12 big knobs plus 40 tiny
                 wobble knobs (true dimension 48 or 52), to test whether "a few
                 big knobs plus many small ones" reproduces MNIST's pattern of a
                 count that keeps rising as images are added.
"""
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

import knob_strokes
import mnist
from grey_ones import RANGES as GREY_RANGES, render as grey_render

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures/mnist_one_intrinsic_dimension"
CACHE = ROOT / "data/intrinsic_dimension"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

KS = [5, 8, 12, 16]
N_FAMILY = 32000
SIZES = [500, 1000, 2000, 4000, 8000, 13533]
BIG = [32000]
REPEATS = 3


# --------------------------------------------------------------------- data
def mnist_ones():
    im, lb = mnist.load("train")
    with gzip.open(ROOT / "data/qmnist/qmnist-test-images-idx3-ubyte.gz", "rb") as f:
        f.read(16)
        q = np.frombuffer(f.read(), np.uint8).reshape(-1, 28, 28)
    with gzip.open(ROOT / "data/qmnist/qmnist-test-labels-idx2-int.gz", "rb") as f:
        f.read(12)
        ql = np.frombuffer(f.read(), ">i4").reshape(-1, 8)[:, 0]
    X = np.concatenate([im[lb == 1], q[ql == 1]]).reshape(-1, 784)
    X = np.unique(X, axis=0).astype(np.float32) / 255.0      # drop exact duplicates
    return np.random.default_rng(0).permutation(X)


def stroke_family(k, n=N_FAMILY):
    path = CACHE / f"strokes_{k}_{n}.npy"
    if not path.exists():
        np.save(path, knob_strokes.make(n, k, seed=100 + k).reshape(n, 784))
    return np.load(path)


def grey_family(n=N_FAMILY):
    path = CACHE / f"grey_{n}.npy"
    if not path.exists():
        rng = np.random.default_rng(7)
        P = rng.uniform(GREY_RANGES[:, 0], GREY_RANGES[:, 1], (n, 5))
        np.save(path, grey_render(P).reshape(n, 784))
    return np.load(path)


WOBBLY = [(8, 40), (12, 40)]     # (big knobs, tiny knobs)
WOBBLE_AMP = 0.3


def wobbly_family(k, t, n=N_FAMILY):
    path = CACHE / f"wobbly_{k}_{t}_{n}.npy"
    if not path.exists():
        X = knob_strokes.make_wobbly(n, k, t, WOBBLE_AMP, seed=200 + k)
        np.save(path, X.reshape(n, 784))
    return np.load(path)


def datasets(only=None):
    makers = {"mnist": mnist_ones, "grey_5": grey_family}
    for k in KS:
        makers[f"strokes_{k}"] = lambda k=k: stroke_family(k)
    for k, t in WOBBLY:
        makers[f"wobbly_{k}_{t}"] = lambda k=k, t=t: wobbly_family(k, t)
    return {name: f() for name, f in makers.items() if only is None or name in only}


# --------------------------------------------------------------------- tools
def knn(X, k, chunk=2000):
    """Distances to the k nearest other rows, ascending. (n, k)."""
    X = np.ascontiguousarray(X, np.float32)
    sq = (X.astype(np.float64) ** 2).sum(1)
    out = np.empty((len(X), k), np.float64)
    idx = np.empty((len(X), k), np.int64)
    for a in range(0, len(X), chunk):
        g = X[a:a + chunk] @ X.T
        d2 = sq[a:a + chunk, None] + sq[None] - 2 * g.astype(np.float64)
        d2[np.arange(len(g)), np.arange(a, a + len(g))] = np.inf
        part = np.argpartition(d2, k, axis=1)[:, :k]
        pd = np.take_along_axis(d2, part, 1)
        o = np.argsort(pd, 1)
        out[a:a + chunk] = np.sqrt(np.maximum(np.take_along_axis(pd, o, 1), 0))
        idx[a:a + chunk] = np.take_along_axis(part, o, 1)
    return out, idx


def twonn(D):
    mu = D[:, 1] / D[:, 0]
    return len(mu) / np.log(mu).sum()


def mle(D, k):
    T = D[:, :k]
    inv = np.log(T[:, -1:] / T[:, :-1]).sum(1) / (k - 1)
    return 1.0 / inv.mean()


def counts(X):
    D, _ = knn(X, 21)
    scale = np.median(np.linalg.norm(X - X.mean(0), axis=1))
    return {"twonn": twonn(D), "mle10": mle(D, 10), "mle20": mle(D, 20),
            "log_r1": float(np.log(D[:, 0]).mean()),
            "nn1_rel": float(np.median(D[:, 0]) / scale)}


def dump(name, obj):
    """Merge into the part's json, so a part can be re-run on a few datasets."""
    path = OUT / f"{name}.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(json.loads(json.dumps(obj)))
    path.write_text(json.dumps(old, indent=1))


# --------------------------------------------------------------------- parts
def part_size(data):
    """TwoNN, MLE and nn-distance against the number of images."""
    rng = np.random.default_rng(1)
    res = {}
    for name, X in data.items():
        sizes = SIZES + (BIG if name != "mnist" else [])
        res[name] = {}
        for n in sizes:
            reps = 1 if n >= len(X) or n >= 13533 else REPEATS
            rows = []
            for _ in range(reps):
                sub = X[rng.choice(len(X), min(n, len(X)), replace=False)]
                rows.append(counts(sub))
            res[name][n] = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
            r = res[name][n]
            print(f"size {name:10s} n={n:6d} twonn {r['twonn']:6.2f} mle10 {r['mle10']:6.2f}"
                  f" mle20 {r['mle20']:6.2f} nn1 {r['nn1_rel']:.3f}", flush=True)
        ns = sorted(res[name])
        for a, b in zip(ns[:-1], ns[1:]):
            dlogr = res[name][b]["log_r1"] - res[name][a]["log_r1"]
            res[name][b]["nn_shrink"] = float(-np.log(b / a) / dlogr) if dlogr < 0 else None
    dump("size", res)


def part_corr(data, n=13533, bins=120):
    """Correlation dimension: slope of log C(r) at every scale."""
    rng = np.random.default_rng(2)
    res = {}
    for name, X in data.items():
        X = X[rng.choice(len(X), min(n, len(X)), replace=False)].astype(np.float32)
        sq = (X.astype(np.float64) ** 2).sum(1)
        # radii relative to the median pair distance
        probe = X[rng.choice(len(X), 400, replace=False)]
        med = np.median(np.sqrt(np.maximum(
            (probe ** 2).sum(1)[:, None] + sq[None] - 2 * probe @ X.T, 0)))
        edges = np.geomspace(0.02, 3.0, bins + 1) * med
        hist = np.zeros(bins, np.int64)
        below = 0
        for a in range(0, len(X), 2000):
            d2 = sq[a:a + 2000, None] + sq[None, :] - 2 * (X[a:a + 2000] @ X.T)
            iu = np.arange(len(d2))[:, None] < (np.arange(len(X))[None] - a)  # pairs j > i
            d = np.sqrt(np.maximum(d2[iu], 0))
            below += int((d < edges[0]).sum())
            hist += np.histogram(d, edges)[0]
        C = (below + np.cumsum(hist)) / (len(X) * (len(X) - 1) / 2)
        r = edges[1:] / med
        res[name] = {"r_rel": r.tolist(), "C": C.tolist()}
        print(f"corr {name:10s} done; pairs below 0.1*median: {C[np.searchsorted(r, 0.1)]:.2e}",
              flush=True)
    dump("corr", res)


def part_local(data, n=13533, centres=600, k=50, jmax=30):
    """Held-out local planes: how many directions point from neighbours to a 1."""
    rng = np.random.default_rng(3)
    res = {}
    for name, X in data.items():
        X = X[rng.choice(len(X), min(n, len(X)), replace=False)]
        _, idx = knn(X, k)
        cs = rng.choice(len(X), centres, replace=False)
        curves = np.empty((centres, jmax + 1))
        spectra = np.empty((centres, min(k, 40)))
        for i, c in enumerate(cs):
            N = X[idx[c]].astype(np.float64)
            m = N.mean(0)
            _, s, Vt = np.linalg.svd(N - m, full_matrices=False)
            off = X[c] - m
            proj = Vt[:jmax] @ off
            tot = off @ off
            curves[i] = 1 - np.concatenate([[0], np.cumsum(proj ** 2)]) / tot
            spectra[i] = (s[:spectra.shape[1]] ** 2) / (s ** 2).sum()
        res[name] = {"median": np.median(curves, 0).tolist(),
                     "q25": np.quantile(curves, 0.25, 0).tolist(),
                     "q75": np.quantile(curves, 0.75, 0).tolist(),
                     "spectrum": np.median(spectra, 0).tolist()}
        med = res[name]["median"]
        print(f"local {name:10s} unexplained at j=0,5,10,20,30: "
              + " ".join(f"{med[j]:.3f}" for j in (0, 5, 10, 20, 30)), flush=True)
    dump("local", res)


def part_blur(data, n=13533, sigmas=(0.0, 0.5, 1.0, 2.0)):
    """Coarse-grain every image by a Gaussian blur and count again."""
    rng = np.random.default_rng(4)
    res = {}
    for name, X in data.items():
        X = X[rng.choice(len(X), min(n, len(X)), replace=False)].reshape(-1, 28, 28)
        res[name] = {}
        for s in sigmas:
            Y = gaussian_filter(X, (0, s, s)) if s > 0 else X
            res[name][s] = counts(Y.reshape(len(Y), 784))
            r = res[name][s]
            print(f"blur {name:10s} sigma={s:.1f} twonn {r['twonn']:6.2f} mle20 {r['mle20']:6.2f}",
                  flush=True)
    dump("blur", res)


def part_blursize(data, sigma=2.0):
    """Does the count still climb with more images once fine detail is blurred away?"""
    rng = np.random.default_rng(5)
    res = {}
    for name, X in data.items():
        sizes = SIZES + (BIG if name != "mnist" else [])
        Y = gaussian_filter(X.reshape(-1, 28, 28), (0, sigma, sigma)).reshape(len(X), 784)
        res[name] = {}
        for n in sizes:
            reps = 1 if n >= 13533 else REPEATS
            rows = [counts(Y[rng.choice(len(Y), min(n, len(Y)), replace=False)])
                    for _ in range(reps)]
            res[name][n] = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
            r = res[name][n]
            print(f"blursize {name:12s} n={n:6d} twonn {r['twonn']:6.2f} mle20 {r['mle20']:6.2f}",
                  flush=True)
    dump("blursize", res)


PARTS = {"size": part_size, "blursize": part_blursize, "corr": part_corr, "local": part_local, "blur": part_blur}

if __name__ == "__main__":
    t0 = time.time()
    args = sys.argv[1:]
    only = next((a[5:].split(",") for a in args if a.startswith("only=")), None)
    parts = [a for a in args if not a.startswith("only=")] or list(PARTS)
    data = datasets(only)
    print({k: v.shape for k, v in data.items()}, f"{time.time() - t0:.0f}s", flush=True)
    for p in parts:
        PARTS[p](data)
        print(f"-- {p} done at {time.time() - t0:.0f}s", flush=True)
