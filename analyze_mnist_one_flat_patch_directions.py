"""How many perpendicular directions does it take to reach every neighbour of a 1?

Run from the repo root:
    OPENBLAS_NUM_THREADS=4 .venv/bin/python analyze_mnist_one_flat_patch_directions.py

The user's procedure, literally:
  1. Pick random 1s ("centres").
  2. Sort every other 1 by pixel distance from the centre.
  3. Grow the neighbourhood: the nearest k neighbours, for k = 2 ... 200.
  4. For each neighbourhood, take the arrows from the centre to each neighbour
     (neighbour image minus centre image, 784 numbers each) and ask how many
     perpendicular directions are needed so that EVERY arrow is reached, i.e.
     what is left of each arrow after walking along those directions is at most
     a stated fraction of the arrow's own length. The directions are the
     uncentred SVD of the arrows (the best directions for the arrows as a group).
  5. "Locally linear" = the neighbourhood sizes over which this count stops
     growing. On a flat d-dimensional patch every new neighbour lies in the same
     d directions, so the count sits at d however many neighbours are added. On a
     bent or noisy patch every new neighbour brings some new direction.

Every arrow must be reached (worst neighbour, not a typical one), and the
tolerance is stated: 10% of the arrow's length left over (the directions carry
99% of that arrow's squared length), with 25% and 5% as alternatives.

Families, all with 13,533 images unless stated (MNIST's count):
  mnist        MNIST train 1s + QMNIST test 1s, exact duplicates dropped.
  gen          generated 1s (`grey_ones`, five knobs, default box). Known answer 5.
  gen_dense    the same five-knob family with 200,000 images, to show what the
               test gives when neighbours are close enough to sit on a flat patch.
  strokes12    `knob_strokes.make(n, 12)`: drawn 1s with 12 knobs. Known answer 12.
"""
import gzip
import json
import time
from pathlib import Path

import numpy as np

import knob_strokes
import mnist
from grey_ones import RANGES, render

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures/mnist_one_flat_patch_directions"
OUT.mkdir(parents=True, exist_ok=True)

N = 13533
N_DENSE = 200_000
CENTRES = 300
KMAX = 200
KS = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 125, 150, 200]
TOLS = [0.25, 0.10, 0.05]


def mnist_ones():
    im, lb = mnist.load("train")
    with gzip.open(ROOT / "data/qmnist/qmnist-test-images-idx3-ubyte.gz", "rb") as f:
        f.read(16)
        q = np.frombuffer(f.read(), np.uint8).reshape(-1, 28, 28)
    with gzip.open(ROOT / "data/qmnist/qmnist-test-labels-idx2-int.gz", "rb") as f:
        f.read(12)
        ql = np.frombuffer(f.read(), ">i4").reshape(-1, 8)[:, 0]
    X = np.concatenate([im[lb == 1], q[ql == 1]]).reshape(-1, 784)
    X = np.unique(X, axis=0).astype(np.float32) / 255.0
    return np.random.default_rng(0).permutation(X)


def generated(n, seed):
    rng = np.random.default_rng(seed)
    P = rng.uniform(RANGES[:, 0], RANGES[:, 1], (n, 5))
    return render(P).reshape(n, 784)


def nearest(X, centres, kmax, chunk=500):
    """Indices and distances of the kmax nearest other images to each centre."""
    X64 = X.astype(np.float64)
    sq = (X64 ** 2).sum(1)
    XT = np.ascontiguousarray(X64.T)
    idx = np.empty((len(centres), kmax), int)
    dist = np.empty((len(centres), kmax))
    for a in range(0, len(centres), chunk):
        c = centres[a:a + chunk]
        d2 = sq[c, None] + sq[None] - 2 * X64[c] @ XT
        d2[np.arange(len(c)), c] = np.inf
        d2[d2 < 1e-10] = np.inf                      # skip identical copies
        part = np.argpartition(d2, kmax, axis=1)[:, :kmax]
        order = np.take_along_axis(d2, part, 1).argsort(1)
        part = np.take_along_axis(part, order, 1)
        idx[a:a + len(c)] = part
        dist[a:a + len(c)] = np.sqrt(np.maximum(np.take_along_axis(d2, part, 1), 0))
    return idx, dist


def count_directions(arrows, tols):
    """Fewest SVD directions so that every arrow keeps at most tol of its length."""
    _, s, Vt = np.linalg.svd(arrows, full_matrices=False)
    coef = arrows @ Vt.T                              # (k, r): arrow along each direction
    length2 = (arrows ** 2).sum(1)
    left = 1 - np.cumsum(coef ** 2, 1) / length2[:, None]   # (k, r) after d = 1..r
    worst = np.sqrt(np.maximum(left.max(0), 0))       # worst arrow's leftover fraction
    worst = np.concatenate([[1.0], worst])            # d = 0
    out = []
    for t in tols:
        ok = np.nonzero(worst <= t)[0]
        out.append(int(ok[0]) if len(ok) else len(arrows))
    return out, worst


def run(name, X, seed):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    centres = rng.choice(len(X), CENTRES, replace=False)
    idx, dist = nearest(X, centres, KMAX)
    norms = np.linalg.norm(X[centres].astype(np.float64), axis=1)
    counts = np.empty((CENTRES, len(KS), len(TOLS)), int)
    worst_at = {k: [] for k in (20, 50, 200)}
    for i, c in enumerate(centres):
        for j, k in enumerate(KS):
            arrows = X[idx[i, :k]].astype(np.float64) - X[c].astype(np.float64)
            counts[i, j], worst = count_directions(arrows, TOLS)
            if k in worst_at:
                worst_at[k].append(worst[:21].tolist() + [None] * (21 - len(worst[:21])))
    rel_radius = dist / norms[:, None]                # neighbour distance / centre's size
    res = {
        "n_images": len(X),
        "ks": KS, "tols": TOLS,
        "median_count": np.median(counts, 0).tolist(),          # [k][tol]
        "q25_count": np.quantile(counts, 0.25, 0).tolist(),
        "q75_count": np.quantile(counts, 0.75, 0).tolist(),
        "median_radius": [float(np.median(rel_radius[:, k - 1])) for k in KS],
        "nearest_rel": float(np.median(rel_radius[:, 0])),
        "worst_leftover_median": {k: np.median(np.array(v, float), 0).tolist()
                                  for k, v in worst_at.items()},
    }
    c10 = np.array(res["median_count"])[:, 1]
    print(f"{name:10s} n={len(X):6d} nearest {res['nearest_rel']:.3f}  count@10%: "
          + " ".join(f"k{k}:{c:g}" for k, c in zip(KS, c10)) + f"  ({time.time()-t0:.0f}s)",
          flush=True)
    return res


ZOOM_STEPS = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
ZOOM_CENTRES = 60


def zoom(name, knob_ranges, draw, n_knobs, seed):
    """Make neighbours on purpose: 200 random knob settings in a small box around
    each centre (half-width = step * each knob's range). Shrinking the box shows
    the neighbourhood size at which the count stops growing, i.e. where the
    family really is flat."""
    rng = np.random.default_rng(seed)
    lo, hi = knob_ranges[:, 0], knob_ranges[:, 1]
    span = hi - lo
    res = {"steps": ZOOM_STEPS, "ks": KS, "known": n_knobs, "rows": {}}
    for step in ZOOM_STEPS:
        counts = np.empty((ZOOM_CENTRES, len(KS)), int)
        radius = np.empty(ZOOM_CENTRES)
        for i in range(ZOOM_CENTRES):
            # keep the centre far enough inside the box that neighbours stay inside it
            c = rng.uniform(lo + min(step, 0.3) * span, hi - min(step, 0.3) * span)
            P = c + rng.uniform(-step, step, (KMAX, n_knobs)) * span
            img = draw(np.vstack([c, P]))
            x0 = img[0].astype(np.float64)
            arrows = img[1:].astype(np.float64) - x0
            d = np.linalg.norm(arrows, axis=1)
            arrows = arrows[np.argsort(d)]
            radius[i] = np.median(d) / np.linalg.norm(x0)
            for j, k in enumerate(KS):
                counts[i, j] = count_directions(arrows[:k], [0.10])[0][0]
        row = {"median_count": np.median(counts, 0).tolist(),
               "q25": np.quantile(counts, 0.25, 0).tolist(),
               "q75": np.quantile(counts, 0.75, 0).tolist(),
               "median_radius": float(np.median(radius))}
        res["rows"][str(step)] = row
        print(f"zoom {name:9s} step {step:<6} radius {row['median_radius']:.4f} count@10%: "
              + " ".join(f"k{k}:{c:g}" for k, c in zip(KS, row["median_count"])), flush=True)
    return res


def main_zoom():
    def draw_gen(P):
        return render(P, dtype=np.float64).reshape(len(P), 784)

    ks = knob_strokes.RANGES.copy()

    def draw_strokes(P):
        full = np.tile(knob_strokes.OFF, (len(P), 1))
        full[:, :12] = P
        return knob_strokes.render(full).reshape(len(P), 784)

    res = {"gen": zoom("gen", RANGES, draw_gen, 5, 21),
           "strokes12": zoom("strokes12", ks[:12], draw_strokes, 12, 22)}
    (OUT / "zoom.json").write_text(json.dumps(res, indent=1))


def main():
    data = {
        "mnist": lambda: mnist_ones(),
        "gen": lambda: generated(N, 1),
        "gen_dense": lambda: generated(N_DENSE, 2),
        "strokes12": lambda: knob_strokes.make(N, 12, seed=3).reshape(N, 784).astype(np.float32),
    }
    res = {}
    for s, (name, make) in enumerate(data.items()):
        res[name] = run(name, make(), 10 + s)
        (OUT / "results.json").write_text(json.dumps(res, indent=1))


GROUPS = {"densest 1%": (0.00, 0.01), "densest 5%": (0.00, 0.05),
          "middle 10%": (0.45, 0.55), "emptiest 5%": (0.95, 1.00)}
DENSITY_K = 10


def dense(name, X, seed):
    """Rank every image by how crowded its surroundings are (distance to its 10th
    nearest neighbour, as a fraction of its own size), then run the same count on
    centres taken only from the most crowded (or emptiest) part of the ranking."""
    t0 = time.time()
    allc = np.arange(len(X))
    idx, dist = nearest(X, allc, KMAX)
    norms = np.linalg.norm(X.astype(np.float64), axis=1)
    rel = dist / norms[:, None]
    order = np.argsort(rel[:, DENSITY_K - 1])          # most crowded first
    rng = np.random.default_rng(seed)
    out = {"ks": KS, "groups": {}}
    for g, (a, b) in GROUPS.items():
        pool = order[int(a * len(X)):int(b * len(X))]
        cs = rng.choice(pool, min(CENTRES, len(pool)), replace=False)
        counts = np.empty((len(cs), len(KS)), int)
        for i, c in enumerate(cs):
            for j, k in enumerate(KS):
                arrows = X[idx[c, :k]].astype(np.float64) - X[c].astype(np.float64)
                counts[i, j] = count_directions(arrows, [0.10])[0][0]
        row = {"n_centres": len(cs), "centres": cs.tolist(),
               "median_count": np.median(counts, 0).tolist(),
               "q25": np.quantile(counts, 0.25, 0).tolist(),
               "q75": np.quantile(counts, 0.75, 0).tolist(),
               "radius": [float(np.median(rel[cs, k - 1])) for k in KS],
               "nearest": float(np.median(rel[cs, 0])),
               "ink": float(np.median(X[cs].sum(1)))}
        out["groups"][g] = row
        print(f"dense {name:9s} {g:12s} nearest {row['nearest']:.3f} r@k200 {row['radius'][-1]:.3f} "
              f"ink {row['ink']:.0f} count: "
              + " ".join(f"k{k}:{c:g}" for k, c in zip(KS, row["median_count"])), flush=True)
    out["ink_all"] = float(np.median(X.sum(1)))
    print(f"   ({time.time() - t0:.0f}s, median ink all {out['ink_all']:.0f})", flush=True)
    return out


def main_dense():
    data = {
        "mnist": lambda: mnist_ones(),
        "gen": lambda: generated(N, 1),
        "strokes12": lambda: knob_strokes.make(N, 12, seed=3).reshape(N, 784).astype(np.float32),
    }
    res = {}
    for s, (name, make) in enumerate(data.items()):
        res[name] = dense(name, make(), 30 + s)
        (OUT / "dense.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    import sys
    parts = sys.argv[1:] or ["neighbours", "zoom"]
    if "neighbours" in parts:
        main()
    if "zoom" in parts:
        main_zoom()
    if "dense" in parts:
        main_dense()
