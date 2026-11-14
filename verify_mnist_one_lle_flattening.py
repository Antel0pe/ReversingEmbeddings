"""Run the unfolding suite's double-layer check (unfolding_suite/verify.py) on every layout saved by
analyze_mnist_one_lle_flattening.py. Uses only the images and the layouts.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python verify_mnist_one_lle_flattening.py              # verify.json
    OPENBLAS_NUM_THREADS=1 .venv/bin/python verify_mnist_one_lle_flattening.py --per-number # verify_per_number_stretch.json

--per-number first stretches each layout number on its own so that each dot's 10 nearest in the
layout keep their image distance (least squares, no truth), and skips the 2-number layouts.
"""
import sys
import json
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.spatial import cKDTree

import mnist
from analyze_mnist_one_lle_flattening import OUT, generated
from unfolding_suite.verify import summary_words, verify


@lru_cache(None)            # the generated sample must be drawn once per process, from the fresh seed
def images(name):
    if name == "mnist":
        X, y = mnist.load("train")
        return X[y == 1].reshape(-1, 784).astype(np.float64) / 255.0
    return generated(6742)[0]       # same seed and order as the analysis run (generated is called first there)


PER_NUMBER = "--per-number" in sys.argv


def per_number_stretch(X, F):
    _, nb = cKDTree(F).query(F, k=11)
    a, b = np.repeat(np.arange(len(F)), 10), nb[:, 1:].ravel()
    s2, _ = nnls((F[a] - F[b]) ** 2, ((X[a] - X[b]) ** 2).sum(1))
    return F * np.sqrt(s2)


def one(path):
    name, label = path.stem.split("__")
    X, F = images(name), np.load(path).astype(float)
    v = verify(X, per_number_stretch(X, F) if PER_NUMBER else F)
    v.pop("per_dot")
    return name, label, v


def main():
    paths = [p for p in sorted(OUT.glob("*__*.npy")) if not (PER_NUMBER and p.stem.endswith("d2"))]
    with Pool(4) as pool:
        rows = pool.map(one, paths)
    out = {}
    for name, label, v in rows:
        out.setdefault(name, {})[label] = v
        print(f"{name:10s} {label:12s} {v['reading']:8s} flagged {100 * v['flagged_share']:5.1f}%  "
              f"stacked {100 * v['stacked_share']:5.1f}%  biggest {100 * v['biggest_patch_share']:5.1f}%  "
              f"typical {v['typical_ratio']:.2f}x  flagged at {v['flagged_ratio'] or 0:.1f}x   [{summary_words(v)}]", flush=True)
    (OUT / ("verify_per_number_stretch.json" if PER_NUMBER else "verify.json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
