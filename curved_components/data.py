"""Shared test sets and scoring for the curved-component experiments.

Every method gets ONLY a matrix of pictures (rows = pictures, columns = pixels),
shuffled, with no knob values -- except the curve-dictionary test, which is
told the knob value on purpose and says so.

Test sets, simplest first:
    circle  a plain circle on flat paper (2 numbers per point). Sanity check:
            a method that cannot find "one number, and it is a circle" here
            will not find it anywhere.
    slide   the root 1 slid sideways, cx 5 -> 23 px, everything else fixed.
    lean    the root 1 leaned, -10 -> 35 degrees, everything else fixed.
    five    random generated 1s with all five knobs free (default box).

Sweeps are split by interleaving: train on every other position, test on the
positions in between. So "held-out miss" asks: does the method know what lies
BETWEEN the pictures it saw?

Miss is always ||rebuilt - true|| / ||true|| over the whole held-out set, as a
percentage of the pictures' own size (same yardstick as the rest of the repo).
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from grey_ones import render, sample  # noqa: E402

ROOT_ONE = np.array([14.0, 14.0, 19.75, 3.2, 0.0])
OUT = os.path.join(ROOT, "curved_components", "results")


def _sweep(knob, lo, hi, step):
    t = np.round(np.arange(lo, hi + 1e-9, step), 6)
    P = np.repeat(ROOT_ONE[None], len(t), 0)
    P[:, knob] = t
    return t, render(P, dtype=np.float64).reshape(len(t), -1)


def load(name, seed=0, fine=1):
    """Return dict(train, test, t_train, t_test, label). t_* = true knob (for grading only).
    fine > 1 divides the sweep step (more, closer pictures)."""
    rng = np.random.default_rng(seed)
    if name == "circle":
        t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        X = np.c_[np.cos(t), np.sin(t)]
        label = "angle (radians)"
    elif name == "slide":
        t, X = _sweep(0, 5.0, 23.0, 0.25 / fine)
        label = "cx (px)"
    elif name == "lean":
        t, X = _sweep(4, -10.0, 35.0, 0.5 / fine)
        label = "lean (degrees)"
    elif name == "five":
        P = sample(2500, rng)
        X = render(P, dtype=np.float64).reshape(len(P), -1)
        idx = rng.permutation(len(P))
        tr, te = idx[:2000], idx[2000:]
        return dict(train=X[tr], test=X[te], t_train=P[tr], t_test=P[te], label="all five knobs")
    else:
        raise ValueError(name)
    tr, te = np.arange(0, len(t), 2), np.arange(1, len(t), 2)
    ptr, pte = rng.permutation(tr), rng.permutation(te)      # hide the order
    return dict(train=X[ptr], test=X[pte], t_train=t[ptr], t_test=t[pte], label=label)


def miss(rebuilt, true):
    """Percent of the pictures' own size that the rebuild gets wrong."""
    return 100 * np.linalg.norm(rebuilt - true) / np.linalg.norm(true)


def pca_misses(train, test, max_k=60):
    """Held-out miss of plain PCA with k = 0..max_k straight components."""
    mu = train.mean(0)
    V = np.linalg.svd(train - mu, full_matrices=False)[2]
    out = []
    for k in range(min(max_k, len(V)) + 1):
        Vk = V[:k]
        out.append(miss(mu + ((test - mu) @ Vk.T) @ Vk, test))
    return np.array(out)


def first_below(curve, level):
    """Smallest count whose miss is under `level` percent (None if never)."""
    hit = np.where(np.asarray(curve) < level)[0]
    return int(hit[0]) if len(hit) else None


def rank_agreement(a, b):
    """|Spearman correlation|: does a recovered number put pictures in knob order?"""
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return abs(np.corrcoef(ra, rb)[0, 1])


def save(name, **arrays):
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, name + ".npz"), **arrays)


def load_spaced(name, every, seed=0):
    """Sweep with training pictures every `every` positions; all others held out.
    Base spacing is 0.25 px (slide) or 0.5 degrees (lean)."""
    rng = np.random.default_rng(seed)
    t, X = _sweep(0, 5.0, 23.0, 0.25) if name == "slide" else _sweep(4, -10.0, 35.0, 0.5)
    tr = np.arange(0, len(t), every)
    te = np.setdiff1d(np.arange(tr[-1] + 1), tr)               # only between seen ends
    ptr, pte = rng.permutation(tr), rng.permutation(te)
    return dict(train=X[ptr], test=X[pte], t_train=t[ptr], t_test=t[pte])
