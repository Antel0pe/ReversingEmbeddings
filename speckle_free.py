"""Random images that obey the negative-space rules, and how close they get to a 1.

The rules (from ideas.md -- "ignore the majority of space that clearly isn't 1"):
    CONNECTED   every nonzero pixel belongs to one 8-connected piece
    SOLID       at least `solid_frac` of the inked pixels are fully black (>= 0.95)
    NO SPECKLE  follows from CONNECTED: nothing floats free

Images are GROWN rather than rejected, because rejection never succeeds: a random
784-pixel image is connected with probability ~0. Growth starts at one pixel and
repeatedly accretes a neighbour, choosing among boundary pixels with weight

    exp(-beta * (number of neighbours already inked))

beta = 0  picks uniformly  -> compact blobs
beta > 0  prefers lonely pixels -> thin, stringy, stroke-like shapes

so beta is a knob from "blob" to "scribble" and everything it makes is legal.
"""
import numpy as np
from scipy.ndimage import label

N = 28
NB = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def grow(n_ink, beta, rng, margin=2):
    """One legal image: n_ink connected pixels, grown with lonely-pixel bias."""
    I = np.zeros((N, N), bool)
    r, c = rng.integers(margin, N - margin, 2)
    I[r, c] = True
    frontier = {}
    def touch(r, c):
        for dr, dc in NB:
            a, b = r + dr, c + dc
            if margin <= a < N - margin and margin <= b < N - margin and not I[a, b]:
                frontier[(a, b)] = frontier.get((a, b), 0) + 1
    touch(r, c)
    while I.sum() < n_ink and frontier:
        keys = list(frontier)
        w = np.exp(-beta * np.array([frontier[k] for k in keys], float))
        k = keys[rng.choice(len(keys), p=w / w.sum())]
        I[k] = True; frontier.pop(k)
        touch(*k)
    return I


def shade(I, solid_frac, rng):
    """Turn the boolean shape into greys, keeping >= solid_frac of ink fully black."""
    X = I.astype(np.float64)
    idx = np.flatnonzero(I)
    n_grey = int(len(idx) * (1 - solid_frac))
    if n_grey:
        # grey the pixels with fewest inked neighbours -- i.e. the edges, like real anti-aliasing
        nb = sum(np.roll(np.roll(I, dr, 0), dc, 1) for dr, dc in NB).ravel()[idx]
        pick = idx[np.argsort(nb)[:n_grey]]
        X.ravel()[pick] = rng.uniform(0.15, 0.94, n_grey)
    return X


def legal(X, solid_frac=0.75):
    """Check an image against the three rules. Returns (ok, reason)."""
    X = np.asarray(X, float).reshape(N, N)
    I = X > 0
    if not I.any(): return False, "empty"
    if label(I, np.ones((3, 3)))[1] != 1: return False, "not one piece"
    if (X[I] >= 0.95).mean() < solid_frac - 1e-9: return False, "too much grey"
    return True, "ok"


def batch(n, n_ink, beta, rng, solid_frac=0.75):
    out = np.empty((n, 784))
    for i in range(n):
        out[i] = shade(grow(n_ink, beta, rng), solid_frac, rng).ravel()
    return out
