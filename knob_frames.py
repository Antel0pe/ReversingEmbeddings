"""The five knob directions at any generated-1 setting, and their metric.

At a setting p, moving knob k changes the image along one 784-vector. The five
of them are the columns of the Jacobian, and they are the natural basis at that
point -- a "frame". This module computes frames, and the small symmetric matrix
that summarises one.

    J(p)       (5, 784)   direction per unit of each knob (pixels, or degrees)
    G = J J^T  (5, 5)     knob speeds on the diagonal, interactions off it

G is what "how does lean vary with width here" means numerically: sqrt(G[k,k])
is how fast knob k moves the image, and G[j,k]/sqrt(G[j,j]G[k,k]) is the cosine
between two knobs' directions. G is stored flattened to its 15 upper-triangular
entries in the order given by `METRIC_IJ`.

Units are RAW: pixels for cx, cy, height, width and degrees for lean. Multiply a
column of J by that knob's range to get the normalised convention instead.

`figures/generated_one_knob_atlas` stores precomputed frames on slices through
the box, and records why the whole box cannot be tabulated.
"""

import numpy as np

import grey_ones
import smooth_ones
from grey_ones import KNOBS, RANGES

# central-difference steps, one ten-thousandth of each knob's range.
# lean's 0.0045 deg sits below the ~0.022 deg crease spacing measured in
# figures/generated_one_crease_origin, so a difference usually stays inside one
# smooth piece rather than averaging across a crease.
DEFAULT_FD = 1e-4 * (RANGES[:, 1] - RANGES[:, 0])

METRIC_IJ = [(i, j) for i in range(5) for j in range(i, 5)]   # 15 entries


def jacobian(P, renderer=None, fd=None):
    """P: (n, 5) settings -> (n, 5, 784) direction vectors, one row per knob."""
    renderer = grey_ones.render if renderer is None else renderer
    fd = DEFAULT_FD if fd is None else np.asarray(fd, float)
    P = np.atleast_2d(np.asarray(P, np.float64))
    n = len(P)
    Q = np.repeat(P, 10, axis=0)                       # minus/plus for each knob
    for k in range(5):
        Q[2 * k::10, k] -= fd[k]
        Q[2 * k + 1::10, k] += fd[k]
    I = np.asarray(renderer(Q), np.float64).reshape(n, 10, -1)
    return (I[:, 1::2] - I[:, 0::2]) / (2 * fd)[None, :, None]


def metric(P, renderer=None, fd=None, flat=False):
    """P: (n, 5) -> (n, 5, 5) Gram matrices of the five knob directions."""
    J = jacobian(P, renderer, fd)
    G = np.einsum("nkp,nlp->nkl", J, J)
    if not flat:
        return G
    return np.stack([G[:, i, j] for i, j in METRIC_IJ], axis=1)


def unflatten_metric(flat):
    """(n, 15) -> (n, 5, 5)."""
    flat = np.atleast_2d(flat)
    G = np.empty((len(flat), 5, 5), flat.dtype)
    for c, (i, j) in enumerate(METRIC_IJ):
        G[:, i, j] = G[:, j, i] = flat[:, c]
    return G


def speeds(G):
    """How far the image moves per unit of each knob."""
    return np.sqrt(np.einsum("nkk->nk", np.atleast_3d(G).reshape(-1, 5, 5)))


def knob_cosines(G):
    """(n, 5, 5) cosines between knob directions; 1 on the diagonal."""
    G = np.asarray(G).reshape(-1, 5, 5)
    s = speeds(G)
    return G / np.maximum(s[:, :, None] * s[:, None, :], 1e-300)


def frame(P, renderer=None, fd=None):
    """Unit-length knob directions, (n, 5, 784)."""
    J = jacobian(P, renderer, fd)
    return J / np.maximum(np.linalg.norm(J, axis=2, keepdims=True), 1e-300)


def smooth_renderer(P):
    """`smooth_ones` with `grey_ones`' call signature."""
    return smooth_ones.render(P)


def load_atlas(path=None):
    """The precomputed slice atlas as a dict. See its README for the schema."""
    from pathlib import Path
    if path is None:
        path = Path(__file__).resolve().parent / "figures/generated_one_knob_atlas/atlas.npz"
    return dict(np.load(path, allow_pickle=False))
