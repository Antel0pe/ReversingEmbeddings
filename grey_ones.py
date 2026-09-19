"""A grey-pixel generator for straight 1s, with defaults taken from MNIST.

Grey here means COVERAGE: a pixel's value is the fraction of its area the stroke
covers. That is how MNIST's greys arose -- LeCun et al. shrank black-and-white
scans into a 20x20 box, and each small pixel became the inked fraction of its
patch. It also makes every knob continuous: nudge the lean and every edge pixel
changes a little, instead of a whole pixel flipping.

The stroke is a parallelogram: horizontal top and bottom, a fixed horizontal
width, leaning at a constant angle. Five knobs, all continuous:

    knob     range            where the numbers come from (MNIST 1s, 5th-95th pct)
    cx, cy   14.0 - 15.0      centre of ink sits within +/-0.5 px of the middle
    height   19.0 - 20.5      visible height is 20 rows for 98% of 1s
    width    1.8 - 4.6 px     ink per row: 5th pct 1.79, 95th pct 4.63
    lean     -10 to +35 deg   + = top to the right; real 1s mostly lean right

Rules for a valid grey 1 (the generator obeys both by construction):
    VISIBLE  = value >= 0.5  (70% of MNIST ink is at or above it)
    SMUDGES  = every nonzero pixel belongs to the one connected piece of ink;
               nothing above 0 may float free. Holds for 94.7% of real 1s.

All numbers are defaults, not truths.
"""

import numpy as np
from scipy.ndimage import label

N_PIX = 28
KNOBS = ["cx", "cy", "height", "width", "lean"]
RANGES = np.array([[14.0, 15.0], [14.0, 15.0], [19.0, 20.5], [1.8, 4.6], [-10.0, 35.0]])
VISIBLE = 0.5
SUB = 16                       # sub-rows per pixel row; x-coverage is exact


def render(P):
    """P: (n, 5) knob settings -> (n, 28, 28) coverage images in [0, 1]."""
    P = np.atleast_2d(np.asarray(P, np.float64))
    out = np.empty((len(P), N_PIX, N_PIX), np.float32)
    ys_lo = np.arange(N_PIX * SUB) / SUB                   # sub-row [lo, lo + 1/SUB]
    ys_mid = ys_lo + 0.5 / SUB
    cols = np.arange(N_PIX)
    for s in range(0, len(P), 512):
        cx, cy, h, w, lean = [P[s:s + 512, k][:, None] for k in range(5)]
        top, bot = cy - h / 2, cy + h / 2
        # fraction of each sub-row inside the stroke's vertical extent (continuous in top/bot)
        wy = np.clip(np.minimum(ys_lo + 1 / SUB, bot) - np.maximum(ys_lo, top), 0, 1 / SUB) * SUB
        centre = cx - np.tan(np.radians(lean)) * (ys_mid - cy)     # top leans right for lean > 0
        L, R = centre - w / 2, centre + w / 2
        ov = np.clip(np.minimum(R[:, :, None], cols + 1) - np.maximum(L[:, :, None], cols), 0, 1)
        img = (ov * wy[:, :, None]).reshape(len(L), N_PIX, SUB, N_PIX).mean(2)
        out[s:s + 512] = img
    return out


def sample(n, rng):
    return RANGES[:, 0] + rng.random((n, 5)) * (RANGES[:, 1] - RANGES[:, 0])


def floating_ink(img):
    """Total ink not in the largest connected piece (8-neighbour). 0 for a valid 1."""
    lab, k = label(img > 0, np.ones((3, 3)))
    if k <= 1:
        return 0.0
    mass = np.bincount(lab.ravel(), img.ravel())[1:]
    return float(mass.sum() - mass.max())


def visible_height(img):
    rows = np.where((img >= VISIBLE).any(1))[0]
    return 0 if not len(rows) else int(rows[-1] - rows[0] + 1)
