"""Move an image along the LEAN knob without ever recovering the generator's knobs.

Leaning is a shear: row i slides sideways by  -(i - pivot) * du,  where du is the
change in tan(lean) and `pivot` is the image's own ink-centroid row. The only
question is how to slide a row of pixels by a fractional amount, and that is where
the edge has to be found -- a pixel grid stores the stroke's edge as a grey value,
not as a position.

Two ways to slide a row, both using nothing but that row's pixels:

  "cumulative"  general. Build the running total of ink along the row, read it back
                shifted, and difference it again. Works on ANY row -- a real MNIST
                1, a curved stroke, anything -- but linear interpolation cannot see
                where inside a pixel the edge sits, so it loses a little each time.

  "interval"    for rows that are one run of ink. Such a row is a plateau of height
                a spanning [L, L+w]: a is the row's largest value, w = (ink in the
                row) / a, and the edge is L = j0 + 1 - x[j0]/a for the first inked
                pixel j0. (The top and bottom rows of a stroke have a < 1, because
                the stroke only covers part of their height -- forgetting that was
                the one thing that made this inexact.) Slide the plateau and
                re-measure its coverage. Exact, and still per-row: it never uses
                cx, cy, height or lean.

Neither knows about the generator. Both will lean an image the generator cannot draw.
"""
import numpy as np

N = 28
ROWS = np.arange(N) + 0.5
COLS = np.arange(N)


def pivot_row(I):
    """The image's own ink-centroid row -- the height the shear turns about."""
    m = I.sum()
    return float((ROWS * I.sum(1)).sum() / m) if m > 0 else N / 2


def _slide_cumulative(row, d):
    C = np.concatenate([[0.0], np.cumsum(row)])
    return np.diff(np.interp(np.arange(N + 1) - d, np.arange(N + 1), C,
                             left=0.0, right=C[-1]))


def _slide_interval(row, d):
    ink = row.sum()
    if ink <= 1e-12:
        return row.copy()
    a = row.max()                              # the plateau height of this row
    w = ink / a                                # its width in pixels
    nz = np.flatnonzero(row > 1e-12)
    L = nz[0] + 1 - row[nz[0]] / a + d         # the edge, to sub-pixel, in one subtraction
    return a * np.clip(np.minimum(L + w, COLS + 1) - np.maximum(L, COLS), 0, 1)


def lean_step(I, du, mode="interval", pivot=None):
    """Return the image sheared by du = change in tan(lean). Input and output are
    both 28x28 images; nothing else is passed in or out."""
    I = np.asarray(I, float).reshape(N, N)
    p = pivot_row(I) if pivot is None else pivot
    d = -(ROWS - p) * du
    slide = _slide_interval if mode == "interval" else _slide_cumulative
    return np.stack([slide(I[i], d[i]) for i in range(N)])
