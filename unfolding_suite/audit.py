"""Audit sheets (2026-11-15): kinds of case the design and held-out sheets did not cover.

Added after auditing the suite for category gaps. Like the held-out sheets, no method was tuned on
them. Each one covers one missing kind of case:

  uneven dots        every other sheet samples the paper evenly; real data never does
  non-convex outline every other sheet is the square, so methods may lean on a convex edge
  long thin paper    a 4:1 strip: dots four times closer across it than along it
  folded flat        a hairpin that lays half the sheet back over the other half

Scope (set by the user, 2026-11-15): only 2-D sheets in 3-D. A 5-D and an 8-D sheet were tried and removed.

The crescent and strip sheets are a different shape from the square, so they are scored like the
stretched sheets: one-to-one and smooth, not "is it the square".

Not covered, and why: holes in the paper (the scorer needs every grid dot), fully random sampling
(same), intrinsic dimension other than 2 (the scorer certifies 2-D triangles), and sheets that cannot
be flattened at all (closed or one-sided surfaces).
"""
import numpy as np

from .sheets import _bend, _curve, sheet


def _uneven(uv, a=0.6):
    """Monotone squeeze of the paper: dot density varies 4x across each direction, edges stay put."""
    return uv + a * np.sin(np.pi * uv) / np.pi


def _spiral_along_x(P, curvature, start):
    """Roll a flat shape (x, y) up along x, without stretching it: x is arc length along a spiral."""
    s, cx, cz = _curve(curvature, float(P[:, 0].max() - start) + 1e-9)
    arc = P[:, 0] - start
    return np.c_[np.interp(arc, s, cx), P[:, 1], np.interp(arc, s, cz)]


def _crescent(uv):
    """The square mapped onto a C shape (a ring sector spanning 270 degrees) in the plane."""
    angle = uv[:, 0] * np.radians(135)
    r = 1.0 + 0.35 * uv[:, 1]
    return np.c_[r * np.sin(angle), r * np.cos(angle)]


@sheet("patchy_scurl", "bent", "The S-curl, with dots 4x denser in some places than others.", group="audit",
       warp=_uneven)
def patchy_scurl(uv):
    return _bend(uv, lambda s: 7 * np.sin(np.pi * s))


@sheet("patchy_roll", "bent", "A roll, with dots 4x denser in some places than others.", group="audit",
       warp=_uneven)
def patchy_roll(uv):
    return _bend(uv, lambda s: 3.5 * s)


@sheet("crescent_flat", "stretched", "Paper shaped like a C (non-convex outline), lying flat.", group="audit")
def crescent_flat(uv):
    return np.c_[_crescent(uv), np.zeros(len(uv))]


@sheet("crescent_roll", "stretched", "C-shaped paper, rolled up.", group="audit")
def crescent_roll(uv):
    P = _crescent(uv)
    return _spiral_along_x(P, lambda s: 2.2 * s, P[:, 0].min())


@sheet("strip_roll", "stretched", "A 4:1 strip of paper, rolled up along its length.", group="audit")
def strip_roll(uv):
    P = np.c_[2.0 * uv[:, 0], 0.5 * uv[:, 1]]
    return _spiral_along_x(P, lambda s: 0.9 * s, -2.0)


@sheet("hairpin_flap", "bent", "Creased, then one end folded right back over the rest (hairpin, gap 0.1).",
       group="audit")
def hairpin_flap(uv):
    r = 0.05
    def k(s):
        mild = sum(np.radians(25) / (0.05 * np.sqrt(2 * np.pi)) * np.exp(-(s - c) ** 2 / (2 * 0.05 ** 2)) * (-1) ** i
                   for i, c in enumerate((0.3, 0.6)))
        return mild + np.where((s > 1.2) & (s < 1.2 + np.pi * r), 1 / r, 0.0)
    return _bend(uv, k)
