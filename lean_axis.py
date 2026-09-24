"""Exact lean coordinate on the five-knob grey-1 image family.

The input is only a 28x28 generated image and a requested lean angle. Geometry
is read from pixel coverage: row sums recover width and vertical bounds, while
cumulative coverage across full rows recovers the moving left edge. The output
uses the same sub-row coverage equation as the family, without calling its
renderer or a numerical optimiser.

See LEAN_AXIS.md for the equations and their domain.
"""

from dataclasses import dataclass

import numpy as np

from grey_ones import N_PIX, RANGES, SUB


@dataclass(frozen=True)
class Geometry:
    cx: float
    cy: float
    height: float
    width: float
    lean: float  # degrees


def _image(x):
    a = np.asarray(x, dtype=np.float64)
    if a.size != N_PIX * N_PIX or a.shape not in ((N_PIX, N_PIX), (N_PIX * N_PIX,)):
        raise ValueError("expected a 28x28 image or a flat 784-vector")
    a = a.reshape(N_PIX, N_PIX)
    if not np.isfinite(a).all() or a.min() < -1e-7 or a.max() > 1 + 1e-7:
        raise ValueError("pixel values must be finite and in [0, 1]")
    return a


def coverage(g, lean=None):
    """The 784-coordinate coverage equation for known image geometry.

    This is written separately from grey_ones.render so image recovery and
    forward coverage can be checked independently against that implementation.
    """
    theta = g.lean if lean is None else float(lean)
    y_lo = np.arange(N_PIX * SUB, dtype=np.float64) / SUB
    y_mid = y_lo + 0.5 / SUB
    top, bottom = g.cy - g.height / 2, g.cy + g.height / 2
    vertical = np.clip(
        (np.minimum(y_lo + 1 / SUB, bottom) - np.maximum(y_lo, top)) * SUB,
        0, 1,
    )
    centre = g.cx - np.tan(np.deg2rad(theta)) * (y_mid - g.cy)
    left, right = centre - g.width / 2, centre + g.width / 2
    cols = np.arange(N_PIX, dtype=np.float64)
    overlap = np.clip(
        np.minimum(right[:, None], cols[None, :] + 1)
        - np.maximum(left[:, None], cols[None, :]),
        0, 1,
    )
    return (overlap * vertical[:, None]).reshape(N_PIX, SUB, N_PIX).mean(axis=1)


def geometry_from_image(x, *, check=True):
    """Recover all five coordinates directly from a generated image.

    The central cumulative boundary of each full row lies inside every one
    of its sub-row stroke intervals throughout the declared knob ranges.
    There the cumulative ink equals boundary minus the mean left edge.
    """
    a = _image(x)
    mass = a.sum(axis=1)
    width = float(mass.max())
    if width <= 0:
        raise ValueError("the image has no ink")

    ink_rows = np.flatnonzero(mass > 0)
    first, last = int(ink_rows[0]), int(ink_rows[-1])
    top = first + 1 - mass[first] / width
    bottom = last + mass[last] / width
    cy = float((top + bottom) / 2)

    # Interior rows are fully covered throughout this family's height range.
    # Exclude the first and last inked rows even when they are nearly full:
    # float32 rounding could otherwise mistake a partial boundary row for one.
    full = np.arange(first + 1, last, dtype=int)
    if len(full) < 2:
        raise ValueError("need at least two full rows to recover lean")
    left_means = np.empty(len(full), dtype=np.float64)
    for i, r in enumerate(full):
        cumulative = np.r_[0.0, np.cumsum(a[r])]
        c = int(np.argmin(np.abs(cumulative - width / 2)))
        left_means[i] = c - cumulative[c]

    y = full.astype(np.float64) + 0.5
    y_centered = y - y.mean()
    slope = np.dot(y_centered, left_means - left_means.mean()) / np.dot(y_centered, y_centered)
    tangent = -slope
    left_at_cy = float(left_means.mean() + tangent * (y.mean() - cy))
    g = Geometry(
        cx=left_at_cy + width / 2,
        cy=cy,
        height=float(bottom - top),
        width=width,
        lean=float(np.rad2deg(np.arctan(tangent))),
    )

    if check and np.max(np.abs(coverage(g) - a)) > 1e-5:
        raise ValueError("image does not follow the five-knob coverage equation")
    return g


def move_lean(x, delta_degrees):
    """Trace from image x by an angle change, returning a 28x28 image.

    The new angle must remain inside the generator's declared lean interval.
    """
    g = geometry_from_image(x)
    target = g.lean + float(delta_degrees)
    lo, hi = RANGES[4]
    if target < lo - 1e-7 or target > hi + 1e-7:
        raise ValueError(f"target lean must remain in [{lo:g}, {hi:g}] degrees")
    return coverage(g, target)


def trace_lean(x, angles=None):
    """Images along the same intrinsic lean axis, at absolute degree values.

    Returns (angles, images), with images shaped (n, 28, 28).
    """
    g = geometry_from_image(x)
    if angles is None:
        angles = np.linspace(*RANGES[4], 181)
    angles = np.atleast_1d(np.asarray(angles, dtype=np.float64))
    lo, hi = RANGES[4]
    if not np.isfinite(angles).all() or (angles < lo - 1e-7).any() or (angles > hi + 1e-7).any():
        raise ValueError(f"all lean angles must lie in [{lo:g}, {hi:g}] degrees")
    return angles, np.stack([coverage(g, a) for a in angles])
