"""The test sheets: each function bends the flat square [-1, 1]^2 into 3-D.

A sheet is a function from paper positions (n, 2) to 3-D positions (n, 3). Register a
new one with the @sheet decorator. The family says what can be expected of it:

  "bent"       Bent without stretching, like real paper. Lengths along the sheet equal
               lengths on the square, so a correct flattening IS the original square.
  "stretched"  Stretched while being shaped. No flat layout need match the square, so
               only one-to-one and smoothness are judged, not squareness.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Sheet:
    name: str
    family: str
    build: Callable[[np.ndarray], np.ndarray]
    note: str


SHEETS: dict[str, Sheet] = {}


def sheet(name, family, note):
    if family not in ("bent", "stretched"):
        raise ValueError(f"unknown family {family!r}")

    def register(fn):
        SHEETS[name] = Sheet(name, family, fn, note)
        return fn

    return register


def _curve(curvature, length, n=40001):
    """A unit-speed curve in a plane: arc positions and x, z coordinates."""
    s = np.linspace(0, length, n)
    k = curvature(s)
    ds = np.diff(s)
    angle = np.r_[0, np.cumsum((k[1:] + k[:-1]) / 2 * ds)]
    x = np.r_[0, np.cumsum((np.cos(angle[1:]) + np.cos(angle[:-1])) / 2 * ds)]
    z = np.r_[0, np.cumsum((np.sin(angle[1:]) + np.sin(angle[:-1])) / 2 * ds)]
    return s, x, z


def _bend(uv, curvature):
    """Bend the paper along u by a curvature profile. Paper bent this way cannot stretch."""
    s, x, z = _curve(curvature, 2.0)
    arc = uv[:, 0] + 1
    return np.c_[np.interp(arc, s, x), uv[:, 1], np.interp(arc, s, z)]


# ------------------------------------------------------------ bent without stretching
@sheet("flat", "bent", "Control: already flat.")
def flat(uv):
    return np.c_[uv, np.zeros(len(uv))]


@sheet("gentle_arc", "bent", "One smooth 120 degree bend; never overhangs.")
def gentle_arc(uv):
    return _bend(uv, lambda s: np.full_like(s, np.radians(120) / 2))


@sheet("wave", "bent", "Curves one way then the other; never overhangs.")
def wave(uv):
    return _bend(uv, lambda s: 5 * np.sin(1.5 * np.pi * s))


@sheet("s_curl", "bent", "Curls over itself in both directions; layers fairly close.")
def s_curl(uv):
    return _bend(uv, lambda s: 7 * np.sin(np.pi * s))


@sheet("roll", "bent", "Rolled into a spiral with room between layers.")
def roll(uv):
    return _bend(uv, lambda s: 3.5 * s)


@sheet("tight_roll", "bent", "Spiral wound so tight its layers almost touch.")
def tight_roll(uv):
    return _bend(uv, lambda s: 8.0 * s)


@sheet("diagonal_roll", "bent", "Rolled about an axis at 45 degrees to the paper's edges.")
def diagonal_roll(uv):
    c = np.sqrt(0.5)
    along, across = c * (uv[:, 0] + uv[:, 1]) + 2 * c, c * (uv[:, 1] - uv[:, 0])
    length = 4 * c
    s, x, z = _curve(lambda t: 1.75 * t, length)
    return np.c_[np.interp(along, s, x), across, np.interp(along, s, z)]


@sheet("accordion", "bent", "Four sharp alternating creases, like a paper fan.")
def accordion(uv):
    creases, width = np.array([0.4, 0.8, 1.2, 1.6]), 0.05
    height = np.radians(125) / (width * np.sqrt(2 * np.pi))

    def k(s):
        return sum((-1) ** i * height * np.exp(-(s - c) ** 2 / (2 * width ** 2)) for i, c in enumerate(creases))

    return _bend(uv, k)


@sheet("cone", "bent", "Wrapped around a cone: bends along a fan of lines, not parallel ones.")
def cone(uv):
    d = uv - np.array([0.0, -3.0])
    r, phi = np.linalg.norm(d, axis=1), np.arctan2(d[:, 0], d[:, 1])
    slope = 0.22                                  # sine of the cone's half-angle
    a = phi / slope
    return np.c_[r * slope * np.cos(a), r * slope * np.sin(a), r * np.sqrt(1 - slope ** 2)]


# ----------------------------------------------------------- stretched while shaped
@sheet("bowl", "stretched", "Pushed into a shallow bowl; mild stretching.")
def bowl(uv):
    return np.c_[uv, 0.35 * (uv ** 2).sum(1)]


@sheet("uneven_roll", "stretched", "A spiral whose paper is stretched more toward the outside.")
def uneven_roll(uv):
    t = 1.5 * np.pi + 3.0 * np.pi * (uv[:, 0] + 1) / 2
    r = 0.25 * t / np.pi
    return np.c_[r * np.cos(t), 1.2 * uv[:, 1], r * np.sin(t)]


@sheet("dents", "stretched", "Random dents and bumps pressed in; never overhangs.")
def dents(uv):
    rng = np.random.default_rng(11)
    z = np.zeros(len(uv))
    for _ in range(14):
        c, width = rng.uniform(-1, 1, 2), rng.uniform(0.2, 0.4)
        z += rng.normal(0, 0.28) * np.exp(-((uv - c) ** 2).sum(1) / (2 * width ** 2))
    return np.c_[uv, z]


def _twisted(uv, n_twists, seed, reach):
    """Carry the flat sheet through local twists of space.

    Each twist turns points about an axis by an angle that depends only on what the
    turn preserves (distance from the axis and height along it), so it can be undone
    exactly and the sheet folds over but never passes through itself.
    """
    rng = np.random.default_rng(seed)
    P = np.c_[uv, np.zeros(len(uv))]
    for _ in range(n_twists):
        centre = np.r_[rng.uniform(-0.8, 0.8, 2), rng.normal(0, 0.15)]
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        Q = P - centre
        along = Q @ axis
        radial = Q - along[:, None] * axis
        angle = rng.choice([-1, 1]) * rng.uniform(2.0, 3.0) * np.exp(
            -((radial ** 2).sum(1) + along ** 2) / (2 * reach ** 2))
        P = (centre + along[:, None] * axis + np.cos(angle)[:, None] * radial
             + np.sin(angle)[:, None] * np.cross(axis, radial))
    return P


@sheet("crumple", "stretched", "Four wide random twists: folds over, stretches unevenly.")
def crumple(uv):
    return _twisted(uv, 4, 3, 0.7)


@sheet("hard_crumple", "stretched", "Eight tighter random twists.")
def hard_crumple(uv):
    return _twisted(uv, 8, 6, 0.4)


# ------------------------------------------------------------- building a test instance
@dataclass(frozen=True)
class Instance:
    """One sheet sampled for testing. Only `points` is ever shown to a method."""

    sheet: Sheet
    grid: int              # nodes per side
    uv: np.ndarray         # (grid^2, 2) true paper positions, row-major on the grid
    xyz: np.ndarray        # (grid^2, 3) true 3-D positions, same order, before hiding
    points: np.ndarray     # (grid^2, 3) what the method sees: shuffled and rotated
    order: np.ndarray      # points[k] is grid node order[k]


def make_instance(s: Sheet, grid=65, seed=0, noise=0.0):
    """Sample a sheet on a jittered grid, then hide its layout from the method.

    The jitter stays under a quarter of a cell, which guarantees that every triangle of
    the grid mesh has positive area on the paper, so there are no slivers at all. Edge
    nodes slide only along their edge, so the true outline is exactly the square.
    `noise` adds random 3-D displacement in units of the grid spacing.
    """
    if (grid - 1) % 4:
        raise ValueError("grid - 1 must be divisible by 4, for the coarser test meshes")
    rng = np.random.default_rng(seed)
    cell = 2 / (grid - 1)
    axis = np.linspace(-1, 1, grid)
    v, u = np.meshgrid(axis, axis, indexing="ij")
    du, dv = rng.uniform(-0.2, 0.2, (2, grid, grid)) * cell
    du[:, [0, -1]] = 0
    dv[[0, -1], :] = 0
    uv = np.c_[(u + du).ravel(), (v + dv).ravel()]
    xyz = s.build(uv)
    hidden = xyz + rng.normal(0, noise * cell, xyz.shape) if noise else xyz
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    order = rng.permutation(len(uv))
    points = (hidden @ rotation + rng.normal(0, 1, 3))[order]
    return Instance(s, grid, uv, xyz, points, order)
