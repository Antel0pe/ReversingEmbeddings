"""Held-out sheets: never used while designing a method, kept for a fair final check.

Written before the final methods were run on them. If a method is changed after looking at
results here, these sheets stop being a fair check for that method; add new ones instead.
Run them with:  python -m unfolding_suite --sheets holdout --first-seed 100 --seeds 3
"""
import numpy as np

from .sheets import _bend, _curve, _twisted, sheet


@sheet("h_roll5", "bent", "Spiral, a little tighter than the roll.", group="holdout")
def h_roll5(uv):
    return _bend(uv, lambda s: 5.0 * s)


@sheet("h_scurl65", "bent", "A slightly gentler S-curl.", group="holdout")
def h_scurl65(uv):
    return _bend(uv, lambda s: 6.5 * np.sin(np.pi * s))


@sheet("h_three_lobes", "bent", "Three curves in turn.", group="holdout")
def h_three_lobes(uv):
    return _bend(uv, lambda s: 7.0 * np.sin(1.5 * np.pi * s))


@sheet("h_fan3", "bent", "Three sharp creases.", group="holdout")
def h_fan3(uv):
    creases, width = np.array([0.5, 1.0, 1.5]), 0.06
    height = np.radians(110) / (width * np.sqrt(2 * np.pi))
    return _bend(uv, lambda s: sum((-1) ** i * height * np.exp(-(s - c) ** 2 / (2 * width ** 2))
                                   for i, c in enumerate(creases)))


@sheet("h_cone30", "bent", "A wider cone.", group="holdout")
def h_cone30(uv):
    d = uv - np.array([0.3, -3.5])
    r, phi = np.linalg.norm(d, axis=1), np.arctan2(d[:, 0], d[:, 1])
    slope = 0.3
    a = phi / slope
    return np.c_[r * slope * np.cos(a), r * slope * np.sin(a), r * np.sqrt(1 - slope ** 2)]


@sheet("h_diag30", "bent", "Rolled about an axis at 30 degrees.", group="holdout")
def h_diag30(uv):
    c, s_ = np.cos(np.radians(30)), np.sin(np.radians(30))
    along = c * uv[:, 0] + s_ * uv[:, 1]
    across = -s_ * uv[:, 0] + c * uv[:, 1]
    low, length = along.min() - 1e-9, along.max() - along.min() + 2e-9
    s, x, z = _curve(lambda t: 2.2 * t, length)
    return np.c_[np.interp(along - low, s, x), across, np.interp(along - low, s, z)]


@sheet("h_arc200", "bent", "One 200 degree bend.", group="holdout")
def h_arc200(uv):
    return _bend(uv, lambda s: np.full_like(s, np.radians(200) / 2))


@sheet("h_dents23", "stretched", "Different random dents.", group="holdout")
def h_dents23(uv):
    rng = np.random.default_rng(23)
    z = np.zeros(len(uv))
    for _ in range(14):
        c, width = rng.uniform(-1, 1, 2), rng.uniform(0.2, 0.4)
        z += rng.normal(0, 0.28) * np.exp(-((uv - c) ** 2).sum(1) / (2 * width ** 2))
    return np.c_[uv, z]


@sheet("h_crumple17", "stretched", "Different wide random twists.", group="holdout")
def h_crumple17(uv):
    return _twisted(uv, 4, 17, 0.7)


@sheet("h_crumple29", "stretched", "Different tighter random twists.", group="holdout")
def h_crumple29(uv):
    return _twisted(uv, 6, 29, 0.5)
