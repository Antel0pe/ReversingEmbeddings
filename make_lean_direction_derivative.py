"""Analyze the exact one-sided derivatives of a generated lean path.

Run: python make_lean_direction_derivative.py
Output: figures/lean_direction_derivative.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from grey_ones import N_PIX, SUB
from lean_axis import Geometry


G = Geometry(14.5, 14.5, 19.75, 3.0, 0.0)
LOW, HIGH = np.tan(np.deg2rad([-10, 35]))
LEFT_AT_CENTRE = G.cx - G.width / 2
SUBROW_LOW = np.arange(N_PIX * SUB, dtype=float) / SUB
SUBROW_MID = SUBROW_LOW + .5 / SUB
DELTA_Y = SUBROW_MID - G.cy
TOP, BOTTOM = G.cy - G.height / 2, G.cy + G.height / 2
VERTICAL = np.clip((np.minimum(SUBROW_LOW + 1/SUB, BOTTOM)
                    - np.maximum(SUBROW_LOW, TOP)) * SUB, 0, 1)
COLS = np.arange(N_PIX)


def tangent(u):
    """Exact dF/du away from pixel-edge boundary events."""
    left = LEFT_AT_CENTRE - u * DELTA_Y
    right = left + G.width
    raw = np.minimum(right[:, None], COLS + 1) - np.maximum(left[:, None], COLS)
    left_inside = (left[:, None] > COLS).astype(float)
    right_inside = (right[:, None] < COLS + 1).astype(float)
    derivative = (left_inside - right_inside) * DELTA_Y[:, None]
    return (np.where(raw > 0, derivative, 0) * VERTICAL[:, None]) \
        .reshape(N_PIX, SUB, N_PIX).mean(axis=1).ravel()


def boundary_events():
    """Every candidate crossing of a stroke edge and integer column line."""
    boundaries = np.arange(N_PIX + 1)
    candidates = []
    for origin in (LEFT_AT_CENTRE, LEFT_AT_CENTRE + G.width):
        for delta, vertical in zip(DELTA_Y, VERTICAL):
            if vertical <= 0 or abs(delta) < 1e-12:
                continue
            u = (origin - boundaries) / delta
            candidates.extend(u[(u > LOW + 1e-10) & (u < HIGH - 1e-10)])
    return np.unique(np.round(candidates, 12))


def colored_vector(ax, values, scale=.3):
    image = values.reshape(N_PIX, N_PIX)
    ax.imshow(image, cmap="RdBu_r",
              norm=TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale),
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(8, 20)
    ax.set_ylim(25, 3)
    for spine in ax.spines.values():
        spine.set_color("#aaa")
        spine.set_linewidth(.8)


def main():
    events = boundary_events()
    endpoints = np.r_[LOW, events, HIGH]
    midpoints = (endpoints[:-1] + endpoints[1:]) / 2
    V = np.stack([tangent(u) for u in midpoints])
    speed = np.linalg.norm(V, axis=1)
    unit = V / speed[:, None]
    cos_turn = np.sum(unit[:-1] * unit[1:], axis=1)
    turns = np.rad2deg(np.arccos(np.clip(cos_turn, -1, 1)))
    event_degrees = np.rad2deg(np.arctan(events))
    interval_degrees = np.rad2deg(np.arctan(midpoints))
    reference_angle = np.rad2deg(np.arccos(np.clip(unit @ unit[0], -1, 1)))
    cumulative = np.r_[0, np.cumsum(turns)]
    corner = np.argmin(np.abs(events))

    fig = plt.figure(figsize=(17.5, 12.2), facecolor="white")
    fig.suptitle("Different ways to differentiate the lean direction",
                 fontsize=20, fontweight="bold", y=.986)
    fig.text(.5, .949,
             "For one generated 1, u = tan(angle). The 784-pixel tangent is constant "
             "between exact boundary events and changes at each corner.",
             ha="center", fontsize=11.2, color="#555")
    gs = fig.add_gridspec(3, 4, left=.06, right=.95, bottom=.075, top=.90,
                          height_ratios=[.9, 1.15, 1.15], hspace=.40, wspace=.32)

    before, after = unit[corner], unit[corner+1]
    for j, (vector, title) in enumerate([
        (before, "Direction just before 0°"),
        (after, "Direction just after 0°"),
        (after - before, "Change in direction at 0°"),
    ]):
        ax = fig.add_subplot(gs[0, j])
        colored_vector(ax, vector, .32 if j < 2 else .35)
        ax.set_title(title, fontsize=10.6, pad=7)
    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    ax.text(.02, .88, "THE DERIVATIVE IDEA", fontsize=11.5,
            fontweight="bold", va="top")
    ax.text(.02, .67,
            "v(u) = dF/du\n"
            "T(u) = v(u) / ||v(u)||\n\n"
            "Between corners: dT/du = 0.\n"
            "At a corner: T jumps.\n\n"
            f"This path has {len(events)} active corners.",
            fontsize=10.2, va="top", linespacing=1.5)

    ax = fig.add_subplot(gs[1, :2])
    ax.scatter(event_degrees, turns, s=10, color="#b94335", alpha=.72)
    ax.scatter([event_degrees[corner]], [turns[corner]], s=80,
               color="#9c211a", zorder=5)
    ax.set(xlabel="lean angle of boundary crossing",
           ylabel="angle between directions before/after",
           title="Local turn at each exact corner",
           xlim=(-10, 35), yscale="log", ylim=(.1, 110))
    ax.grid(alpha=.2)
    ax.annotate(f"{turns[corner]:.3f}° at zero lean",
                (event_degrees[corner], turns[corner]), (5, 35),
                fontsize=9.5, arrowprops=dict(arrowstyle="->", color="#333"))
    ax.text(.98, .05, f"median corner {np.median(turns):.2f}°",
            transform=ax.transAxes, ha="right", fontsize=9.3,
            bbox=dict(facecolor="white", edgecolor="#ddd", pad=4))

    ax = fig.add_subplot(gs[1, 2:])
    ax.stairs(speed, np.rad2deg(np.arctan(endpoints)),
              color="#2b6f9f", lw=1.6)
    ax.set(xlabel="lean angle", ylabel="||dF/du||",
           title="Speed is constant on each straight piece",
           xlim=(-10, 35), ylim=(25, 38))
    ax.grid(alpha=.2)
    ax.text(.98, .06,
            "Speed and direction are separate information.",
            transform=ax.transAxes, ha="right", fontsize=9.3,
            bbox=dict(facecolor="white", edgecolor="#ddd", pad=4))

    ax = fig.add_subplot(gs[2, :2])
    ax.plot(interval_degrees, reference_angle, color="#76549c", lw=1.7)
    ax.set(xlabel="lean angle", ylabel="angle to initial tangent",
           title="Where the current direction points relative to −10°",
           xlim=(-10, 35), ylim=(0, 125))
    ax.grid(alpha=.2)

    ax = fig.add_subplot(gs[2, 2:])
    ax.step(np.rad2deg(np.arctan(endpoints)), np.r_[cumulative, cumulative[-1]],
            where="post", color="#16835c", lw=1.9)
    ax.set(xlabel="lean angle", ylabel="sum of local turn angles",
           title="Accumulated turning along the path",
           xlim=(-10, 35), ylim=(0, 1300))
    ax.grid(alpha=.2)
    ax.text(.98, .06, f"total {cumulative[-1]:.0f}° across 784D",
            transform=ax.transAxes, ha="right", fontsize=9.3,
            bbox=dict(facecolor="white", edgecolor="#ddd", pad=4))

    fig.text(.5, .026,
             "Turn angles compare one-sided tangents at algebraic pixel-edge crossings. "
             "Accumulated turning is not a planar winding count or a topological invariant.",
             ha="center", fontsize=9.5, color="#555")
    fig.savefig("figures/lean_direction_derivative.png", dpi=150,
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
