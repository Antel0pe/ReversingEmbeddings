"""Show lean as an exact 2D pixel-boundary phase map rather than a projection.

Run: python make_exact_lean_phase_map.py
Output: figures/exact_lean_phase_map.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_lean_direction_derivative import (
    DELTA_Y, G, LEFT_AT_CENTRE, VERTICAL, LOW, HIGH,
    boundary_events, tangent,
)


def main():
    theta = np.linspace(-10, 35, 901)
    u = np.tan(np.deg2rad(theta))
    active = VERTICAL > 0
    y = DELTA_Y[active] + G.cy
    left = LEFT_AT_CENTRE - np.outer(DELTA_Y[active], u)

    events = boundary_events()
    ends = np.r_[LOW, events, HIGH]
    before_after = np.stack([tangent((a+b)/2) for a, b in zip(ends[:-1], ends[1:])])
    unit = before_after / np.linalg.norm(before_after, axis=1)[:, None]
    turn = np.rad2deg(np.arccos(np.clip(np.sum(unit[:-1]*unit[1:], axis=1), -1, 1)))
    event_angles = np.rad2deg(np.arctan(events))
    zero = np.argmin(np.abs(events))

    fig = plt.figure(figsize=(15.6, 10.7), facecolor="white")
    fig.suptitle("An exact 2D map of where the lean path changes direction",
                 fontsize=19, fontweight="bold", y=.984)
    fig.text(.5, .947,
             "Horizontal = lean angle; vertical = height within the stroke. Color gives the left edge's exact "
             "column. White contours mark integer pixel boundaries.",
             ha="center", fontsize=10.5, color="#555")
    gs = fig.add_gridspec(2, 2, left=.08, right=.94, bottom=.10, top=.89,
                          height_ratios=[1.55, .72], width_ratios=[1.15, 1],
                          hspace=.37, wspace=.27)
    ax = fig.add_subplot(gs[0, :])
    im = ax.imshow(left, origin="lower", aspect="auto", cmap="viridis",
                   extent=(-10, 35, y[0], y[-1]), interpolation="nearest")
    levels = np.arange(np.ceil(left.min()), np.floor(left.max())+1)
    ax.contour(theta, y, left, levels=levels, colors="white",
               linewidths=.8, alpha=.85)
    ax.axvline(0, color="#f4d35e", lw=2.2, ls="--")
    ax.axhline(8.5, color="#ffad75", lw=1.4, ls=":")
    ax.axhline(20.5, color="#93c9ff", lw=1.4, ls=":")
    ax.text(-9.6, 8.8, "top row", color="white", fontsize=9,
            bbox=dict(facecolor="#444", alpha=.7, edgecolor="none", pad=2))
    ax.text(-9.6, 20.8, "bottom row", color="white", fontsize=9,
            bbox=dict(facecolor="#444", alpha=.7, edgecolor="none", pad=2))
    ax.annotate("At 0°, every subrow left edge aligns at column 13",
                xy=(0, 15), xytext=(4, 22.8), fontsize=9.7, color="white",
                arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
                bbox=dict(facecolor="#333", alpha=.75, edgecolor="none", pad=4))
    ax.set(xlabel="lean angle (degrees)", ylabel="vertical position y",
           title="Pixel-edge events are contour crossings of L(angle, y)")
    cb = fig.colorbar(im, ax=ax, fraction=.025, pad=.015)
    cb.set_label("left edge column L", fontsize=9)

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(event_angles, turn, s=8, color="#aa3d32", alpha=.68)
    ax.scatter([event_angles[zero]], [turn[zero]], s=68, color="#921d16", zorder=4)
    ax.set(xlabel="lean angle of pixel-boundary event",
           ylabel="direction change (degrees)",
           title="Each contour event changes the 784D tangent",
           xlim=(-10, 35), yscale="log", ylim=(.5, 120))
    ax.grid(alpha=.2)
    ax.text(.97, .86, f"{len(events)} active corners\n"
            f"largest: {turn[zero]:.2f}° at 0°", transform=ax.transAxes,
            ha="right", fontsize=9.3,
            bbox=dict(facecolor="white", edgecolor="#ddd", pad=5))

    ax = fig.add_subplot(gs[1, 1])
    for row_y, color, label in [(8.5, "#c65d33", "top: moves right"),
                                (20.5, "#2f70aa", "bottom: moves left")]:
        edge = LEFT_AT_CENTRE - u * (row_y-G.cy)
        ax.plot(theta, edge, color=color, lw=2, label=label)
    for b in range(8, 20):
        ax.axhline(b, color="#999", lw=.45, alpha=.45)
    ax.axvline(0, color="#777", ls=":", lw=1)
    ax.set(xlabel="lean angle (degrees)", ylabel="left edge column",
           title="Two horizontal cuts through the phase map",
           xlim=(-10, 35), ylim=(8, 20))
    ax.grid(alpha=.13)
    ax.legend(fontsize=8.8, loc="lower left")

    fig.text(.5, .037,
             "For this fixed-geometry 1, L(angle,y) = 13 − tan(angle)(y − 14.5); "
             "the right edge is L+3. With the fixed top/bottom and 16-subrow coverage rule, "
             "this map determines every image on the lean path without PCA.",
             ha="center", fontsize=9.2, color="#555")
    fig.savefig("figures/exact_lean_phase_map.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
