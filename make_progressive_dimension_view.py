"""Illustrate scale-dependent visible dimensions of a local five-knob chart.

Run: python make_progressive_dimension_view.py
Output: figures/progressive_dimension_view.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from grey_ones import RANGES
from make_direction_atlas import jacobian


def box(ax, xy, width, height, label, fill):
    ax.add_patch(FancyBboxPatch(xy, width, height,
                                boxstyle="round,pad=0.02,rounding_size=0.025",
                                linewidth=1.5, edgecolor="#555",
                                facecolor=fill))
    ax.text(xy[0]+width/2, xy[1]+height/2, label,
            ha="center", va="center", fontsize=9.4, color="#222")


def main():
    q = np.array([14.43, 14.62, 19.77, 3.13,
                  np.tan(np.deg2rad(17.3))])
    spans = RANGES[:, 1] - RANGES[:, 0]
    spans[4] = np.tan(np.deg2rad(RANGES[4, 1])) \
             - np.tan(np.deg2rad(RANGES[4, 0]))
    local_window = .01 * spans
    extents = np.linalg.svd(jacobian(q, step=1e-6) * local_window,
                             compute_uv=False)
    thresholds = np.geomspace(.5, .005, 401)
    visible = np.sum(extents[None, :] > thresholds[:, None], axis=1)

    fig = plt.figure(figsize=(16, 8.6), facecolor="white")
    fig.suptitle("A five-dimensional family can look lower-dimensional from far away",
                 fontsize=20, fontweight="bold", y=.983)
    fig.text(.5, .939,
             "Example: a small cell around one generated image, with each knob allowed "
             "1% of its declared range. The five numbers below are local pixel-space extents.",
             ha="center", fontsize=10.8, color="#555")
    gs = fig.add_gridspec(2, 2, left=.065, right=.95, bottom=.10, top=.87,
                          width_ratios=[1.15, 1], height_ratios=[1, .90],
                          hspace=.38, wspace=.31)

    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar(range(1, 6), extents, color=["#266ea4", "#438db0", "#6aa7b2",
                                                "#a481af", "#bd94b3"], width=.65)
    for bar, value in zip(bars, extents):
        ax.text(bar.get_x()+bar.get_width()/2, value+.009, f"{value:.3f}",
                ha="center", fontsize=9.2)
    for level, label in [(.2, "coarse"), (.04, "medium"), (.01, "fine")]:
        ax.axhline(level, color="#888", ls="--", lw=1)
        ax.text(5.45, level, label, va="bottom", fontsize=8.7, color="#555")
    ax.set(xticks=range(1, 6), xlabel="local singular direction",
           ylabel="approximate 784-pixel extent",
           title="Five independent directions, with unequal sizes",
           xlim=(.5, 5.8), ylim=(0, .35))
    ax.grid(alpha=.15, axis="y")

    ax = fig.add_subplot(gs[0, 1])
    ax.step(thresholds, visible, where="post", color="#1d7759", lw=2.5)
    ax.set(xscale="log", xlim=(.5, .005), ylim=(-.2, 5.4),
           yticks=range(6), xlabel="smallest pixel-space change the view can resolve",
           ylabel="directions above that threshold",
           title="Zooming in reveals previously thin directions")
    ax.grid(alpha=.18)
    for eps in (.2, .04, .01):
        count = int(np.sum(extents > eps))
        ax.plot(eps, count, "o", color="#ae4739", markersize=7)
        ax.annotate(f"{count} visible", (eps, count),
                    (eps*1.10, count+.22), fontsize=9)
    ax.text(.04, .07, "This is a resolution rule, not a change\n"
            "in the family's intrinsic dimension.",
            transform=ax.transAxes, fontsize=9.5,
            bbox=dict(facecolor="white", edgecolor="#ddd", pad=5))

    ax = fig.add_subplot(gs[1, :])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("A possible zoomable atlas", fontsize=13, pad=10)
    box(ax, (.06, .35), .25, .38,
        "COARSE 3D VIEW\n\nlocal neighborhoods\nshown as one region", "#e7eef5")
    ax.add_patch(FancyArrowPatch((.34, .54), (.49, .54),
                                 arrowstyle="-|>", mutation_scale=20,
                                 lw=2, color="#555"))
    ax.text(.42, .62, "zoom into a\nsmall region", ha="center", fontsize=9.2)
    for x, y, label, fill in [
        (.53, .60, "3D chart A", "#e6f1ec"),
        (.72, .60, "3D chart B", "#e8edf7"),
        (.53, .32, "3D chart C", "#f3e9ef"),
        (.72, .32, "3D chart D", "#f3eddf"),
    ]:
        box(ax, (x, y), .16, .15, label, fill)
    ax.text(.62, .17,
            "Color, slider, or linked panels retain the extra coordinates.\n"
            "One fixed panel is a slice; the linked collection covers a region.",
            ha="center", fontsize=10, color="#444")

    fig.text(.5, .035,
             "Local singular extents use a linear approximation near one image. "
             "The continuous generated family remains five-dimensional at generic points; "
             "larger views can also gain apparent axes from curvature.",
             ha="center", fontsize=9.3, color="#555")
    fig.savefig("figures/progressive_dimension_view.png", dpi=150,
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
