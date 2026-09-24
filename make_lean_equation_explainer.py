"""Explain the exact image-to-image lean equation with one concrete example.

Run: python make_lean_equation_explainer.py
Output: figures/lean_equation_explainer.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import grey_ones
from lean_axis import coverage, geometry_from_image, trace_lean


INK = "#202020"
TOP = "#cf6128"
BOTTOM = "#2165a3"
NEW = "#16825d"


def row_edge_data(x, width):
    mass = x.sum(axis=1)
    rows = np.arange(np.flatnonzero(mass > 0)[0] + 1,
                     np.flatnonzero(mass > 0)[-1])
    positions, boundaries, cumulative_values = [], [], []
    for r in rows:
        cumulative = np.r_[0, np.cumsum(x[r])]
        c = int(np.argmin(np.abs(cumulative - width / 2)))
        boundaries.append(c)
        cumulative_values.append(cumulative[c])
        positions.append(c - cumulative[c])
    return rows, np.array(positions), np.array(boundaries), np.array(cumulative_values)


def image_panel(ax, image, title, border=INK):
    ax.imshow(image, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=12, fontweight="bold", pad=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(border)
        spine.set_linewidth(2 if border != INK else .8)


def main():
    # The input image is the sole input to the recovery and motion equations.
    p = np.array([14.5, 14.5, 19.75, 3.0, 20.0])
    x = grey_ones.render(p)[0].astype(np.float64)
    g = geometry_from_image(x)
    y = coverage(g, g.lean + 10)
    expected = grey_ones.render(np.array([[p[0], p[1], p[2], p[3], 30.0]]))[0]
    err = np.abs(y - expected)
    mass = x.sum(axis=1)
    top, bottom = g.cy - g.height / 2, g.cy + g.height / 2
    rows, left, boundaries, cv = row_edge_data(x, g.width)

    fig = plt.figure(figsize=(16.2, 11.6), facecolor="white")
    fig.suptitle("From one generated 1 to a new 1, using only its pixels and +10°",
                 fontsize=18, fontweight="bold", y=.985)
    fig.text(.5, .949,
             "Read width and edges from the 784 input values; find the current lean; "
             "calculate every new pixel by stroke coverage. No PCA or fitted inverse.",
             ha="center", fontsize=10.3, color="#555")
    gs = fig.add_gridspec(3, 4, left=.055, right=.97, bottom=.052, top=.895,
                          height_ratios=[1.12, 1.08, .74], hspace=.47, wspace=.35)

    # 1. The image itself, with two illustrative rows marked.
    ax = fig.add_subplot(gs[0, 0])
    image_panel(ax, x, "1  Input image: 20°")
    ax.axhline(8, color=TOP, lw=1.7, alpha=.85)
    ax.axhline(20, color=BOTTOM, lw=1.7, alpha=.85)
    ax.text(1, 7, "row 8", color=TOP, fontsize=9, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1))
    ax.text(1, 19, "row 20", color=BOTTOM, fontsize=9, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1))

    # 2. Row sums recover the full width and the top/bottom cutoffs.
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(np.arange(28) + .5, mass, "-o", color=INK, ms=3.3, lw=1.7)
    ax.axhline(g.width, color=NEW, ls="--", lw=1.4)
    ax.axvline(top, color="#888", ls=":", lw=1)
    ax.axvline(bottom, color="#888", ls=":", lw=1)
    ax.annotate(f"full-row ink = width {g.width:.2f}", xy=(13, g.width),
                xytext=(4.2, 2.25), fontsize=9.2, color=NEW,
                arrowprops=dict(arrowstyle="->", color=NEW))
    ax.text(.03, .06, f"top {top:.3f}     bottom {bottom:.3f}",
            transform=ax.transAxes, fontsize=8.9, color="#555")
    ax.set(xlabel="row position", ylabel="sum of pixel values",
           title="2  Row ink gives width and height", xlim=(0, 28), ylim=(0, 3.45))
    ax.grid(alpha=.2)

    # 3. The cumulative midpoint boundary gives mean left-edge positions.
    ax = fig.add_subplot(gs[0, 2])
    for r, color in [(8, TOP), (20, BOTTOM)]:
        C = np.r_[0, np.cumsum(x[r])]
        i = int(np.where(rows == r)[0][0])
        ax.plot(np.arange(29), C, "-o", color=color, lw=1.5, ms=2.4,
                label=f"row {r}: c={boundaries[i]}, z={left[i]:.3f}")
        ax.plot(boundaries[i], cv[i], "o", color=color, ms=8,
                markeredgecolor="white", markeredgewidth=1)
    ax.axhline(g.width / 2, color="#888", ls="--", lw=1)
    ax.text(.02, .52, "half the ink", transform=ax.transAxes,
            fontsize=8.5, color="#777")
    ax.set(xlabel="column boundary c", ylabel="ink to the left, Cᵣ(c)",
           title="3  Cumulative ink reveals edges", xlim=(7, 22), ylim=(-.1, 3.3))
    ax.legend(fontsize=8.1, loc="upper left", framealpha=.95)
    ax.grid(alpha=.2)

    # 4. A line across the recovered left edges yields tan(lean).
    ax = fig.add_subplot(gs[0, 3])
    yy = rows + .5
    fit = g.cx - g.width / 2 - np.tan(np.deg2rad(g.lean)) * (yy - g.cy)
    ax.plot(yy, fit, color=INK, lw=1.8)
    ax.scatter(yy, left, color=NEW, s=23, zorder=3, label="read from pixels")
    for r, color in [(8, TOP), (20, BOTTOM)]:
        i = int(np.where(rows == r)[0][0])
        ax.scatter(yy[i], left[i], color=color, s=78, zorder=4,
                   edgecolor="white", linewidth=.8)
    ax.text(.04, .05, f"slope = −{np.tan(np.deg2rad(g.lean)):.3f}\n"
            f"lean = arctan(0.364) = {g.lean:.2f}°",
            transform=ax.transAxes, fontsize=9.6,
            bbox=dict(facecolor="white", edgecolor="#ddd", pad=5))
    ax.set(xlabel="row centre, r + ½", ylabel="mean left edge, zᵣ",
           title="4  The edge line yields lean")
    ax.grid(alpha=.2)

    # 5-6. Two rows illustrate the opposite horizontal motions about cy.
    for slot, r, color, title, limits in [
        (0, 8, TOP, "5  Top row moves right", (14, 21)),
        (1, 20, BOTTOM, "6  Bottom row moves left", (7, 15)),
    ]:
        ax = fig.add_subplot(gs[1, slot])
        cols = np.arange(28)
        ax.bar(cols - .18, x[r], width=.36, color="#555", label="input 20°")
        ax.bar(cols + .18, y[r], width=.36, color=color, label="output 30°")
        ax.set(xlabel="pixel column", ylabel="coverage", title=title,
               xlim=limits, ylim=(0, 1.19))
        ax.set_xticks(np.arange(limits[0], limits[1] + 1))
        ax.grid(alpha=.18, axis="y")
        if slot == 0:
            ax.legend(fontsize=8.2, loc="upper left")

    ax = fig.add_subplot(gs[1, 2])
    image_panel(ax, y, "7  Output image: 30°", border=NEW)
    ax.text(.5, -.09, "same width, height and centre",
            transform=ax.transAxes, ha="center", fontsize=8.9, color="#555")

    ax = fig.add_subplot(gs[1, 3])
    im = ax.imshow(err * 1e7, cmap="magma", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("8  Difference from generator", fontsize=12,
                 fontweight="bold", pad=9)
    ax.text(.5, -.09, f"largest pixel error {err.max():.1e}",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    cb = fig.colorbar(im, ax=ax, fraction=.05, pad=.025)
    cb.set_label("absolute error ×10⁷", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)

    # A longer sweep shows this is one reusable equation, not an isolated pair.
    sweep_angles = np.array([-10, 0, 10, 20, 25, 30, 35], dtype=float)
    _, sweep = trace_lean(x, sweep_angles)
    bottom_grid = gs[2, :].subgridspec(1, len(sweep_angles), wspace=.35)
    for j, (angle, image) in enumerate(zip(sweep_angles, sweep)):
        ax = fig.add_subplot(bottom_grid[0, j])
        image_panel(ax, image, f"{angle:g}°",
                    border=NEW if angle == 30 else BOTTOM if angle == 20 else INK)
    fig.text(.5, .264, "THE SAME INPUT TRACED ALONG ITS COMPLETE LEAN AXIS",
             ha="center", fontsize=11, fontweight="bold", color="#333")
    fig.text(.5, .017,
             "All outputs are calculated directly from the boxed input's pixels. "
             "The match to the generator is limited only by float32 rounding.",
             ha="center", fontsize=9.6, color="#555")

    fig.savefig("figures/lean_equation_explainer.png", dpi=145, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
