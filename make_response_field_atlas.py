"""Visual atlases of generated-1 image and direction changes across knobs.

Run: python make_response_field_atlas.py
Outputs four figures in figures/response_*.png.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
import numpy as np

from grey_ones import RANGES
from lean_axis import Geometry, coverage


OUT = Path("figures")
OUT.mkdir(exist_ok=True)
INK = "#2f3944"
RED = "#b74237"
BLUE = "#326fa8"
ORANGE = "#d18830"
BASE = Geometry(14.43, 14.62, 19.77, 3.2, 17.3)


def with_values(g, **changes):
    return Geometry(**{**g.__dict__, **changes})


def signed_image(ax, v, limit, reference=None):
    ax.imshow(v.reshape(28, 28), cmap="RdBu_r",
              norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
              interpolation="nearest")
    if reference is not None:
        ax.contour(reference.reshape(28, 28), levels=[.5], colors=[INK],
                   linewidths=.55)
    ax.set_xticks([])
    ax.set_yticks([])


def grey_image(ax, x):
    ax.imshow(x.reshape(28, 28), cmap="gray_r", vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def lean_steps(g, degrees=5):
    x0 = coverage(g)
    past = x0 - coverage(g, g.lean-degrees)
    future = coverage(g, g.lean+degrees) - x0
    return x0, past, future


def vector_angle(a, b):
    a, b = a.ravel(), b.ravel()
    cos = np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))
    return np.rad2deg(np.arccos(np.clip(cos, -1, 1)))


def make_width_lean_sweep():
    widths = [2.0, 2.6, 3.2, 3.8, 4.4]
    states = [lean_steps(with_values(BASE, width=w)) for w in widths]
    limit = max(np.max(np.abs(v)) for _, past, future in states
                for v in (past, future))
    fig = plt.figure(figsize=(17.5, 12.0), facecolor="white")
    fig.suptitle("As width changes, which pixels respond to leaning?",
                 y=.987, fontsize=19, fontweight="bold")
    fig.text(.5, .955,
             "Controlled generated 1 · same center, height, and starting lean 17.3° in every column · "
             "each tile uses the same 28×28 pixel positions",
             ha="center", fontsize=10.5, color="#555")
    gs = fig.add_gridspec(4, 5, left=.14, right=.97, bottom=.12, top=.90,
                          hspace=.38, wspace=.17)
    row_labels = ["The starting 1", "Lean from 12.3° to 17.3°",
                  "Lean from 17.3° to 22.3°",
                  "Support switch"]
    for row, label in enumerate(row_labels):
        fig.text(.033, .814 - row*.197, label, va="center", ha="left",
                 fontsize=10.5, color=INK, fontweight="bold", rotation=90)
    mask_cmap = ListedColormap([BLUE, "white", "#ced4d8", RED])
    for col, (w, (x, past, future)) in enumerate(zip(widths, states)):
        ax = fig.add_subplot(gs[0, col])
        grey_image(ax, x)
        ax.set_title(f"width = {w:.1f} px", fontsize=11.6, pad=7)
        ax = fig.add_subplot(gs[1, col])
        signed_image(ax, past, limit, x)
        ax.text(.5, -.075, f"pixel L2 = {np.linalg.norm(past):.2f}",
                ha="center", transform=ax.transAxes, fontsize=9.2, color="#555")
        ax = fig.add_subplot(gs[2, col])
        signed_image(ax, future, limit, x)
        ax.text(.5, -.075,
                f"L2 = {np.linalg.norm(future):.2f}; past→future = {vector_angle(past,future):.1f}°",
                ha="center", transform=ax.transAxes, fontsize=8.7, color="#555")
        ax = fig.add_subplot(gs[3, col])
        if col == 0:
            ax.axis("off")
            ax.text(.5, .57, "reference\nwidth", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11.5, color="#777")
        else:
            old = np.abs(states[col-1][2]) > 1e-8
            new = np.abs(future) > 1e-8
            category = np.zeros((28, 28), dtype=int)
            category[old & ~new] = -1
            category[old & new] = 1
            category[~old & new] = 2
            ax.imshow(category, cmap=mask_cmap, vmin=-1, vmax=2,
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(.5, -.075,
                    f"+{np.count_nonzero(~old&new)} enter  "
                    f"−{np.count_nonzero(old&~new)} leave  "
                    f"turn {vector_angle(states[col-1][2],future):.1f}°",
                    ha="center", transform=ax.transAxes, fontsize=8.2, color="#555")
    fig.text(.5, .064,
             "Rows 2–3: red pixels gain ink and blue lose ink during a +5° lean move. Color strength uses one scale across the sweep.",
             ha="center", fontsize=10, color=INK)
    fig.text(.5, .038,
             "Last row compares forward-lean participation with the preceding width: red enters, blue leaves, gray stays active. It does not show changes in rate.",
             ha="center", fontsize=9.2, color="#555")
    fig.savefig(OUT / "response_width_lean_sweep.png", dpi=155, facecolor="white")
    plt.close(fig)


def make_knob_plane_glyphs():
    center_values = [14.0, 14.25, 14.5, 14.75, 15.0]
    width_values = [2.0, 2.6, 3.2, 3.8, 4.4]
    center = with_values(BASE, cx=14.5, width=3.2)
    origin = coverage(center)
    tile_data = {}
    for w in width_values:
        for cx in center_values:
            tile_data[(w, cx)] = coverage(with_values(center, cx=cx, width=w))-origin
    limit = max(np.max(np.abs(d)) for d in tile_data.values())
    fig = plt.figure(figsize=(14.4, 14.5), facecolor="white")
    fig.suptitle("A 2D knob plane whose points contain full pixel-change images",
                 y=.985, fontsize=18, fontweight="bold")
    fig.text(.5, .953,
             "Each square is a 28×28 vector: this generated 1 minus the center-square 1. "
             "Only width and horizontal center vary.",
             ha="center", fontsize=10.2, color="#555")
    gs = fig.add_gridspec(5, 5, left=.11, right=.965, bottom=.105, top=.902,
                          hspace=.27, wspace=.10)
    for i, w in enumerate(width_values[::-1]):
        for j, cx in enumerate(center_values):
            ax = fig.add_subplot(gs[i, j])
            difference = tile_data[(w, cx)]
            signed_image(ax, difference, limit, origin)
            if i == 0:
                ax.set_title(f"center x = {cx:.2f}", fontsize=10.7, pad=6)
            if j == 0:
                ax.set_ylabel(f"width\n{w:.1f}", rotation=0, ha="right",
                              va="center", labelpad=11, fontsize=10.2, color=INK)
            ax.text(.5, -.065, f"distance = {np.linalg.norm(difference):.2f}",
                    ha="center", transform=ax.transAxes, fontsize=8.5,
                    color="#555")
            if w == 3.2 and cx == 14.5:
                for spine in ax.spines.values():
                    spine.set_edgecolor(ORANGE)
                    spine.set_linewidth(3)
                ax.text(.5, .08, "starting 1", ha="center",
                        transform=ax.transAxes, fontsize=9.2,
                        bbox=dict(facecolor="white", edgecolor=ORANGE, pad=3))
    fig.text(.5, .053,
             "Outer horizontal/vertical positions are knob settings; inner red/blue positions are pixels. "
             "Red = more ink, blue = less than the center-square image.",
             ha="center", fontsize=9.7, color=INK)
    fig.text(.5, .028,
             "Numbers are 784-pixel distances from the center square. Equal spacing on the knob grid need not mean equal image-space distance; three knobs are fixed.",
             ha="center", fontsize=9.1, color="#555")
    fig.savefig(OUT / "response_knob_plane_glyphs.png", dpi=155, facecolor="white")
    plt.close(fig)


def make_pixel_response_landscapes():
    center_values = np.linspace(14, 15, 101)
    width_values = np.linspace(2, 4.4, 101)
    selected = [(8, 14, "upper left edge"), (8, 18, "upper right edge"),
                (20, 10, "lower left edge"), (20, 14, "lower right edge")]
    landscapes = np.zeros((4, len(width_values), len(center_values)))
    for i, w in enumerate(width_values):
        for j, cx in enumerate(center_values):
            g = with_values(BASE, cx=cx, width=w)
            effect = coverage(g, g.lean+5)-coverage(g)
            for k, (r, c, _) in enumerate(selected):
                landscapes[k, i, j] = effect[r, c]
    limit = float(np.abs(landscapes).max())
    fig = plt.figure(figsize=(14.4, 9.3), facecolor="white")
    fig.suptitle("Turn the view around: one pixel across a whole knob plane",
                 y=.983, fontsize=18.5, fontweight="bold")
    fig.text(.5, .942,
             "A 28×28 direction image gives 784 pixel responses at one knob setting. "
             "Each colored map here follows just one pixel through many settings.",
             ha="center", fontsize=10.1, color="#555")
    gs = fig.add_gridspec(2, 3, left=.055, right=.95, bottom=.14, top=.88,
                          width_ratios=[.8, 1, 1], hspace=.35, wspace=.27)
    ax = fig.add_subplot(gs[:, 0])
    example = coverage(with_values(BASE, cx=14.5, width=3.2))
    grey_image(ax, example)
    for k, (r, c, _) in enumerate(selected):
        ax.plot(c, r, "o", markersize=12, markerfacecolor="none",
                markeredgewidth=2, markeredgecolor=[ORANGE, RED, BLUE, "#2c9872"][k])
        ax.text(c+1.3, r+.4, str(k+1), color=INK, fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=1))
    ax.set_title("The four selected image pixels", fontsize=11, pad=7)
    ax.text(.5, -.07, "Numbers match the maps on the right",
            ha="center", transform=ax.transAxes, fontsize=9.4, color="#555")
    for k, (r, c, name) in enumerate(selected):
        ax = fig.add_subplot(gs[k//2, 1+k%2])
        im = ax.imshow(landscapes[k], origin="lower", aspect="auto", cmap="RdBu_r",
                       norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                       extent=(14, 15, 2, 4.4), interpolation="nearest")
        ax.set(xlabel="horizontal center knob", ylabel="width knob",
               title=f"{k+1} · {name} (row {r}, col {c})")
        ax.tick_params(labelsize=8.8)
        ax.axhline(3.2, color="#777", alpha=.4, lw=.7)
        ax.axvline(14.5, color="#777", alpha=.4, lw=.7)
        cb = fig.colorbar(im, ax=ax, shrink=.77, pad=.02)
        cb.set_label("pixel change for +5° lean", fontsize=8.2)
    fig.text(.5, .086,
             "Red = that one pixel gains ink when lean increases 5°; blue = it loses ink; white = no response. "
             "The response switches as an edge enters or leaves it.",
             ha="center", fontsize=9.6, color=INK)
    fig.text(.5, .054,
             "This transpose of the direction field suggests grouping pixels with similar response maps. Four are displayed here; the complete field has 784 maps.",
             ha="center", fontsize=9.2, color="#555")
    fig.savefig(OUT / "response_pixel_landscapes.png", dpi=155, facecolor="white")
    plt.close(fig)


def equal_pixel_steps(base, target):
    q = np.array([base.cx, base.cy, base.height, base.width, base.lean], dtype=float)
    x0 = coverage(base)
    steps = np.empty(5)
    effects = []
    for j in range(5):
        lo, hi = 0., RANGES[j, 1]-q[j]
        for _ in range(48):
            mid = (lo+hi)/2
            after = q.copy()
            after[j] += mid
            effect = coverage(Geometry(*after))-x0
            if np.linalg.norm(effect) < target:
                lo = mid
            else:
                hi = mid
        steps[j] = hi
        after = q.copy()
        after[j] += hi
        effects.append(coverage(Geometry(*after))-x0)
    return q, x0, steps, np.stack(effects)


def make_equal_effect_sum():
    base = Geometry(14.4, 14.4, 19.6, 3.0, 17.3)
    q, origin, steps, effects = equal_pixel_steps(base, .8)
    summed = effects.sum(axis=0)
    actual = coverage(Geometry(*(q+steps)))-origin
    residual = actual-summed
    outside = int(np.count_nonzero((origin+summed < 0) | (origin+summed > 1)))
    limit_top = float(np.abs(effects).max())
    limit_bottom = float(max(np.abs(summed).max(), np.abs(actual).max(),
                             np.abs(residual).max()))
    names = ["center x", "center y", "height", "width", "lean"]
    units = ["px", "px", "px", "px", "°"]
    fig = plt.figure(figsize=(17.8, 9.0), facecolor="white")
    fig.suptitle("Put all five knob moves on the same image-space scale",
                 y=.982, fontsize=19, fontweight="bold")
    fig.text(.5, .942,
             "Choose a positive move of each knob that changes the 784-pixel image by exactly 0.80 in L2 distance. "
             "Then add the five change images.",
             ha="center", fontsize=10.2, color="#555")
    top = fig.add_gridspec(1, 6, left=.045, right=.965, bottom=.53, top=.85,
                           wspace=.14)
    ax = fig.add_subplot(top[0, 0])
    grey_image(ax, origin)
    ax.set_title("starting 1", fontsize=10.8, pad=7)
    for j in range(5):
        ax = fig.add_subplot(top[0, j+1])
        signed_image(ax, effects[j], limit_top, origin)
        ax.set_title(f"{names[j]} +{steps[j]:.3f}{units[j]}", fontsize=10.3, pad=7)
        ax.text(.5, -.095, f"pixel distance {np.linalg.norm(effects[j]):.2f}",
                transform=ax.transAxes, ha="center", fontsize=9.1, color="#555")
    lower = fig.add_gridspec(1, 4, left=.09, right=.91, bottom=.15, top=.45,
                             wspace=.23)
    for k, (arr, title, note) in enumerate([
        (summed, "sum of five separate moves", f"L2={np.linalg.norm(summed):.2f}; {outside} invalid pixels"),
        (actual, "move all five knobs together", f"L2={np.linalg.norm(actual):.2f}"),
        (residual, "actual minus separate sum", f"L2={np.linalg.norm(residual):.2f}"),
    ]):
        ax = fig.add_subplot(lower[0, k])
        signed_image(ax, arr, limit_bottom, origin)
        ax.set_title(title, fontsize=10.7, pad=7)
        ax.text(.5, -.09, note, ha="center", transform=ax.transAxes,
                fontsize=9.2, color="#555")
    ax = fig.add_subplot(lower[0, 3])
    grey_image(ax, origin+actual)
    ax.set_title("actual resulting 1", fontsize=10.7, pad=7)
    fig.text(.5, .083,
             "Red = gain ink, blue = lose ink. Within each row, color intensity uses one common scale. "
             "The sum and the true joint move differ because the pixel responses change as the knobs move.",
             ha="center", fontsize=9.5, color=INK)
    fig.text(.5, .054,
             "Equal pixel-space length makes the five effects comparable, but it does not make five finite moves linear or preserve the [0,1] image bounds.",
             ha="center", fontsize=9.2, color="#555")
    fig.savefig(OUT / "response_equal_effect_sum.png", dpi=155, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make_width_lean_sweep()
    make_knob_plane_glyphs()
    make_pixel_response_landscapes()
    make_equal_effect_sum()
