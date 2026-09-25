"""Four reader-first figures for the generated grey-1 direction field.

Run: python make_direction_field_explained.py
Outputs: figures/pixel_edge_switch.png, figures/lean_phase_map_guide.png,
figures/five_pixel_directions.png, figures/lean_direction_changes.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

from lean_axis import Geometry, coverage
from make_direction_atlas import jacobian, lean_direction


OUT = Path("figures")
OUT.mkdir(exist_ok=True)
RED = "#b83d32"
BLUE = "#2f6da5"
INK = "#34383e"
ORANGE = "#d1842f"


def pixel_image(ax, x, title=None):
    ax.imshow(x, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11.5, pad=9)


def vector_image(ax, v, background, limit, title=None):
    ax.imshow(v.reshape(28, 28), cmap="RdBu_r",
              norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
              interpolation="nearest")
    ax.contour(background, levels=[.5], colors=[INK], linewidths=.72)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11.4, pad=9)


def strip_values(left):
    cols = np.arange(12, 17)
    return np.clip(np.minimum(left + 3, cols + 1) - np.maximum(left, cols), 0, 1)


def strip(ax, left, title):
    values = strip_values(left)
    ax.imshow(values[None, :], cmap="gray_r", vmin=0, vmax=1,
              extent=(12, 17, 0, 1), aspect="auto", interpolation="nearest")
    for c in range(12, 18):
        ax.axvline(c, color="#777", lw=1)
    ax.add_patch(Rectangle((left, 0), 3, 1, fill=False,
                           edgecolor=ORANGE, linewidth=3))
    for c, value in zip(range(12, 17), values):
        ax.text(c+.5, .5, f"{value:.1f}", ha="center", va="center",
                color="white" if value > .6 else INK, fontsize=12,
                fontweight="bold")
    ax.set(xlim=(12, 17), ylim=(0, 1), title=title)
    ax.set_yticks([])
    ax.set_xticks(np.arange(12.5, 17.5), labels=[f"pixel {c}" for c in range(12, 17)])
    ax.tick_params(axis="x", labelsize=8.5)
    ax.spines[:].set_visible(False)


def delta_strip(ax, a, b, label):
    delta = strip_values(b) - strip_values(a)
    ax.imshow(delta[None, :], cmap="RdBu_r", vmin=-.2, vmax=.2,
              extent=(12, 17, 0, 1), aspect="auto", interpolation="nearest")
    for c in range(12, 18):
        ax.axvline(c, color="#999", lw=.8)
    for c, value in zip(range(12, 17), delta):
        ax.text(c+.5, .5, f"{value:+.1f}" if abs(value) > 1e-9 else "0",
                ha="center", va="center", fontsize=11,
                color="white" if abs(value) > .1 else INK, fontweight="bold")
    ax.set(xlim=(12, 17), ylim=(0, 1), title=label)
    ax.set_yticks([])
    ax.set_xticks(np.arange(12.5, 17.5), labels=[str(c) for c in range(12, 17)])
    ax.tick_params(axis="x", labelsize=9)
    ax.spines[:].set_visible(False)


def make_pixel_edge_switch():
    base = Geometry(14.5, 14.5, 19.75, 3.0, 0)
    image = coverage(base)
    fig = plt.figure(figsize=(15.2, 9.4), facecolor="white")
    fig.suptitle("A pixel edge crossing makes the lean direction switch pixels",
                 y=.976, fontsize=18, fontweight="bold")
    fig.text(.5, .936,
             "A generated 1 is one inked stroke on a 28×28 grid. "
             "A pixel value is the fraction of its square covered by ink: 0 = white, 1 = black.",
             ha="center", fontsize=10.8, color="#555")
    gs = fig.add_gridspec(2, 2, left=.07, right=.96, bottom=.12, top=.86,
                          width_ratios=[.8, 1.85], height_ratios=[1.3, .8],
                          hspace=.30, wspace=.18)
    ax = fig.add_subplot(gs[0, 0])
    pixel_image(ax, image, "1 · Start with one generated 1")
    ax.axhspan(8-.5, 8+.5, facecolor="none", edgecolor=ORANGE, lw=2.4)
    ax.annotate("inspect a thin slice\nnear row 8", xy=(15, 8), xytext=(2, 2),
                fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2),
                bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=3))
    strips = gs[0, 1].subgridspec(3, 1, hspace=.60)
    for sub, left, title in [
        (0, 12.8, "2 · Slight lean left: edge at 12.8; pixel 12 is 20% covered"),
        (1, 13.0, "3 · No lean: edge reaches boundary 13; pixel 12 is empty"),
        (2, 13.2, "4 · Slight lean right: edge at 13.2; pixel 13 loses coverage"),
    ]:
        strip(fig.add_subplot(strips[sub]), left, title)
    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    ax.text(0, .95, "How to read the strips", fontsize=12, fontweight="bold", va="top")
    ax.text(0, .78,
            "Each box is one adjacent pixel in the same thin row.\n"
            "The orange outline is the moving ink stroke.\n"
            "The printed number is that pixel's ink fraction.\n\n"
            "A vector here is just five numbers because\n"
            "we cropped to five pixels. The complete\n"
            "image-change vector has 784 numbers.",
            fontsize=10.6, linespacing=1.45, va="top")
    deltas = gs[1, 1].subgridspec(2, 1, hspace=.64)
    delta_strip(fig.add_subplot(deltas[0]), 12.8, 13.0,
                "5 · Before the boundary: pixel 12 loses ink; pixel 15 gains it")
    delta_strip(fig.add_subplot(deltas[1]), 13.0, 13.2,
                "6 · After the boundary: pixel 13 loses ink; pixel 16 gains it")
    fig.text(.62, .072,
             "Blue = ink lost; red = ink gained. The old pattern cannot keep moving the stroke after the crossing.",
             ha="center", fontsize=10.1, color=INK)
    fig.text(.62, .044,
             "This is one thin horizontal slice with width 3. The generator averages 16 such slices per pixel row.",
             ha="center", fontsize=9.2, color="#666")
    fig.savefig(OUT / "pixel_edge_switch.png", dpi=155, facecolor="white")
    plt.close(fig)


def make_phase_map_guide():
    theta = np.linspace(-5, 5, 601)
    y = np.linspace(6, 23, 341)
    left = 13 - np.outer(y-14.5, np.tan(np.deg2rad(theta)))
    fig = plt.figure(figsize=(15, 8.2), facecolor="white")
    fig.suptitle("How to read the 2D lean map", y=.974,
                 fontsize=19, fontweight="bold")
    fig.text(.5, .93,
             "One location on this map means: choose a lean angle, then look at one thin horizontal slice of that 1.",
             ha="center", fontsize=10.8, color="#555")
    gs = fig.add_gridspec(1, 2, left=.08, right=.95, bottom=.17, top=.85,
                          width_ratios=[1.35, 1], wspace=.31)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.pcolormesh(theta, y, left, shading="auto", cmap="viridis",
                       vmin=12.45, vmax=13.55)
    ax.contour(theta, y, left, levels=[13], colors="white", linewidths=2.2)
    ax.plot(theta, np.full_like(theta, 8.5), color="#f4b05c", lw=1.7, ls="--")
    ax.plot(theta, np.full_like(theta, 20.5), color="#79c7ed", lw=1.7, ls="--")
    ax.scatter([-2, 0, 2], [8.5]*3, s=72, c=[ORANGE]*3,
               edgecolors="white", linewidths=1.2, zorder=5)
    ax.text(-4.8, 7.9, "top slice y=8.5", color="white", fontsize=9.6,
            bbox=dict(facecolor=INK, alpha=.75, edgecolor="none", pad=3))
    ax.text(-4.8, 21.6, "bottom slice y=20.5", color="white", fontsize=9.6,
            bbox=dict(facecolor=INK, alpha=.75, edgecolor="none", pad=3))
    ax.text(-4.75, 14.05, "middle slice does not shift sideways", color="white",
            fontsize=8.7,
            bbox=dict(facecolor=INK, alpha=.72, edgecolor="none", pad=2))
    ax.annotate("white line: left edge is\nexactly at pixel boundary 13",
                xy=(0, 10.5), xytext=(1.05, 6.8), color="white", fontsize=9.6,
                arrowprops=dict(arrowstyle="->", color="white", lw=1.4),
                bbox=dict(facecolor=INK, alpha=.8, edgecolor="none", pad=4))
    ax.set(xlabel="lean angle of the 1 (degrees)",
           ylabel="vertical position in its image (top is smaller y)",
           xlim=(-5, 5), ylim=(23, 6),
           title="Color = horizontal position of the left ink edge")
    cb = fig.colorbar(im, ax=ax, shrink=.79, pad=.02)
    cb.set_label("left edge x coordinate, in pixels", fontsize=9)
    right = gs[0, 1].subgridspec(2, 1, hspace=.48)
    for i, row_y, color, label in [
        (0, 8.5, ORANGE, "Top: edge moves right as lean increases"),
        (1, 20.5, BLUE, "Bottom: edge moves left as lean increases"),
    ]:
        ar = fig.add_subplot(right[i])
        edge = 13 - np.tan(np.deg2rad(theta)) * (row_y-14.5)
        ar.plot(theta, edge, color=color, lw=2.5)
        ar.axhline(13, color=INK, lw=1.4, ls="--")
        ar.axvline(0, color="#888", lw=1, ls=":")
        ar.scatter([0], [13], s=75, color=INK, zorder=5)
        ar.set(xlim=(-5, 5), ylim=(12.35, 13.65),
               xlabel="lean angle (degrees)", ylabel="left edge x",
               title=label)
        ar.grid(alpha=.16)
        ar.text(.04, .08, "crossing at 0° changes\nwhich pixel the edge occupies",
                transform=ar.transAxes, fontsize=9,
                bbox=dict(facecolor="white", edgecolor="#ddd", pad=4))
    fig.text(.5, .09,
             "Read across a horizontal dashed line as you change lean. Crossing a white line means that slice's ink edge enters the next pixel.",
             ha="center", fontsize=10, color=INK)
    fig.text(.5, .056,
             "At y=14.5 the edge stays at 13, so that horizontal white line is stationary. The earlier wide map extends the angle range to −10°…35°.",
             ha="center", fontsize=9.4, color="#666")
    fig.savefig(OUT / "lean_phase_map_guide.png", dpi=155, facecolor="white")
    plt.close(fig)


def make_five_pixel_directions():
    q = np.array([14.43, 14.62, 19.77, 3.13, np.tan(np.deg2rad(17.3))])
    g = Geometry(q[0], q[1], q[2], q[3], 17.3)
    x = coverage(g)
    J = jacobian(q)
    unit = J / np.linalg.norm(J, axis=0)[None, :]
    limit = float(np.abs(unit).max())
    names = ["Move right", "Move down", "Grow taller", "Grow wider", "Lean more"]
    descriptions = [
        "left side loses;\nright side gains",
        "mostly top loses;\nbottom gains",
        "top and bottom\ngain ink",
        "both side edges\ngain ink",
        "top goes right;\nbottom goes left",
    ]
    fig = plt.figure(figsize=(18.3, 5.9), facecolor="white")
    fig.suptitle("Five independent pixel-change patterns at one generated 1",
                 y=.993, fontsize=18.5, fontweight="bold")
    fig.text(.5, .907,
             "Each knob gives one 784-number vector. Red pixels darken and blue pixels lighten if that knob increases a tiny amount.",
             ha="center", fontsize=10.8, color="#555")
    gs = fig.add_gridspec(1, 6, left=.035, right=.965, bottom=.25, top=.82,
                          wspace=.16)
    ax = fig.add_subplot(gs[0, 0])
    pixel_image(ax, x, "Starting image")
    ax.text(.5, -.08, "28 × 28 = 784 pixels", ha="center", va="top",
            transform=ax.transAxes, fontsize=10.5, color=INK)
    ax.text(.5, -.17, "The black outline on each\nmap marks this stroke.",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=9.2, color="#555")
    for j, (name, desc) in enumerate(zip(names, descriptions)):
        ax = fig.add_subplot(gs[0, j+1])
        vector_image(ax, unit[:, j], x, limit, f"{j+1} · {name}")
        count = int(np.count_nonzero(np.abs(J[:, j]) > 1e-5))
        ax.text(.5, -.08, f"{count} pixels change", ha="center", va="top",
                transform=ax.transAxes, fontsize=10.2,
                fontweight="bold", color=INK)
        ax.text(.5, -.18, desc, ha="center", va="top",
                transform=ax.transAxes, fontsize=9.5, color="#555")
    fig.text(.5, .082,
             "These are five ways to move the image, not five pixels. Each panel is still located in the original 784 pixel coordinates.",
             ha="center", fontsize=10.2, color=INK)
    fig.text(.5, .051,
             "The five vectors are separately normalized to length 1 so their pixel patterns can be compared. The same red/blue scale is used in every panel.",
             ha="center", fontsize=9.2, color="#666")
    fig.savefig(OUT / "five_pixel_directions.png", dpi=155, facecolor="white")
    plt.close(fig)


def make_lean_direction_changes():
    q = np.array([14.43, 14.62, 19.77, 3.13, 17.3])
    steps = [.3, .3, .4, .5, 5]
    names = ["center right +0.3 px", "center down +0.3 px",
             "height +0.4 px", "width +0.5 px", "lean +5°"]

    def unit_lean(coords):
        g = Geometry(*coords)
        v = lean_direction(g.lean, g)
        return v / np.linalg.norm(v)

    original = unit_lean(q)
    original_image = coverage(Geometry(*q))
    updated = []
    new_images = []
    angles = []
    for j, step in enumerate(steps):
        after = q.copy()
        after[j] += step
        updated.append(unit_lean(after))
        new_images.append(coverage(Geometry(*after)))
        angles.append(np.rad2deg(np.arccos(np.clip(np.dot(original, updated[-1]), -1, 1))))
    top_limit = max(np.abs(z).max() for z in [original, *updated])
    bottom_limit = max(np.abs(z-original).max() for z in updated)

    fig = plt.figure(figsize=(18.5, 8.4), facecolor="white")
    fig.suptitle("How each knob changes the lean direction", y=.993,
                 fontsize=19, fontweight="bold")
    fig.text(.5, .920,
             "The top row asks: which pixels would change if lean increased now? "
             "Each column changes one knob by the stated amount, then asks the same question.",
             ha="center", fontsize=10.5, color="#555")
    gs = fig.add_gridspec(2, 6, left=.04, right=.965, bottom=.16, top=.84,
                          hspace=.39, wspace=.16)
    ax = fig.add_subplot(gs[0, 0])
    vector_image(ax, original, original_image, top_limit, "Starting lean action")
    ax.text(.5, -.08, "red = more ink when leaning",
            ha="center", transform=ax.transAxes, fontsize=8.8, color=RED)
    ax.text(.5, -.15, "blue = less ink when leaning",
            ha="center", transform=ax.transAxes, fontsize=8.8, color=BLUE)
    ax = fig.add_subplot(gs[1, 0])
    pixel_image(ax, original_image, "Starting generated 1")
    ax.text(.5, -.08, "Bottom row = new lean action",
            ha="center", transform=ax.transAxes, fontsize=9.1, color=INK)
    ax.text(.5, -.15, "minus starting lean action",
            ha="center", transform=ax.transAxes, fontsize=9.1, color=INK)
    local_change = [False, True, True, False, False]
    for j, (name, new, new_image, angle) in enumerate(zip(names, updated, new_images, angles)):
        ax = fig.add_subplot(gs[0, j+1])
        vector_image(ax, new, new_image, top_limit, name)
        ax.text(.5, -.08, f"{angle:.1f}° from start",
                ha="center", transform=ax.transAxes, fontsize=10,
                color=INK, fontweight="bold")
        ax.text(.5, -.16,
                "tiny step: direction changes" if local_change[j] else
                "tiny step: direction unchanged",
                ha="center", transform=ax.transAxes, fontsize=8.7,
                color="#555")
        ax = fig.add_subplot(gs[1, j+1])
        vector_image(ax, new-original, original_image, bottom_limit,
                     "Change in lean action")
        ax.text(.5, -.08, "red = shifts toward darkening",
                ha="center", transform=ax.transAxes, fontsize=8.8, color=RED)
        ax.text(.5, -.15, "blue = shifts toward lightening",
                ha="center", transform=ax.transAxes, fontsize=8.8, color=BLUE)
    fig.text(.5, .074,
             "Every tile compares 784 pixel coordinates. A knob changes which edges are active and how fast each pixel responds to future lean.",
             ha="center", fontsize=10, color=INK)
    fig.text(.5, .043,
             "Lean vectors have length 1; each row uses one shared color scale. Finite moves can cross boundaries; five outcomes do not imply five local lean-vector dimensions.",
             ha="center", fontsize=9.1, color="#666")
    fig.savefig(OUT / "lean_direction_changes.png", dpi=155, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make_pixel_edge_switch()
    make_phase_map_guide()
    make_five_pixel_directions()
    make_lean_direction_changes()
