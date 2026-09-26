"""Explain what consecutive two-vector views preserve and omit.

Run: python make_lean_pairwise_planes.py
Outputs: figures/lean_pairwise_planes.png and figures/lean_pairwise_unfoldings.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch
import numpy as np

from lean_axis import Geometry, coverage


OUT = Path("figures")
OUT.mkdir(exist_ok=True)
BASE = Geometry(14.5, 14.5, 19.75, 3.0, 0.0)
ANGLES = np.arange(-10, 36, 5)
BLUE = "#286998"
ORANGE = "#d27724"
INK = "#25323a"


def data():
    images = np.array([coverage(BASE, float(a)) for a in ANGLES])
    steps = np.diff(images.reshape(len(images), -1), axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    directions = steps / lengths[:, None]
    pair_angles = np.arccos(np.clip(
        np.sum(directions[:-1] * directions[1:], axis=1), -1, 1
    ))
    return images, steps, lengths, directions, pair_angles


def clean_image_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_pair(ax, index, radians):
    endpoint = np.array([1 + np.cos(radians), np.sin(radians)])
    ax.add_patch(FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>",
                                 mutation_scale=16, linewidth=3, color=BLUE))
    ax.add_patch(FancyArrowPatch((1, 0), endpoint, arrowstyle="-|>",
                                 mutation_scale=16, linewidth=3, color=ORANGE))
    ax.add_patch(Arc((1, 0), .40, .40, theta1=0,
                     theta2=float(np.degrees(radians)), color="#777", lw=1.1))
    ax.plot([0, 1], [0, 0], "o", ms=4, color=INK)
    ax.text(.37, -.17, f"v{index}", ha="center", va="top", color=BLUE,
            fontsize=10, fontweight="bold")
    ax.text(1.35, .69, f"v{index+1}", ha="center", color=ORANGE,
            fontsize=10, fontweight="bold")
    ax.text(1.15, .16, f"{np.degrees(radians):.1f}°", fontsize=9.5,
            color=INK, bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set(xlim=(-.13, 2.1), ylim=(-.35, 1.35), aspect="equal")
    clean_image_axes(ax)
    ax.set_title(f"pair {index+1}:  {ANGLES[index]}→{ANGLES[index+1]}°  then  "
                 f"{ANGLES[index+1]}→{ANGLES[index+2]}°", fontsize=9.5, pad=3)


def make_pairs(images, steps, lengths, pair_angles):
    fig = plt.figure(figsize=(16.8, 12.2), facecolor="white")
    fig.suptitle("Consecutive lean moves: each pair fits exactly in 2D",
                 y=.985, fontsize=19, fontweight="bold", color=INK)
    fig.text(.5, .953,
             "Controlled generated 1. Only lean changes, in 5° steps from −10° to 35°. "
             "A pixel is its fraction covered by ink (0 = white, 1 = black).",
             ha="center", fontsize=10.5, color="#505c63")
    gs = fig.add_gridspec(4, 4, left=.06, right=.96, top=.913, bottom=.11,
                          height_ratios=[1.25, 1.2, 1.2, .85],
                          hspace=.58, wspace=.31)

    for col, (a, title) in enumerate([(images[0], "Start: lean −10°"),
                                       (images[1], "After one move: −5°")]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(a, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        clean_image_axes(ax)
        ax.set_title(title, fontsize=11, pad=5)
    limit = np.abs(steps).max()
    for col, k in [(2, 0), (3, 2)]:
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(steps[k].reshape(28, 28), cmap="RdBu_r", vmin=-limit,
                  vmax=limit, interpolation="nearest")
        clean_image_axes(ax)
        ax.set_title(f"v{k}: {ANGLES[k]}→{ANGLES[k+1]}° pixel change",
                     fontsize=10.6, pad=5)
    fig.text(.5, .692,
             "Above: each v is a 784-pixel difference image. Red pixels gain ink; blue lose ink. "
             "Both difference images use the same color scale.",
             ha="center", fontsize=10, color=INK)

    for k, radians in enumerate(pair_angles):
        ax = fig.add_subplot(gs[1 + k // 4, k % 4])
        draw_pair(ax, k, radians)

    ax = fig.add_subplot(gs[3, :])
    bars = ax.bar(np.arange(len(lengths)), lengths, color="#778d9a", width=.64)
    bars[0].set_color(BLUE)
    bars[-1].set_color(ORANGE)
    ax.set_xticks(np.arange(len(lengths)),
                  [f"v{k}\n{ANGLES[k]}→{ANGLES[k+1]}°"
                   for k in range(len(lengths))], fontsize=8.8)
    ax.set_ylabel("784-pixel distance\nfor each 5° move", fontsize=10)
    ax.set_ylim(0, max(lengths) * 1.24)
    for k, length in enumerate(lengths):
        ax.text(k, length + .08, f"{length:.2f}", ha="center", fontsize=8.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)

    fig.text(.5, .066,
             "In each pair box, blue and orange arrows are unit length and joined tip-to-tail. "
             "Their angle is exact, computed from all 784 pixel values.",
             ha="center", fontsize=10.2, color=INK)
    fig.text(.5, .039,
             "The orange v in one box is the blue v in the next. Each box rotates its own 2D axes "
             "to put blue horizontally, so the boxes do not share a global plane.",
             ha="center", fontsize=9.6, color="#505c63")
    fig.savefig(OUT / "lean_pairwise_planes.png", dpi=160,
                facecolor="white")
    plt.close(fig)


def unfold(lengths, pair_angles, signs):
    headings = np.r_[0, np.cumsum(signs * pair_angles)]
    steps = lengths[:, None] * np.column_stack((np.cos(headings),
                                                 np.sin(headings)))
    return np.vstack((np.zeros(2), np.cumsum(steps, axis=0)))


def novelty_ratios(directions):
    basis = []
    ratios = []
    for direction in directions:
        residual = direction.copy()
        for _ in range(2):
            for previous in basis:
                residual -= np.dot(residual, previous) * previous
        ratio = np.linalg.norm(residual)
        ratios.append(ratio)
        if ratio > 1e-10:
            basis.append(residual / ratio)
    return np.array(ratios)


def draw_unfold(ax, points, title, color, actual_endpoint):
    ax.plot(points[:, 0], points[:, 1], color=color, lw=2.4, alpha=.9)
    ax.scatter(points[:, 0], points[:, 1], color=color, s=22, zorder=3)
    ax.scatter(*points[0], color=INK, marker="s", s=65, zorder=4)
    ax.scatter(*points[-1], color=ORANGE, marker="*", s=150, zorder=5)
    ax.text(points[0, 0], points[0, 1] + .7, "start", fontsize=9,
            color=INK, ha="center")
    ax.text(points[-1, 0], points[-1, 1] + .8, "end", fontsize=9,
            color=ORANGE, ha="center")
    ax.set(xlim=(-3.5, 22.5), ylim=(-8, 8), aspect="equal",
           xlabel="constructed 2D x", ylabel="constructed 2D y")
    ax.set_title(f"{title}\n2D start-to-end = {np.linalg.norm(points[-1]):.2f}; "
                 f"actual 784D = {actual_endpoint:.2f}",
                 fontsize=11, pad=8)
    ax.grid(alpha=.14)
    ax.tick_params(labelsize=8.5)


def make_global(images, steps, lengths, directions, pair_angles):
    true_end = float(np.linalg.norm(images[-1] - images[0]))
    left = unfold(lengths, pair_angles, np.ones(len(pair_angles)))
    alternate = unfold(lengths, pair_angles,
                       np.array([1, -1] * 4, dtype=float))
    pairwise = np.degrees(np.arccos(np.clip(directions @ directions.T,
                                            -1, 1)))
    novelty = novelty_ratios(directions)

    fig = plt.figure(figsize=(17.2, 12.4), facecolor="white")
    fig.suptitle("Why the exact 2D pairs do not determine one global 2D path",
                 y=.987, fontsize=18.5, fontweight="bold", color=INK)
    fig.text(.5, .954,
             "Same generated 1 and nine consecutive +5° lean moves as the pair figure. "
             "Each point after the square marks the end of one move.",
             ha="center", fontsize=10.4, color="#505c63")
    gs = fig.add_gridspec(3, 2, left=.08, right=.96, top=.918, bottom=.105,
                          height_ratios=[.53, 1.43, 1.22],
                          hspace=.46, wspace=.24)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(images[0], cmap="gray_r", vmin=0, vmax=1,
              interpolation="nearest")
    clean_image_axes(ax)
    ax.set_title("Starting image: lean −10°", fontsize=10.5, pad=4)
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.text(0, .9, "Two ways to join the same nine arrows in 2D", fontsize=13,
            fontweight="bold", color=INK, transform=ax.transAxes)
    ax.text(0, .6,
            "Both preserve every move's true length and every angle between\n"
            "neighboring moves. A pair of vectors has no inherent left/right\n"
            "orientation, or rule for placing the next pair's plane.",
            fontsize=10.5, color="#505c63", va="top", transform=ax.transAxes)

    ax = fig.add_subplot(gs[1, 0])
    draw_unfold(ax, left, "Turn left at every pair", BLUE, true_end)
    ax = fig.add_subplot(gs[1, 1])
    draw_unfold(ax, alternate, "Alternate left and right", ORANGE, true_end)

    ax = fig.add_subplot(gs[2, 0])
    im = ax.imshow(pairwise, cmap="viridis_r", vmin=0, vmax=120,
                   interpolation="nearest")
    ax.set(xticks=range(9), yticks=range(9),
           xlabel="lean move v (earlier to later)",
           ylabel="lean move v (earlier to later)",
           title="Actual angles between every pair in 784D")
    ax.set_xticklabels([f"v{i}" for i in range(9)])
    ax.set_yticklabels([f"v{i}" for i in range(9)])
    ax.tick_params(labelsize=8)
    for i in range(9):
        for j in range(9):
            if i != j:
                ax.text(j, i, f"{pairwise[i,j]:.0f}", ha="center", va="center",
                        color="white" if pairwise[i,j] > 57 else INK,
                        fontsize=7)
    cb = fig.colorbar(im, ax=ax, shrink=.75, pad=.04)
    cb.set_label("angle in degrees", fontsize=9)

    ax = fig.add_subplot(gs[2, 1])
    bars = ax.bar(range(9), novelty, color="#527b9c", width=.7)
    bars[0].set_color("#8a9ba6")
    ax.set(xticks=range(9), xlabel="lean move, from v0 to v8",
           ylabel="fraction of this direction outside\nspan of previous directions",
           ylim=(0, 1.16), title="How much new linear direction each move adds")
    ax.set_xticklabels([f"v{i}" for i in range(9)])
    for i, value in enumerate(novelty):
        ax.text(i, value + .035, f"{value:.2f}", ha="center", fontsize=8.7)
    ax.grid(axis="y", alpha=.17)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .058,
             "The two 2D paths use the same axis scale and exact neighbor angles, "
             "yet predict different endpoints. The true endpoint is measured from all 784 pixels.",
             ha="center", fontsize=9.8, color=INK)
    fig.text(.5, .031,
             "Angle grid and new-direction bars use the original vectors without projection. "
             "Nine sampled moves can span nine linear directions while the lean path needs one parameter.",
             ha="center", fontsize=9.3, color="#505c63")
    fig.savefig(OUT / "lean_pairwise_unfoldings.png", dpi=155,
                facecolor="white")
    plt.close(fig)
    return true_end, left, alternate, novelty


def main():
    images, steps, lengths, directions, pair_angles = data()
    make_pairs(images, steps, lengths, pair_angles)
    true_end, left, alternate, novelty = make_global(
        images, steps, lengths, directions, pair_angles
    )
    print("step lengths:", np.round(lengths, 3))
    print("adjacent angles in degrees:", np.round(np.degrees(pair_angles), 1))
    print("outside-previous-span fractions:", np.round(novelty, 3))
    print("start-to-end distance: actual 784D", round(true_end, 3),
          "all-left 2D", round(np.linalg.norm(left[-1]), 3),
          "alternating 2D", round(np.linalg.norm(alternate[-1]), 3))


if __name__ == "__main__":
    main()
