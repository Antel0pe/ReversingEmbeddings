"""Draw the causal steps of the exact generated-image lean equation.

Run: python make_lean_process_diagram.py
Output: figures/lean_process_diagram.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon
import numpy as np

from grey_ones import render
from lean_axis import coverage, geometry_from_image


ORANGE = "#cc6329"
BLUE = "#2a6dba"
GREEN = "#167d54"


def frame(ax, number, title):
    ax.set_title(f"{number}  {title}", fontsize=12, fontweight="bold", pad=9,
                 loc="left")
    for spine in ax.spines.values():
        spine.set_color("#bbb")
        spine.set_linewidth(1)


def image(ax, x):
    ax.imshow(x, cmap="gray_r", vmin=0, vmax=1, extent=(0, 28, 28, 0),
              interpolation="nearest")
    ax.set(xlim=(0, 28), ylim=(28, 0))
    ax.set_xticks([]); ax.set_yticks([])


def connect(fig, a, b, color="#888"):
    arrow = FancyArrowPatch(a, b, transform=fig.transFigure,
                            arrowstyle="-|>", mutation_scale=17,
                            lw=1.7, color=color)
    fig.add_artist(arrow)


def main():
    p = np.array([14.5, 14.5, 19.75, 3.0, 20.0])
    x = render(p)[0].astype(float)
    g = geometry_from_image(x)
    y = coverage(g, 30)
    top = g.cy - g.height / 2
    bottom = g.cy + g.height / 2
    u = np.tan(np.deg2rad(g.lean))
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.6), facecolor="white")
    fig.subplots_adjust(left=.055, right=.97, top=.85, bottom=.07,
                        hspace=.35, wspace=.20)
    fig.suptitle("How one generated image becomes a new image at +10°",
                 fontsize=20, fontweight="bold", y=.975)
    fig.text(.5, .932,
             "Input: 784 pixel values and +10°. Each box uses the result of the previous one.",
             ha="center", fontsize=11.5, color="#555")

    # Read top left to right, then follow the arrows down and back left.
    a1, a2, a3 = axes[0]
    a6, a5, a4 = axes[1]

    image(a1, x)
    for r, color in ((8, ORANGE), (20, BLUE)):
        a1.axhspan(r, r+1, fill=False, edgecolor=color, lw=2.2)
        a1.text(1.0, r-.35, f"row {r}", fontsize=9.5, color=color,
                bbox=dict(facecolor="white", alpha=.9, edgecolor="none", pad=1))
    frame(a1, 1, "Start with the pixels")
    a1.text(.5, -.075, "Look at the ink in each row", transform=a1.transAxes,
            ha="center", fontsize=9.5, color="#555")

    mass = x.sum(axis=1)
    a2.plot(np.arange(28)+.5, mass, color="#333", lw=2)
    a2.scatter([8.5, 20.5], [mass[8], mass[20]], s=55,
               color=[ORANGE, BLUE], zorder=4)
    a2.axhline(g.width, color=GREEN, ls="--", lw=1.7)
    a2.axvline(top, color="#777", ls=":", lw=1.2)
    a2.axvline(bottom, color="#777", ls=":", lw=1.2)
    a2.annotate("width = 3.00", (14, g.width), (11, 2.38), fontsize=10,
                color=GREEN, arrowprops=dict(arrowstyle="->", color=GREEN))
    a2.text(.06, .12, f"top = {top:.3f}\nbottom = {bottom:.3f}",
            transform=a2.transAxes, fontsize=10,
            bbox=dict(facecolor="white", edgecolor="#ddd", pad=5))
    a2.set(xlim=(0, 28), ylim=(-.1, 3.55), xlabel="image row",
           ylabel="sum of the 28 pixel values")
    a2.grid(alpha=.17)
    frame(a2, 2, "Row totals give the size")

    for r, color in ((8, ORANGE), (20, BLUE)):
        C = np.r_[0, np.cumsum(x[r])]
        c = int(np.argmin(np.abs(C - g.width/2)))
        z = c - C[c]
        a3.step(np.arange(29), C, where="post", color=color, lw=2.1,
                label=f"row {r}: c={c}; C={C[c]:.3f}; left={z:.3f}")
        a3.scatter([c], [C[c]], color=color, s=75, zorder=4,
                   edgecolor="white")
        a3.axvline(c, color=color, alpha=.22, lw=1)
    a3.axhline(g.width/2, color="#888", ls="--")
    a3.set(xlim=(8, 21), ylim=(-.05, 3.35), xlabel="integer column boundary c",
           ylabel="ink to the left, C(c)")
    a3.legend(loc="upper left", fontsize=8.7, framealpha=.96)
    a3.grid(alpha=.17)
    frame(a3, 3, "Find each row's left edge")
    a3.text(.98, .06, "At a marked boundary: left = c − C(c)",
            transform=a3.transAxes, ha="right", fontsize=9.2,
            bbox=dict(facecolor="white", edgecolor="#ddd", pad=4))

    rows = np.arange(5, 24)
    lefts = []
    for r in rows:
        C = np.r_[0, np.cumsum(x[r])]
        c = int(np.argmin(np.abs(C - g.width/2)))
        lefts.append(c-C[c])
    lefts = np.array(lefts)
    a4.scatter(rows+.5, lefts, s=23, color=GREEN, zorder=3)
    a4.plot(rows+.5, g.cx-g.width/2-u*(rows+.5-g.cy),
            color="#222", lw=2)
    a4.scatter([8.5, 20.5], [lefts[3], lefts[15]], s=90,
               color=[ORANGE, BLUE], edgecolor="white", zorder=5)
    a4.annotate("top edge", (8.5, lefts[3]), (7, 14.2),
                fontsize=9.5, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE))
    a4.annotate("bottom edge", (20.5, lefts[15]), (17, 10.0),
                fontsize=9.5, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE))
    a4.text(.06, .07, "slope = −tan(20°)\ncentre = (14.5, 14.5)",
            transform=a4.transAxes, fontsize=10,
            bbox=dict(facecolor="white", edgecolor="#ddd", pad=5))
    a4.set(xlabel="row position", ylabel="recovered left edge",
           xlim=(4, 25), ylim=(9.4, 16.7))
    a4.grid(alpha=.17)
    frame(a4, 4, "Connect edges to recover 20°")

    # Show the exact geometric action before any pixels are calculated.
    image(a5, x)
    for theta, color, lw, alpha in ((20, "#555", 1.7, .8),
                                    (30, GREEN, 2.7, 1)):
        t = np.tan(np.deg2rad(theta))
        xs = [g.cx-g.width/2-t*(top-g.cy),
              g.cx+g.width/2-t*(top-g.cy),
              g.cx+g.width/2-t*(bottom-g.cy),
              g.cx-g.width/2-t*(bottom-g.cy)]
        a5.add_patch(Polygon(np.c_[xs, [top, top, bottom, bottom]], closed=True,
                             fill=False, edgecolor=color, linewidth=lw, alpha=alpha))
    a5.plot(g.cx, g.cy, "o", color=GREEN, markersize=6)
    a5.annotate("top → right", (18.2, 8.5), (7.0, 5.4), color=ORANGE,
                fontsize=10, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2))
    a5.annotate("bottom → left", (10.5, 20.5), (14.3, 24.7), color=BLUE,
                fontsize=10, arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
    frame(a5, 5, "Change only the lean: 20° → 30°")
    a5.text(.5, -.075, "Grey outline: old   ·   green outline: new",
            transform=a5.transAxes, ha="center", fontsize=9.5, color="#555")

    image(a6, y)
    delta = y-x
    colors = np.zeros((*x.shape, 4))
    colors[delta>1e-5, :3] = (.78, .16, .12)
    colors[delta< -1e-5, :3] = (.12, .38, .78)
    colors[..., 3] = np.minimum(np.abs(delta)/.7, 1)*.55
    a6.imshow(colors, extent=(0, 28, 28, 0), interpolation="nearest")
    a6.text(1, 2.2, "row 8, col 18: 0.184 → 1.000 (red)\n"
            "row 8, col 15: 0.816 → 0.000 (blue)",
            fontsize=9.2, color="#333",
            bbox=dict(facecolor="white", edgecolor="#ccc", alpha=.94, pad=4))
    frame(a6, 6, "Fill pixels by new coverage")
    a6.text(.5, -.10,
            "Split each row into 16 strips; average stroke overlap\n"
            "in each pixel. Red gains ink; blue loses ink.",
            transform=a6.transAxes, ha="center", fontsize=9.2, color="#555")

    # Figure-coordinate arrows explain the order, not just the analytical panels.
    fig.canvas.draw()
    boxes = [ax.get_position() for ax in (a1, a2, a3, a4, a5, a6)]
    for i, j in ((0, 1), (1, 2)):
        connect(fig, (boxes[i].x1+.005, boxes[i].y0+boxes[i].height/2),
                (boxes[j].x0-.005, boxes[j].y0+boxes[j].height/2))
    connect(fig, (boxes[2].x0+boxes[2].width/2, boxes[2].y0-.025),
            (boxes[3].x0+boxes[3].width/2, boxes[3].y1+.015))
    for i, j in ((3, 4), (4, 5)):
        connect(fig, (boxes[i].x0-.005, boxes[i].y0+boxes[i].height/2),
                (boxes[j].x1+.005, boxes[j].y0+boxes[j].height/2))

    fig.savefig("figures/lean_process_diagram.png", dpi=155, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
