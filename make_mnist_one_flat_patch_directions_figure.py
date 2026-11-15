"""Figure for analyze_mnist_one_flat_patch_directions.py (run that first).

    .venv/bin/python make_mnist_one_flat_patch_directions_figure.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_mnist_one_flat_patch_directions import mnist_ones
from grey_ones import RANGES, render

D = Path(__file__).resolve().parent / "figures/mnist_one_flat_patch_directions"
R = json.loads((D / "results.json").read_text())
Z = json.loads((D / "zoom.json").read_text())

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]  # far -> near
INK, MUTED, GRID = "#222222", "#6b6b66", "#e4e3de"


def strip(fig, top, title, centre, neigh, dists, colour, prec=0):
    fig.text(0.05, top, title, fontsize=13, color=INK, weight="bold", va="bottom")
    w, gap, left = 0.07, 0.02, 0.05
    h = w * 15 / 12.5
    imgs = [centre] + list(neigh)
    for i, im in enumerate(imgs):
        x = left + i * (w + gap) + (0.03 if i else 0)
        ax = fig.add_axes([x, top - 0.008 - h, w, h])
        ax.imshow(im.reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(colour if i == 0 else "0.6"), s.set_linewidth(2.5 if i == 0 else 0.8)
        lab = "this 1" if i == 0 else f"#{i}: {100 * dists[i - 1]:.{prec}f}% away"
        ax.text(0.5, -0.1, lab, transform=ax.transAxes, ha="center", va="top", fontsize=10,
                color=INK if i == 0 else MUTED)


def main():
    # --- example neighbourhoods -------------------------------------------------
    X = mnist_ones().astype(np.float64)
    rng = np.random.default_rng(5)
    c = rng.integers(len(X))
    d = np.linalg.norm(X - X[c], axis=1)
    d[d < 1e-6] = np.inf
    order = np.argsort(d)[:6]
    m_d = d[order] / np.linalg.norm(X[c])

    rng = np.random.default_rng(7)
    span = RANGES[:, 1] - RANGES[:, 0]
    pc = rng.uniform(RANGES[:, 0] + 0.1 * span, RANGES[:, 1] - 0.1 * span)
    P = pc + rng.uniform(-0.01, 0.01, (200, 5)) * span
    G = render(np.vstack([pc, P]), dtype=np.float64).reshape(201, 784)
    gd = np.linalg.norm(G[1:] - G[0], axis=1)
    go = np.argsort(gd)[:6]
    g_d = gd[go] / np.linalg.norm(G[0])

    fig = plt.figure(figsize=(15, 12.5))
    fig.text(0.05, 0.965, "How many perpendicular directions does it take to reach every neighbour of a 1?",
             fontsize=17, weight="bold", color=INK)
    fig.text(0.05, 0.94, "Distance = how big the difference image is, as a % of the 1 itself. "
             "Every number below is the middle (median) of 300 random starting 1s (60 for the left plot).",
             fontsize=11, color=MUTED)
    strip(fig, 0.895, "A real MNIST 1 and its 6 nearest real 1s (out of 13,533)",
          X[c], X[order], m_d, BLUE)
    strip(fig, 0.725, "A generated 1 (5 knobs) and the 6 nearest of 200 neighbours made by nudging its knobs "
          "by up to 1% of each knob's range",
          G[0], G[1:][go], g_d, ORANGE, prec=1)

    # --- plots --------------------------------------------------------------------
    ks = np.array(R["mnist"]["ks"])
    axL = fig.add_axes([0.06, 0.085, 0.35, 0.45])
    axR = fig.add_axes([0.54, 0.085, 0.33, 0.45])
    for ax in (axL, axR):
        ax.plot(ks, ks, ls="--", color="0.55", lw=1.5)
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20, 50, 100, 200])
        ax.set_xticklabels(["2", "5", "10", "20", "50", "100", "200"])
        ax.set_xlim(2, 200), ax.set_ylim(0, 90)
        ax.set_xlabel("how many nearest neighbours are included", fontsize=12)
        ax.set_ylabel("directions needed to reach every one of them", fontsize=12)
        ax.grid(True, color=GRID, lw=0.8), ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.text(9, 60, "dashed line: every neighbour\nneeds its own new direction\n(no flat patch at all)",
                fontsize=10, color=MUTED, ha="left")

    # left: generated 1s, neighbours made on purpose at shrinking distance
    axL.set_title("Test the test: generated 1s (5 knobs)\nwith neighbours made at chosen distances",
                  fontsize=13, loc="left", color=INK)
    axL.axhline(5, color=ORANGE, lw=1, ls=":")
    rows = Z["gen"]["rows"]
    steps = Z["gen"]["steps"]
    near = []
    for col, st in zip(RAMP, steps):
        r = rows[str(st)]
        y = np.array(r["median_count"])
        axL.plot(ks, y, color=col, lw=2.2)
        if y[-1] <= 6:
            near.append(f"{100 * r['median_radius']:.1f}%")
            continue
        axL.text(205, y[-1], f"{100 * r['median_radius']:.0f}% away", fontsize=10, color=INK,
                 va="center", ha="left", clip_on=False)
    axL.text(205, 4.0, ", ".join(near) + "\naway", fontsize=10, color=INK, va="center",
             ha="left", clip_on=False)
    axL.text(205, 84, "typical\nneighbour\ndistance:", fontsize=10, color=MUTED, va="center",
             ha="left", clip_on=False)
    axL.text(6, 1.4, "flat: stays at 5 (= the 5 knobs)", fontsize=10.5, color=INK)

    # right: the real datasets at their actual density
    axR.set_title("Real MNIST 1s: their actual nearest neighbours\n(beside families with the same number of images, 13,533)",
                  fontsize=13, loc="left", color=INK)
    series = [("mnist", "MNIST 1s", BLUE),
              ("strokes12", "drawn 1s, 12 knobs", AQUA),
              ("gen", "generated 1s, 5 knobs", ORANGE)]
    for key, name, col in series:
        r = R[key]
        y = np.array(r["median_count"])[:, 1]
        lo, hi = np.array(r["q25_count"])[:, 1], np.array(r["q75_count"])[:, 1]
        axR.fill_between(ks, lo, hi, color=col, alpha=0.15, lw=0)
        axR.plot(ks, y, color=col, lw=2.4, label=name)
        rad = r["median_radius"]
        axR.text(205, y[-1], f"{name}\n{100 * rad[0]:.0f}%-{100 * rad[-1]:.0f}% away",
                 fontsize=10, color=INK, va="center", ha="left", clip_on=False)
    axR.legend(loc="upper left", frameon=False, fontsize=10.5)

    fig.text(0.05, 0.015,
             "A neighbour counts as reached when less than 10% of the arrow from this 1 to it is left over after "
             "moving along the chosen directions (the best directions for the whole group of arrows).\n"
             "Every neighbour must be reached, not just a typical one. Shaded bands on the right: middle half of the "
             "starting 1s. With 25% or 5% allowed instead, MNIST at 200 neighbours needs 49 or 104 directions.",
             fontsize=9.5, color=MUTED)
    fig.savefig(D / "flat_patch_directions.png", dpi=110)
    print("wrote", D / "flat_patch_directions.png")


if __name__ == "__main__":
    main()
