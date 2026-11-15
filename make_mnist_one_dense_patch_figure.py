"""Figure for the `dense` part of analyze_mnist_one_flat_patch_directions.py.

    .venv/bin/python make_mnist_one_dense_patch_figure.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_mnist_one_flat_patch_directions import mnist_ones

D = Path(__file__).resolve().parent / "figures/mnist_one_flat_patch_directions"
R = json.loads((D / "dense.json").read_text())
Z = json.loads((D / "zoom.json").read_text())

INK, MUTED, GRID = "#222222", "#6b6b66", "#e4e3de"
SHADES = {"densest 1%": "#0d366b", "densest 5%": "#2a78d6",
          "middle 10%": "#6da7ec", "emptiest 5%": "#b7d3f6"}
ORANGE = "#eb6834"
NAMES = {"densest 1%": "most crowded 1%", "densest 5%": "most crowded 5%",
         "middle 10%": "middle of the pack", "emptiest 5%": "loneliest 5%"}


def main():
    X = mnist_ones().astype(np.float64)
    g = R["mnist"]["groups"]
    crowded = g["densest 1%"]["centres"][:4]

    fig = plt.figure(figsize=(15, 11.5))
    fig.text(0.05, 0.965, "Only the most crowded MNIST 1s: is there a flat patch there?",
             fontsize=17, weight="bold", color=INK)
    fig.text(0.05, 0.94, "Crowded = its 10 nearest 1s are unusually close. Distance = how big the difference "
             "image is, as a % of the 1 itself.", fontsize=11, color=MUTED)

    # --- top: four of the most crowded 1s, each with its 5 nearest ---------------
    fig.text(0.05, 0.905, "Four 1s from the most crowded 1% (left, outlined) and their 5 nearest real 1s",
             fontsize=13, weight="bold", color=INK)
    w, h = 0.055, 0.055 * 15 / 11.5
    for r, c in enumerate(crowded):
        d = np.linalg.norm(X - X[c], axis=1)
        d[d < 1e-6] = np.inf
        nn = np.argsort(d)[:5]
        rel = d[nn] / np.linalg.norm(X[c])
        col = r % 2
        row = r // 2
        x0 = 0.05 + col * 0.47
        y0 = 0.80 - row * 0.135
        for i, im in enumerate([X[c]] + list(X[nn])):
            ax = fig.add_axes([x0 + i * (w + 0.015) + (0.012 if i else 0), y0, w, h])
            ax.imshow(im.reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#0d366b" if i == 0 else "0.6"), s.set_linewidth(2.5 if i == 0 else 0.8)
            ax.text(0.5, -0.08, "this 1" if i == 0 else f"{100 * rel[i - 1]:.0f}%",
                    transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
                    color=INK if i == 0 else MUTED)

    # --- bottom left: how close are the neighbours? ------------------------------
    ks = np.array(R["mnist"]["ks"])
    axL = fig.add_axes([0.06, 0.11, 0.36, 0.46])
    axR = fig.add_axes([0.54, 0.11, 0.33, 0.46])
    axL.set_title("How far away are the neighbours?\n(the k-th nearest 1, median over starting 1s)",
                  fontsize=13, loc="left", color=INK)
    for name, col in SHADES.items():
        axL.plot(ks, 100 * np.array(g[name]["radius"]), color=col, lw=2.4, label=NAMES[name])
    axL.plot(ks, 100 * np.array(R["gen"]["groups"]["densest 1%"]["radius"]), color=ORANGE, lw=2.4,
             label="generated 1s (5 knobs), most crowded 1%")
    flat = 100 * Z["gen"]["rows"]["0.01"]["median_radius"]
    axL.axhspan(0, flat, color=ORANGE, alpha=0.18, lw=0)
    axL.text(2.2, flat + 1.2, f"generated 1s were flat only inside about {flat:.0f}%",
             fontsize=10.5, color=INK)
    axL.set_ylim(0, 100)
    axL.set_ylabel("distance to the k-th nearest 1 (% of the 1's size)", fontsize=12)
    axL.legend(loc="upper left", frameon=False, fontsize=10.5)

    # --- bottom right: the count -------------------------------------------------
    axR.set_title("Directions needed to reach every neighbour", fontsize=13, loc="left", color=INK)
    axR.plot(ks, ks, ls="--", color="0.55", lw=1.5)
    axR.text(9, 60, "dashed line: every neighbour\nneeds its own new direction", fontsize=10, color=MUTED)
    for name, col in SHADES.items():
        y = np.array(g[name]["median_count"])
        axR.plot(ks, y, color=col, lw=2.4)
        if name == "emptiest 5%":
            axR.text(205, y[-1], f"{NAMES[name]}: {y[-1]:.0f}", fontsize=10, color=INK, va="center",
                     ha="left", clip_on=False)
    top = [g[n]["median_count"][-1] for n in ("densest 1%", "densest 5%", "middle 10%")]
    axR.text(205, 88, f"most crowded 1%, 5%\nand middle: {min(top):.0f}-{max(top):.0f}", fontsize=10,
             color=INK, va="center", ha="left", clip_on=False)
    gy = np.array(R["gen"]["groups"]["densest 1%"]["median_count"])
    axR.plot(ks, gy, color=ORANGE, lw=2.4)
    axR.text(205, gy[-1], f"generated 1s (5 knobs),\nmost crowded 1%: {gy[-1]:.0f}", fontsize=10,
             color=INK, va="center", ha="left", clip_on=False)
    axR.set_ylim(0, 90)
    axR.set_ylabel("directions needed (10% of each arrow may be left over)", fontsize=12)

    for ax in (axL, axR):
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20, 50, 100, 200])
        ax.set_xticklabels(["2", "5", "10", "20", "50", "100", "200"])
        ax.set_xlim(2, 200)
        ax.set_xlabel("how many nearest neighbours are included (k)", fontsize=12)
        ax.grid(True, color=GRID, lw=0.8), ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.text(0.05, 0.015, "All 13,533 MNIST 1s were ranked by the distance to their 10th nearest 1. Each blue line "
             "uses up to 300 starting 1s from one slice of that ranking (135 for the most crowded 1%).\nThe same "
             "neighbours (all 13,533 1s) are used in every case; only the choice of starting 1 changes.",
             fontsize=9.5, color=MUTED)
    fig.savefig(D / "dense_patches.png", dpi=110)
    print("wrote", D / "dense_patches.png")


if __name__ == "__main__":
    main()
