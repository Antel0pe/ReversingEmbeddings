"""One figure for idea B: the "up" side found from the dots, on a fold-back and on a roll.

Run: .venv/bin/python -m unfolding_suite.experiments.up_loops_figure
Writes unfolding_suite/figures/up_loops/up_loops.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from unfolding_suite.diagnose import FAR
from unfolding_suite.experiments.up_loops import closest_links, consistent_up, facing, perpendiculars
from unfolding_suite.sheets import SHEETS, make_instance

OUT = Path(__file__).resolve().parent.parent / "figures" / "up_loops"


def panel(ax, name, title):
    inst = make_instance(SHEETS[name], 65, 100)
    links = closest_links(inst.points)
    up = consistent_up(perpendiculars(inst.points), links)
    # the method saw the sheet turned at a random angle; turn its answer back to draw it
    back = np.argsort(inst.order)
    A = inst.xyz - inst.xyz.mean(0)
    B = inst.points[back] - inst.points.mean(0)
    u, _, vt = np.linalg.svd(A.T @ B)
    up_grid = up[back] @ (u @ vt).T
    G = inst.grid
    row = (G // 2) * G + np.arange(G)
    P = inst.xyz
    ax.plot(P[row, 0], P[row, 2], color="0.55", lw=2.5, zorder=1)
    every = row[::2]
    ax.quiver(P[every, 0], P[every, 2], up_grid[every, 0], up_grid[every, 2], color="tab:blue",
              angles="xy", scale_units="xy", scale=12, width=0.004, zorder=2)
    g = inst.order[links]
    wrong = np.linalg.norm(inst.uv[g[:, 0]] - inst.uv[g[:, 1]], axis=1) > FAR
    f = facing(up, links)
    caught = wrong & (f < 0)
    missed = wrong & (f >= 0)
    for mask, colour in ((caught, "tab:green"), (missed, "red")):
        for a, b in g[mask]:
            ax.plot(*np.c_[P[a, [0, 2]], P[b, [0, 2]]], color=colour, lw=2, zorder=3)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f"{title}\n{wrong.sum()} wrong connections: {caught.sum()} caught (green), "
                 f"{missed.sum()} missed (red)", fontsize=11.5)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    panel(axes[0], "s_curl", "A. S-shaped curl: the paper folds BACK on itself,\n"
                             "so the two layers' up sides point different ways")
    panel(axes[1], "h_roll5", "B. Medium roll: the paper WRAPS around,\n"
                              "so neighbouring layers' up sides point the same way")
    fig.suptitle("Idea B: give every dot an \"up\" side (blue arrows), carried from dot to dot along the paper "
                 "using only the dot coordinates", fontsize=12.5)
    fig.text(0.5, 0.02, "Side view: the grey line is the paper (it looks the same all along its length). "
             "A connection is thrown out when the ups at its two ends point in opposite directions.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(OUT / "up_loops.png", dpi=110)
    print("wrote", OUT / "up_loops.png")


if __name__ == "__main__":
    main()
