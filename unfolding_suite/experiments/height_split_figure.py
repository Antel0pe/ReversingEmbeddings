"""One figure for the height split: where it separates the layers, where it cannot, and the tally.

Run: .venv/bin/python -m unfolding_suite.experiments.height_split_figure
Writes unfolding_suite/figures/height_split/height_split.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from unfolding_suite.diagnose import FAR
from unfolding_suite.experiments.height_split import OUT as RUNS, heights, my_layer
from unfolding_suite.sheets import SHEETS, make_instance

OUT = Path(__file__).resolve().parent.parent / "figures" / "height_split"
K, GAP = 40, 0.5


def neighbourhood_panel(ax, inst, node, title):
    """Side view of the roll (it is the same all along its length), zoomed on one dot."""
    k = int(np.where(inst.order == node)[0][0])                         # that dot's index in the shuffled points
    near, h = heights(inst.points, K, quadratic=True)
    spacing = np.median(cKDTree(inst.points).query(inst.points, k=2)[0][:, 1])
    keep = my_layer(h, GAP * spacing)[k]
    nodes = inst.order[near[k]]
    other = np.linalg.norm(inst.uv[nodes] - inst.uv[node], axis=1) > FAR
    P = inst.xyz
    row = (node // inst.grid) * inst.grid + np.arange(inst.grid)        # one row of the paper traces the spiral
    ax.plot(P[row, 0], P[row, 2], color="0.75", lw=2, zorder=0, label="the paper (side view)")
    groups = [(keep & ~other, "tab:blue", "kept, really my layer"),
              (~keep & other, "tab:orange", "dropped, really another layer  (right)"),
              (keep & other, "red", "KEPT but another layer  (wrong)"),
              (~keep & ~other, "purple", "dropped but my layer  (lost)")]
    for mask, colour, label in groups:
        mask = mask.copy()
        mask[0] = False
        ax.scatter(P[nodes[mask], 0], P[nodes[mask], 2], s=28, color=colour, lw=0,
                   label=f"{label}: {mask.sum()}")
    ax.scatter(*P[node, [0, 2]], s=160, marker="*", color="black", zorder=5, label="the dot being linked")
    c = P[node, [0, 2]]
    r = 0.16
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)


def tally_panel(ax):
    scores = json.loads((RUNS / "scores.json").read_text())
    sheets = ["s_curl", "h_roll5", "tight_roll"]
    methods = [("isomap_k10", "connect to 10 closest dots", "0.6"),
               ("height_K40_g0.5_bowl", "height split (your idea)", "tab:blue"),
               ("check_k10", "older surface filter", "tab:green"),
               ("both_K40_g0.5", "both together", "black")]
    width = 0.2
    for j, (m, label, colour) in enumerate(methods):
        for i, s in enumerate(sheets):
            r = scores[f"{s} / {m}"]["100"]
            x = i + (j - 1.5) * width
            ax.bar(x, max(r["jumps"], 0.7), width, color=colour, label=label if i == 0 else None)
            word = "PASS" if r["passed"] else f"fold\n{100 * max(r['fold_share'], r['overlap_share']):.0f}%"
            ax.text(x, max(r["jumps"], 0.7) * 1.15, f"{r['jumps']}\n{word}", ha="center", va="bottom", fontsize=8,
                    color="darkgreen" if r["passed"] else "firebrick")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 1e5)
    ax.set_xticks(range(len(sheets)), ["S-shaped curl\n(curl's end comes\nin at an angle)", "medium roll",
                                       "tight roll\n(layers closer together\nthan neighbouring dots)"])
    ax.set_ylabel("wrong connections (dot joined to another layer)")
    ax.set_title("C. Wrong connections left, and did the sheet fully unfold?\n"
                 "(number above each bar = wrong connections left)", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inst = make_instance(SHEETS["tight_roll"], 65, 100)
    G = inst.grid
    mid = G // 2
    outer = mid * G + int(np.argmin(np.abs(np.linspace(-1, 1, G) - 0.05)))   # an outer turn, middle of the roll's length
    inner = mid * G + (G - 1)                                                 # the paper's free end, innermost turn
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.6), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    neighbourhood_panel(axes[0], inst, outer, "A. Tight roll, a dot in an outer turn:\n"
                        "heights split into clean groups, the other layers are dropped")
    neighbourhood_panel(axes[1], inst, inner, "B. Tight roll, a dot at the paper's free end\n"
                        "(innermost turn): the curl is too tight to split")
    tally_panel(axes[2])
    fig.suptitle("Height split: fit a curved plane to a dot's 40 closest dots, measure each one's height above it, "
                 "and keep only the group of heights the dot itself is in", fontsize=12.5)
    fig.text(0.5, 0.01, "A and B: side view of the roll (it looks the same all along its length). "
             "Grey line = the paper; coloured = the dot's 40 closest dots, judged against the true paper.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(OUT / "height_split.png", dpi=110)
    print("wrote", OUT / "height_split.png")


if __name__ == "__main__":
    main()
