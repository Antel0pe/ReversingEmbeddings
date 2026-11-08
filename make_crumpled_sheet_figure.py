"""Figures for analyze_crumpled_sheet_unrolling.py: a printed checkerboard, before and after.

1. sheet_flattening_main.png: one crumpled sheet, three big panels.
2. sheet_flattening_all.png: every test sheet, same three columns.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

from analyze_crumpled_sheet_unrolling import one_to_one, squash

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("figures/crumpled_sheet_unrolling")
res = json.loads((OUT / "results.json").read_text())
data = np.load(OUT / "sheets.npz")
UV = data["UV"]
INK, MUTED = "#0b0b0b", "#52514e"
GOOD, BAD = "#008300", "#d03b3b"
BLUE, ORANGE = np.array(matplotlib.colors.to_rgb("#2a78d6")), np.array(matplotlib.colors.to_rgb("#eb6834"))

# Printed on the paper: colour runs blue -> orange from left edge to right edge, and
# shade runs pale -> deep from bottom edge to top edge. Every spot has its own colour,
# so a clean flattening shows a smooth blend and an overlap shows as speckle.
a = ((UV[:, 0] + 1) / 2)[:, None]
shade = (0.30 + 0.70 * (UV[:, 1] + 1) / 2)[:, None]
COL = ((1 - a) * BLUE + a * ORANGE) * shade + (1 - shade) * 1.0
ORDER = np.random.default_rng(0).permutation(len(UV))   # draw in random order: overlaps show as speckle


def flat_panel(ax, Y, verdict, size=3):
    Y = Y - Y.mean(0)
    ax.scatter(Y[ORDER, 0], Y[ORDER, 1], c=COL[ORDER], s=size, linewidths=0, rasterized=True)
    ax.set_aspect("equal", "datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#c9c8c2")
    good = verdict < 0.005
    ax.text(0.5, -0.04, "one-to-one" if good else f"{100 * verdict:.0f}% overlap",
            transform=ax.transAxes, ha="center", va="top", fontsize=13, color=GOOD if good else BAD,
            fontweight="bold")


def sheet_panel(ax, X, elev=28, azim=-60, size=3):
    ax.scatter(X[ORDER, 0], X[ORDER, 1], X[ORDER, 2], c=COL[ORDER], s=size, linewidths=0, depthshade=False,
               rasterized=True)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect(np.ptp(X, 0))
    ax.set_axis_off()


def pct(x):
    return "0%" if x == 0 else f"{100 * x:.2g}%"


def score(r, method):
    return r["squash_best_of_3000_tilts_overlap"] if method == "squash" else r["unroll"].get("overlap", 1.0)


# ------------------------------------------------------------------ figure 1
NAME = "crumpled, layers apart"
r = res[NAME]
fig = plt.figure(figsize=(16, 7.4), facecolor="white")
fig.text(0.03, 0.945, "Flattening a crumpled sheet that is 2D but sits in 3D", fontsize=21, fontweight="bold",
         color=INK)
fig.text(0.03, 0.895, "The paper is printed blue to orange left to right, pale to deep bottom to top. Each method sees only "
         "4,000 dots in 3D and is told the sheet is 2D.\nWorked: the colours come back as a smooth blend. Failed: speckle, "
         "where different parts of the paper land on the same spot (\"overlap\").", fontsize=12.5, color=MUTED,
         va="top", linespacing=1.4)
ax = fig.add_axes([0.01, 0.10, 0.33, 0.66], projection="3d")
sheet_panel(ax, data[f"{NAME}|sheet"], elev=35, azim=-50)
fig.text(0.03, 0.775, "1. The crumpled sheet", fontsize=15, color=INK)
ax = fig.add_axes([0.37, 0.20, 0.27, 0.54])
flat_panel(ax, data[f"{NAME}|squash"], r["squash_at_40_random_points"]["median_overlap"])
fig.text(0.37, 0.775, "2. Squashed straight onto a flat plane", fontsize=15, color=INK)
fig.text(0.37, 0.045, f"Plane touching the sheet at a random dot (typical of 40 tried).\n"
         f"Best of 3,000 plane tilts still overlaps {100 * r['squash_best_of_3000_tilts_overlap']:.0f}%.",
         fontsize=11.5, color=MUTED, va="bottom")
ax = fig.add_axes([0.69, 0.20, 0.27, 0.54])
flat_panel(ax, data[f"{NAME}|unroll"], r["unroll"]["overlap"])
fig.text(0.69, 0.775, "3. Unrolled by walking along the sheet", fontsize=15, color=INK)
fig.text(0.69, 0.045, "Distance = shortest walk hopping between nearby dots\n"
         "(each dot's 10 nearest), then laid flat keeping those\ndistances. Turned or mirrored doesn't matter.",
         fontsize=11.5, color=MUTED, va="bottom")
fig.savefig(OUT / "sheet_flattening_main.png", dpi=130, facecolor="white")

# ------------------------------------------------------------------ figure 2
names = list(res)
SHOW = {"gently curved (bowl)": "Gently\ncurved\n(bowl)", "rolled up": "Rolled up",
        "crumpled, layers apart": "Crumpled", "tightly rolled": "Tightly\nrolled",
        "crumpled, layers close": "Crumpled\nharder"}
fig = plt.figure(figsize=(14, 3.3 * len(names) + 1.4), facecolor="white")
H = 3.3 * len(names) + 1.4
fig.text(0.03, 1 - 0.45 / H, "Every test sheet: where each method breaks", fontsize=20, fontweight="bold",
         color=INK)
fig.text(0.03, 1 - 0.85 / H, "Same printed colours and methods. Squashed column: the best of 3,000 plane tilts. "
         "Overlap: share of the paper landing on top of a different part.", fontsize=12, color=MUTED)
heads = ["The sheet", "Squashed flat (best tilt)", "Unrolled by walking along it"]
for j, h in enumerate(heads):
    fig.text(0.23 + 0.235 * j, 1 - 1.25 / H, h, fontsize=14, color=INK, ha="center")
_r = np.random.default_rng(0)   # the same 3,000 tilts the analysis used
_r.uniform(-1, 1, (len(UV), 2))
dirs = _r.normal(size=(3000, 3))
for i, name in enumerate(names):
    r = res[name]
    top = 1 - (1.45 + 3.3 * i) / H
    hgt = 2.55 / H
    fig.text(0.03, top - hgt / 2, SHOW[name], fontsize=14, color=INK, va="center", fontweight="bold",
             linespacing=1.3)
    ax = fig.add_axes([0.12, top - hgt, 0.22, hgt], projection="3d")
    sheet_panel(ax, data[f"{name}|sheet"], size=1.5)
    X = data[f"{name}|sheet"]
    # recompute the best tilt's picture for display (same 3000 tilts as the analysis)
    d = dirs.copy()
    d[d[:, 2] < 0] *= -1
    best = min(d, key=lambda n: one_to_one(squash(X, n), UV)["overlap"])
    ax = fig.add_axes([0.37, top - hgt + 0.03 / H * 10, 0.17, hgt - 0.35 / H])
    flat_panel(ax, squash(X, best), r["squash_best_of_3000_tilts_overlap"], size=1.5)
    ax = fig.add_axes([0.605, top - hgt + 0.03 / H * 10, 0.17, hgt - 0.35 / H])
    flat_panel(ax, data[f"{name}|unroll"], r["unroll"].get("overlap", 1.0), size=1.5)
    gap = r["closest_approach_of_different_parts"] / r["neighbour_spacing"]
    note = (f"Closest two layers come:\n{gap:.1f}× the gap between\nneighbouring dots.\n\n"
            f"{pct(r['unroll_share_of_steps_that_jump_layers'])} of the short steps\njump to another layer.")
    more = r.get("more_points")
    if more:
        n_best = max(more, key=int)
        m = more[n_best]
        ov = m.get("overlap", 1.0)
        note += (f"\n\nWith {int(n_best):,} dots: steps that\njump {pct(m['share_of_steps_that_jump_layers'])}, "
                 f"overlap {100 * ov:.0f}%.")
    fig.text(0.81, top - hgt / 2, note, fontsize=11, color=MUTED, va="center", linespacing=1.35)
fig.savefig(OUT / "sheet_flattening_all.png", dpi=110, facecolor="white")
print("saved")
