"""One uncluttered picture of the best unwrappings and biggest failures, from saved results.

Run from the repo root: .venv/bin/python -m unfolding_suite.experiments.make_twitter_graphic
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

from unfolding_suite.report import box_shape, print_colours
from unfolding_suite.scoring import align_to_square, triangles
from unfolding_suite.sheets import SHEETS, make_instance

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUN = Path("figures/unfolding_suite/all")
OUT = Path("figures/unfolding_suite/twitter_unwrapping.png")
INK, MUTED, GOOD, BAD = "#0b0b0b", "#6a6963", "#008300", "#d03b3b"
ROWS = [("s_curl", "S-curl", (16, -74)), ("roll", "Roll", (16, -74)),
        ("diagonal_roll", "Diagonal roll", (20, -35)), ("tight_roll", "Tight roll", (16, -74))]
COLUMNS = [("best_fit_plane", "Squash it flat"), ("isomap_k6", "Walk along it,\nshort steps"),
           ("isomap_k20", "Walk along it,\nlonger steps")]
VIEW = 1.5
DOT = 3.4

tests = json.loads((RUN / "results.json").read_text())["tests"]
layouts = np.load(RUN / "layouts.npz")
stored = {k.rsplit("|", 1)[0]: k for k in layouts.files}
rng = np.random.default_rng(0)

cell, top, left = 3.0, 1.75, 0.15
width, height = left + cell * 4 + 0.15, top + (cell + 0.42) * len(ROWS) + 0.1
fig = plt.figure(figsize=(width, height), facecolor="white")


def box(x, y, w, h):
    return [x / width, 1 - (y + h) / height, w / width, h / height]


fig.text(left / width + 0.012, 1 - 0.47 / height, "Can you uncrumple paper from 3D dots alone?", fontsize=27,
         fontweight="bold", color=INK, va="center")
fig.text(left / width + 0.012, 1 - 0.98 / height, "Same colour = same spot on the paper. A clean colour blend means it came back flat.",
         fontsize=15.5, color=MUTED, va="center")
heads = ["Crumpled sheet"] + [c[1] for c in COLUMNS]
for j, head in enumerate(heads):
    fig.text((left + cell * (j + 0.5)) / width, 1 - (top - 0.27) / height, head, ha="center", va="center", fontsize=16.5,
             fontweight="bold", color=INK, linespacing=1.1)

for i, (name, label, view) in enumerate(ROWS):
    y = top + (cell + 0.42) * i
    seed = int(stored[f"{name} / {COLUMNS[1][0]}"].rsplit("|", 1)[1])
    inst = make_instance(SHEETS[name], 65, seed)
    T = triangles(65)
    ax = fig.add_axes(box(left - 0.1, y - 0.1, cell + 0.2, cell + 0.2), projection="3d")
    shuffle = rng.permutation(len(inst.uv))
    ax.scatter(*inst.xyz[shuffle].T, c=print_colours(inst.uv[shuffle]), s=DOT, linewidths=0, depthshade=False, rasterized=True)
    lo, hi = inst.xyz.min(0), inst.xyz.max(0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(box_shape(inst.xyz))
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_axis_off()
    fig.text((left + cell * 0.5) / width, 1 - (y + cell + 0.12) / height, label, ha="center", va="center", fontsize=15,
             color=MUTED)
    for j, (method, _) in enumerate(COLUMNS):
        test = f"{name} / {method}"
        key = stored[test]
        I = make_instance(SHEETS[name], 65, int(key.rsplit("|", 1)[1]))
        aligned, _ = align_to_square(layouts[key], I.uv)
        ax = fig.add_axes(box(left + cell * (j + 1) + 0.16, y + 0.05, cell - 0.32, cell - 0.32))
        order = rng.permutation(len(I.uv))    # shuffled, so overlapping layers show as speckle
        ax.scatter(*aligned[order].T, c=print_colours(I.uv[order]), s=DOT, linewidths=0, rasterized=True)
        ax.set_xlim(-VIEW, VIEW)
        ax.set_ylim(-VIEW, VIEW)
        ax.set_aspect("equal")
        ax.axis("off")
        passed = tests[test]["passed"]
        words = "Unfolded" if passed else tests[test]["words"].capitalize()
        fig.text((left + cell * (j + 1.5)) / width, 1 - (y + cell + 0.12) / height, words, ha="center", va="center", fontsize=16,
                 fontweight="bold", color=GOOD if passed else BAD)
fig.savefig(OUT, dpi=140, facecolor="white", metadata={"Software": None})
print(OUT, fig.get_size_inches() * 140)
