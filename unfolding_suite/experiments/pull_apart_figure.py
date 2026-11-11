"""One figure for the pulling experiment: three cases, before and after pulling.

Run: OMP_NUM_THREADS=1 .venv/bin/python -m unfolding_suite.experiments.pull_apart_figure
Writes unfolding_suite/figures/pull_apart/pull_apart.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from unfolding_suite.diagnose import FAR
from unfolding_suite.experiments.pull_apart import pull
from unfolding_suite.link_check import checked_links
from unfolding_suite.scoring import score_layout, signed_area, to_grid_order, triangles
from unfolding_suite.sheets import SHEETS, make_instance

OUT = Path(__file__).resolve().parent.parent / "figures" / "pull_apart"


def true_links(inst):
    _, near = cKDTree(inst.uv).query(inst.uv, k=11)
    g = np.c_[np.repeat(np.arange(len(inst.uv)), 10), near[:, 1:].ravel()]
    where = np.empty(len(inst.order), int)
    where[inst.order] = np.arange(len(inst.order))
    return where[g]


CASES = [
    ("tight_roll", "true", "Tight roll, strings = the paper's\nreal neighbours (a cheat, to test pulling)"),
    ("tight_roll", "dots", "Tight roll, strings chosen\nfrom the dots alone (fair)"),
    ("hard_crumple", "true", "Hard crumple, strings = the paper's\nreal neighbours (a cheat)"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15, 10.2))
    for col, (name, which, title) in enumerate(CASES):
        inst = make_instance(SHEETS[name], 65, 100)
        links = true_links(inst) if which == "true" else checked_links(inst.points, 10, 12, 0.5)
        X, _, _ = pull(inst.points, links)
        c = X - X.mean(0)
        flat = to_grid_order(c @ np.linalg.svd(c, full_matrices=False)[2][:2].T, inst)
        r = score_layout(flat, inst.uv, inst.xyz, inst.grid, inst.sheet.family)
        g = inst.order[links]
        jumps = g[np.linalg.norm(inst.uv[g[:, 0]] - inst.uv[g[:, 1]], axis=1) > FAR]

        # before: the crumpled sheet in 3-D, coloured by left-right position on the paper
        ax = fig.add_subplot(2, 3, col + 1, projection="3d")
        P = inst.xyz
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=inst.uv[:, 0], cmap="viridis", s=2, linewidths=0)
        for a, b in jumps[:: max(1, len(jumps) // 150)]:
            ax.plot(*np.c_[P[a], P[b]], color="red", lw=0.8)
        ax.set_title(f"{title}\n\nBEFORE: the crumpled sheet\n"
                     f"{len(jumps)} strings join different layers (red)", fontsize=11)
        ax.set_axis_off()
        ax.view_init(18, -60 if name == "tight_roll" else -40)

        # after: the pulled sheet seen face-on; red = triangles of paper that came out mirror-image (folded)
        ax = fig.add_subplot(2, 3, col + 4)
        T = triangles(inst.grid)
        area = signed_area(flat, T)
        major = np.sign(np.sum(signed_area(inst.uv, T) * np.sign(area))) or 1
        bad = major * area < 0
        ax.scatter(flat[:, 0], flat[:, 1], c=inst.uv[:, 0], cmap="viridis", s=2, linewidths=0)
        for t in T[bad]:
            ax.fill(*flat[t].T, color="red", lw=0)
        ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        verdict = "PASS: one-to-one, nothing folded" if r["passed"] else \
            f"FAIL: {100 * max(r['fold_share'], r['overlap_share']):.0f}% of the paper folded over"
        ax.set_title(f"AFTER pulling, seen face-on\n{verdict}", fontsize=11,
                     color="black" if r["passed"] else "firebrick")
    fig.suptitle("Pulling the paper apart: strings keep neighbouring dots at their original distance, "
                 "and every dot is pushed away from the middle", fontsize=13)
    fig.text(0.5, 0.015, "Colour = where the dot sits along one edge of the original paper "
             "(a smooth colour sweep after pulling means it unfolded; which way it runs does not matter).\n"
             "Red in the bottom row = paper that ended up mirror-image, i.e. folded over.  Seed 100.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(OUT / "pull_apart.png", dpi=110)
    print("wrote", OUT / "pull_apart.png")


if __name__ == "__main__":
    main()
