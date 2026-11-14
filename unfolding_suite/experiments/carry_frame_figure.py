"""Figure for carry_frame: what carrying two directions outward does to a bent sheet and a crumpled one.

Three panels, all flat results drawn as the paper's own graph paper (every 8th row and column of dots):
  1. s_curl (only bent), version A: one walk outward, rotation only.
  2. crumple (stretched), version A: the same walk.
  3. crumple, version C: every link votes, and each dot may stretch.
Dots are coloured by the loop miss of version A; triangles that came out mirror-flipped are filled red.

Run: OMP_NUM_THREADS=4 .venv/bin/python -m unfolding_suite.experiments.carry_frame_figure
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from unfolding_suite.experiments.carry_frame import carry, stretch, up_checked_links
from unfolding_suite.run import failure_words
from unfolding_suite.scoring import score_layout, signed_area, to_grid_order, triangles
from unfolding_suite.sheets import SHEETS, make_instance

OUT = Path(__file__).resolve().parent.parent / "figures" / "carry_frame"


def per_dot_miss(c, n):
    nt = ~c["in_tree"]
    total, count = np.zeros(n), np.zeros(n)
    for col in (0, 1):
        np.add.at(total, c["links"][nt, col], c["miss"][nt])
        np.add.at(count, c["links"][nt, col], 1)
    return total / np.maximum(count, 1)


def draw(ax, inst, flat, miss, root, title, note):
    G = inst.grid
    F = to_grid_order(flat, inst)
    r = score_layout(F, inst.uv, inst.xyz, G, inst.sheet.family)
    F = F - F[inst.order[root]]
    T = triangles(G)
    area = signed_area(F, T)
    flipped = np.sign(area) != np.sign(np.median(area))
    for t in T[flipped]:
        ax.fill(*F[t].T, color="#d62728", lw=0, alpha=0.8, zorder=1)
    P = F.reshape(G, G, 2)
    for k in range(0, G, 8):
        ax.plot(*P[k].T, color="0.55", lw=0.6, zorder=2)
        ax.plot(*P[:, k].T, color="0.55", lw=0.6, zorder=2)
    m = to_grid_order(np.c_[miss, miss], inst)[:, 0]
    sc = ax.scatter(*F.T, c=np.clip(m, 1e-4, 3), s=2.5, cmap="viridis", norm=LogNorm(1e-4, 3), zorder=3, lw=0)
    ax.plot(*F[inst.order[root]], marker="*", ms=16, color="white", mec="black", zorder=4)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title, fontsize=12, loc="left")
    verdict = (("PASS" if r["passed"] else "FAIL") + f"   flipped (red): {r['fold_share']:.0%}"
               f"   on top of other paper: {r['overlap_share']:.0%}")
    ax.text(0.02, -0.02, f"{verdict}\n{note}", transform=ax.transAxes, va="top", fontsize=10.5)
    return sc


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(17, 7.2))
    fig.subplots_adjust(left=0.02, right=0.9, top=0.74, bottom=0.17, wspace=0.08)
    rows = []
    for name in ("s_curl", "crumple"):
        inst = make_instance(SHEETS[name], 65, 100)
        links = up_checked_links(inst.points)
        c = carry(inst.points, links)
        rows.append((inst, c, per_dot_miss(c, len(inst.points))))
    (bi, bc, bm), (ci, cc, cm) = rows
    flatC, scale = stretch(cc, len(ci.points))

    def miss_note(c):
        nt = ~c["in_tree"]
        return (f"loop miss: typical {np.median(c['miss'][nt]):.1%} of a step,\n"
                f"worst 5% of links over {np.quantile(c['miss'][nt], 0.95):.0%}")

    draw(axs[0], bi, bc["flat"], bm, bc["root"], "1. Bent paper (s_curl)\n    one walk outward, turning only", miss_note(bc))
    draw(axs[1], ci, cc["flat"], cm, cc["root"], "2. Crumpled, stretched paper (crumple)\n    the same walk", miss_note(cc))
    sc = draw(axs[2], ci, flatC, cm, cc["root"],
              "3. Same crumple: every link votes,\n    each dot may stretch",
              f"stretch used: {np.quantile(scale, 0.05):.2f}x to {np.quantile(scale, 0.95):.2f}x (5th-95th pct)\n"
              "dot colours: loop miss from panel 2")
    cax = fig.add_axes([0.915, 0.25, 0.012, 0.5])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("loop miss at this dot: how far its links' own steps disagree\n"
                 "with where the walk placed their ends (share of a step)", fontsize=10)
    fig.suptitle("Carry two directions outward from one dot (white star), write each step in them, place the dot",
                 fontsize=14, x=0.02, ha="left", y=0.98)
    fig.text(0.02, 0.91, "Each panel: the flat result. Grey lines = graph paper printed on the sheet (every 8th row and "
             "column of dots). Red = triangles of paper that came out mirror-flipped (folded over).\n"
             "Walking order and which side is 'up' come from up_loops. 4,225 dots per sheet, seed 100.",
             fontsize=10.5, va="top")
    fig.savefig(OUT / "carry_frame.png", dpi=110)
    print(OUT / "carry_frame.png")


if __name__ == "__main__":
    main()
