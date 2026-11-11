"""Figure for fold_flatten.py: the two folded sheets, where flattening ends up, and the check.

Run after fold_flatten.py:
  .venv/bin/python -m unfolding_suite.experiments.fold_flatten_figure
Writes unfolding_suite/figures/fold_flatten/fold_flatten.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from unfolding_suite.experiments import fold_flatten as ff
from unfolding_suite.scoring import align_to_square
from unfolding_suite.sheets import SHEETS, make_instance

OUT = Path(__file__).resolve().parent.parent / "figures" / "fold_flatten"
METHOD = "check_k10"
COLOURS = ["#4C78A8", "#9D9D9D", "#54A24B", "#E45756"]     # creased, base, fold/loop, far part
BINS = np.linspace(0, 2, 41)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L = np.load(ff.OUT / "layouts.npz")
    fig = plt.figure(figsize=(19, 11.5))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1.25], hspace=0.42, wspace=0.28,
                            left=0.04, right=0.98, top=0.86, bottom=0.07)
    fig.suptitle("A sheet creased on one side and folded on the other: where does flattening end up, "
                 "and can we tell from the dots alone?", fontsize=17, y=0.975)
    fig.text(0.5, 0.905, f"Paper is never stretched, so the right answer is the square. Dots are 0.031 apart.\n"
             f"Gap = closest distance between two parts of the paper that are far apart on it. "
             f"Method shown: walking along nearby dots, with the link check ({METHOD}).",
             ha="center", fontsize=11.5, color="#333")

    for row, (kind, title) in enumerate([("flap", "Case 1: folded flat, the whole flap equally close"),
                                         ("loop", "Case 2: a loop that only swings close in one strip")]):
        name = f"{kind}_gap02"
        names = ff.REGION_NAMES[kind]
        # ---- side view
        ax = fig.add_subplot(grid[row, 0])
        s, x, z = ff.side_view(name)
        lab = ff.regions(name, s)
        for r in range(4):
            m = lab == r
            ax.plot(np.where(m, x, np.nan), np.where(m, z, np.nan), color=COLOURS[r], lw=3.5, label=names[r])
        ax.set_aspect("equal")
        ax.set_title(f"{title}\nside view of the sheet (gap 0.02)", fontsize=12.5, loc="left")
        ax.set_xlabel("across (paper units)")
        ax.set_ylabel("height")
        ax.legend(loc="upper left", fontsize=10, frameon=False, ncol=2)
        ax.set_ylim(-0.03, 0.52 if kind == "loop" else 0.42)
        if kind == "flap":
            ax.annotate("flap lies 0.02 above the base\nall along its length", xy=(1.0, 0.045), xytext=(0.62, 0.2),
                        arrowprops=dict(arrowstyle="->", color="#333"), fontsize=10.5)
        else:
            ax.annotate("only here is it 0.02\nfrom the base", xy=(0.66, 0.045), xytext=(0.12, 0.2),
                        arrowprops=dict(arrowstyle="->", color="#333"), fontsize=10.5)

        # ---- flat result
        ax = fig.add_subplot(grid[row, 1])
        inst = make_instance(SHEETS[name], seed=ff.SEEDS[0])
        uv = inst.uv[inst.order]
        dot_lab = ff.regions(name, uv[:, 0] + 1)
        flat, _ = align_to_square(L[f"{name} / {METHOD}|flat"].astype(float), uv)   # truth used only to line it up
        ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], "--", color="#555", lw=1.2)
        for r in (0, 1, 2, 3):
            m = dot_lab == r
            ax.scatter(flat[m, 0], flat[m, 1], s=5, color=COLOURS[r], alpha=0.65, lw=0)
        reg = ff.json.loads((ff.OUT / "regions.json").read_text())[f"{name} / {METHOD}"]
        far = reg[names[3]]
        mid = reg[names[2]]
        if kind == "flap":
            msg = (f"creased part: flat, {reg[names[0]]['shape_miss_pct']:.1f}% off\n"
                   f"flap (red): right size and shape, but lying\nmirror-image ON the base, hiding it "
                   f"({far['on_top_of_base_pct']:.0f}% of its dots\nwithin half a dot spacing of a base dot)")
        else:
            msg = (f"creased part: flat, {reg[names[0]]['shape_miss_pct']:.1f}% off\n"
                   f"loop (green): squashed flat, its two halves on top of\neach other ({mid['flipped_pct']:.0f}% of it mirror-image); "
                   f"the end (red)\nlies on the base ({far['on_top_of_base_pct']:.0f}%)")
        ax.text(0.0, -1.12, msg, fontsize=10.5, va="top", ha="center")
        ax.set_aspect("equal")
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.75, 1.15)
        ax.axis("off")
        ax.set_title("where flattening ends up\n(dashed = the correct square)", fontsize=12.5, loc="left")

    # ---- the check, one panel for both cases
    ax = fig.add_subplot(grid[:, 2])
    lines = [("flap_gap02", "case 1, gap 0.02", COLOURS[3], "-"), ("flap_gap05", "case 1, gap 0.05", COLOURS[3], "--"),
             ("loop_gap02", "case 2, gap 0.02", COLOURS[2], "-"), ("loop_gap05", "case 2, gap 0.05", COLOURS[2], "--"),
             ("accordion", "control: four sharp creases, no fold (flattens correctly)", "#333", "-"),
             ("wave", "control: wave, no fold (flattens correctly)", "#999", "-")]
    centres = (BINS[1:] + BINS[:-1]) / 2 - 1
    for name, label, colour, style in lines:
        inst = make_instance(SHEETS[name], seed=ff.SEEDS[0])
        u = inst.uv[inst.order][:, 0]
        c = L[f"{name} / {METHOD}|closer"]
        idx = np.digitize(u + 1, BINS) - 1
        ax.plot(centres, [np.median(c[idx == i]) for i in range(len(centres))], style, color=colour, lw=2.4, label=label)
    ax.axhline(2, color="#888", lw=1, ls=":")
    ax.text(-0.98, 2.1, "flag line: put 2x closer than they really are", fontsize=10, color="#555")
    ax.axvspan(-1, ff.CREASE_END - 1, color=COLOURS[0], alpha=0.08)
    ax.text(-0.75, 60, "creased part\n(both cases)", fontsize=10.5, color=COLOURS[0], ha="center")
    ax.set_yscale("log")
    ax.set_ylim(0.8, 120)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("position along the paper, left edge to right edge\n(uses the truth only to line up the x-axis)")
    ax.set_ylabel("how much CLOSER the flat result puts a dot's flat neighbours than they are in 3-D\n"
                  "(worst of its 10 nearest in the flat result; median per strip; 1 = honest)")
    ax.set_title("Can we see it from the dots alone?\nCompare each dot's flat neighbours with their real 3-D distance",
                 fontsize=12.5, loc="left")
    ax.legend(loc="center left", fontsize=10, frameon=False, bbox_to_anchor=(0.0, 0.73))
    fig.savefig(OUT / "fold_flatten.png", dpi=110)
    print("wrote", OUT / "fold_flatten.png")


if __name__ == "__main__":
    main()
