"""Figure for fold_topology.py: where the loop detector finds folds, and why it misses the crumples.

Run after `fold_topology truths`, `blobs` and `loops`:
  .venv/bin/python -m unfolding_suite.experiments.fold_topology_figure
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from unfolding_suite.experiments.fold_topology import OUT, SEED, load_truth, paper
from unfolding_suite.sheets import SHEETS, make_instance

FIG = Path(__file__).resolve().parent.parent / "figures" / "fold_topology"
TRUTH, HIT, MISS, PAPER = "#e8a33d", "#1f4e9c", "#c0392b", "#d9d9d9"


def paper_map(ax, name, loops, title):
    inst = make_instance(SHEETS[name], 65, SEED)
    uv = paper(inst)
    links, walk, _ = load_truth(name)
    shortcut_dot = np.zeros(len(uv), bool)
    shortcut_dot[links[walk > 2, 0]] = True
    ax.scatter(*uv[~shortcut_dot].T, s=2, c=PAPER, lw=0)
    ax.scatter(*uv[shortcut_dot].T, s=6, c=TRUTH, lw=0)
    bars = loops[name]["knn"]["bars"]
    real = [b for b in bars if b["walk"] > 2]
    fake = [b for b in bars if b["walk"] <= 2]
    for group, colour, marker in ((real, HIT, "o"), (fake, MISS, "x")):
        if group:
            ends = uv[np.array([b["ends"] for b in group]).ravel()]
            if marker == "o":
                ax.scatter(*ends.T, s=34, marker="o", facecolors="none", edgecolors=colour, lw=1.4)
            else:
                ax.scatter(*ends.T, s=34, marker="x", c=colour, lw=1.4)
    ax.set_title(title, fontsize=12, loc="left")
    ax.text(0.0, -0.13, f"{shortcut_dot.sum()} dots have a real shortcut among their 10 nearest (orange).\n"
            f"{len(bars)} loops never fill in: {len(real)} start on a real shortcut (blue circles),\n"
            f"{len(fake)} start on an ordinary link (red crosses).",
            transform=ax.transAxes, fontsize=9.5, va="top")
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def score_hist(ax, blobs):
    bins = np.linspace(1, 3.5, 51)
    groups = {"one layer only, bent sheets": [], "one layer only, crumples": [],
              "real shortcut, bent sheets": [], "real shortcut, crumples": []}
    for name, r in blobs.items():
        if "crumple" in name:
            kind = "crumples"
        elif SHEETS[name].family == "bent" and name != "tight_roll":
            kind = "bent sheets"
        else:
            continue
        score = np.array(r["score"])
        groups[f"one layer only, {kind}"].append(score[~np.array(r["in_reach_mask"])])
        groups[f"real shortcut, {kind}"].append(score[np.array(r["touched_mask"])])
    style = {"one layer only, bent sheets": ("#9e9e9e", "-"), "one layer only, crumples": ("#9e9e9e", "--"),
             "real shortcut, bent sheets": (HIT, "-"), "real shortcut, crumples": (TRUTH, "-")}
    for label, arrays in groups.items():
        v = np.concatenate(arrays)
        h, _ = np.histogram(np.clip(v, bins[0], bins[-1]), bins)
        colour, ls = style[label]
        ax.step(bins[:-1], h / h.sum(), where="post", color=colour, ls=ls, lw=2, label=f"{label}\n({len(v):,} dots)")
    ax.set_xlabel("gap score: how far apart the two biggest clumps of nearby dots are,\n"
                  "in units of the usual dot spacing (1 = one smooth clump)", fontsize=10)
    ax.set_ylabel("share of dots in each group", fontsize=10)
    ax.set_title("C. Gap test on each dot's 40 nearest dots", fontsize=12, loc="left")
    ax.legend(fontsize=9, frameon=False, loc="upper right", handlelength=1.5)
    ax.text(0.0, -0.2, "Bent sheets: shortcut dots score 1.9-2.6; one-layer dots never\n"
            "above 1.38. A clean split.\n"
            "Crumples: shortcut dots score 1.2-2.6, inside the range of one-layer\n"
            "dots where the paper is stretched (up to 3.9). No clean split.\n"
            "(tight_roll left out: its layers are closer than the dot spacing.)",
            transform=ax.transAxes, fontsize=9.5, va="top")


def main():
    loops = json.loads((OUT / "loops.json").read_text())
    blobs = json.loads((OUT / "blobs.json").read_text())
    fig = plt.figure(figsize=(17, 7.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.18, left=0.02, right=0.98, top=0.83, bottom=0.27)
    paper_map(fig.add_subplot(gs[0]), "h_roll5", loops, "A. Rolled sheet (h_roll5), on the flat paper")
    paper_map(fig.add_subplot(gs[1]), "h_crumple29", loops, "B. Crumpled sheet (h_crumple29), on the flat paper")
    score_hist(fig.add_subplot(gs[2]), blobs)
    fig.suptitle("Can topology find where a dot's nearest dots jump to another layer?", fontsize=15, x=0.02, ha="left")
    fig.text(0.02, 0.905, "A and B: every dot drawn at its true place on the unfolded paper (the method never sees this). "
             "A loop that never fills in = a ring of links with no paper inside it.\nIts first link marks where the "
             "fold is. 'Real shortcut' = walking along the actual 3-D sheet is more than twice the straight-line length.",
             fontsize=10, ha="left", va="top")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fold_topology.png", dpi=110)
    print("wrote", FIG / "fold_topology.png")


if __name__ == "__main__":
    main()
