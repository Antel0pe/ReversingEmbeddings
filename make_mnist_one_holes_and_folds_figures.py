"""Figures for analyze_mnist_one_holes_and_folds.py (run that first).

    .venv/bin/python make_mnist_one_holes_and_folds_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mnist

OUT = Path(__file__).resolve().parent / "figures/mnist_one_holes_and_folds"
R = json.loads((OUT / "results.json").read_text())
T = np.load(OUT / "toys.npz")
S = np.load(OUT / "shell_scores.npz")
X, y = mnist.load("train")
I = X[y == 1].astype(np.float64) / 255.0
GREY, RED, BLUE, ORANGE = "#b8b8b8", "#d1495b", "#2e6fba", "#e08a1e"


def toy_figure():
    import check_shell_test_touch as C
    from scipy.spatial.distance import pdist, squareform
    import analyze_mnist_one_holes_and_folds as A
    thr = float(T["flat_thr"])
    P2, touch2 = C.curl(0.25, 160)
    sc2 = A.shell_split(squareform(pdist(P2)), np.arange(len(P2)))[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    cases = ((axes[0], T["curl"], T["curl_score"],
              "A. Curls over and RESTS on itself\n0% of dots near the touch flagged"),
             (axes[1], P2, sc2,
              "B. Same curl, stopped 3 dot-spacings short of touching\n"
              f"{100 * (sc2[touch2 < 0.3] > thr).mean():.0f}% of dots near the gap flagged"))
    for ax, P, sc, title in cases:
        flag = sc > thr
        ax.scatter(P[~flag, 0], P[~flag, 2], s=4, c=GREY, lw=0, label="shell is one piece")
        ax.scatter(P[flag, 0], P[flag, 2], s=10, c=RED, lw=0,
                   label="shell splits in two (flagged)")
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-0.45, 1.15)
        ax.set_xlabel("left - right")
        ax.set_ylabel("height")
    axes[0].annotate("touch", xy=(0, 0), xytext=(0.5, -0.38), fontsize=10,
                     arrowprops=dict(arrowstyle="->"))
    axes[1].annotate("gap", xy=(0, -0.12), xytext=(0.5, -0.38), fontsize=10,
                     arrowprops=dict(arrowstyle="->"))
    axes[0].legend(loc="upper left", fontsize=9, markerscale=2, frameon=False)
    fig.suptitle("Side view of a toy sheet (the paper runs straight back into the page). "
                 "Both also show one strong loop (A: 8.5x, B: 10.4x; chance loops at most 2.8x).",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "toy_check.png", dpi=130)
    plt.close(fig)


def loop_figure():
    hd = R["high_dim"]
    names = [("mnist", "MNIST 1s"), ("mnist_twin", "cloud with MNIST's spread\n(no holes by construction)"),
             ("generated", "generated 1s\n(no holes by construction)"),
             ("generated_twin", "cloud with generated\n1s' spread (no holes)")]
    fig = plt.figure(figsize=(13, 6.4))
    ax = fig.add_axes([0.09, 0.47, 0.86, 0.45])
    for k, (key, lab) in enumerate(names):
        s = hd[key]["loops"]["top_strengths"]
        ax.scatter([k] * len(s), s, s=60, c=[RED if key == "mnist" else BLUE], zorder=3)
        ax.text(k + 0.08, s[0], f"{s[0]:.2f}x", va="center", fontsize=10)
    toy = R["toys"]["curl"]["loops"]["top_strengths"][0]
    ax.axhline(toy, color=ORANGE, ls="--")
    ax.text(-0.3, toy - 0.25, f"the real loop in the toy curl: {toy:.1f}x", color=ORANGE,
            va="top", fontsize=10)
    ax.set_xticks(range(4), [lab for _, lab in names], fontsize=9)
    ax.set_xlim(-0.4, 4.1)
    ax.set_ylim(0.8, 9.3)
    ax.set_ylabel("loop strength\n(scale it fills in / scale it appears)")
    ax.set_title("Five strongest loops in each cloud (all 6,742 points). A real hole would "
                 "stand far above the clouds with no holes.", fontsize=11)
    st = hd["mnist"]["loops"]["strongest"]
    path = st["path"]
    pick = np.unique(np.linspace(0, len(path) - 1, min(len(path), 14)).round().astype(int))
    imgs = [path[p] for p in pick]
    n = len(imgs)
    w = 0.88 / n
    for k, idx in enumerate(imgs):
        a = fig.add_axes([0.07 + k * w, 0.03, w * 0.92, 0.2])
        a.imshow(I[idx], cmap="gray_r", vmin=0, vmax=1)
        a.set_xticks([]), a.set_yticks([])
        if k in (0, n - 1):
            for sp in a.spines.values():
                sp.set_color(RED), sp.set_linewidth(2.5)
    fig.text(0.07, 0.27, f"MNIST's strongest loop, walked the long way round "
             f"({len(path)} images, every step shorter than the shortcut; {n} shown).\n"
             f"Red frames: the two images whose direct link closed the loop.", fontsize=10)
    fig.savefig(OUT / "loops.png", dpi=130)
    plt.close(fig)


def shell_figure():
    fig = plt.figure(figsize=(13, 8.2))
    ax = fig.add_axes([0.08, 0.6, 0.86, 0.32])
    bins = np.linspace(1, 2.0, 60)
    ax.hist(S["mnist_twin"], bins, color=BLUE, alpha=0.55,
            label="cloud with MNIST's spread (no folds)")
    ax.hist(S["generated"], bins, color=ORANGE, alpha=0.55, label="generated 1s (no folds)")
    ax.hist(S["mnist"], bins, color=RED, alpha=0.55, label="MNIST 1s")
    for key, col in (("mnist_twin", BLUE), ("generated", ORANGE), ("mnist", RED)):
        ax.axvline(np.median(S[key]), color=col, lw=2)
    ax.set_xlabel("how cleanly each centre's shell splits in two  (biggest gap / typical gap; "
                  "1 = no split; thick lines = middle value; a few above 2 not shown)")
    ax.set_ylabel("centre images (of 1,500)")
    ax.legend(fontsize=10, frameon=False)
    ax.set_title("Generated 1s have no folds, yet their shells split MORE than MNIST's. "
                 "So a split here is not a fold.", fontsize=11)
    ex = R["mnist_shell_examples"][:3]
    for r_, e in enumerate(ex):
        row = [("centre", e["centre"])] + [("A", i) for i in e["piece_a"]] + \
              [("B", i) for i in e["piece_b"]]
        for k, (lab, idx) in enumerate(row):
            gap = 0.02 if k >= 1 else 0
            gap += 0.02 if k >= 5 else 0
            a = fig.add_axes([0.07 + k * 0.095 + gap, 0.36 - r_ * 0.15, 0.085, 0.13])
            a.imshow(I[idx], cmap="gray_r", vmin=0, vmax=1)
            a.set_xticks([]), a.set_yticks([])
            if r_ == 0:
                a.set_title({"centre": "centre", "A": "piece A", "B": "piece B"}[lab], fontsize=9)
        fig.text(0.02, 0.42 - r_ * 0.15, f"{e['score']:.1f}x", fontsize=9)
    fig.text(0.07, 0.52, "The three most cleanly split MNIST centres, with the four images "
             "of each piece nearest the centre:", fontsize=10)
    fig.savefig(OUT / "shells.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    toy_figure()
    loop_figure()
    shell_figure()
    print("figures in", OUT)
