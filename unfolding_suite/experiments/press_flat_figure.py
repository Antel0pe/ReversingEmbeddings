"""Figures for the press and grow-and-join experiments (2026-11-13).

press_flat.png   : side views of one strip of dots while two plates close (perfect strings).
                   Left: a wavy sheet. Right: a loosely rolled sheet.
patch_stitch.png : the S-shaped sheet, grown from 8 random seed points.
                   A: pieces on the original paper with perfect strings; B: the same with
                   strings to the 10 nearest dots; C: the joined result from B, from above.

Run: OMP_NUM_THREADS=4 .venv/bin/python -m unfolding_suite.experiments.press_flat_figure
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from unfolding_suite.diagnose import FAR
from unfolding_suite.experiments.patch_stitch import flatten_piece, patch_stitch
from unfolding_suite.experiments.press_flat import knn, press, true_links
from unfolding_suite.report import print_colours
from unfolding_suite.scoring import align_to_square, signed_area, to_grid_order, triangles
from unfolding_suite.sheets import SHEETS, make_instance

OUT = Path(__file__).resolve().parent.parent / "figures" / "press_flat"
MOMENTS = [(0, "before"), (6, "plates partway"), (12, "plates nearly closed")]


def side_frames(name):
    inst = make_instance(SHEETS[name], 65, 100)
    rec = []
    press(inst.points, true_links(inst), record=rec)
    X0 = inst.points - inst.points.mean(0)
    axes = np.linalg.svd(X0, full_matrices=False)[2]
    strip = np.abs(inst.uv[inst.order, 1]) < 0.02
    frames = [X0] + rec
    u = inst.uv[inst.order, 0][strip]
    out = []
    for k, label in MOMENTS:
        F = (frames[k] - frames[k].mean(0))[strip] @ axes.T
        d = np.linalg.svd(F[:, :2] - F[:, :2].mean(0), full_matrices=False)[2][0]
        x = F[:, :2] @ d
        if np.corrcoef(x, u)[0, 1] < 0:
            x = -x
        out.append((x - x.mean(), F[:, 2] - F[:, 2].mean(), label))
    return u, out


def press_figure():
    cases = [("wave", "A wavy sheet, pressed", "Result: flat and unfolded (passes)", "#1f7a45"),
             ("roll", "A loosely rolled sheet, pressed", "Result: squashed into a flat tube;\n"
              "54% of the paper ends up lying on other paper", "#b3361f")]
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.6), constrained_layout=True)
    for ax, (name, title, verdict, colour) in zip(axs, cases):
        u, views = side_frames(name)
        gap = 0.62
        for row, (x, h, label) in enumerate(views):
            y0 = -row * gap
            order = np.argsort(u)
            ax.plot(x[order], h[order] + y0, color="#bbbbbb", lw=0.8, zorder=1)
            ax.scatter(x, h + y0, c=u, cmap="viridis", vmin=-1, vmax=1, s=14, zorder=2)
            ax.text(-0.98, y0 + 0.22, label, fontsize=11, color="#333333")
            ax.annotate("", xy=(0, y0 - 0.25), xytext=(0, y0 - 0.17),
                        arrowprops=dict(arrowstyle="-|>", color="#999999")) if row < 2 else None
        ax.text(0, -2 * gap - 0.3, verdict, ha="center", va="top", fontsize=11.5, color=colour, weight="bold")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-2 * gap - 0.55, 0.38)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=14)
    fig.suptitle("Pressing between two plates, seen from the side. Each dot is one dot of a single strip across the paper; "
                 "colour = where it sits along that strip\non the original paper (purple at one end, yellow at the other). "
                 "The plates are above and below; they close from top to bottom of each column.\n"
                 "Strings are the paper's true neighbours, so only the pressing itself is being tested.", fontsize=11)
    fig.savefig(OUT / "press_flat.png", dpi=110)


def grow(inst, links, n_seeds, rng):
    n = len(inst.points)
    links = np.unique(np.sort(links, 1), axis=0)
    w = np.linalg.norm(inst.points[links[:, 0]] - inst.points[links[:, 1]], axis=1)
    g = csr_matrix((np.r_[w, w], (np.r_[links[:, 0], links[:, 1]], np.r_[links[:, 1], links[:, 0]])), shape=(n, n))
    seeds = rng.choice(n, n_seeds, replace=False)
    return dijkstra(g, directed=False, indices=seeds).argmin(0), seeds


def stitch_figure():
    inst = make_instance(SHEETS["s_curl"], 65, 100)
    uv = inst.uv[inst.order]
    near = knn(inst.points)
    g = inst.order[near]
    jumps = near[np.linalg.norm(inst.uv[g[:, 0]] - inst.uv[g[:, 1]], axis=1) > FAR]
    fig, axs = plt.subplots(1, 3, figsize=(17, 7.2), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    owner, seeds = grow(inst, near, 8, np.random.default_rng(100))
    axs[0].scatter(uv[:, 0], uv[:, 1], c=[cmap(o) for o in owner], s=7)
    axs[0].scatter(uv[seeds, 0], uv[seeds, 1], marker="*", s=260, c="black", edgecolor="white", zorder=3)
    axs[0].set_title("A. Grow: 8 pieces from 8 random seed points\n(each colour = one piece)", fontsize=13)
    # B: is each piece's OWN flat layout folded? (flipped triangles inside the piece)
    n = len(uv)
    lk = np.unique(np.sort(near, 1), axis=0)
    w = np.linalg.norm(inst.points[lk[:, 0]] - inst.points[lk[:, 1]], axis=1)
    graph = csr_matrix((np.r_[w, w], (np.r_[lk[:, 0], lk[:, 1]], np.r_[lk[:, 1], lk[:, 0]])), shape=(n, n))
    hops = csr_matrix((np.ones(2 * len(lk)), (np.r_[lk[:, 0], lk[:, 1]], np.r_[lk[:, 1], lk[:, 0]])), shape=(n, n))
    where = np.empty(n, int)
    where[inst.order] = np.arange(n)
    T_all = where[triangles(65, 1)]
    for s_ in range(8):
        core = owner == s_
        m = core.copy()
        for _ in range(2):
            m = m | (hops @ m.astype(float) > 0)
        idx, xy = flatten_piece(graph, np.flatnonzero(m))
        pos = -np.ones(n, int)
        pos[idx] = np.arange(len(idx))
        TT = T_all[(pos[T_all] >= 0).all(1)]
        ar = signed_area(xy, pos[TT])
        folded = (np.sign(ar) != np.sign(np.median(ar))).mean()
        colour = "#d03b3b" if folded > 0 else "#4caf6e"
        axs[1].scatter(uv[core, 0], uv[core, 1], c=colour, s=7, alpha=0.55)
        cx, cy = np.median(uv[core], 0)
        axs[1].text(cx, cy, f"{100 * folded:.1f}%", ha="center", va="center", fontsize=12, weight="bold",
                    bbox=dict(boxstyle="round", fc="white", ec=colour))
    for a_, b_ in jumps:
        axs[1].plot(uv[[a_, b_], 0], uv[[a_, b_], 1], color="black", lw=1.6, zorder=3)
        axs[1].scatter(uv[[a_, b_], 0], uv[[a_, b_], 1], color="black", s=22, zorder=4)
    axs[1].set_title("B. Flatten each piece on its own:\nhow much of each piece comes out folded?", fontsize=13)
    axs[1].text(0, -1.12, "black = the 12 wrong strings, along the paper's top and bottom edges\n"
                "(each joins dots far apart on the paper). Green = piece came out right;\n"
                "red = piece came out partly folded. Every red piece contains a wrong string.",
                ha="center", va="top", fontsize=10.5)
    for ax in axs[:2]:
        ax.set_xlabel("drawn on the original flat paper", fontsize=10.5)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.45, 1.08)
    flat, _ = patch_stitch(inst.points, near, 8, rng=np.random.default_rng(100))
    F = align_to_square(to_grid_order(flat, inst), inst.uv)[0]
    T = triangles(65, 1)
    area = signed_area(F, T)
    flipped = np.sign(area) != np.sign(np.median(area))
    col = print_colours(inst.uv[T].mean(1))
    col[flipped] = (0.82, 0.23, 0.23)
    axs[2].add_collection(PolyCollection(F[T], facecolors=col, edgecolors="none"))
    axs[2].plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color="#777777", lw=1, ls="--")
    axs[2].set_xlim(-1.6, 1.6); axs[2].set_ylim(-1.6, 1.6)
    axs[2].set_aspect("equal"); axs[2].set_xticks([]); axs[2].set_yticks([])
    axs[2].set_title("C. Join the pieces in pairs, then pairs of pairs...\nresult seen from above", fontsize=13)
    axs[2].set_xlabel("blue-to-orange print = paper position; red = mirror-image (folded);\n"
                      "dashed = where the square should be. 63% ends up lying on other paper", fontsize=10.5)
    fig.savefig(OUT / "patch_stitch.png", dpi=110)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    stitch_figure()
    press_figure()
