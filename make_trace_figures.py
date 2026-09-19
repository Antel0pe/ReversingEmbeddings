"""Figures for trace_structure.py: recovering knobs from a complete point cloud.

traced_axes.png  -- move and size rungs: the knobs found from coordinates alone
traced_tilt.png  -- why the same method breaks on tilt

The tilt part rebuilds a 214,860-step graph; expect ~5 minutes.
"""

import itertools
from collections import defaultdict, Counter
import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import networkx as nx

import equation_ladder as L
import trace_structure as T

OUT = "figures"
GREEN, RED, GREY, BLUE = "#1a7f37", "#c62828", "#444", "#1565c0"


def pix(ax, x, border=None, lw=2.0):
    x = x.reshape(10, 10)
    ax.set_facecolor("white")
    ax.imshow(np.ma.masked_where(x == 0, x), cmap="gray_r", vmin=0, vmax=1.6,
              interpolation="nearest", zorder=2)
    ax.set_xticks(np.arange(-.5, 10, 1), minor=True); ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
    ax.grid(which="minor", color="#d0d0d0", lw=.45)
    ax.set_xticks([]); ax.set_yticks([]); ax.tick_params(which="minor", length=0)
    ax.set_xlim(-.5, 9.5); ax.set_ylim(9.5, -.5)
    if border:
        for sp in ax.spines.values(): sp.set_edgecolor(border); sp.set_linewidth(lw)


def pca95(V):
    X = V - V.mean(0); s = np.linalg.svd(X.astype(float), compute_uv=False)
    v = s ** 2 / (s ** 2).sum(); return int(np.searchsorted(np.cumsum(v), .95) + 1)


def traced_line(C, axis):
    """Points that differ from a base point ONLY along `axis`, in order.
    Longest such line, ties broken towards the middle of the cloud -- the edges
    of the range are where thresholds can be assigned to either of two axes."""
    centre = C.mean(0); best, key = None, None
    for base in range(len(C)):
        others = [a for a in range(C.shape[1]) if a != axis]
        same = np.where((C[:, others] == C[base, others]).all(1))[0]
        k = (len(same), -np.abs(C[same].mean(0) - centre)[others].sum())
        if key is None or k > key: best, key = same, k
    return best[np.argsort(C[best, axis])]


NAMES = {"move": ["left edge", "top edge"],
         "size": None}


def describe(V, idx):
    x = V[idx].reshape(10, 10); rows = np.where(x.any(1))[0]; cols = np.where(x.any(0))[0]
    return rows[0], rows[-1], cols[0], cols[-1]


def fig_axes():
    data = {}
    for rung, cap in [("move", 34), ("size", 24)]:
        V = np.array([np.frombuffer(b, np.int8) for b in sorted(L.valid_set(rung))])
        R = T.lattice_axes(V, T.elementary_graph(V, cap))
        data[rung] = (V, R)
    rows = []
    for rung in ["move", "size"]:
        V, R = data[rung]
        for i in range(len(R["axes"])):
            line = traced_line(R["coords"], i)
            e = np.array([describe(V, j) for j in range(len(V))], float)
            c = R["coords"][:, i]
            cor = [abs(np.corrcoef(c, e[:, q])[0, 1]) if e[:, q].std() > 0 else 0 for q in range(4)]
            q = int(np.argmax(cor))
            moving = f"{['top', 'bottom', 'left', 'right'][q]} edge, |corr| {cor[q]:.3f}"
            rows.append((rung, i, line, moving, V, R))
    NC = max(len(r[2]) for r in rows)
    FW = 1.25 * NC + 4.6; FH = 1.62 * len(rows) + 4.3
    fig = plt.figure(figsize=(FW, FH))
    top0 = 2.85
    for k, (rung, i, line, moving, V, R) in enumerate(rows):
        extra = 0.75 if rung == "size" else 0.0
        yt = top0 + k * 1.62 + extra
        gs = fig.add_gridspec(1, NC, left=3.9 / FW, right=1 - 0.25 / FW, top=1 - yt / FH,
                              bottom=1 - (yt + 1.12) / FH, wspace=0.10)
        for c, j in enumerate(line):
            ax = fig.add_subplot(gs[0, c]); pix(ax, V[j], BLUE, 1.4)
            ax.text(0.5, -0.07, str(int(R["coords"][j, i])), transform=ax.transAxes, ha="center",
                    va="top", fontsize=9.5, color=BLUE, weight="bold")
        fig.text(3.75 / FW, 1 - (yt + 0.56) / FH, f"axis {i + 1}: {len(R['axes'][i]['cuts'])} thresholds\n"
                 f"(tracks the {moving})", ha="right", va="center", fontsize=11, linespacing=1.45)
    for rung, first in [("move", 0), ("size", 2)]:
        V, R = data[rung]
        yt = top0 + first * 1.62 + (0.75 if rung == "size" else 0) - 0.30
        fig.text(0.25 / FW, 1 - yt / FH,
                 f"{rung.upper()} rung: {len(V)} images.  Found {len(R['axes'])} axes from {R['cuts']} "
                 f"yes/no thresholds, reproducing every step distance exactly.\n"
                 f"PCA would need {pca95(V)} straight directions to cover 95% of the same cloud."
                 + ("  Left/right lines are short: with one edge held, width 1-3 lets the other move "
                    "only 3 places." if rung == "size" else ""),
                 fontsize=11.2, weight="bold", color=GREEN, va="bottom", linespacing=1.5)
    fig.text(0.5, 1 - 0.42 / FH, "TRACING THE KNOBS FROM THE DATA ALONE:  no generator, no pixel grid, "
             "just the complete list of valid images", ha="center", fontsize=16, weight="bold")
    fig.text(0.5, 0.955 - 0.55 / FH, "Each row is one traced line: images that differ from each other "
             "along ONE recovered axis and agree on all the others. Blue number = the recovered "
             "coordinate on that axis.", ha="center", fontsize=11.5)
    fig.text(0.5, 0.955 - 0.92 / FH, "Recovered by: join images that share ink with no third image "
             "closer to both -> split the steps into yes/no thresholds -> chain nested thresholds "
             "into axes (Eppstein's lattice dimension).", ha="center", fontsize=11, style="italic",
             color=GREY)
    fig.savefig(f"{OUT}/traced_axes.png", dpi=110, facecolor="white"); plt.close(fig)
    print("wrote traced_axes.png")


def tilt_stats():
    V = np.array([np.frombuffer(b, np.int8) for b in sorted(L.valid_set("tilt"))])
    edges = T.elementary_graph(V, 12)
    adj = defaultdict(set)
    for a, b in edges: adj[a].add(b); adj[b].add(a)
    ntri = sum(len(adj[a] & adj[b]) for a, b in edges) // 3
    index = {v.tobytes(): i for i, v in enumerate(V)}
    tri = [L.draw(2, 5, 1, [5] * 5), L.draw(2, 5, 1, [5, 5, 5, 5, 4]), L.draw(2, 5, 2, [4] * 5)]
    ids = [index[t.ravel().astype(np.int8).tobytes()] for t in tri]
    assert all(ids[q] in adj[ids[p]] for p, q in [(0, 1), (1, 2), (0, 2)]), "not a triangle"
    rng = np.random.default_rng(0); dims = []
    for x in rng.choice(len(V), 1500, replace=False):
        g = nx.Graph(); g.add_nodes_from(adj[x])
        for y, z in itertools.combinations(adj[x], 2):
            if z in adj[y]: continue
            if any(w != x and w not in adj[x] for w in adj[y] & adj[z]): g.add_edge(y, z)
        dims.append(max(len(c) for c in nx.find_cliques(g)))
    return dict(n=len(V), steps=len(edges), ntri=ntri, tri=[V[i] for i in ids], dims=np.array(dims))


def fig_tilt(st):
    dims, tri = st["dims"], st["tri"]
    FW, FH = 17.0, 10.4
    fig = plt.figure(figsize=(FW, FH))
    fy = lambda t: 1 - t / FH
    fig.text(0.5, fy(0.42), "WHY TILT BREAKS THE TRACING:  on a pixel grid, leaning and widening "
             "share pixels", ha="center", fontsize=16, weight="bold")
    fig.text(0.5, fy(0.88), f"TILT rung: {st['n']:,} images, {st['steps']:,} steps.  A grid of integer "
             f"knobs never has three points that are all one step apart.  Tilt has {st['ntri']:,} "
             "such triangles.", ha="center", fontsize=11.8)
    fig.text(0.255, fy(1.55), "ONE OF THEM: three valid 1s, each a single step from the other two",
             ha="center", fontsize=12.5, weight="bold")
    S = 2.3 / FW                                   # image size, figure fraction (width)
    SH = 2.3 / FH
    centres = [(0.105, fy(3.45)), (0.405, fy(3.45)), (0.255, fy(7.55))]
    labels = ["upright, 1 wide", "bottom pixel\nleaned one column", "upright, 2 wide"]
    for (cx, cy), x, lab in zip(centres, tri, labels):
        ax = fig.add_axes([cx - S / 2, cy - SH / 2, S, SH]); pix(ax, x, RED, 2.4)
        ax.set_title(lab, fontsize=11.5, pad=6, linespacing=1.3)
    def link(p, q, text, off):
        (x1, y1), (x2, y2) = centres[p], centres[q]
        vx, vy = (x2 - x1) * FW, (y2 - y1) * FH; Ln = np.hypot(vx, vy)
        ux, uy = vx / Ln, vy / Ln; r = 1.55                     # inches from centre to line end
        a = (x1 + ux * r / FW, y1 + uy * r / FH); b = (x2 - ux * r / FW, y2 - uy * r / FH)
        fig.add_artist(plt.Line2D([a[0], b[0]], [a[1], b[1]], color=RED, lw=2.6))
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        fig.text(mx + off[0] / FW, my + off[1] / FH, text, ha="center", va="center", fontsize=10.5,
                 color=RED, weight="bold", linespacing=1.3)
    d = lambda p, q: int((tri[p] != tri[q]).sum())
    link(0, 1, f"1 step: {d(0, 1)} pixels\n(lean)", (0, 0.42))
    link(0, 2, f"1 step:\n{d(0, 2)} pixels\n(widen)", (-0.75, 0))
    link(1, 2, f"1 step:\n{d(1, 2)} pixels", (0.75, 0))
    fig.text(0.255, fy(9.75), "Leaning the bottom pixel is part of the way to widening, so all three "
             "are single steps.\nThe two moves are not independent directions -- on the grid they "
             "overlap.", ha="center", fontsize=11, style="italic", color=GREY, linespacing=1.5)

    ax = fig.add_axes([0.58, fy(8.55), 0.39, (8.55 - 1.95) / FH])
    cnt = Counter(dims.tolist()); ks = sorted(cnt)
    ax.bar(ks, [cnt[k] for k in ks], color="#ef9a9a", edgecolor=RED, zorder=3,
           label="TILT: largest commuting set, per point")
    ax.axvline(2, color=GREEN, lw=2.4, ls="--", zorder=4, label="MOVE rung: 2 at every point")
    ax.axvline(4, color=BLUE, lw=2.4, ls="--", zorder=4, label="SIZE rung: 4 at interior points")
    ax.axvline(6, color="#222", lw=2.0, ls=":", zorder=4, label="knobs in the tilt generator: 6")
    ax.legend(loc="upper right", fontsize=10.2, framealpha=.95)
    ax.set_xlabel("largest set of moves at a point that all commute", fontsize=11.5)
    ax.set_ylabel("points (1,500 sampled)", fontsize=11.5); ax.grid(alpha=.3, axis="y")
    ax.set_title(f"LOCAL DIMENSION: no single answer (median {np.median(dims):.0f}, "
                 f"range {dims.min()} to {dims.max()})", fontsize=12.5, pad=10)
    fig.text(0.775, fy(9.4), "Tilt on a grid is not two smooth knobs (slope, offset). It shatters\n"
             "into many small shears of single rows, most of which commute.", ha="center",
             va="top", fontsize=11, style="italic", color=GREY, linespacing=1.5)
    fig.savefig(f"{OUT}/traced_tilt.png", dpi=110, facecolor="white"); plt.close(fig)
    print("wrote traced_tilt.png")


if __name__ == "__main__":
    fig_axes()
    fig_tilt(tilt_stats())
