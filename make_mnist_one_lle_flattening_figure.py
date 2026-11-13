"""Figure for analyze_mnist_one_lle_flattening.py (run that first).

    .venv/bin/python make_mnist_one_lle_flattening_figure.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

import mnist

D = Path(__file__).resolve().parent / "figures/mnist_one_lle_flattening"
R = json.loads((D / "results.json").read_text())["mnist"]


def ones():
    X, y = mnist.load("train")
    return X[y == 1].astype(np.float64) / 255.0


def slant(X):
    yy, xx = np.mgrid[0:28, 0:28] + 0.5
    ink = X.sum((1, 2))
    mx, my = (X * xx).sum((1, 2)) / ink, (X * yy).sum((1, 2)) / ink
    dy, dx = yy - my[:, None, None], xx - mx[:, None, None]
    return -(X * dy * dx).sum((1, 2)) / (X * dy ** 2).sum((1, 2))     # sideways per downward; + = top to the right


def nearest(P, i, k):
    d = ((P - P[i]) ** 2).sum(1)
    d[i] = np.inf
    return np.argsort(d)[:k]


def main():
    X = ones()
    flat = X.reshape(len(X), -1)
    s = np.degrees(np.arctan(slant(X)))
    F2 = np.load(D / "mnist__lle_k20_d2.npy")
    F15 = np.load(D / "mnist__lle_k20_d15.npy")

    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], width_ratios=[1.25, 1], hspace=0.38, wspace=0.62)

    # A: the 2-number layout
    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(F2[:, 0], F2[:, 1], c=s, cmap="coolwarm", s=4, vmin=-35, vmax=35, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("slant of the 1 (degrees; + = top leans right)")
    lo, hi = np.percentile(F2[:, 0], [1, 99])
    for t in np.linspace(lo, hi, 7):
        i = np.argmin(np.abs(F2[:, 0] - t) + 0.0 * F2[:, 1])
        ab = AnnotationBbox(OffsetImage(X[i], cmap="gray_r", zoom=1.3), (F2[i, 0], F2[i, 1]),
                            frameon=True, pad=0.1, bboxprops=dict(ec="0.4"))
        ax.add_artist(ab)
    r2 = R["layouts"]["lle k=20 d=2"]["copy_r2"][0]
    ax.set_title("A.  Each dot is one MNIST 1, placed where the method put it\n(2 numbers, 20-dot pieces). It comes out as one bent line ordered\n"
                 f"by slant: the 2nd number is a bent copy of the 1st (R² {r2:.2f})", fontsize=10.5, loc="left")
    ax.set_xlabel("1st number")
    ax.set_ylabel("2nd number")

    # B: how close each flattening gets, at 15 numbers
    ax = fig.add_subplot(gs[0, 1])
    L = R["layouts"]
    rows = [("10 most similar images\n(best possible)", R["ceiling_similar"], "0.55", None),
            ("best flat plane", L["plane d=15"]["rebuild"], "#4c72b0", L["plane d=15"]["new"]),
            ("walking along links\n(suite's walking method)", L["walk k=10 d=15"]["rebuild"], "#55a868",
             L["walk k=10 d=15"]["new"]),
            ("locally linear pieces,\n20 dots each", L["lle k=20 d=15"]["rebuild"], "#c44e52", L["lle k=20 d=15"]["new"]),
            ("locally linear pieces,\n40 dots each", L["lle k=40 d=15"]["rebuild"], "#dd8452", L["lle k=40 d=15"]["new"]),
            ("10 random images\n(no information)", R["ceiling_random"], "0.8", None)]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows])
    for yi, (lab, v, _, new) in zip(y, rows):
        extra = f"   ({new} of 15 numbers are new directions)" if new is not None else ""
        ax.text(v + 1, yi, f"{v:.1f}%{extra}", va="center", fontsize=9)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, 130)
    ax.set_xlabel("miss when each image is rebuilt from the 10 images\nplaced nearest it (% of the image)")
    ax.set_title("B.  How close does each flattening get?\n(15 numbers each; lower is better;\nnothing can beat the top grey bar)", fontsize=10.5, loc="left")

    # C: one typical 1 and its neighbours
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    sq = (flat ** 2).sum(1)
    Dp = sq[:, None] + sq[None] - 2 * flat @ flat.T
    np.fill_diagonal(Dp, np.inf)
    cut = np.partition(Dp, 99, 1)[:, 99]                 # distance of each image's 100th most similar
    Df = ((F15[:, None] - F15[None]) ** 2).sum(2)
    np.fill_diagonal(Df, np.inf)
    nb = np.argpartition(Df, 8, 1)[:, :8]
    strangers = (Dp[np.arange(len(X))[:, None], nb] > cut[:, None]).sum(1)
    typical = np.where(strangers == int(np.median(strangers)))[0]
    pick = typical[np.argmin(np.abs(s[typical] - np.median(s)))]   # a typical 1: median stranger count, median slant
    del Dp, Df
    dpix = sq + sq[pick] - 2 * flat @ flat[pick]
    dpix[pick] = np.inf
    rank = np.empty(len(X), int)
    rank[np.argsort(dpix)] = np.arange(1, len(X) + 1)
    sim = np.argsort(dpix)[:8]
    fl = nearest(F15, pick, 8)
    ax.set_title("C.  One MNIST 1 (left), its 8 most similar images (top row), and the 8 images the method put "
                 "nearest it with 15 numbers (bottom row)\n"
                 "Number under each = its rank among all 6,742 1s by similarity to the left image (1 = most similar). "
                 f"Across all 1s, {100 * L['lle k=20 d=15']['strangers']:.0f}% of layout neighbours rank worse than 100",
                 fontsize=10.5, loc="left")
    W = 1 / 10.5
    def put(img, x, y, label, color="k"):
        a = ax.inset_axes([x, y, W * 0.9, 0.36])
        a.imshow(img, cmap="gray_r", vmin=0, vmax=1)
        a.set_xticks([]); a.set_yticks([])
        a.set_xlabel(label, fontsize=9, color=color)
    put(X[pick], 0.0, 0.27, "the 1")
    for j, i in enumerate(sim):
        put(X[i], (j + 1.6) * W, 0.55, f"rank {rank[i]}")
    for j, i in enumerate(fl):
        put(X[i], (j + 1.6) * W, 0.02, f"rank {rank[i]}", "#c44e52" if rank[i] > 100 else "k")
    ax.text(1.55 * W, 0.97, "most similar images", fontsize=9, transform=ax.transAxes)
    ax.text(1.55 * W, 0.42, "nearest in the flat layout (red = not in its 100 most similar)", fontsize=9,
            transform=ax.transAxes)

    fig.suptitle("Flattening the 6,742 MNIST training 1s by treating each small piece as flat", fontsize=13, y=0.995)
    fig.savefig(D / "lle_flattening.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
