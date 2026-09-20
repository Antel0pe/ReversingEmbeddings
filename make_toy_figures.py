"""figures/toy_shapes.png -- a playground of shapes with known answers, measured blind.

Each shape is built in 3-D (left column, what it really is) and then rotated into
50-D before any measurement, so the tools get no help from the picture. Takes ~3 min.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import toyshapes as T

OUT = "figures"
GREEN, RED, GREY, BLUE, ORANGE = "#1a7f37", "#c62828", "#444", "#1565c0", "#ef6c00"
KS = [20, 40, 80, 160, 320]
RADII = [3, 4, 5, 6, 8]
TRUE_DIM = {k: 2 for k in T.SHAPES}
TRUE_LOOPS = {"swiss roll": 0, "sphere": 0, "torus": 2, "drain": 1, "cone": 0, "crossing sheets": 0}
VERDICT = {
    "swiss roll": "curled, so straight axes need 3.\nUp close it is flat: 2.",
    "sphere": "closed surface, no loops.\nThe void inside needs H2 to see.",
    "torus": "both loops appear together, in a\nwindow of scales. Red here is curvature.",
    "drain": "one loop that never closes,\nand the neighbourhood narrows.",
    "cone": "2 everywhere except the tip,\nwhere the count jumps.",
    "crossing sheets": "2 on each sheet, 3 along the join --\nthe same defect as MNIST tilt.",
}


def main():
    rng = np.random.default_rng(0)
    rows = []
    for name, fn in T.SHAPES.items():
        X3 = fn(800, rng); D = T.embed(X3, 50, rng, noise=0.015)
        s = T.nn_scale(D)
        ld = T.local_dim(D, np.arange(len(D)), k=20)   # small enough that curvature does not fire
        dims = [np.median(T.local_dim(D, rng.choice(len(D), 50, replace=False), k=k)) for k in KS]
        loops = [T.betti01(D, k * s)[1] for k in RADII]
        Z = D - D.mean(0); lam = np.linalg.eigvalsh(Z.T @ Z)[::-1].clip(0)
        pca = int(np.searchsorted(np.cumsum(lam) / lam.sum(), .95) + 1)
        rows.append((name, X3, ld, dims, loops, pca))
        print(f"  {name:16s} local dim {dims}  loops {loops}  PCA {pca}", flush=True)

    n = len(rows); RH = 2.45
    FW, FH = 17.0, RH * n + 2.9
    fig = plt.figure(figsize=(FW, FH))
    fig.text(0.5, 1 - 0.4 / FH, "A PLAYGROUND OF SHAPES WITH KNOWN ANSWERS, MEASURED BLIND IN 50 DIMENSIONS",
             ha="center", fontsize=16, weight="bold")
    fig.text(0.5, 1 - 0.83 / FH, "Each shape is built in 3-D (left), then rotated into 50-D. Every "
             "measurement is made there -- the tools never see the picture.", ha="center", fontsize=11.5)
    fig.text(0.5, 1 - 1.18 / FH, "Red points = places where the local count is 3 or more: the shape is "
             "not flat there (20 neighbours). Dashed lines = the true answer.\nRed scattered all over "
             "is ordinary curvature and noise; red gathered in ONE place is a real defect -- see the cone's "
             "tip and the crossing sheets' join.", ha="center", va="top", fontsize=11,
             style="italic", color=GREY, linespacing=1.5)
    for r, (name, X3, ld, dims, loops, pca) in enumerate(rows):
        top = 2.15 + r * RH
        y0, y1 = 1 - (top + RH - 0.45) / FH, 1 - top / FH
        ax = fig.add_axes([0.035, y0, 0.20, y1 - y0], projection="3d")
        c = np.where(ld >= 3, RED, BLUE)
        ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], c=c, s=4, alpha=.75, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.grid(False)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(22, 35)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis): pane.pane.set_alpha(0.03)
        fig.text(0.017, (y0 + y1) / 2, name.upper(), rotation=90, va="center", ha="center",
                 fontsize=12, weight="bold")

        ax = fig.add_axes([0.30, y0 + 0.012, 0.19, (y1 - y0) * 0.86])
        ax.plot(KS, dims, "o-", color=BLUE, lw=2.4, ms=7)
        ax.axhline(TRUE_DIM[name], ls="--", color=GREEN, lw=1.8)
        ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS], fontsize=9)
        ax.minorticks_off()
        ax.set_ylim(0, 6); ax.grid(alpha=.3); ax.tick_params(labelsize=9)
        if r == 0: ax.set_title("how many directions, as the\nneighbourhood grows", fontsize=11, pad=8)
        if r == n - 1: ax.set_xlabel("neighbours used", fontsize=10.5)
        ax.set_ylabel("dimension", fontsize=10)

        ax = fig.add_axes([0.565, y0 + 0.012, 0.19, (y1 - y0) * 0.86])
        ax.plot(RADII, loops, "s-", color=ORANGE, lw=2.4, ms=7)
        ax.axhline(TRUE_LOOPS[name], ls="--", color=GREEN, lw=1.8)
        ax.set_yscale("symlog", linthresh=1); ax.set_xticks(RADII); ax.grid(alpha=.3)
        ax.tick_params(labelsize=9); ax.set_ylim(-0.3, 60)
        ax.set_ylabel("loops", fontsize=10)
        if r == 0: ax.set_title("loops that survive, as the\nconnection radius grows", fontsize=11, pad=8)
        if r == n - 1: ax.set_xlabel("radius (x nearest-neighbour gap)", fontsize=10.5)

        fig.text(0.80, (y0 + y1) / 2 + 0.012, f"truth: {T.TRUTH[name]}", fontsize=11, weight="bold",
                 va="bottom")
        fig.text(0.80, (y0 + y1) / 2 - 0.004, f"straight axes (PCA) would need {pca}.\n{VERDICT[name]}",
                 fontsize=10.5, va="top", color=GREY, linespacing=1.45)
    fig.savefig(f"{OUT}/toy_shapes.png", dpi=110, facecolor="white"); plt.close(fig)
    print("wrote toy_shapes.png")


if __name__ == "__main__":
    main()
