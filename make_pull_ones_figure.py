"""Figure for analyze_pull_ones.py: the 1s laid out by their first two pulled directions.

Run after analyze_pull_ones.py. Writes figures/pull_ones/pull_ones.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

import analyze_pull_ones as A

OUT = Path(__file__).resolve().parent / "figures" / "pull_ones"


def thumbnails(ax, coords, images, title, cells=9):
    """Grey dots: every image. On top: for each cell of a grid, the image nearest the cell's centre."""
    x, y = coords[:, 0], coords[:, 1]
    ax.scatter(x, y, s=4, color="0.8", lw=0)
    gx = np.linspace(np.percentile(x, 1), np.percentile(x, 99), cells)
    gy = np.linspace(np.percentile(y, 1), np.percentile(y, 99), cells)
    tree = cKDTree(np.c_[x / np.ptp(x), y / np.ptp(y)])
    used = set()
    for a in gx:
        for b in gy:
            d, i = tree.query([a / np.ptp(x), b / np.ptp(y)])
            if d > 0.6 / cells or i in used:
                continue
            used.add(i)
            box = OffsetImage(images[i], cmap="gray_r", zoom=1.15, norm=plt.Normalize(0, 1))
            ax.add_artist(AnnotationBbox(box, (x[i], y[i]), frameon=True, pad=0.05,
                                         bboxprops={"edgecolor": "0.7", "lw": 0.5}))
    ax.set_title(title, fontsize=11.5)
    ax.set_xticks([]), ax.set_yticks([])


def main():
    gen_images, gen_knobs = A.generated()
    mn = A.mnist_ones()
    g = np.load(OUT / "generated_push0.001.npz")["pulled"]
    m = np.load(OUT / "mnist_push0.001.npz")["pulled"]
    props = A.moment_properties(mn)
    names = ["slant", "width", "left-right position", "up-down position", "height"]
    order = [4, 3, 0, 1, 2]                                   # PROPS indices in the order of `names`

    def label(C, P, cols, k):
        r = [abs(spearmanr(C[:, k], P[:, c])[0]) for c in cols]
        best = int(np.argmax(r))
        return best, r[best]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.6))
    gx = label(g, gen_knobs, [4, 3, 0, 1, 2], 0)
    gy = label(g, gen_knobs, [4, 3, 0, 1, 2], 1)
    thumbnails(axes[0], g, gen_images,
               "A. Generated 1s (the five knobs are known)\n"
               f"across = pulled direction 1, follows {names[gx[0]]} (rank match {gx[1]:.2f})\n"
               f"up = pulled direction 2, follows {names[gy[0]]} (rank match {gy[1]:.2f})")
    mx = label(m, props, order, 0)
    my = label(m, props, order, 1)
    thumbnails(axes[1], m, mn,
               "B. Real MNIST 1s (no known knobs; properties read off the ink)\n"
               f"across = pulled direction 1, follows {names[mx[0]]} (rank match {mx[1]:.2f})\n"
               f"up = pulled direction 2, follows {names[my[0]]} (rank match {my[1]:.2f})")
    fig.suptitle("1,000 images pulled apart, then laid out by the two directions the pulled cloud is most spread along",
                 fontsize=13)
    fig.text(0.5, 0.015, "Grey dots: every image. Thumbnails: the image nearest each point of a 9x9 grid. "
             "Rank match: 1 = the direction puts images in exactly the order of that property, 0 = unrelated.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(OUT / "pull_ones.png", dpi=110)
    print("wrote", OUT / "pull_ones.png")


if __name__ == "__main__":
    main()
