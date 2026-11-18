"""How many other MNIST 1s lie within distance r of one start 1, as r grows.

Start = the most crowded training 1: the one whose 10th-nearest other 1 is
closest, measured relative to its own length. A typical 1 (median crowdedness)
is drawn as the comparison. Distances come from the scrambled data
(`scrambled_ones.load()`); they equal the pixel distances, so the curve is the
same either way. No image is shown: the start is chosen by measurement only.

Output: figures/scrambled_one_neighbour_growth/neighbour_growth.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from scrambled_ones import load

OUT = Path(__file__).resolve().parent / "figures" / "scrambled_one_neighbour_growth"
K = 10   # crowdedness = distance to the 10th-nearest other 1

X = np.asarray(load(), dtype=np.float64)
norms = np.linalg.norm(X, axis=1)
D = cdist(X, X)
np.fill_diagonal(D, np.inf)
rel = D / norms[:, None]                   # distance as a share of the start's length
kth = np.sort(rel, axis=1)[:, K - 1]
dense = int(np.argmin(kth))
typical = int(np.argsort(kth)[len(kth) // 2])

r = np.linspace(0, 1.65, 826)
def growth(i):
    d = np.sort(rel[i][np.isfinite(rel[i])])
    return d, np.searchsorted(d, r, side="right")

fig, ax1 = plt.subplots(figsize=(11, 5.5))

for i, colour, name in [(typical, "#9a9a9a", "a typical 1 (middle crowdedness)"),
                        (dense, "#1f5fbf", "the most crowded 1")]:
    d, n = growth(i)
    ax1.plot(100 * r, n, color=colour, lw=2.4, label=name)
    for k in (1, 10, 100):
        ax1.plot(100 * d[k - 1], k, "o", color=colour, ms=6)
    print(f"{name}: 1st {100*d[0]:.1f}%  10th {100*d[9]:.1f}%  "
          f"100th {100*d[99]:.1f}%  half ({len(d)//2}) {100*d[len(d)//2]:.1f}%  "
          f"all {100*d[-1]:.1f}%")

d, _ = growth(dense)
ax1.annotate(f"nearest other 1: {100*d[0]:.1f}% away\n10th: {100*d[9]:.1f}%   100th: {100*d[99]:.1f}%",
             xy=(100 * d[99], 100), xytext=(3, 2600), fontsize=10, color="#1f5fbf",
             arrowprops=dict(arrowstyle="->", color="#1f5fbf"))
ax1.axhline(len(X) - 1, color="k", lw=0.8, ls=":")
ax1.text(1, len(X) - 1 + 80, f"all {len(X) - 1:,} other 1s", fontsize=9)
ax1.set_xlim(0, 165); ax1.set_ylim(0, len(X) * 1.08)
ax1.set_xlabel("distance out from the start, as % of the start image's own length\n"
               "(straight-line distance over all 784 numbers)")
ax1.set_ylabel("number of other 1s within that distance")
ax1.set_title("How many neighbours you collect as you widen the circle\n"
              f"(dots mark the 1st, 10th and 100th neighbour; all {len(X):,} MNIST training 1s, scrambled)", fontsize=11)
ax1.legend(loc="center right", frameon=False)
ax1.grid(alpha=0.25)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "neighbour_growth.png", dpi=130)
print("saved", OUT / "neighbour_growth.png")
