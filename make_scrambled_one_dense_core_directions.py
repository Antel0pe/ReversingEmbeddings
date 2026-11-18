"""Directions needed vs number of neighbours, inside the dense group only.

Reads figures/scrambled_one_dense_core/dense_core.json written by
`make_scrambled_one_dense_core.py` (run that first). Neighbours come only from
the 220-member dense group; start = its most crowded 1. Directions = fewest SVD
directions reaching every arrow with at most 10% of its length left over.

Output: figures/scrambled_one_dense_core/dense_core_directions.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR = Path(__file__).resolve().parent / "figures" / "scrambled_one_dense_core"
res = json.loads((DIR / "dense_core.json").read_text())["group"]
k = np.array(res["k"])
dirs = np.array(res["directions"])
rad = 100 * np.array(res["radius"])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(k, k, "k--", lw=1.1, label="one direction per neighbour (no sharing at all)")
ax.plot(k, dirs, color="#1f5fbf", lw=2.6, label="directions actually needed")
rows = ["neighbours   reach    directions"]
for m in (5, 10, 20, 40, 80, 120, 160, k[-1]):
    ax.plot(m, dirs[m - 1], "o", color="#1f5fbf", ms=5)
    rows.append(f"{m:>6}     {rad[m - 1]:>5.1f}%   {dirs[m - 1]:>6}")
ax.text(0.66, 0.31, "\n".join(rows), transform=ax.transAxes, va="top",
        family="monospace", fontsize=9.5, color="#1f5fbf",
        bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
ax.set_xlim(0, k[-1] + 1); ax.set_ylim(0, k[-1] + 1)
ax.set_xlabel("number of neighbours, nearest first (only 1s from the dense group)")
ax.set_ylabel("directions needed to reach every neighbour")
ax.set_title(f"Inside the dense group of {k[-1] + 1} MNIST 1s: directions needed as neighbours are added\n"
             "(reached = at most 10% of each arrow's length left over)\n"
             "'reach' in the table = distance to the furthest neighbour, as % of the start's length",
             fontsize=11)
ax.legend(loc="upper left", frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(DIR / "dense_core_directions.png", dpi=130)
print("\n".join(rows))
