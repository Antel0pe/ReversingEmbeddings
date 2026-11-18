"""How many directions it takes to reach every neighbour, as the circle widens.

Same two start 1s as `make_scrambled_one_neighbour_growth.py` (most crowded and
middle crowdedness), on the scrambled data (`scrambled_ones.load()`). For a
circle of radius r around the start:
  arrows     = (neighbour - start) for every other 1 within r (784 numbers each)
  directions = the fewest uncentred-SVD directions such that EVERY arrow keeps at
               most 10% of its own length outside them (the worst neighbour, not
               a typical one). Same rule as `analyze_mnist_one_flat_patch_directions`.
Reference: one direction per neighbour (no saving at all), which can never
exceed 784 because there are only 784 numbers.

Output: figures/scrambled_one_neighbour_growth/direction_growth.png (+ .json)
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from scrambled_ones import load

OUT = Path(__file__).resolve().parent / "figures" / "scrambled_one_neighbour_growth"
K = 10          # crowdedness = distance to the 10th-nearest other 1
TOL = 0.10      # leftover allowed for every arrow
MARKS = (0.20, 0.30, 0.40, 0.60, 1.00)   # radii labelled on the plot


def count_directions(arrows):
    """Fewest SVD directions so every arrow keeps at most TOL of its length."""
    _, _, Vt = np.linalg.svd(arrows, full_matrices=False)
    coef2 = (arrows @ Vt.T) ** 2
    left = 1 - np.cumsum(coef2, 1) / (arrows ** 2).sum(1)[:, None]
    worst = np.sqrt(np.maximum(left.max(0), 0))
    ok = np.nonzero(worst <= TOL)[0]
    return int(ok[0]) + 1 if len(ok) else len(Vt)


X = np.asarray(load(), dtype=np.float64)
norms = np.linalg.norm(X, axis=1)
D = cdist(X, X)
np.fill_diagonal(D, np.inf)
rel = D / norms[:, None]
kth = np.sort(rel, axis=1)[:, K - 1]
starts = {"most crowded 1": int(np.argmin(kth)),
          "typical 1 (middle crowdedness)": int(np.argsort(kth)[len(kth) // 2])}

# neighbourhood sizes to evaluate: every k up to 60, then ~log-spaced to everyone
ks = np.unique(np.concatenate([np.arange(1, 61),
                               np.geomspace(60, len(X) - 1, 90).round().astype(int)]))
results = {}
for name, i in starts.items():
    order = np.argsort(rel[i])[:len(X) - 1]
    arrows_all = X[order] - X[i]
    t = time.perf_counter()
    counts = [count_directions(arrows_all[:k]) for k in ks]
    results[name] = {"k": ks.tolist(), "radius": rel[i][order][ks - 1].tolist(),
                     "directions": counts}
    print(f"{name}: {time.perf_counter() - t:.0f} s")
    for m in MARKS:
        j = np.searchsorted(results[name]["radius"], m, side="right") - 1
        if j >= 0:
            print(f"   within {m:.0%}: {ks[j]:5d} neighbours -> {counts[j]:4d} directions")
    print(f"   everyone ({ks[-1]}): {counts[-1]} directions")

fig, ax = plt.subplots(figsize=(11, 6))
for name, colour in [("typical 1 (middle crowdedness)", "#9a9a9a"),
                     ("most crowded 1", "#1f5fbf")]:
    res = results[name]
    r = 100 * np.array(res["radius"])
    k = np.array(res["k"])
    ax.plot(r, np.minimum(k, 784), color=colour, lw=1.2, ls="--")
    ax.plot(r, res["directions"], color=colour, lw=2.6, label=f"{name}: directions needed")
    if name == "most crowded 1":
        rows = ["Most crowded 1, inside a circle of:"]
        for m in MARKS:
            j = np.searchsorted(res["radius"], m, side="right") - 1
            ax.plot(r[j], res["directions"][j], "o", color=colour, ms=6)
            rows.append(f"{m:>5.0%}   {k[j]:>5,} neighbours  →  {res['directions'][j]:>3} directions")
        rows.append(f"  all   {k[-1]:>5,} neighbours  →  {res['directions'][-1]:>3} directions")
        ax.text(0.56, 0.80, "\n".join(rows), transform=ax.transAxes, va="top",
                family="monospace", fontsize=9.5, color=colour,
                bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
ax.plot([], [], color="k", lw=1.2, ls="--", label="number of neighbours (one direction each, capped at 784)")
ax.axhline(784, color="k", lw=0.8, ls=":")
ax.text(1, 795, "784 = every possible direction", fontsize=9)
ax.set_xlim(0, 165); ax.set_ylim(0, 860)
ax.set_xlabel("distance out from the start, as % of the start image's own length\n"
              "(the circle includes every other 1 at least this close)")
ax.set_ylabel("number of directions")
ax.set_title("How many directions it takes to reach every neighbour inside the circle\n"
             f"(reached = at most {TOL:.0%} of each arrow's length left over; "
             f"all {len(X):,} MNIST training 1s, scrambled)", fontsize=11)
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "direction_growth.png", dpi=130)
(OUT / "direction_growth.json").write_text(json.dumps(results))
print("saved figures/scrambled_one_neighbour_growth/direction_growth.png")
