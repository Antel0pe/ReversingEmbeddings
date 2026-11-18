"""Find a dense, well-connected group of MNIST 1s and count directions inside it.

Data: all 7,877 MNIST 1s (train + test), scrambled (`scrambled_ones.load`).
Distances are relative to the start image's own length, as before.

1. Crowdedness of a 1 = distance to its 10th-nearest other 1. Keep the most
   crowded 5%.
2. Link two kept 1s if each is among the other's 10 nearest kept 1s (mutual).
   Repeatedly drop 1s with fewer than 3 links (weakly connected), then split
   into connected groups. Lone 1s and small groups are ignored.
3. In the biggest group, start at its most crowded 1. Add neighbours one at a
   time, nearest first, either (a) only from the group or (b) from all 1s, and
   count the fewest SVD directions that reach every arrow with at most 10% of
   its length left over (`analyze_mnist_one_flat_patch_directions` rule).

Output: figures/scrambled_one_dense_core/dense_core.png (+ .json)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

from analyze_mnist_one_flat_patch_directions import count_directions
from scrambled_ones import load

OUT = Path(__file__).resolve().parent / "figures" / "scrambled_one_dense_core"
SHARE, K, MIN_LINKS, TOL = 0.05, 10, 3, 0.10

X = np.vstack([load("train"), load("test")]).astype(np.float64)
n = len(X)
D = cdist(X, X)
np.fill_diagonal(D, np.inf)
rel = D / np.linalg.norm(X, axis=1)[:, None]
kth = np.sort(rel, axis=1)[:, K - 1]

dense = np.argsort(kth)[:int(SHARE * n)]
nb = np.argsort(D[np.ix_(dense, dense)], axis=1)[:, :K]
A = np.zeros((len(dense),) * 2, bool)
A[np.repeat(np.arange(len(dense)), K), nb.ravel()] = True
M = A & A.T
keep = np.ones(len(dense), bool)
while True:
    weak = keep & (M[:, keep].sum(1) < MIN_LINKS)
    if not weak.any():
        break
    keep &= ~weak
_, lab = connected_components(csr_matrix(M[np.ix_(keep, keep)]), directed=False)
sizes = np.bincount(lab)
groups = [dense[keep][lab == g] for g in np.argsort(sizes)[::-1]]
print(f"{len(dense)} most crowded 1s -> {len(dense) - keep.sum()} dropped as weakly linked; "
      f"groups of size {sorted(sizes.tolist(), reverse=True)}")
for g in groups[:2]:
    R = rel[np.ix_(g, g)]
    f = R[np.isfinite(R)]
    print(f"  group of {len(g)}: nearest-neighbour median {np.median(R.min(1)):.3f}, "
          f"typical pair {np.median(f):.2f}, farthest pair {f.max():.2f}")
print(f"  closest 1s in the two groups: {rel[np.ix_(groups[0], groups[1])].min():.3f} apart")

group = groups[0]
centre = group[np.argmin(kth[group])]
res = {}
for name, pool in [("group", group), ("all", np.arange(n))]:
    pool = pool[pool != centre]
    order = pool[np.argsort(rel[centre, pool])]
    arrows = X[order] - X[centre]
    ks = np.arange(1, len(group))
    res[name] = {"k": ks.tolist(), "radius": rel[centre, order[ks - 1]].tolist(),
                 "directions": [count_directions(arrows[:k], [TOL])[0][0] for k in ks]}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4))
ks = np.array(res["group"]["k"])
for name, colour, label in [("all", "#9a9a9a", "neighbours taken from all 7,877 1s"),
                            ("group", "#1f5fbf", f"neighbours taken only from the dense group")]:
    a1.plot(ks, 100 * np.array(res[name]["radius"]), color=colour, lw=2.4, label=label)
    a2.plot(ks, res[name]["directions"], color=colour, lw=2.4, label=label)
a1.set_xlabel("number of neighbours (nearest first)")
a1.set_ylabel("distance to that neighbour, % of the start's length")
a1.set_title("1. How far you must reach for the k-th neighbour\n"
             f"(the dense group has {len(group)} 1s; start = its most crowded one)", fontsize=11)
a1.legend(loc="upper left", frameon=False, fontsize=9)
a2.plot(ks, ks, "k--", lw=1.1, label="one direction per neighbour")
a2.set_xlabel("number of neighbours (nearest first)")
a2.set_ylabel("directions needed")
a2.set_title(f"2. Directions needed to reach every neighbour\n"
             f"(at most {TOL:.0%} of each arrow's length left over)", fontsize=11)
a2.legend(loc="upper left", frameon=False, fontsize=9)
for a in (a1, a2):
    a.grid(alpha=0.25)
    a.set_xlim(0, len(group))
a2.set_ylim(0, len(group))
fig.suptitle(f"Most crowded 5% of MNIST 1s, weakly linked ones removed: the biggest connected group "
             f"({len(group)} 1s, scrambled data)", fontsize=12)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "dense_core.png", dpi=130)
(OUT / "dense_core.json").write_text(json.dumps(res))
for name in res:
    d = res[name]["directions"]
    print(name, {k: (round(100 * res[name]["radius"][k - 1], 1), d[k - 1])
                 for k in (5, 10, 20, 40, 80, 150, len(ks))})
print("saved figures/scrambled_one_dense_core/dense_core.png")
