"""Thinning test: is the direction count set by HOW FAR you reach, or by HOW MANY 1s?

Picture a bent sheet of paper with dots on it. Draw a circle of radius r around
one dot. The number of flat directions needed to cover the paper inside that
circle is set by how much the paper bends within r. Erase half the dots and the
same circle still holds the same piece of paper, so it should need about the
same number of directions (as long as enough dots remain to show the bend).

If instead every dot sticks out in a direction of its own (a fuzzy cloud, or
more ways of varying than the dots can show), the count is set by how many dots
are inside: erase half and the same circle needs about half as many.

Test: same starts (60 members of each dataset's dense group, as in
`make_scrambled_one_many_starts.py`), neighbours taken from the full data, a
random half, and a random quarter. At each radius, count the directions needed
to reach every neighbour inside it (<= 10% of each arrow left over).

  paper (bending only):         the three curves lie on top of each other
  every dot its own direction:  half ~ 1/2 of full, quarter ~ 1/4

Datasets: MNIST 1s (scrambled), generated 1s (5 knobs), drawn 1s (12 knobs),
shapeless cloud with MNIST's spread. 7,877 images each.

Output: figures/scrambled_one_thinning_test/thinning.png (+ .json; delete the
json to recompute, otherwise only the figure is redrawn)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

import knob_strokes
from grey_ones import render, sample
from make_scrambled_one_many_starts import count_directions, dense_group
from scrambled_ones import load

OUT = Path(__file__).resolve().parent / "figures" / "scrambled_one_thinning_test"
N_STARTS, KMAX, N_R = 60, 300, 40
KEEP = {"all 1s": 1.0, "half": 0.5, "quarter": 0.25}

def compute():
    rng = np.random.default_rng(0)
    mn = np.vstack([load("train"), load("test")]).astype(np.float64)
    n = len(mn)
    data = {
        "MNIST 1s": mn,
        "generated 1s, 5 knobs (known: bending only)": render(sample(n, rng)).reshape(n, 784).astype(np.float64),
        "drawn 1s, 12 knobs (known: bending only)": knob_strokes.make(n, 12).reshape(n, 784).astype(np.float64),
        "shapeless cloud (known: every dot its own direction)":
            rng.multivariate_normal(mn.mean(0), np.cov(mn.T), size=n),
    }
    subsets = {name: np.sort(rng.permutation(n)[:int(f * n)]) for name, f in KEEP.items()}

    results = {}
    for dname, X in data.items():
        group, _, _ = dense_group(X)
        starts = rng.choice(group, size=min(N_STARTS, len(group)), replace=False)
        rel = cdist(X[starts], X) / np.linalg.norm(X[starts], axis=1)[:, None]
        rel[np.arange(len(starts)), starts] = np.inf
        # radius grid: up to where the FULL data has KMAX neighbours for the median start
        r_max = np.median(np.sort(rel, axis=1)[:, KMAX - 1])
        radii = np.linspace(0, r_max, N_R + 1)[1:]
        res = {"radius": radii.tolist()}
        for sname, idx in subsets.items():
            dirs = np.zeros((len(starts), N_R))
            nbrs = np.zeros((len(starts), N_R))
            for s, c in enumerate(starts):
                d = rel[s, idx]
                order = idx[np.argsort(d)]
                dsort = np.sort(d)
                arrows = X[order[:KMAX]] - X[c]
                for j, r in enumerate(radii):
                    k = int(np.searchsorted(dsort, r, side="right"))
                    nbrs[s, j] = k
                    dirs[s, j] = count_directions(arrows[:k]) if k else 0
            res[sname] = {"directions": np.median(dirs, 0).tolist(),
                          "neighbours": np.median(nbrs, 0).tolist()}
        results[dname] = res
        j = N_R - 1
        print(f"{dname}\n   at radius {100 * radii[j]:.1f}%: " + ", ".join(
            f"{s}: {res[s]['neighbours'][j]:.0f} nbrs -> {res[s]['directions'][j]:.0f} dirs" for s in KEEP))
        j = N_R // 2 - 1
        print(f"   at radius {100 * radii[j]:.1f}%: " + ", ".join(
            f"{s}: {res[s]['neighbours'][j]:.0f} nbrs -> {res[s]['directions'][j]:.0f} dirs" for s in KEEP))

    return results


def plot(results):
    names = {"generated 1s, 5 knobs (known: bending only)": ("#2a9d5c", "generated 1s, 5 knobs (bending only)"),
             "drawn 1s, 12 knobs (known: bending only)": ("#c98a1b", "drawn 1s, 12 knobs (bending only)"),
             "MNIST 1s": ("#1f5fbf", "MNIST 1s"),
             "shapeless cloud (known: every dot its own direction)": ("#9a9a9a", "shapeless cloud (no structure)")}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.2))
    res = results["MNIST 1s"]
    r = 100 * np.array(res["radius"])
    shades = {"all 1s": "#0b2f6b", "half": "#3f7fd9", "quarter": "#9cc3f0"}
    for sname, colour in shades.items():
        a1.plot(r, res[sname]["directions"], color=colour, lw=2.6, label={"all 1s": "all 1s kept", "half": "half of the 1s kept", "quarter": "a quarter of the 1s kept"}[sname])
    a1.set_xlim(17, r[-1]); a1.set_ylim(0, 110)
    a1.set_xlabel("radius of the circle around the start, % of the start image's length")
    a1.set_ylabel("directions needed to reach every 1 inside\n(median over 60 starts)")
    a1.set_title("1. MNIST: same circles, with fewer 1s kept", fontsize=11)
    a1.legend(loc="upper left", frameon=False)
    for dname, (colour, label) in names.items():
        res = results[dname]
        full = np.array(res["all 1s"]["directions"])
        x = np.array(res["all 1s"]["neighbours"])
        ok = full >= 10
        for sname, ls in (("half", "--"), ("quarter", "-")):
            ratio = np.array(res[sname]["directions"])[ok] / full[ok]
            a2.plot(x[ok], ratio, color=colour, lw=2.6 if ls == "-" else 1.4, ls=ls,
                    label=label if sname == "quarter" else None)
    a2.axhline(1, color="k", lw=0.8)
    a2.axhline(0.5, color="k", lw=0.8, ls="--")
    a2.axhline(0.25, color="k", lw=0.8)
    a2.text(302, 1.0, " pure bending:\n same count", va="center", fontsize=9)
    a2.text(302, 0.5, " every 1 its own\n direction (half)", va="center", fontsize=9)
    a2.text(302, 0.25, " every 1 its own\n direction (quarter)", va="center", fontsize=9)
    a2.set_xlim(0, 300); a2.set_ylim(0, 1.1)
    a2.set_xlabel("size of the circle, as the number of 1s inside it with ALL 1s kept")
    a2.set_ylabel("directions with fewer 1s kept  /  directions with all 1s")
    a2.set_title("2. Share of the directions that survive thinning\n"
                 "(thick = a quarter of the 1s kept, thin dashed = half kept)", fontsize=11)
    a2.legend(loc="lower left", frameon=False, fontsize=9.5)
    for ax in (a1, a2):
        ax.grid(alpha=0.25)
    fig.suptitle("Thinning test: erase 1s and recount in the SAME circles "
                 "(reached = at most 10% of each arrow's length left over)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "thinning.png", dpi=125)
    print("saved figures/scrambled_one_thinning_test/thinning.png")


if __name__ == "__main__":
    path = OUT / "thinning.json"
    if path.exists():
        results = json.loads(path.read_text())
    else:
        results = compute()
        OUT.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results))
    plot(results)
