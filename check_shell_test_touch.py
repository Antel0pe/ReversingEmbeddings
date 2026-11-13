"""Why did the shell test miss the toy curl's touch?  (follow-up to
analyze_mnist_one_holes_and_folds.py)

    .venv/bin/python check_shell_test_touch.py

The curl is flat paper, then a loop that comes back down. Variants:
  gap      the loop lands this far below where it left (0 = resting on itself)
  density  dots per unit area
For each, the share of dots within 0.3 of the touch whose shell splits in two,
against a flat sheet at the same density (its top 1% sets the bar).
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform

import analyze_mnist_one_holes_and_folds as A

OUT = Path(__file__).resolve().parent / "figures/mnist_one_holes_and_folds"
rng = np.random.default_rng(1)


def curl(gap, density, R=0.5, flat=1.5, L=2.0):
    loop = 2 * np.pi * (R + gap / 2)            # roughly the spiral's length
    total = 2 * flat + loop
    n = int(density * total * L)
    s = rng.uniform(0, total, n)
    x, z = np.empty(n), np.empty(n)
    a, c = s < flat, s >= flat + loop
    b = ~a & ~c
    x[a], z[a] = s[a] - flat, 0.0
    th = (s[b] - flat) / loop * 2 * np.pi
    r = R + gap * th / (2 * np.pi)
    x[b], z[b] = r * np.sin(th), R - r * np.cos(th)
    x[c], z[c] = s[c] - flat - loop, -gap
    touch = np.minimum(np.abs(s - flat), np.abs(s - flat - loop))
    return np.c_[x, rng.uniform(0, L, n), z], touch


def flat_sheet(density):
    n = int(density * 4)
    return np.c_[rng.uniform(-1, 1, (n, 2)), np.zeros(n)]


def main():
    rows = []
    for density in (160, 480):
        F = flat_sheet(density)
        thr = np.quantile(A.shell_split(squareform(pdist(F)), np.arange(len(F)))[0], 0.99)
        spacing = 1 / np.sqrt(density)
        for gap in (0.0, 0.1, 0.25):
            P, touch = curl(gap, density)
            near = np.where(touch < 0.3)[0]
            far = np.where(touch > 0.8)[0]
            D = squareform(pdist(P))
            sc_near = A.shell_split(D, near)[0]
            sc_far = A.shell_split(D, rng.choice(far, min(400, len(far)), replace=False))[0]
            row = dict(density=density, gap=gap, gap_in_dot_spacings=round(gap / spacing, 2),
                       n=len(P), flagged_near_touch_pct=round(100 * float((sc_near > thr).mean()), 1),
                       flagged_far_pct=round(100 * float((sc_far > thr).mean()), 1))
            print(row, flush=True)
            rows.append(row)
    (OUT / "touch_check.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
