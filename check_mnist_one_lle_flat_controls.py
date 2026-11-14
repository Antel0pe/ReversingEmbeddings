"""Is the locally linear method limited to 2-D, or to square sheets? Two quick controls.

    OPENBLAS_NUM_THREADS=4 .venv/bin/python check_mnist_one_lle_flat_controls.py

1. Flat rectangles (long:short 1 to 6), 4,225 dots, 2 layout numbers, 10-dot pieces:
   both sides are found every time, so a long direction alone does not make copies.
2. The same 6,742 generated 1s, 5 layout numbers, 20-dot pieces, described three ways:
   the five dial settings (flat by construction), six ink averages (nearly flat), and the
   784 pixels (from analyze_mnist_one_lle_flattening.py). Flat descriptions recover all five
   dials at 0.98; pixels recover three (up/down 0.05, height about 0).
"""
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsRegressor

from grey_ones import KNOBS, RANGES, render
from unfolding_suite.experiments.fold_flatten import lle

RNG = np.random.default_rng(0)


def r2(target, source):
    pred = cross_val_predict(KNeighborsRegressor(10), source, target, cv=5)
    return 1 - ((pred - target) ** 2).sum() / ((target - target.mean()) ** 2).sum()


def rectangles(n=4225):
    for ratio in (1, 1.5, 2, 3, 4, 6):
        W, H = 2 * np.sqrt(ratio), 2 / np.sqrt(ratio)
        nx = int(round(np.sqrt(n * ratio)))
        ny = n // nx
        gx, gy = np.meshgrid((np.arange(nx) + 0.5) / nx * W, (np.arange(ny) + 0.5) / ny * H)
        uv = np.c_[gx.ravel(), gy.ravel()] + RNG.uniform(-0.3, 0.3, (nx * ny, 2)) * [W / nx, H / ny]
        pts = np.c_[uv, np.zeros(len(uv))] @ np.linalg.qr(RNG.standard_normal((3, 3)))[0]
        F = lle(pts, 2, None, k=10)[0]
        print(f"long:short {ratio:>4}  number 2 from number 1 {r2(F[:, 1], F[:, [0]]):5.2f}  "
              f"long side {r2(uv[:, 0], F):.2f}  short side {r2(uv[:, 1], F):.2f}")


def generated_descriptions(n=6742):
    p = RANGES[:, 0] + RNG.random((n, 5)) * (RANGES[:, 1] - RANGES[:, 0])
    X = render(p, dtype=np.float64)
    yy, xx = np.mgrid[0:28, 0:28] + 0.5
    ink = X.sum((1, 2))
    mx, my = (X * xx).sum((1, 2)) / ink, (X * yy).sum((1, 2)) / ink
    dy, dx = yy - my[:, None, None], xx - mx[:, None, None]
    mom = np.c_[ink, mx, my, (X * dy ** 2).sum((1, 2)) / ink, (X * dx * dy).sum((1, 2)) / ink,
                (X * dx ** 2).sum((1, 2)) / ink]
    reps = {"dial settings": (p - RANGES[:, 0]) / (RANGES[:, 1] - RANGES[:, 0]),
            "six ink averages": (mom - mom.mean(0)) / mom.std(0)}
    for name, P in reps.items():
        F = lle(P, 5, None, k=20)[0]
        new = 1 + sum(r2(F[:, j], F[:, :j]) < 0.5 for j in range(1, 5))
        print(f"{name:18s} new {new}/5  " + " ".join(f"{k} {r2(p[:, i], F):.2f}" for i, k in enumerate(KNOBS)))


if __name__ == "__main__":
    rectangles()
    generated_descriptions()
