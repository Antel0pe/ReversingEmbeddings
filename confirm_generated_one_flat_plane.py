"""Re-check the planes that passed, on fresh 1s, because they were picked as the best.

`analyze_generated_one_flat_plane.py` ran six plane searches and two came out clean on
its own check images, so that score is flattered by selection. Here every image, every
setting and every search for a look-alike is new.

Run from the repo root: .venv/bin/python confirm_generated_one_flat_plane.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from grey_ones import RANGES, render

OUT = Path("figures/generated_one_flat_plane")
SPAN = RANGES[:, 1] - RANGES[:, 0]
CHECK = ["five weighted sums", "searched: best-fit start, images A", "searched: random start, images C",
         "YOURS at middle of the ranges"]


def R1(P):
    return render(P, dtype=np.float64).reshape(len(np.atleast_2d(P)), -1)


def jac(P):
    h = 1e-4 * SPAN
    out = []
    for k in range(5):
        lo, hi = P.copy(), P.copy()
        lo[:, k] -= h[k]
        hi[:, k] += h[k]
        out.append((R1(hi) - R1(lo)) / 2e-4)
    return np.stack(out, 1)


rng = np.random.default_rng(2026)
P_pool = RANGES[:, 0] + rng.random((150_000, 5)) * SPAN
P_test = RANGES[:, 0] + rng.random((1000, 5)) * SPAN
pool, test = R1(P_pool), R1(P_test)
JJ = jac(RANGES[:, 0] + rng.random((4000, 5)) * SPAN)
typical = float(np.median(np.linalg.norm(test[:500] - test[500:], axis=1)))
planes = np.load(OUT / "planes.npz")
results = {}
for name in CHECK:
    U = planes[name]
    A = JJ @ U
    weak = np.linalg.svd(A, compute_uv=False)[:, -1] / np.linalg.svd(JJ, compute_uv=False)[:, -1]
    sign = np.sign(np.linalg.det(A))
    Z = pool @ U
    mean, (val, vec) = Z.mean(0), np.linalg.eigh(np.cov(Z.T))
    T = vec / np.sqrt(val)
    tree = cKDTree((Z - mean) @ T)
    found = []
    for t in range(len(test)):
        target = (test[t] @ U - mean) @ T
        _, idx = tree.query(target, k=300)
        gaps = np.linalg.norm(pool[idx] - test[t], axis=1)
        best = 0.0
        for c in idx[gaps >= 0.25 * typical][:5]:
            sol = least_squares(lambda p: (R1(p[None])[0] @ U - mean) @ T - target, P_pool[c],
                                bounds=(RANGES[:, 0], RANGES[:, 1]), x_scale=SPAN, diff_step=1e-4,
                                max_nfev=60, xtol=1e-14, ftol=1e-14, gtol=1e-14)
            if np.linalg.norm(sol.fun) <= 1e-6:
                best = max(best, np.linalg.norm(R1(sol.x[None])[0] - test[t]) / typical)
        found.append(best)
    found = np.array(found)
    results[name] = {"lookalikes_checked": len(found),
                     "share_with_lookalike": float((found >= 0.25).mean()),
                     "largest_gap": float(found.max()),
                     "settings_checked": len(JJ),
                     "share_dead": float((weak < 1e-6).mean()),
                     "smaller_orientation_share": float(min((sign > 0).mean(), (sign < 0).mean())),
                     "weakest_response_min": float(weak.min())}
    print(name, results[name], flush=True)
(OUT / "fresh_confirmation.json").write_text(json.dumps(results, indent=1))
