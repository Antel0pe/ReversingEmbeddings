"""Does projecting the generated 1s onto a flat 5-D plane give a usable address?

The idea under test (the user's): crumple a sheet in 3-D, pick a point, take the
flat plane there, and project everything onto it. Each spot on the plane then
labels one spot on the sheet. Do the same for the generated 1s, assuming 5-D.

No decoder is trained or needed. The only question is whether the projection is
one-to-one: does each position on the plane belong to exactly one 1?

  data only   Project a large sample. Compare each 1 with the images whose plane
              positions land closest to it. Are any of them very different?
  look-alikes Take such a near-miss and refine its knobs with the renderer until
              both 1s have the same five plane numbers to one part in a million,
              then render it to confirm. One such pair proves the plane fails.
              Finding none proves nothing, so failing to find one is only evidence.
  folds       From the slope of the projection at sampled settings: does some
              direction of change stop registering, or does the plane flip over?

The paper example runs first as a check on the tests themselves: a gently curved
sheet has to pass and a crumpled sheet has to fail.

Run from the repo root: .venv/bin/python analyze_generated_one_flat_plane.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from grey_ones import RANGES, render

OUT = Path("figures/generated_one_flat_plane")
OUT.mkdir(parents=True, exist_ok=True)

ROOT = np.array([14.0, 14.0, 19.75, 3.2, 0.0])
BOX_MIDDLE = RANGES.mean(1)
SPAN = RANGES[:, 1] - RANGES[:, 0]
SAME_POSITION = 1e-6     # one part in a million of the spread of plane positions
CLEARLY_DIFFERENT = 0.25 # image gap, as a share of the distance between two typical 1s
VERY_DIFFERENT = 0.5
RESULTS = {}


def R1(P):
    return render(P, dtype=np.float64).reshape(len(np.atleast_2d(P)), -1)


def jac(f, P, ranges):
    """(n, d, D) image change per full range of each dial, by central difference."""
    P = np.atleast_2d(P)
    h = 1e-4 * (ranges[:, 1] - ranges[:, 0])
    out = []
    for k in range(P.shape[1]):
        lo, hi = P.copy(), P.copy()
        lo[:, k] -= h[k]
        hi[:, k] += h[k]
        out.append((f(hi) - f(lo)) / 2e-4)
    return np.stack(out, 1)


def plane_at(f, p, ranges):
    """Orthonormal axes of the flat plane touching the family at p: the user's recipe."""
    return np.linalg.svd(jac(f, p[None], ranges)[0].T, full_matrices=False)[0]


class Scale:
    """Rescale plane positions so each uncorrelated direction has unit spread."""

    def __init__(self, Z):
        self.mean = Z.mean(0)
        val, vec = np.linalg.eigh(np.cov(Z.T))
        self.T = vec / np.sqrt(np.maximum(val, val.max() * 1e-14))

    def __call__(self, Z):
        return (Z - self.mean) @ self.T


def data_only_check(U, pool, test, typical, k=20):
    """Using only images: how different are the 1s landing closest on the plane?"""
    s = Scale(pool @ U)
    tree = cKDTree(s(pool @ U))
    _, idx = tree.query(s(test @ U), k=k, workers=-1)
    gaps = np.linalg.norm(pool[idx] - test[:, None], axis=2).max(1) / typical
    return {"median_largest_gap_among_closest": float(np.median(gaps)),
            "p95_largest_gap_among_closest": float(np.percentile(gaps, 95)),
            "worst_largest_gap_among_closest": float(gaps.max()),
            "share_with_clearly_different_neighbour": float((gaps >= CLEARLY_DIFFERENT).mean())}


def fold_check(U, JJ):
    """Share of settings where a direction stops registering, or the plane has flipped."""
    A = JJ @ U
    weakest = np.linalg.svd(A, compute_uv=False)[:, -1]
    dead = weakest / np.linalg.svd(JJ, compute_uv=False)[:, -1] < 1e-6
    sign = np.sign(np.linalg.det(A))[~dead]
    pos = float((sign > 0).mean()) if len(sign) else 0.0
    return {"share_direction_stops_registering": float(dead.mean()),
            "share_plane_flipped": float((~dead).mean() * min(pos, 1 - pos))}


def lookalike_check(f, U, P_pool, pool, P_test, test, typical, ranges, n_targets=600,
                    n_candidates=5, near=300):
    """Search for a clearly different 1 at the same plane position."""
    s = Scale(pool @ U)
    tree = cKDTree(s(pool @ U))
    found, pairs = [], []
    for t in range(min(n_targets, len(test))):
        target = s(test[t:t + 1] @ U)[0]
        _, idx = tree.query(target, k=near)
        gaps = np.linalg.norm(pool[idx] - test[t], axis=1)
        best, best_p = 0.0, None
        for c in idx[gaps >= CLEARLY_DIFFERENT * typical][:n_candidates]:
            sol = least_squares(lambda p: s(f(p[None]) @ U)[0] - target, P_pool[c],
                                bounds=(ranges[:, 0], ranges[:, 1]),
                                x_scale=ranges[:, 1] - ranges[:, 0], diff_step=1e-4,
                                max_nfev=60, xtol=1e-14, ftol=1e-14, gtol=1e-14)
            if np.linalg.norm(sol.fun) > SAME_POSITION:
                continue
            gap = np.linalg.norm(f(sol.x[None])[0] - test[t])
            if gap > best:
                best, best_p = gap, sol.x
        found.append(best / typical)
        if best_p is not None and best / typical >= CLEARLY_DIFFERENT:
            pairs.append((float(best / typical), P_test[t].tolist(), best_p.tolist()))
    found = np.array(found)
    pairs.sort(key=lambda e: -e[0])
    return {"checked": len(found),
            "share_with_lookalike": float((found >= CLEARLY_DIFFERENT).mean()),
            "share_with_very_different_lookalike": float((found >= VERY_DIFFERENT).mean()),
            "largest_gap": float(found.max()),
            "pairs": pairs[:5]}


# ---------------------------------------------------------------- 1. the paper
def gentle(P):
    u, v = P[:, 0], P[:, 1]
    return np.c_[u, v, 0.3 * (u * u + v * v)]


def crumpled(P):
    u, v = P[:, 0], P[:, 1]
    a = 3 * np.pi
    return np.c_[u + 0.28 * np.sin(a * u) * np.cos(np.pi * v / 2),
                 v + 0.22 * np.sin(2.5 * np.pi * v + 1) * np.cos(np.pi * u / 2),
                 0.45 * np.cos(a * u) * np.cos(2 * v) + 0.3 * np.sin(3 * v + u)]


print("--- paper in 3-D: the tests themselves must pass and fail correctly")
PAPER = np.array([[-1.0, 1.0], [-1.0, 1.0]])
rng = np.random.default_rng(1)
RESULTS["paper"] = {}
for name, sheet in [("gently curved", gentle), ("crumpled", crumpled)]:
    Pp, Pt = -1 + 2 * rng.random((40_000, 2)), -1 + 2 * rng.random((1000, 2))
    Xp, Xt = sheet(Pp), sheet(Pt)
    typ = float(np.median(np.linalg.norm(Xt[:500] - Xt[500:], axis=1)))
    U = plane_at(sheet, np.array([0.03, -0.02]), PAPER)
    r = {**data_only_check(U, Xp, Xt, typ), **fold_check(U, jac(sheet, Pt[:600], PAPER)),
         **lookalike_check(sheet, U, Pp, Xp, Pt, Xt, typ, PAPER, n_targets=200)}
    RESULTS["paper"][name] = r
    print(name, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)})


# --------------------------------------------------------- 2. the generated 1s
print("--- generated 1s: rendering")
rng = np.random.default_rng(0)
P_pool = RANGES[:, 0] + rng.random((150_000, 5)) * SPAN
P_test = RANGES[:, 0] + rng.random((2_000, 5)) * SPAN
pool, test = R1(P_pool), R1(P_test)
JJ = jac(R1, RANGES[:, 0] + rng.random((1500, 5)) * SPAN, RANGES)
typical = float(np.median(np.linalg.norm(test[:1000] - test[1000:], axis=1)))
RESULTS["typical_distance_between_two_1s"] = typical

# Reference for the data-only check: neighbours chosen by the real knobs, each
# scaled by how fast it moves the picture. Their gaps reflect sampling density alone.
speed = np.sqrt((JJ ** 2).sum(2)).mean(0)
Zk, Zt = P_pool / SPAN * speed, P_test / SPAN * speed
_, idx = cKDTree(Zk).query(Zt, k=20, workers=-1)
g = np.linalg.norm(pool[idx] - test[:, None], axis=2).max(1) / typical
RESULTS["real_knobs_reference"] = {"median_largest_gap_among_closest": float(np.median(g)),
                                   "p95_largest_gap_among_closest": float(np.percentile(g, 95)),
                                   "worst_largest_gap_among_closest": float(g.max())}
print("real knobs reference", RESULTS["real_knobs_reference"])

# Good faith: many starting points. Pick the winner on separate images, data only.
rng_s = np.random.default_rng(5)
starts = RANGES[:, 0] + rng_s.random((200, 5)) * SPAN
scores = [data_only_check(plane_at(R1, p, RANGES), pool[:30_000], pool[30_000:30_400], typical)
          ["p95_largest_gap_among_closest"] for p in starts]
best_start = starts[int(np.argmin(scores))]
RESULTS["best_of_200_points"] = {"point": best_start.tolist(),
                                 "validation_p95_best": float(min(scores)),
                                 "validation_p95_median": float(np.median(scores)),
                                 "validation_p95_worst": float(max(scores))}
print(RESULTS["best_of_200_points"])

U_bestfit = np.linalg.svd(pool[:20_000] - pool[:20_000].mean(0), full_matrices=False)[2][:5].T

# Not the user's recipe: five pixel weightings known from the moment chart.
yy, xx = np.mgrid[0:28, 0:28] + 0.5
W = np.stack([np.ones(784), xx.ravel(), yy.ravel(), (yy ** 2).ravel(), (xx * yy).ravel()], 1)
U_sums = np.linalg.qr(W)[0]


def knobs_from_sums(X):
    """Exact formula for the continuous stroke: 5 plane numbers -> 5 knobs."""
    m = X @ W
    cx, cy = m[:, 1] / m[:, 0], m[:, 2] / m[:, 0]
    vy = m[:, 3] / m[:, 0] - cy ** 2 - 1 / 12
    h = np.sqrt(12 * vy)
    lean = np.degrees(np.arctan(-(m[:, 4] / m[:, 0] - cx * cy) / vy))
    return np.c_[cx, cy, h, m[:, 0] / h, lean]


err = np.abs(knobs_from_sums(test) - P_test)
miss = np.linalg.norm(R1(knobs_from_sums(test)) - test, axis=1) / typical
RESULTS["sums_plane_formula"] = {"worst_error_as_share_of_each_range": (err.max(0) / SPAN).tolist(),
                                 "image_gap_after_formula_median": float(np.median(miss)),
                                 "image_gap_after_formula_worst": float(miss.max())}
print(RESULTS["sums_plane_formula"])


# Not the user's recipe: search for a plane from images alone, asking directly that
# no two different 1s land close together.
def pairs_for_search(offset, n_images, seed):
    r = np.random.default_rng(seed)
    X = pool[offset:offset + n_images]
    B = np.linalg.svd(X[:8000] - X[:8000].mean(0), full_matrices=False)[2][:20].T
    m = min(15_000, n_images)
    _, nn = cKDTree(X @ B).query(X[:m] @ B, k=31, workers=-1)
    A = np.concatenate([X[:m] - X[nn[:, 1]], X[:m] - X[nn[:, 6]], X[:m] - X[nn[:, 30]],
                        X[r.integers(0, n_images, m)] - X[r.integers(0, n_images, m)]])
    A = A[np.linalg.norm(A, axis=1) > 1e-9]
    return torch.tensor(A / np.linalg.norm(A, axis=1, keepdims=True), dtype=torch.float32)


def plane_search(start, S, seed, steps=1500):
    torch.manual_seed(seed)
    A = torch.tensor(start, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([A], lr=0.01)
    for i in range(steps):
        beta = 2 + 18 * i / steps
        Q = torch.linalg.qr(A)[0]
        r = (S[torch.randint(0, len(S), (30_000,))] @ Q).pow(2).sum(1).clamp_min(1e-12).log() / 2
        loss = torch.logsumexp(-beta * r, 0) / beta
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.linalg.qr(A.detach())[0].numpy().astype(np.float64)


rng_p = np.random.default_rng(11)
PLANES = {
    "YOURS at root 1": plane_at(R1, ROOT, RANGES),
    "YOURS at middle of the ranges": plane_at(R1, BOX_MIDDLE, RANGES),
    "YOURS at best of 200 points": plane_at(R1, best_start, RANGES),
    "best-fit plane for all 1s": U_bestfit,
    "random plane": np.linalg.qr(rng_p.normal(size=(784, 5)))[0],
    "five weighted sums": U_sums,
}
SEARCHES = [("searched: best-fit start, images A", U_bestfit, 0, 40_000, 0),
            ("searched: best-fit start, images B", U_bestfit, 40_000, 40_000, 1),
            ("searched: best-fit start, images C", U_bestfit, 80_000, 40_000, 2),
            ("searched: random start, images A", None, 0, 40_000, 3),
            ("searched: random start, images B", None, 40_000, 40_000, 4),
            ("searched: random start, images C", None, 80_000, 40_000, 5)]
for name, start, offset, n_img, seed in SEARCHES:
    if start is None:
        start = np.random.default_rng(100 + seed).normal(size=(784, 5))
    PLANES[name] = plane_search(start, pairs_for_search(offset, n_img, seed), seed)
    print("found", name, flush=True)

RESULTS["planes"] = {}
for name, U in PLANES.items():
    r = {**data_only_check(U, pool, test, typical), **fold_check(U, JJ),
         **lookalike_check(R1, U, P_pool, pool, P_test, test, typical, RANGES)}
    RESULTS["planes"][name] = r
    print(f"{name:36s} neighbour gap med {r['median_largest_gap_among_closest']:.3f}"
          f" p95 {r['p95_largest_gap_among_closest']:.3f} worst {r['worst_largest_gap_among_closest']:.3f}"
          f" | dead {r['share_direction_stops_registering']:.2f} flip {r['share_plane_flipped']:.2f}"
          f" | lookalike {r['share_with_lookalike']:.3f} very {r['share_with_very_different_lookalike']:.3f}"
          f" max {r['largest_gap']:.2f}", flush=True)

np.savez_compressed(OUT / "planes.npz", **{n: U for n, U in PLANES.items()})
(OUT / "results.json").write_text(json.dumps(RESULTS, indent=1))
print("saved")
