"""Flattening a 2-D sheet that sits in 3-D: straight projection vs. walking along it.

Toy sheets only, no generated 1s. Each sheet is a square of paper, coordinates
(u, v) in [-1, 1]^2, bent into 3-D. Two ways to flatten it, both told only that the
sheet is 2-D and given only the 3-D points:

  squash   Pick a point, take the flat plane touching the sheet there (from its
           nearest neighbours), and drop every point straight onto that plane.
           Also: try every possible tilt of the plane and keep the best.
  unroll   The user's fix. Join each point to its nearest neighbours, measure
           distance between any two points as the shortest walk along those short
           joins (so distance is measured along the sheet, not through the air),
           then lay the points out flat so those walking distances are kept as
           well as possible. (This is a known method, Isomap.)

What counts as one-to-one, checked using the true (u, v) the method never saw:

  overlap  Points that land next to each other in the flat picture but come from
           different parts of the sheet. Squashing two layers together does this.
  tear     Points that are next to each other on the sheet but land far apart in
           the flat picture.

Run from the repo root: .venv/bin/python analyze_crumpled_sheet_unrolling.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial import cKDTree

OUT = Path("figures/crumpled_sheet_unrolling")
OUT.mkdir(parents=True, exist_ok=True)
N = 4000
NEAR = 10          # neighbours used to judge overlap and tear
FAR = 0.3          # "a different part of the sheet": 15% of the sheet's side


# ------------------------------------------------------------------- the sheets
def bowl(u, v):
    return np.c_[u, v, 0.35 * (u ** 2 + v ** 2)]


def rolled(u, v):
    """Paper rolled up one and a half turns. Bending without stretching."""
    s = (u + 1) / 2                      # 0..1 along the roll
    t = 1.5 * np.pi + 3.0 * np.pi * s    # angle
    r = 0.25 * t / np.pi                 # radius grows as it unrolls
    return np.c_[r * np.cos(t), 1.2 * v, r * np.sin(t)]


def crumpled(u, v, n_twists=8, seed=4, turn=(2.0, 3.0), reach=0.4):
    """Flat sheet carried through random local twists of space.

    Each twist turns everything about a random axis by an angle that fades with
    distance from a centre. A rotation about an axis keeps each point's distance
    from that axis and its height along it, so the twist can be undone exactly:
    no two points of the sheet are ever sent to the same place. The sheet folds
    over itself but, like real paper, never passes through itself.
    """
    rng = np.random.default_rng(seed)
    P = np.c_[u, v, np.zeros_like(u)]
    for _ in range(n_twists):
        c = np.r_[rng.uniform(-0.8, 0.8, 2), rng.normal(0, 0.15)]
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        Q = P - c
        along = Q @ axis
        radial = Q - along[:, None] * axis
        # angle must depend only on things the rotation keeps: distance from axis and height along it
        angle = rng.choice([-1, 1]) * rng.uniform(*turn) * np.exp(
            -((radial ** 2).sum(1) + along ** 2) / (2 * reach ** 2))
        cross = np.cross(axis, radial)
        P = c + along[:, None] * axis + np.cos(angle)[:, None] * radial + np.sin(angle)[:, None] * cross
    return P


def tight_roll(u, v):
    """The same roll wound tighter, so its layers pass close to one another."""
    s = (u + 1) / 2
    t = 2.0 * np.pi + 4.0 * np.pi * s
    r = 0.12 + 0.045 * t / np.pi
    return np.c_[r * np.cos(t), 1.2 * v, r * np.sin(t)]


def loosely_crumpled(u, v):
    """Four wide twists: folds over, but its layers stay about two point-spacings apart.
    Chosen from 72 settings as the hardest to squash among those with that much room."""
    return crumpled(u, v, n_twists=4, seed=3, turn=(2.0, 3.0), reach=0.7)


def tightly_crumpled(u, v):
    """Eight tight twists: layers come closer together than neighbouring points."""
    return crumpled(u, v, n_twists=8, seed=6, turn=(2.0, 3.0), reach=0.4)


SHEETS = {"gently curved (bowl)": bowl, "rolled up": rolled, "crumpled, layers apart": loosely_crumpled,
          "tightly rolled": tight_roll, "crumpled, layers close": tightly_crumpled}


# ---------------------------------------------------------------- the two methods
def squash(X, normal):
    """Drop every point straight onto the plane with this normal."""
    normal = normal / np.linalg.norm(normal)
    a = np.cross(normal, [1.0, 0, 0] if abs(normal[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    return X @ np.c_[a, np.cross(normal, a)]


def tangent_normal(X, i, k=20):
    """Normal of the plane touching the sheet at point i, from its neighbours only."""
    _, idx = cKDTree(X).query(X[i], k=k)
    return np.linalg.svd(X[idx] - X[idx].mean(0))[2][2]


SHORTCUTS = {}


def unroll(X, k=10, UV=None):
    """Walk-along-the-sheet distances, then the flat layout that keeps them best."""
    dist, idx = cKDTree(X).query(X, k=k + 1)
    if UV is not None:   # checking only: share of short steps that jump to another part
        SHORTCUTS[k] = float((np.linalg.norm(UV[idx[:, 1:]] - UV[:, None], axis=2) > FAR).mean())
    rows = np.repeat(np.arange(len(X)), k)
    G = csr_matrix((dist[:, 1:].ravel(), (rows, idx[:, 1:].ravel())), shape=(len(X), len(X)))
    n_pieces = connected_components(G, directed=False)[0]
    if n_pieces > 1:
        return None, n_pieces
    D = shortest_path(G, method="D", directed=False)
    D2 = D ** 2
    B = -0.5 * (D2 - D2.mean(0) - D2.mean(1)[:, None] + D2.mean())
    val, vec = eigh(B, subset_by_index=[len(X) - 2, len(X) - 1])
    return vec * np.sqrt(np.maximum(val, 0)), 1


# ---------------------------------------------------------------- the checks
def one_to_one(Y, UV):
    """Share of points with an overlap, and with a tear. Both 0 means one-to-one."""
    Y = (Y - Y.mean(0)) * np.sqrt(((UV - UV.mean(0)) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
    _, near_flat = cKDTree(Y).query(Y, k=NEAR + 1)
    overlap = (np.linalg.norm(UV[near_flat[:, 1:]] - UV[:, None], axis=2) > FAR).any(1)
    _, near_sheet = cKDTree(UV).query(UV, k=NEAR + 1)
    tear = (np.linalg.norm(Y[near_sheet[:, 1:]] - Y[:, None], axis=2) > FAR).any(1)
    return {"overlap": float(overlap.mean()), "tear": float(tear.mean())}


def layer_gap(X, UV, k=10):
    """How close different parts of the sheet come, against the neighbour spacing."""
    tree = cKDTree(X)
    spacing = np.median(tree.query(X, k=k + 1)[0][:, -1])
    d, idx = tree.query(X, k=200)
    other = np.linalg.norm(UV[idx] - UV[:, None], axis=2) > FAR
    closest = np.where(other, d, np.inf).min()
    return float(spacing), float(closest)



def main():
    global RESULTS
    rng = np.random.default_rng(0)
    UV = rng.uniform(-1, 1, (N, 2))
    dirs = rng.normal(size=(3000, 3))
    dirs[dirs[:, 2] < 0] *= -1
    RESULTS, FLAT = {}, {}
    for name, sheet in SHEETS.items():
        X = sheet(UV[:, 0], UV[:, 1])
        starts = rng.choice(N, 40, replace=False)
        sq = [one_to_one(squash(X, tangent_normal(X, i)), UV) for i in starts]
        over = np.array([s["overlap"] for s in sq])
        every = np.array([one_to_one(squash(X, d), UV)["overlap"] for d in dirs])
        Y_unroll, pieces = unroll(X, 10, UV)
        spacing, closest = layer_gap(X, UV)
        r = {"squash_at_40_random_points": {"best_overlap": float(over.min()),
                                            "median_overlap": float(np.median(over)),
                                            "worst_overlap": float(over.max())},
             "squash_best_of_3000_tilts_overlap": float(every.min()),
             "unroll": one_to_one(Y_unroll, UV) if Y_unroll is not None else {"pieces": pieces},
             "neighbour_spacing": spacing,
             "closest_approach_of_different_parts": closest,
             "unroll_share_of_steps_that_jump_layers": SHORTCUTS[10]}
        # how the unrolling depends on the number of neighbours joined
        r["unroll_by_neighbours"] = {}
        for k in [4, 6, 10, 20]:
            Yk, p = unroll(X, k, UV)
            r["unroll_by_neighbours"][k] = {**(one_to_one(Yk, UV) if Yk is not None else {"pieces": p}),
                                            "share_of_steps_that_jump_layers": SHORTCUTS[k]}
        RESULTS[name] = r
        FLAT[name] = (X, squash(X, tangent_normal(X, starts[np.argsort(over)[len(over) // 2]])), Y_unroll)
        print(name, json.dumps(r, indent=None), flush=True)

    # Does the walk method recover when points are packed closer than the layers?
    print("--- more points on the sheets that failed", flush=True)
    for name in ["tightly rolled", "crumpled, layers close"]:
        RESULTS[name]["more_points"] = {}
        for n in [8000, 12000]:
            UVn = np.random.default_rng(n).uniform(-1, 1, (n, 2))
            Xn = SHEETS[name](UVn[:, 0], UVn[:, 1])
            sp, cl = layer_gap(Xn, UVn)
            Yn, p = unroll(Xn, 10, UVn)
            RESULTS[name]["more_points"][n] = {**(one_to_one(Yn, UVn) if Yn is not None else {"pieces": p}),
                                               "neighbour_spacing": sp, "closest_approach_of_different_parts": cl,
                                               "share_of_steps_that_jump_layers": SHORTCUTS[10]}
            print(name, n, RESULTS[name]["more_points"][n], flush=True)
            del Yn

    np.savez_compressed(OUT / "sheets.npz", UV=UV, **{f"{n}|{part}": a for n, v in FLAT.items()
                                                    for part, a in zip(["sheet", "squash", "unroll"], v)})
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=1))


if __name__ == "__main__":
    main()
