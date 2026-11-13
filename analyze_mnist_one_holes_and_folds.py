"""Holes and folds in MNIST 1s, by persistent homology plus a local shell test.

Run from the repo root:
    OPENBLAS_NUM_THREADS=4 .venv/bin/python analyze_mnist_one_holes_and_folds.py

Two tests, the same on every dataset:

  LOOPS   Persistent homology of the whole cloud (ripser, Vietoris-Rips).
          Link every pair of points closer than a scale e, fill in every triangle
          of links, and watch loops appear and fill in as e grows. A loop's
          strength is fill-in scale / appearance scale. Noise loops have strengths
          near 1; a real hole is far above. For the strongest loop we also name
          the link that closed it (the shortcut) and the path the long way round.
          A cavity check (one dimension up) runs on 600-point subsamples.

  SHELLS  For each centre point, keep its 8th-48th nearest neighbours (a hollow
          shell around it, centre removed). On one sheet the shell is a single
          piece. Where two layers pass close, it splits into two well-separated
          pieces. Score = the longest link in the shell's shortest-link tree
          whose removal leaves at least 5 points on each side, divided by the
          median link. A big score means a clean split.

Datasets:
  toy flat sheet / toy tube / toy curl that touches itself (3-D, known answers)
  generated 1s       (five knobs in a box: no holes, no folds, by construction)
  MNIST twin         (a Gaussian cloud with MNIST 1s' exact mean and spread in
                      every pixel direction: no holes or folds, by construction)
  generated twin     (the same for the generated 1s)
  MNIST 1s           (training set, all 6,742)

A real dataset is compared against its own twin: anything the twin also shows is
explained by sparseness alone. The generated 1s test that comparison itself,
because their true answer is "nothing".
"""
import json
import time
from pathlib import Path

import numpy as np
from ripser import ripser
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree, shortest_path
from scipy.spatial.distance import pdist, squareform

import mnist
from grey_ones import RANGES, render

OUT = Path(__file__).resolve().parent / "figures/mnist_one_holes_and_folds"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)
K_IN, K_OUT, MIN_SIDE = 8, 48, 5
DENSITY = 160.0          # toy dots per unit area


# --------------------------------------------------------------------- datasets
def toy_flat():
    n = int(DENSITY * 4)
    u = RNG.uniform(-1, 1, (n, 2))
    return np.c_[u, np.zeros(n)], None


def toy_tube(R=0.5, L=2.0):
    n = int(DENSITY * 2 * np.pi * R * L)
    t = RNG.uniform(0, 2 * np.pi, n)
    return np.c_[R * np.sin(t), RNG.uniform(0, L, n), R - R * np.cos(t)], t


def toy_curl(R=0.5, flat=1.5, L=2.0):
    """Flat, then a full loop of radius R that comes back down and rests on the
    paper exactly where it left (touching, tangent), then flat again. Every other
    part of the loop curves away. s = distance along the paper."""
    total = 2 * flat + 2 * np.pi * R
    n = int(DENSITY * total * L)
    s = RNG.uniform(0, total, n)
    x = np.empty(n)
    z = np.empty(n)
    a = s < flat
    x[a], z[a] = s[a] - flat, 0.0
    b = (s >= flat) & (s < flat + 2 * np.pi * R)
    th = (s[b] - flat) / R
    x[b], z[b] = R * np.sin(th), R - R * np.cos(th)
    c = s >= flat + 2 * np.pi * R
    x[c], z[c] = s[c] - flat - 2 * np.pi * R, 0.0
    return np.c_[x, RNG.uniform(0, L, n), z], s


def mnist_ones():
    X, y = mnist.load("train")
    return X[y == 1].reshape(-1, 784).astype(np.float64) / 255.0


def generated(n):
    p = RANGES[:, 0] + RNG.random((n, 5)) * (RANGES[:, 1] - RANGES[:, 0])
    return render(p, dtype=np.float64).reshape(n, -1), p


def twin(X):
    """Gaussian with X's mean and covariance (same spread in every direction)."""
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    z = RNG.standard_normal((len(X), len(S)))
    return mu + (z * S / np.sqrt(len(X) - 1)) @ Vt


# ------------------------------------------------------------------- the tests
def loops(D, label, cocycles=True, thresh=np.inf):
    """thresh caps the scale. Needed for the 3-D toys: in 3-D every point is near
    every other at large scales and the triangle count explodes (1,000 dots took
    40 s uncapped). A loop still open at the cap is given death = cap."""
    t = time.time()
    r = ripser(D, distance_matrix=True, maxdim=1, do_cocycles=cocycles, thresh=thresh)
    h1 = r["dgms"][1].copy()
    h1[np.isinf(h1[:, 1]), 1] = thresh
    strength = h1[:, 1] / h1[:, 0]
    order = np.argsort(-strength)
    out = {"n": len(D), "seconds": round(time.time() - t, 1), "n_loops": int(len(h1)),
           "top_strengths": [round(float(v), 3) for v in strength[order[:5]]],
           "median_strength": round(float(np.median(strength)), 3) if len(h1) else None}
    if cocycles and len(h1):
        k = order[0]
        birth, death = h1[k]
        cc = r["cocycles"][1][k]
        e = cc[np.argmin(np.abs(D[cc[:, 0], cc[:, 1]] - birth)), :2]
        i, j = int(e[0]), int(e[1])
        # the long way round: path from i to j using only links shorter than birth
        A = np.where(D < birth, D, 0.0)
        A[i, j] = A[j, i] = 0.0
        _, pred = shortest_path(csr_matrix(A), indices=i, return_predecessors=True)
        path, v = [j], j
        while v != i and v >= 0:
            v = pred[v]
            path.append(int(v))
        out.update(strongest=dict(birth=float(birth), death=float(death),
                                  shortcut=[i, j], path=path[::-1]))
    print(f"  [{label}] loops: n={len(D)} {out['n_loops']} loops, top strengths "
          f"{out['top_strengths']} median {out['median_strength']} ({out['seconds']} s)")
    return out


def cavities(X, label, n=600):
    idx = RNG.choice(len(X), min(n, len(X)), replace=False)
    D = squareform(pdist(X[idx]))
    r = ripser(D, distance_matrix=True, maxdim=2)
    h2 = r["dgms"][2]
    s = np.sort(h2[:, 1] / h2[:, 0])[::-1] if len(h2) else np.array([])
    print(f"  [{label}] cavities on {len(idx)}: {len(h2)} found, top {np.round(s[:3], 3)}")
    return {"n": len(idx), "n_cavities": int(len(h2)),
            "top_strengths": [round(float(v), 3) for v in s[:5]]}


def shell_split(D, centres):
    """Shell score and the two pieces for each centre."""
    scores, pieces = [], []
    for c in centres:
        nb = np.argsort(D[c])[K_IN:K_OUT + 1]
        S = D[np.ix_(nb, nb)]
        T = minimum_spanning_tree(csr_matrix(S)).tocoo()
        w = T.data
        med = np.median(w)
        best, best_sides = 0.0, None
        for e in np.argsort(-w):
            keep = np.ones(len(w), bool)
            keep[e] = False
            G = csr_matrix((w[keep], (T.row[keep], T.col[keep])), shape=S.shape)
            _, lab = connected_components(G, directed=False)
            if min((lab == 0).sum(), (lab == 1).sum()) >= MIN_SIDE:
                best, best_sides = w[e] / med, (nb[lab == 0], nb[lab == 1])
                break
        scores.append(best)
        pieces.append(best_sides)
    return np.array(scores), pieces


def summarise_shells(scores, ref_scores, label):
    thr = np.quantile(ref_scores, 0.99)
    frac = float((scores > thr).mean())
    print(f"  [{label}] shells: median score {np.median(scores):.2f}, "
          f"above control's 99th pct ({thr:.2f}): {100 * frac:.1f}%")
    return {"median": round(float(np.median(scores)), 3),
            "p99": round(float(np.quantile(scores, 0.99)), 3),
            "control_p99": round(float(thr), 3), "flagged_pct": round(100 * frac, 2)}


# ------------------------------------------------------------------------ main
def main():
    R = {}
    print("TOYS")
    flat, _ = toy_flat()
    tube, tube_t = toy_tube()
    curl, curl_s = toy_curl()
    toy = {}
    Ds = {name: squareform(pdist(P)) for name, P in
          (("flat", flat), ("tube", tube), ("curl", curl))}
    sc = {name: shell_split(D, np.arange(len(D)))[0] for name, D in Ds.items()}
    for name, P in (("flat", flat), ("tube", tube), ("curl", curl)):
        toy[name] = {"loops": loops(Ds[name], name, cocycles=False, thresh=1.0),
                     "shells": summarise_shells(sc[name], sc["flat"], name)}
    # where on the curl are the flagged dots?  (distance along paper from the touch)
    thr = np.quantile(sc["flat"], 0.99)
    flag = sc["curl"] > thr
    flat_len, R0 = 1.5, 0.5
    touch_dist = np.minimum(np.abs(curl_s - flat_len),
                            np.abs(curl_s - flat_len - 2 * np.pi * R0))
    toy["curl"]["flagged_touch_distance"] = {
        "median_flagged": round(float(np.median(touch_dist[flag])), 3) if flag.any() else None,
        "median_all": round(float(np.median(touch_dist)), 3),
        "pct_flagged_within_0.3": round(100 * float(flag[touch_dist < 0.3].mean()), 1),
        "pct_flagged_beyond_0.6": round(100 * float(flag[touch_dist > 0.6].mean()), 1)}
    print("  curl flagged dots vs distance from touch:", toy["curl"]["flagged_touch_distance"])
    R["toys"] = toy
    np.savez_compressed(OUT / "toys.npz", curl=curl, curl_s=curl_s, curl_score=sc["curl"],
                        tube=tube, tube_score=sc["tube"], flat_thr=thr)

    print("HIGH-DIMENSIONAL")
    I = mnist_ones()
    G, Gp = generated(len(I))
    sets = {"mnist": I, "mnist_twin": twin(I), "generated": G, "generated_twin": twin(G)}
    centres = RNG.choice(len(I), 1500, replace=False)
    hd, scores, keep = {}, {}, {}
    for name, X in sets.items():
        D = squareform(pdist(X))
        nn = np.sort(D, 1)[:, 1]
        hd[name] = {"median_nearest_neighbour": round(float(np.median(nn)), 4),
                    "median_nn_over_median_size": round(
                        float(np.median(nn) / np.median(np.linalg.norm(X - X.mean(0), axis=1))), 4),
                    "loops": loops(D, name), "cavities": cavities(X, name)}
        scores[name], pieces = shell_split(D, centres)
        if name == "mnist":
            keep["pieces"] = pieces
        del D
    for name, ref in (("mnist", "mnist_twin"), ("mnist_twin", "mnist_twin"),
                      ("generated", "generated_twin"), ("generated_twin", "generated_twin")):
        hd[name]["shells"] = summarise_shells(scores[name], scores[ref], name)
    R["high_dim"] = hd

    # save the MNIST pictures we want to look at
    s = scores["mnist"]
    top = np.argsort(-s)[:6]
    ex = []
    for t in top:
        a, b = keep["pieces"][t]
        D_c = np.linalg.norm(I[[centres[t]]] - I, axis=1)
        a = a[np.argsort(D_c[a])][:4]
        b = b[np.argsort(D_c[b])][:4]
        ex.append({"centre": int(centres[t]), "score": float(s[t]),
                   "piece_a": a.tolist(), "piece_b": b.tolist()})
    R["mnist_shell_examples"] = ex
    np.savez_compressed(OUT / "shell_scores.npz", centres=centres,
                        **{k: v for k, v in scores.items()})
    (OUT / "results.json").write_text(json.dumps(R, indent=1))
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
