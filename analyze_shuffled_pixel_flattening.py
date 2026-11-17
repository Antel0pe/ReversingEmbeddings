"""Flatten generated 1s when the method is NOT told where any pixel sits (rule B).

The method receives an (n, 784) matrix whose pixel order is shuffled. It may use
only that matrix. Steps:
  1. learn which pixels are neighbours (pixels inked together across images),
  2. blur every image along those learned neighbours (the fixed "ruler"),
  3. walking-distance flattening (Isomap) to 5 numbers.
The true knobs are used ONLY to grade.

Predictions recorded before running (2026-11-16):
  P1 learned neighbours right sideways, confused vertically with along-stroke pixels
  P2 raw pixel ruler: finds lean/width/cx, misses cy and height
  P3 blur on the TRUE grid (reference, rule A): recovers cy and height
  P4 blur on LEARNED neighbours: between P2 and P3
Pass: every knob predicted from the 5 numbers with R^2 >= 0.95 AND every one of
the 5 numbers predicted from the knobs with R^2 >= 0.95.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from sklearn.manifold import Isomap
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

from grey_ones import KNOBS, RANGES, render, sample

OUT = Path("figures/generated_one_shuffled_flattening")
N = 6742          # same count as the earlier MNIST/generated flattening runs
N_NEIGH_PIX = 8   # learned neighbours per pixel (a grid pixel has 8)


def r2_cv(X, Y):
    """Cross-validated kNN R^2 of each column of Y predicted from X."""
    pred = cross_val_predict(KNeighborsRegressor(10), X, Y,
                             cv=KFold(5, shuffle=True, random_state=0))
    return [float(1 - ((Y[:, j] - pred[:, j]) ** 2).mean() / Y[:, j].var())
            for j in range(Y.shape[1])]


def graph_laplacian(nbrs, n_pix):
    W = np.zeros((n_pix, n_pix))
    for i, js in enumerate(nbrs):
        W[i, js] = 1
    W = np.maximum(W, W.T)
    return np.diag(W.sum(1)) - W


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    P = sample(N, rng)
    X_true = render(P, dtype=np.float64).reshape(N, -1)
    perm = rng.permutation(784)
    X = X_true[:, perm]                      # what the method sees
    rows, cols = np.divmod(perm, 28)         # hidden, for grading only

    # ---- step 1: learn pixel neighbours from co-inking --------------------
    live = X.std(0) > 1e-9
    C = np.full((784, 784), -np.inf)
    Cl = np.corrcoef(X[:, live].T)
    idx = np.flatnonzero(live)
    C[np.ix_(idx, idx)] = Cl
    np.fill_diagonal(C, -np.inf)
    learned = [np.argsort(-C[i])[:N_NEIGH_PIX] if live[i] else np.array([], int)
               for i in range(784)]
    cheb = [np.maximum(abs(rows[js] - rows[i]), abs(cols[js] - cols[i]))
            for i, js in enumerate(learned) if live[i]]
    cheb = np.concatenate(cheb)
    dr = np.concatenate([abs(rows[js] - rows[i]) for i, js in enumerate(learned) if live[i]])
    dc = np.concatenate([abs(cols[js] - cols[i]) for i, js in enumerate(learned) if live[i]])
    step1 = {
        "live_pixels": int(live.sum()),
        "frac_learned_are_true_neighbours": float((cheb == 1).mean()),
        "median_true_distance_px": float(np.median(cheb)),
        "frac_within_2px": float((cheb <= 2).mean()),
        "mean_row_offset": float(dr.mean()),
        "mean_col_offset": float(dc.mean()),
        "row_offset_hist": np.bincount(dr, minlength=8)[:8].tolist(),
        "col_offset_hist": np.bincount(dc, minlength=8)[:8].tolist(),
    }
    print("step 1:", json.dumps(step1), flush=True)

    # true 8-neighbour grid in the same shuffled order (rule A reference)
    true_nbrs = [np.flatnonzero((np.maximum(abs(rows - rows[i]), abs(cols - cols[i])) == 1))
                 for i in range(784)]
    L_learned = graph_laplacian(learned, 784)
    L_true = graph_laplacian(true_nbrs, 784)

    # t = 1/6 ~ blur sigma 1 px on an 8-neighbour unit grid; t = 4/6 ~ 2 px
    rulers = {"raw pixels": X}
    for t, tag in [(1 / 6, "~1px"), (4 / 6, "~2px")]:
        rulers[f"learned blur {tag}"] = X @ expm(-t * L_learned)
        rulers[f"true-grid blur {tag}"] = X @ expm(-t * L_true)

    Pn = (P - RANGES[:, 0]) / (RANGES[:, 1] - RANGES[:, 0])
    results = {"step1": step1, "runs": []}
    for name, Z in rulers.items():
        for k in (10, 20):
            t0 = time.time()
            Y = Isomap(n_neighbors=k, n_components=5).fit_transform(Z)
            knobs_from_layout = r2_cv(Y, Pn)
            layout_from_knobs = r2_cv(Pn, (Y - Y.mean(0)) / Y.std(0))
            # strangers: layout nearest neighbour not among 20 nearest in knob space
            nn_lay = NearestNeighbors(n_neighbors=2).fit(Y).kneighbors(Y)[1][:, 1]
            nn_knob = NearestNeighbors(n_neighbors=21).fit(Pn).kneighbors(Pn)[1][:, 1:]
            strangers = float(np.mean([nn_lay[i] not in nn_knob[i] for i in range(N)]))
            passed = min(knobs_from_layout) >= 0.95 and min(layout_from_knobs) >= 0.95
            run = {"ruler": name, "k": k,
                   "knobs_from_layout": dict(zip(KNOBS, knobs_from_layout)),
                   "layout_from_knobs": layout_from_knobs,
                   "strangers": strangers, "pass": passed,
                   "seconds": round(time.time() - t0, 1)}
            results["runs"].append(run)
            np.save(OUT / f"layout_{name.replace(' ', '_').replace('~', '')}_k{k}.npy", Y)
            print(json.dumps(run), flush=True)
            (OUT / "results.json").write_text(json.dumps(results, indent=1))
    np.save(OUT / "settings.npy", P)
    np.save(OUT / "perm.npy", perm)


if __name__ == "__main__":
    sys.exit(main())
