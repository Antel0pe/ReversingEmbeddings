"""Warp scorecard: run the MNIST warp method on drawn 1s whose knobs are KNOWN.

Run (each `fit` is independent, so run them in parallel):
    OPENBLAS_NUM_THREADS=1 .venv/bin/python analyze_drawn_one_warp_scorecard.py fit drawn5
    ... fit drawn8 / fit drawn12 / fit mnist
    .venv/bin/python analyze_drawn_one_warp_scorecard.py score

Why: on MNIST the warp method (`warp_charts`, one super-resolved template pushed
through a polynomial warp of the plane) leaves 28% / 21% / 18% at 5 / 12 / 20
numbers and barely sees height or vertical position. Neither number can be read
without knowing what the same method does on a family of comparable richness
whose true answer is known. The 5-knob generated family is too easy (it is an
exact affine orbit). The drawn 1s (`knob_strokes`) are the missing rung:

    drawn5   cx cy height width lean                (soft edges, round ends)
    drawn8   + bend wiggle taper                    (all of them bend the stroke)
    drawn12  + bulge hook_len hook_ang bar_len      (hook and bar are NEW ink)

Same counts as MNIST (3000 train / 1500 held out), same fitting settings.

Scored:
  1. leftover   held-out relative pixel error at 5, 12, 20 warp numbers, beside
                PCA with the same count.
  2. knob card  how much of each TRUE knob's spread the warp numbers predict on
                held-out images (quadratic ridge regression, R^2). Beside it the
                same regression from PCA numbers of the same count, and from the
                five moment numbers alone.
  3. where      where each held-out image's leftover sits, measured in that
                image's own stroke frame from its moments (top end, middle,
                bottom end, beyond the ends; left of, on, right of the stroke).
                Only measurements: no pictures of MNIST.

Fits are cached in figures/drawn_one_warp_scorecard/fit_<name>.npz.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

import knob_strokes as ks
import mnist
import warp_charts as wc

OUT = Path(__file__).resolve().parent / "figures" / "drawn_one_warp_scorecard"
N_TRAIN, N_TEST = 3000, 1500
MODELS = {5: wc.model(1, True), 12: wc.model(2), 20: wc.model(3)}
KNOB_COUNT = {"drawn5": 5, "drawn8": 8, "drawn12": 12}
NAMES = ["drawn5", "drawn8", "drawn12", "wobbly_0.4", "wobbly_0.8", "sharp5", "sharp12", "mnist"]
# Edge blur (px) at which drawn 1s put the same share of their ink in part-inked
# pixels as MNIST 1s (10th/50th/90th percentile 0.28/0.38/0.57 vs 0.26/0.40/0.57),
# chosen from that image statistic alone, never from the leftover.
SHARP_SOFTNESS = 0.18


def load(name):
    """Images (n, 28, 28) float64 and true knobs (n, k) or None."""
    n = N_TRAIN + N_TEST
    if name == "mnist":
        X, y = mnist.load("train")
        return X[y == 1][:n].astype(np.float64) / 255.0, None
    if name.startswith("sharp"):          # "sharp12": drawnK with MNIST's edge sharpness
        k = int(name[5:])
        P = ks.sample(n, k, np.random.default_rng(0))
        P[:, 14] = SHARP_SOFTNESS
        with np.errstate(over="ignore"):
            return ks.render(P).astype(np.float64), P[:, :k]
    if name.startswith("wobbly"):          # "wobbly_0.3": 12 knobs + 8 small wobbles
        amp = float(name.split("_")[1])
        rng = np.random.default_rng(0)
        P = ks.sample(n, 12, rng)
        modes = np.arange(2, 6)                                  # same as make_wobbly
        W = rng.uniform(-amp, amp, (n, 8)) * np.tile(2.0 / modes, 2)
        return ks.render(np.c_[P, W]).astype(np.float64), P[:, :12]
    k = KNOB_COUNT[name]
    P = ks.sample(n, k, np.random.default_rng(0))
    return ks.render(P).astype(np.float64), P[:, :k]


def fit(name, rounds=25):
    I, P = load(name)
    tr, te = np.arange(N_TRAIN), np.arange(N_TRAIN, N_TRAIN + N_TEST)
    out = {"knobs": P if P is not None else np.zeros((0, 0))}
    for k, m in MODELS.items():
        T, THtr, THte = wc.fit_chart(I, m, tr, te, rounds=rounds)
        D = wc.decode(T, THte, m)
        out[f"T{k}"], out[f"tr{k}"], out[f"te{k}"], out[f"dec{k}"] = T, THtr, THte, D
        print(f"{name} {k:>2} numbers: held-out leftover "
              f"{wc.rel_error(I[te], D) * 100:.2f}%", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"fit_{name}.npz", **out)


# ---------------------------------------------------------------- scoring

def quad_ridge_r2(Xtr, Ytr, Xte, Yte, lams=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
    """Held-out R^2 of each column of Y from a quadratic function of X.

    The ridge strength is chosen per knob on the last fifth of the TRAINING
    rows, then refitted on all of them; the held-out rows are never seen.
    """
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12

    def feats(X):
        Z = (X - mu) / sd
        iu = np.triu_indices(Z.shape[1])
        Q = (Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]]
        return np.c_[np.ones(len(Z)), Z, Q]

    def r2(Y, pred):
        return 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)

    def solve(A, Y, lam):
        H = A.T @ A
        reg = lam * np.trace(H) / len(H) * np.eye(len(H))
        reg[0, 0] = 0.0
        return np.linalg.solve(H + reg, A.T @ Y)

    A, B = feats(Xtr), feats(Xte)
    cut = int(0.8 * len(A))
    val = np.array([r2(Ytr[cut:], A[cut:] @ solve(A[:cut], Ytr[:cut], lam))
                    for lam in lams])                       # (n_lam, n_knobs)
    best = np.array(lams)[val.argmax(0)]
    out = np.empty(Ytr.shape[1])
    for j, lam in enumerate(best):
        W = solve(A, Ytr[:, j:j + 1], lam)
        out[j] = r2(Yte[:, j:j + 1], B @ W)[0]
    return out


def pca_coords(F, tr, k):
    mu = F[tr].mean(0)
    _, _, V = np.linalg.svd(F[tr] - mu, full_matrices=False)
    return (F - mu) @ V[:k].T, mu, V[:k]


def stroke_frame(I):
    """Per image: centre, half-length, slant (dx per dy) and width, from moments."""
    P = wc.moment_chart(I)
    cx, cy, a, b, d = P.T
    vyy = d * d
    slant = (b * d) / vyy
    return cx, cy, np.sqrt(12 * vyy) / 2, slant, np.sqrt(12) * a


def where(I, R):
    """Share of squared leftover in each part of each image's own stroke frame."""
    cx, cy, half, slant, width = stroke_frame(I)
    X, Y = wc.PIX_X[None], wc.PIX_Y[None]
    t = (Y - cy[:, None, None]) / half[:, None, None]          # -1 top end, +1 bottom
    lat = (X - cx[:, None, None]) - slant[:, None, None] * (Y - cy[:, None, None])
    hw = width[:, None, None] / 2
    e = R ** 2
    tot = e.sum()
    along = {"beyond the top end": t < -1, "top quarter": (t >= -1) & (t < -0.5),
             "middle half": (t >= -0.5) & (t <= 0.5),
             "bottom quarter": (t > 0.5) & (t <= 1), "beyond the bottom end": t > 1}
    across = {"left of the stroke": lat < -hw, "on the stroke": np.abs(lat) <= hw,
              "right of the stroke": lat > hw}
    out = {"along": {k: float(e[m].sum() / tot) for k, m in along.items()},
           "across": {k: float(e[m].sum() / tot) for k, m in across.items()},
           "missing ink (real has more)": float(e[R > 0].sum() / tot)}
    # the two corners where a hook (top left) or a base bar (bottom, sideways) lives
    out["top-left corner"] = float(e[(t < -0.5) & (lat < -hw)].sum() / tot)
    out["bottom, off the stroke"] = float(e[(t > 0.5) & (np.abs(lat) > hw)].sum() / tot)
    return out


def score():
    res = {}
    for name in NAMES:
        path = OUT / f"fit_{name}.npz"
        if not path.exists():
            print(f"missing {path.name}, skipping")
            continue
        Z = np.load(path)
        I, P = load(name)
        tr, te = np.arange(N_TRAIN), np.arange(N_TRAIN, N_TRAIN + N_TEST)
        F = I.reshape(len(I), -1)
        r = {"leftover": {}, "pca_leftover": {}, "per_image_leftover": {},
             "where": {}}
        for k in MODELS:
            D = Z[f"dec{k}"]
            r["leftover"][k] = wc.rel_error(I[te], D)
            per = (np.linalg.norm((I[te] - D).reshape(len(te), -1), axis=1)
                   / np.linalg.norm(F[te], axis=1))
            r["per_image_leftover"][k] = {q: float(np.percentile(per, q))
                                         for q in (10, 50, 90, 99)}
            r["per_image_leftover"][k]["max"] = float(per.max())
            C, mu, V = pca_coords(F, tr, k)
            r["pca_leftover"][k] = wc.rel_error(F[te], mu + C[te] @ V)
            r["where"][k] = where(I[te], I[te] - D)
        if P is not None and P.size:
            names = ks.NAMES[:P.shape[1]]
            card = {}
            mom = wc.moment_chart(I)
            card["5 moment sums"] = quad_ridge_r2(mom[tr], P[tr], mom[te], P[te])
            for k in MODELS:
                card[f"{k} warp numbers"] = quad_ridge_r2(Z[f"tr{k}"], P[tr],
                                                          Z[f"te{k}"], P[te])
                C, _, _ = pca_coords(F, tr, k)
                card[f"{k} PCA numbers"] = quad_ridge_r2(C[tr], P[tr], C[te], P[te])
            sh = np.random.default_rng(1).permutation(len(te))
            card["20 warp numbers, SHUFFLED"] = quad_ridge_r2(
                Z["tr20"], P[tr], Z["te20"][sh], P[te])
            r["knob_r2"] = {src: dict(zip(names, map(float, v)))
                            for src, v in card.items()}
        res[name] = r

        print(f"\n=== {name} ===")
        for k in MODELS:
            pi = r["per_image_leftover"][k]
            print(f"  {k:>2} numbers: warp {r['leftover'][k] * 100:6.2f}%   "
                  f"PCA {r['pca_leftover'][k] * 100:6.2f}%   per image median "
                  f"{pi[50] * 100:.1f}%, 90th {pi[90] * 100:.1f}%")
        if "knob_r2" in r:
            names = list(next(iter(r["knob_r2"].values())))
            print("  " + " " * 26 + "".join(f"{n[:9]:>10}" for n in names))
            for src, row in r["knob_r2"].items():
                print(f"  {src:>26}" + "".join(f"{row[n]:>10.2f}" for n in names))
        w = r["where"][20]
        print("  leftover at 20 numbers, along:",
              ", ".join(f"{k} {v * 100:.0f}%" for k, v in w["along"].items()))
        print("                         across:",
              ", ".join(f"{k} {v * 100:.0f}%" for k, v in w["across"].items()),
              f"| missing ink {w['missing ink (real has more)'] * 100:.0f}%",
              f"| top-left {w['top-left corner'] * 100:.0f}%",
              f"| bottom off-stroke {w['bottom, off the stroke'] * 100:.0f}%")
    (OUT / "scorecard.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'scorecard.json'}")


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit(sys.argv[2])
    else:
        score()
