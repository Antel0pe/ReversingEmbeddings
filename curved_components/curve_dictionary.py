"""Symbolic-regression style: build the pictures from a LIST OF CURVE SHAPES.

This is the user's "define a basic set of curves and fit those instead of
straight lines" idea, in the form SINDy / sparse symbolic regression uses
(PySR itself needs Julia, which is not installed; this is the same idea with
a fixed list):

    picture(t) = average picture + f1(t) * pattern1 + f2(t) * pattern2 + ...

where every f is picked from a list of ready-made curve shapes of the ONE
number t (scaled to run from -1 to 1 over the sweep):
    powers      t, t^2, ..., t^5
    waves       sin(w t), cos(w t) for 160 speeds w from 0.25 to 40
    growth      exp(a t) for a = +-0.5 ... +-5
Greedy: add whichever shape explains the most of what is still unexplained
(across all 784 pixels at once), refit, repeat. Count how many shapes are
needed versus how many straight PCA components.

IMPORTANT difference from the other tests: this one is TOLD the knob value t.
PCA is not. A second run replaces the true knob with the principal curve's own
number (found without any knob values), to see what is lost without the cheat.

Two held-out tests:
    between  train on every other position, test the ones in between
    outside  train only on the middle third, test everything outside it

Run:  .venv/bin/python curved_components/curve_dictionary.py
"""

import numpy as np

import principal_curves
from data import first_below, load, miss, pca_misses, save

SPEEDS = np.linspace(0.25, 40, 160)
GROWTH = np.r_[-np.geomspace(0.5, 5, 6), np.geomspace(0.5, 5, 6)]


def library(s):
    cols, names = [], []
    for p in range(1, 6):
        cols.append(s ** p); names.append(f"t^{p}")
    for w in SPEEDS:
        cols.append(np.sin(w * s)); names.append(f"sin({w:.2f}t)")
        cols.append(np.cos(w * s)); names.append(f"cos({w:.2f}t)")
    for a in GROWTH:
        cols.append(np.exp(a * s)); names.append(f"exp({a:+.2f}t)")
    return np.stack(cols, 1), names


def greedy(s_tr, X_tr, n_terms):
    F, names = library(s_tr)
    F = np.c_[np.ones(len(s_tr)), F]
    chosen = [0]
    for _ in range(n_terms):
        Q = np.linalg.qr(F[:, chosen])[0]
        R = X_tr - Q @ (Q.T @ X_tr)
        G = F - Q @ (Q.T @ F)
        norm = np.linalg.norm(G, axis=0)
        score = np.where(norm > 1e-8 * np.sqrt(len(s_tr)), np.linalg.norm(G.T @ R, axis=1) / np.maximum(norm, 1e-300), 0)
        score[chosen] = 0
        chosen.append(int(score.argmax()))
    return chosen, ["constant"] + names


def held_out_curve(s_tr, X_tr, s_te, X_te, chosen):
    Ftr, Fte = np.c_[np.ones(len(s_tr)), library(s_tr)[0]], np.c_[np.ones(len(s_te)), library(s_te)[0]]
    out = []
    for m in range(1, len(chosen) + 1):
        W = np.linalg.lstsq(Ftr[:, chosen[:m]], X_tr, rcond=None)[0]
        out.append(miss(Fte[:, chosen[:m]] @ W, X_te))
    return np.array(out)                    # index 0 = average picture only, index m = m shapes


def scale(t, lo, hi):
    return 2 * (t - lo) / (hi - lo) - 1


def main(n_terms=40):
    res = {}
    for name in ["slide", "lean"]:
        d = load(name)
        lo, hi = min(d["t_train"].min(), d["t_test"].min()), max(d["t_train"].max(), d["t_test"].max())
        str_, ste = scale(d["t_train"], lo, hi), scale(d["t_test"], lo, hi)
        pca = pca_misses(d["train"], d["test"], n_terms)
        chosen, names = greedy(str_, d["train"], n_terms)
        between = held_out_curve(str_, d["train"], ste, d["test"], chosen)
        # same but with the principal curve's own number instead of the true knob
        nodes = principal_curves.fit(d["train"])
        ltr, _ = principal_curves.project(d["train"], nodes)
        lte, _ = principal_curves.project(d["test"], nodes)
        L0, L1 = ltr.min(), ltr.max()
        chosen_u, _ = greedy(scale(ltr, L0, L1), d["train"], n_terms)
        between_u = held_out_curve(scale(ltr, L0, L1), d["train"], scale(lte, L0, L1), d["test"], chosen_u)
        # outside: train on middle third only
        allt = np.r_[d["t_train"], d["t_test"]]; allX = np.r_[d["train"], d["test"]]
        a, b = lo + (hi - lo) / 3, lo + 2 * (hi - lo) / 3
        mid = (allt >= a) & (allt <= b)
        ch_o, _ = greedy(scale(allt[mid], lo, hi), allX[mid], 12)
        outside = held_out_curve(scale(allt[mid], lo, hi), allX[mid], scale(allt[~mid], lo, hi), allX[~mid], ch_o)
        pca_out = pca_misses(allX[mid], allX[~mid], 12)
        res.update({f"{name}_pca": pca, f"{name}_between": between, f"{name}_between_unsup": between_u,
                    f"{name}_outside": outside, f"{name}_pca_outside": pca_out})
        print(f"== {name}")
        print(f"  first 10 shapes picked: {', '.join(names[c] for c in chosen[1:11])}")
        for lab, c in [("PCA, straight (no knob)", pca), ("shape list, told the knob", between),
                       ("shape list, principal-curve number", between_u)]:
            print(f"  {lab:36s}: 1 -> {c[1]:6.2f}%  5 -> {c[5]:6.2f}%  10 -> {c[10]:6.2f}%  "
                  f"<10% at {first_below(c, 10)}  <1% at {first_below(c, 1)}")
        print(f"  OUTSIDE the middle third it was trained on: shape list best {outside.min():6.2f}% "
              f"(at {outside.argmin()} shapes), PCA best {pca_out.min():6.2f}%")
    save("curve_dictionary", **res)
    return res


if __name__ == "__main__":
    main()
