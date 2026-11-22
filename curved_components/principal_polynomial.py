"""Principal polynomial analysis (Laparra et al. 2014): PCA with bent components.

PCA keeps one straight direction at a time. This method keeps one BENT direction
at a time:
  1. Find PCA's best straight direction in what is left; each picture gets one
     number a = its position along that direction.
  2. Predict everything ELSE about the picture (all the other directions) as a
     polynomial in that same number: c1*a + c2*a^2 + ... + cd*a^d.
     That bends the straight direction into a curve.
  3. Subtract the curve and repeat on what is left.

Degree 1 is exactly PCA. If the bending soaks up what PCA would have spent extra
components on, the bent version needs FEWER components for the same miss.

Its known weakness: step 2 needs everything else to be a single-valued function
of the number a. On a loop, the same a occurs twice (going out and coming back),
so no polynomial can predict both. The circle tests that directly.

Run:  .venv/bin/python curved_components/principal_polynomial.py
"""

import numpy as np

from data import first_below, load, miss, pca_misses, save


def powers(a, d, scale):
    z = np.clip(a / scale, -1.0, 1.0)      # never extrapolate the polynomial past the training range
    return np.stack([z ** k for k in range(1, d + 1)], 1)


def fit(train, n_comp, degree, ridge=1e-8):
    mu = train.mean(0)
    R = train - mu
    steps = []
    for _ in range(n_comp):
        e = np.linalg.svd(R, full_matrices=False)[2][0]
        a = R @ e
        scale = np.abs(a).max() + 1e-12     # training range of the number
        rest = R - np.outer(a, e)
        if degree > 1:
            Phi = np.c_[np.ones(len(a)), powers(a, degree, scale)[:, 1:]]   # constant + a^2..a^d
            W = np.linalg.solve(Phi.T @ Phi + ridge * np.eye(Phi.shape[1]), Phi.T @ rest)
        else:
            W = None
        steps.append((e, scale, W))
        R = rest - (Phi @ W if W is not None else 0)
    return mu, steps


def held_out_misses(mu, steps, degree, test):
    """Miss after keeping 0, 1, 2, ... bent components (miss = what is still left)."""
    R = test - mu
    out = [miss(mu + 0 * R, test)]
    for e, scale, W in steps:
        a = R @ e
        rest = R - np.outer(a, e)
        if W is not None:
            Phi = np.c_[np.ones(len(a)), powers(a, degree, scale)[:, 1:]]
            rest = rest - Phi @ W
        R = rest
        out.append(miss(test - R, test))
    return np.array(out)


def main():
    res = {}
    for name, n_comp in [("circle", 2), ("slide", 40), ("lean", 40), ("five", 60)]:
        d = load(name)
        pca = pca_misses(d["train"], d["test"], n_comp)
        res[name + "_pca"] = pca
        line = f"{name:6s} PCA (straight): 1 comp {pca[1]:6.2f}%  5 comps {pca[min(5, n_comp)]:6.2f}%  " \
               f"<10% at {first_below(pca, 10)}  <1% at {first_below(pca, 1)}"
        print(line)
        for degree in [2, 3, 5, 9]:
            mu, steps = fit(d["train"], n_comp, degree)
            curve = held_out_misses(mu, steps, degree, d["test"])
            res[f"{name}_deg{degree}"] = curve
            print(f"{'':6s} bent, degree {degree}: 1 comp {curve[1]:6.2f}%  5 comps {curve[min(5, n_comp)]:6.2f}%  "
                  f"<10% at {first_below(curve, 10)}  <1% at {first_below(curve, 1)}")
    save("principal_polynomial", **res)
    return res


if __name__ == "__main__":
    main()
