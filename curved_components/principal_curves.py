"""Principal curves (Hastie & Stuetzle 1989): bend PCA's first line through the data.

The method, in plain steps:
  1. Start with PCA's first straight line, chopped into K joined segments.
  2. Drop every picture onto its nearest point of the bent line; that gives each
     picture ONE number: how far along the line it lands.
  3. Move each joint of the line to the average of the pictures that landed near
     it (a sliding average along the line).
  4. Repeat, letting the averaging window shrink so the line can bend more.

Each picture is then described by one number (how far along), and rebuilt as the
point on the curve. The curve itself is stored as K pictures (the joints).

Control (labelled "told the order"): skip the fitting and join the training
pictures in their TRUE knob order. That is the best a one-number curve through
these pictures could hope to do; it is cheating and is only a ceiling.

Run:  .venv/bin/python curved_components/principal_curves.py
"""

import numpy as np

from data import load, load_spaced, miss, pca_misses, rank_agreement, save


def project(X, nodes):
    """Nearest point on the polyline for every row of X -> (arc-length, points)."""
    A, B = nodes[:-1], nodes[1:]
    seg = B - A
    seglen2 = np.maximum((seg ** 2).sum(1), 1e-18)
    # parameter along each segment, clipped to the segment
    u = np.clip(((X[:, None, :] - A[None]) * seg[None]).sum(2) / seglen2, 0, 1)
    P = A[None] + u[..., None] * seg[None]
    d2 = ((X[:, None, :] - P) ** 2).sum(2)
    j = d2.argmin(1)
    cum = np.r_[0, np.cumsum(np.sqrt(seglen2))]
    lam = cum[j] + u[np.arange(len(X)), j] * np.sqrt(seglen2[j])
    return lam, P[np.arange(len(X)), j]


def fit(X, K=60, iters=60, start=0.3, end=0.015):
    mu = X.mean(0)
    v = np.linalg.svd(X - mu, full_matrices=False)[2][0]
    s = (X - mu) @ v
    nodes = mu + np.linspace(s.min(), s.max(), K)[:, None] * v
    for f in np.geomspace(start, end, iters):
        lam, _ = project(X, nodes)
        L = lam.max() - lam.min()
        grid = np.linspace(lam.min(), lam.max(), K)
        w = np.exp(-0.5 * ((grid[:, None] - lam[None]) / (f * L)) ** 2)
        nodes = (w @ X) / w.sum(1, keepdims=True)
    return nodes


def circular_agreement(lam, angle):
    """For a closed loop the start point is arbitrary: best rank agreement over start points."""
    return max(rank_agreement(lam, (angle - s) % (2 * np.pi)) for s in np.linspace(0, 2 * np.pi, 72))


def main():
    rows = []
    for name in ["circle", "slide", "lean"]:
        d = load(name)
        nodes = fit(d["train"])
        lam, rebuilt = project(d["test"], nodes)
        order = np.argsort(d["t_train"])
        lam_c, rebuilt_c = project(d["test"], d["train"][order])
        agree = circular_agreement if name == "circle" else rank_agreement
        pca = pca_misses(d["train"], d["test"], 2)
        r = dict(name=name, curve_miss=miss(rebuilt, d["test"]), curve_order=agree(lam, d["t_test"]),
                 told_miss=miss(rebuilt_c, d["test"]), told_order=agree(lam_c, d["t_test"]),
                 pca1=pca[1], pca2=pca[2])
        rows.append(r)
        save(f"principal_curve_{name}", nodes=nodes, lam=lam, t_test=d["t_test"])
        print(f"{name:7s} principal curve (1 number): held-out miss {r['curve_miss']:6.2f}%  "
              f"knob-order agreement {r['curve_order']:.3f} | told the order: miss {r['told_miss']:6.2f}% "
              f"agreement {r['told_order']:.3f} | PCA 1 straight: {r['pca1']:6.2f}%  2 straight: {r['pca2']:6.2f}%")
    print("\nHow far apart can the training pictures be? (held-out = every position in between)")
    spacing = {}
    for name, base, unit in [("slide", 0.25, "px"), ("lean", 0.5, "deg")]:
        out = []
        for every in [2, 4, 8, 12, 16]:
            d = load_spaced(name, every)
            nodes = fit(d["train"], K=max(60, 2 * len(d["train"])))
            lam, rebuilt = project(d["test"], nodes)
            order = np.argsort(d["t_train"])
            _, rebuilt_c = project(d["test"], d["train"][order])
            out.append((every * base, len(d["train"]), miss(rebuilt, d["test"]),
                        rank_agreement(lam, d["t_test"]), miss(rebuilt_c, d["test"])))
            print(f"  {name:5s} pictures {every * base:4.1f} {unit} apart ({len(d['train']):2d} seen): "
                  f"curve miss {out[-1][2]:6.2f}%  order agreement {out[-1][3]:.3f}  | told the order {out[-1][4]:6.2f}%")
        spacing[name] = np.array(out)
    save("principal_curve_spacing", **spacing)
    return rows, spacing


if __name__ == "__main__":
    main()
