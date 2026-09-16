"""Generate figures/axes.gif and figures/routes.gif.

Two routes between the same pair of 1s: along the manifold (hopping between real
neighbours) versus straight through pixel space. The motion panels are the point --
along the manifold the red/blue moves as a coherent edge (ink sliding), on the
straight line red and blue sit in unrelated places (one image fading under another).

Each row also carries the cubic that describes that manifold curve, in the curve's
own three principal axes. Twelve numbers per curve.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.sparse.csgraph import connected_components, dijkstra

import mnist, paths

AXES = ["tilt", "stroke thickness", "overall size", "horizontal position"]
K = 10
YY, XX = np.mgrid[0:28, 0:28]


def shape_features(I):
    """Tilt, thickness, size and position, straight from image moments."""
    t = I.sum(); ax = (XX * I).sum() / t; ay = (YY * I).sum() / t
    dx, dy = XX - ax, YY - ay
    m20 = (dx * dx * I).sum() / t; m02 = (dy * dy * I).sum() / t; m11 = (dx * dy * I).sum() / t
    w, V = np.linalg.eigh(np.array([[m20, m11], [m11, m02]]))
    vx, vy = V[:, 1]
    a = np.degrees(np.arctan2(vx, vy))
    a = a - 180 if a > 90 else (a + 180 if a <= -90 else a)
    return [a, t / (4 * np.sqrt(max(w[1], 1e-9))), np.sqrt(m20 + m02), ax]


def main():
    X, y = mnist.load("train")
    X = X.reshape(len(X), -1).astype(np.float32) / 255.0
    ones = X[y == 1].astype(np.float64)
    n = len(ones); sq = (ones ** 2).sum(1)
    D, G = paths.neighbour_graph(ones, 2.5)
    theta = (D + np.eye(n) * 1e9).min(1).mean()
    deg = (D <= theta).sum(1)
    _, lab = connected_components(G, directed=False)
    giant = np.where(lab == np.argmax(np.bincount(lab)))[0]

    Z = np.array([shape_features(ones[i].reshape(28, 28)) for i in range(n)])
    Z = (Z - Z.mean(0)) / Z.std(0)

    rng = np.random.default_rng(0)
    pool = giant[deg[giant] >= 12]
    chosen = {}
    for col, name in enumerate(AXES):
        best = -np.inf
        for a in rng.choice(pool, 150, replace=False):
            a = int(a)
            dd, pp = dijkstra(G, indices=[a], directed=False, return_predecessors=True)
            dd, pp = dd[0], pp[0]
            ok = np.where((D[a] >= 4.0) & (D[a] <= 7.0) & np.isfinite(dd) & (deg >= 10))[0]
            if not len(ok):
                continue
            # pick the partner that differs most on this axis and least on the others
            score = np.abs(Z[ok, col] - Z[a, col]) - 0.7 * np.abs(np.delete(Z[ok] - Z[a], col, 1)).sum(1)
            b = int(ok[np.argmax(score)])
            if score.max() > best:
                p = paths.geodesic(G, D, a, b)
                if p and 5 <= len(p) <= 16:
                    best = score.max(); chosen[name] = (a, b, p)
        if name not in chosen:
            raise SystemExit(f"no suitable pair found for axis '{name}'")

    data = {}
    for name, (a, b, p) in chosen.items():
        P = ones[p]
        s = paths.arclength_param(P)
        ts = np.linspace(0, 1, K)
        man = np.array([[np.interp(u, s, P[:, j]) for j in range(784)] for u in ts])
        lin = paths.straight_line(ones[a], ones[b], K)
        co, _, _, _, cap = paths.fit_curve(P)
        eq = "  ".join(f"c{j+1}={co[0,j]:+.1f}{co[1,j]:+.1f}t{co[2,j]:+.1f}t2{co[3,j]:+.1f}t3"
                       for j in range(2))
        near = lambda F: np.array([np.sqrt(np.maximum((f**2).sum() + sq - 2*ones@f, 0)).min() for f in F])
        data[name] = (man, lin, near(man), near(lin), eq, cap)
        print(f"{name:20s} {len(p)} hops | straight {D[a,b]:.2f} | manifold "
              f"{np.linalg.norm(np.diff(P,axis=0),axis=1).sum():.2f} | 3 axes hold {cap:.0%}")

    fig, ax = plt.subplots(len(AXES), 4, figsize=(7.6, 2.05 * len(AXES)))
    for r, name in enumerate(AXES):
        ax[r, 0].set_ylabel(name, fontsize=8)
        ax[r, 0].set_xlabel(data[name][4], fontsize=5.2, family="monospace")
        for c, t_ in enumerate(["manifold", "its motion", "straight line", "its motion"]):
            if r == 0:
                ax[r, c].set_title(t_, fontsize=8, color="#1a7f37" if c < 2 else "#b3261e")
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    H = [[ax[r, 0].imshow(data[nm][0][0].reshape(28, 28), cmap="gray", vmin=0, vmax=1),
          ax[r, 1].imshow(np.zeros((28, 28)), cmap="bwr", vmin=-1, vmax=1),
          ax[r, 2].imshow(data[nm][1][0].reshape(28, 28), cmap="gray", vmin=0, vmax=1),
          ax[r, 3].imshow(np.zeros((28, 28)), cmap="bwr", vmin=-1, vmax=1)]
         for r, nm in enumerate(AXES)]

    def frame(k):
        out = []
        for r, nm in enumerate(AXES):
            man, lin, _, _, _, _ = data[nm]
            j = min(k, K - 2)
            H[r][0].set_data(man[min(k, K - 1)].reshape(28, 28))
            d1 = (man[j + 1] - man[j]).reshape(28, 28)
            H[r][1].set_data(d1 / max(np.abs(d1).max(), 1e-9))
            H[r][2].set_data(lin[min(k, K - 1)].reshape(28, 28))
            d2 = (lin[j + 1] - lin[j]).reshape(28, 28)
            H[r][3].set_data(d2 / max(np.abs(d2).max(), 1e-9))
            out += H[r]
        return out

    fig.suptitle("Along the manifold vs straight through pixel space\n"
                 "(the cubic under each row is that manifold curve, in its own axes)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    FuncAnimation(fig, frame, frames=list(range(K)) + list(range(K - 2, 0, -1)),
                  interval=280).save("figures/axes.gif", writer=PillowWriter(fps=4))
    print("wrote figures/axes.gif")
    for nm in AXES:
        _, _, dm, dl, _, _ = data[nm]
        print(f"  {nm:20s} max distance to real data: manifold {dm.max():.2f}  straight {dl.max():.2f}")


if __name__ == "__main__":
    main()
