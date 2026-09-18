"""What the shape of a path between two 1s controls.

Hold A and B fixed -- a hard-left-leaning 1 and a right-leaning one -- and vary
the arc between them. Three points pin a quadratic, so the whole family is
parameterised by where the halfway point sits: a magnitude (bulge) and a
direction. The two have completely different effects.

Writes figures/bulge_size.png, bulge_size_extended.png, bulge_direction.png.
Takes a few minutes; the distance-to-data query is the expensive part.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.sparse.csgraph import connected_components

import mnist, paths

YY, XX = np.mgrid[0:28, 0:28]
OUT = "figures"


def tilt(I):
    t = I.sum(); ax = (XX * I).sum() / t; ay = (YY * I).sum() / t
    dx, dy = XX - ax, YY - ay
    m20 = (dx*dx*I).sum()/t; m02 = (dy*dy*I).sum()/t; m11 = (dx*dy*I).sum()/t
    w, V = np.linalg.eigh(np.array([[m20, m11], [m11, m02]]))
    vx, vy = V[:, 1]
    a = np.degrees(np.arctan2(vx, vy))
    return a - 180 if a > 90 else (a + 180 if a <= -90 else a)


def setup():
    X, y = mnist.load("train")
    X = X.reshape(len(X), -1).astype(np.float32) / 255.0
    ones = X[y == 1].astype(np.float64)
    n = len(ones)
    D, G = paths.neighbour_graph(ones, 2.5)
    theta = (D + np.eye(n) * 1e9).min(1).mean()
    deg = (D <= theta).sum(1)
    _, lab = connected_components(G, directed=False)
    giant = np.where(lab == np.argmax(np.bincount(lab)))[0]
    ok = giant[deg[giant] >= 12]
    T = np.array([tilt(ones[i].reshape(28, 28)) for i in ok])
    a, b = int(ok[np.argmin(T)]), int(ok[np.argmax(T)])
    route = ones[paths.geodesic(G, D, a, b)]
    sq = (ones ** 2).sum(1)

    def near(F):
        F = np.clip(np.atleast_2d(F), 0, None)
        return np.array([np.sqrt(np.maximum((f**2).sum() + sq - 2*ones@f, 0)).min() for f in F])

    return ones, D, theta, a, b, route, near


def strip(fig, gridspec, rows, ts, theta, near, ncol):
    """One image row per variation, distance printed under every frame, never beside it."""
    for r, (name, F, colour, heavy) in enumerate(rows):
        d = near(F)
        for c in range(ncol):
            ax = fig.add_subplot(gridspec[r, c])
            ax.imshow(np.clip(F[c], 0, 1).reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5, -0.11, f"{d[c]:.1f}", transform=ax.transAxes, ha="center", va="top",
                    fontsize=9, weight="bold",
                    color="#1a7f37" if d[c] <= theta else "#b3261e")
            for sp in ax.spines.values():
                sp.set_edgecolor(colour); sp.set_linewidth(1.6 if heavy else 0.9)
            if r == 0:
                ax.set_title(f"t = {ts[c]:.2f}", fontsize=10, pad=7)
            if c == 0:
                ax.text(-0.16, 0.5, name, transform=ax.transAxes, ha="right", va="center",
                        fontsize=10.5, color=colour, linespacing=1.5,
                        weight="bold" if heavy else "normal")


def main():
    ones, D, theta, ai, bi, route, near = setup()
    A, B = ones[ai], ones[bi]
    s = paths.arclength_param(route)
    mid = (A + B) / 2
    gm = np.array([np.interp(0.5, s, route[:, j]) for j in range(784)])
    h0, u0 = paths.bulge_of(A, B, gm)
    print(f"A tilt {tilt(A.reshape(28,28)):+.1f}  B tilt {tilt(B.reshape(28,28)):+.1f}  "
          f"chord {D[ai,bi]:.2f}  manifold bulge h0 = {h0:.2f}")

    NC = 9
    ts = np.linspace(0, 1, NC)
    real = np.array([[np.interp(v, s, route[:, j]) for j in range(784)] for v in ts])
    arc = lambda f: paths.arc_through(A, B, mid + f * h0 * u0, ts)

    # --- figure 1: bulge size, baselines first then variations -------------
    rows = [("REAL MANIFOLD ROUTE\n(the target)", real, "#1a7f37", True),
            ("h = 0\nstraight line\n(pixel averaging)", arc(0.0), "#b3261e", True)]
    rows += [(f"h = {f:g} x h0\nbulge {f*h0:.1f}", arc(f), "#333", False) for f in [0.5, 1.0, 1.5, 2.5]]
    rows += [("h = -1 x h0\nbulge the OTHER way", arc(-1.0), "#7b1fa2", False)]
    R = len(rows)
    fig = plt.figure(figsize=(1.62*NC + 3.0, 1.78*R))
    gs = fig.add_gridspec(R, NC, left=0.215, right=0.985, top=0.885, bottom=0.055,
                          wspace=0.06, hspace=0.30)
    strip(fig, gs, rows, ts, theta, near, NC)
    fig.text(0.5, 0.955, "BULGE SIZE:  same two 1s, arcs of increasing curvature between them",
             ha="center", fontsize=15, weight="bold")
    fig.text(0.5, 0.925, "A = a 1 leaning 35 degrees left  ->  B = a 1 leaning 10 degrees right. "
                         "Every row starts and ends at the same two images.", ha="center", fontsize=11)
    fig.text(0.5, 0.903, "The number under each frame is its distance to the nearest REAL 1. "
                         f"Green means still on the manifold (under {theta:.2f}).",
             ha="center", fontsize=10, style="italic", color="#444")
    fig.savefig(f"{OUT}/bulge_size.png", dpi=135, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote bulge_size.png")

    # --- figure 2: push it much further, plus the trade-off ----------------
    real21 = np.array([[np.interp(v, s, route[:, j]) for j in range(784)] for v in np.linspace(0, 1, 21)])
    rows = [("REAL MANIFOLD\n(the target)", real, "#1a7f37", True),
            ("h = 0\nstraight line", arc(0.0), "#b3261e", True)]
    rows += [(f"h = {f:g} x h0", arc(f), "#333", False) for f in [1.0, 2.5, 4.0, 6.0, 10.0]]
    R = len(rows)
    fig = plt.figure(figsize=(1.62*NC + 3.4, 1.78*R + 5.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.78*R, 5.0], hspace=0.13,
                             left=0.20, right=0.985, top=0.905, bottom=0.055)
    strip(fig, outer[0].subgridspec(R, NC, wspace=0.06, hspace=0.30), rows, ts, theta, near, NC)

    ax = fig.add_subplot(outer[1])
    F = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 4.0, 6.0, 10.0, 20.0]
    tt = np.linspace(0, 1, 21)
    gh = [paths.ghosting(paths.arc_through(A, B, mid + f*h0*u0, tt)).mean() for f in F]
    dd = [near(paths.arc_through(A, B, mid + f*h0*u0, tt)).mean() for f in F]
    ax.plot(F, gh, "o-", lw=2.6, ms=8, color="#6a1b9a", label="ghosting (fraction of ink at half strength)")
    ax.axhline(paths.ghosting(real21).mean(), ls="--", lw=2, color="#1a7f37")
    ax.text(20.4, paths.ghosting(real21).mean(), "  real manifold route", fontsize=10, color="#1a7f37", va="center")
    ax.axhline(paths.ghosting(ones[:3000]).mean(), ls=":", lw=2, color="#1a7f37")
    ax.text(20.4, paths.ghosting(ones[:3000]).mean(), "  a real MNIST 1\n  (its anti-aliasing)",
            fontsize=10, color="#1a7f37", va="center")
    ax.set_xscale("symlog", linthresh=1); ax.set_xticks(F); ax.set_xticklabels([f"{f:g}" for f in F])
    ax.set_xlabel("bulge, in multiples of h0", fontsize=12); ax.set_ylim(0, 0.62); ax.grid(alpha=.3)
    ax.set_ylabel("ghosting", color="#6a1b9a", fontsize=12); ax.tick_params(axis="y", labelcolor="#6a1b9a")
    ax2 = ax.twinx()
    ax2.plot(F, dd, "s-", lw=2.6, ms=8, color="#b3261e", label="distance to real data")
    ax2.axhline(theta, ls="--", lw=2, color="#b3261e", alpha=.6)
    ax2.text(20.4, theta, "  THETA", fontsize=10, color="#b3261e", va="center")
    ax2.set_ylabel("mean distance to the nearest real 1", color="#b3261e", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#b3261e"); ax2.set_yscale("log")
    ax.set_title("THE TRADE-OFF: more bulge always removes ghosting, and always costs distance",
                 fontsize=13, pad=12)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=11)
    fig.text(0.5, 0.968, "BULGE SIZE, PUSHED FURTHER:  does more curvature keep removing the smudge?",
             ha="center", fontsize=16, weight="bold")
    fig.text(0.5, 0.941, "Yes -- monotonically, out to 20x. But the stroke stops ghosting only by swinging "
                         "to the wrong angle. The manifold gets both; one arc never does.",
             ha="center", fontsize=11.5)
    fig.savefig(f"{OUT}/bulge_size_extended.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote bulge_size_extended.png")

    # --- figure 3: direction, at CONSTANT bulge ----------------------------
    e = (B - A) / np.linalg.norm(B - A)
    rng = np.random.default_rng(1)
    v = rng.normal(size=784); v -= np.dot(v, e)*e; v -= np.dot(v, u0)*u0; v /= np.linalg.norm(v)
    U = lambda p: np.cos(np.radians(p))*u0 + np.sin(np.radians(p))*v
    half = lambda p: paths.arc_through(A, B, mid + h0*U(p), [0.5])

    FW, FH = 15.2, 16.2
    fig = plt.figure(figsize=(FW, FH))
    CY, RX = 0.478, 0.335
    RY = RX*FW/FH; IMG = 0.098; IMGY = IMG*FW/FH
    fine = np.arange(0, 361, 4.0)
    q = np.array([near(half(p))[0] for p in fine])
    norm = Normalize(1.0, 6.2); cmap = plt.get_cmap("RdYlGn_r")

    PW = 0.315; PH = PW*FW/FH
    axp = fig.add_axes([0.5-PW/2, CY-PH/2, PW, PH], projection="polar")
    axp.scatter(np.radians(fine), np.ones_like(fine), c=q, cmap=cmap, norm=norm, s=90, zorder=3)
    axp.plot(np.radians(fine), np.ones_like(fine), lw=1.0, color="#999", zorder=2)
    axp.plot([0], [1.0], "o", ms=20, mfc="none", mec="#1a7f37", mew=3.0, zorder=6)
    axp.plot([0], [0], "o", ms=9, color="#111", zorder=6)
    axp.text(np.radians(200), 0.24, "the straight\nline A-B\n(the pole)", fontsize=10, ha="center")
    axp.set_rlim(0, 1.42); axp.set_rticks([]); axp.grid(alpha=.3)
    axp.set_thetagrids(range(0, 360, 45), fontsize=10)
    axp.set_title(f"the rope path: radius is CONSTANT at {h0:.1f}\ncolour = distance to the nearest real 1",
                  fontsize=11, pad=18)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axp, fraction=0.045, pad=0.16, shrink=0.78)
    cb.set_label("distance to nearest real 1", fontsize=10)
    cb.ax.axhline(theta, color="#1a7f37", lw=2.5)

    for k in range(16):
        p = k*360/16; r = np.radians(p)
        img = half(p)[0]; d = near([img])[0]; good = d <= theta
        ax = fig.add_axes([0.5+RX*np.cos(r)-IMG/2, CY+RY*np.sin(r)-IMGY/2, IMG, IMGY])
        ax.imshow(np.clip(img, 0, 1).reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#1a7f37" if good else "#b3261e"); sp.set_linewidth(3.2 if good else 1.2)
        ax.text(0.5, -0.13, f"angle {int(p)}°", transform=ax.transAxes, ha="center", va="top",
                fontsize=10.5, color="#1a7f37" if good else "#444",
                weight="bold" if good else "normal")

    axl = fig.add_axes([0.075, 0.075, 0.855, 0.145])
    axl.plot(fine, q, lw=2.8, color="#b3261e")
    axl.axhline(theta, ls="--", lw=2, color="#1a7f37")
    axl.text(362, theta, "  THETA", color="#1a7f37", fontsize=10, va="center")
    axl.plot([0], [q[0]], "o", ms=11, color="#1a7f37")
    axl.annotate("only here", xy=(0, q[0]), xytext=(38, 3.6), fontsize=11, color="#1a7f37",
                 weight="bold", arrowprops=dict(arrowstyle="->", color="#1a7f37", lw=1.7))
    axl.set_xlim(-4, 378); axl.set_xticks(range(0, 361, 45)); axl.grid(alpha=.3)
    axl.set_xlabel("angle around the pole (degrees)", fontsize=11.5)
    axl.set_ylabel("distance to\nnearest real 1", fontsize=11)
    axl.set_title("the same thing as a plain graph: rope length is fixed, only the angle changes", fontsize=11.5)
    fig.text(0.5, 0.977, "BULGE DIRECTION:  the rope length never changes -- only the angle does",
             ha="center", fontsize=16.5, weight="bold")
    fig.text(0.5, 0.953, f"The straight line A-B is the pole. The arc's halfway point is tied {h0:.1f} away "
                         "from it and walked once around.", ha="center", fontsize=11.5)
    fig.text(0.5, 0.931, "Only the direction of the bulge changes. Green border = the halfway image is still "
                         "a valid 1.", ha="center", fontsize=11, style="italic", color="#444")
    fig.savefig(f"{OUT}/bulge_direction.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote bulge_direction.png")

    rand = []
    for _ in range(200):
        w = rng.normal(size=784); w -= np.dot(w, e)*e; w /= np.linalg.norm(w)
        rand.append(near(paths.arc_through(A, B, mid + h0*w, [0.5]))[0])
    print(f"\n200 random perpendicular directions: best {min(rand):.2f}, median {np.median(rand):.2f}")
    print(f"the manifold's own direction: {near(half(0))[0]:.2f}")


if __name__ == "__main__":
    main()
