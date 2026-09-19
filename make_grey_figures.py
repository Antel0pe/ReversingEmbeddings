"""Figures for grey_ones.py / trace_grey.py.

grey_generator.png  -- real MNIST 1s first, then the grey generator's 1s, with the
                       statistics that set its defaults
grey_dimension.png  -- tracing the dimension from images alone: the cliff after 5,
                       the traced directions as images, and the same test on real 1s

Takes ~4 minutes.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt

import grey_ones as G, trace_grey as TG, mnist

OUT = "figures"
GREEN, RED, GREY, BLUE, PURPLE = "#1a7f37", "#c62828", "#444", "#1565c0", "#6a1b9a"


def show(ax, x, border=None, lw=1.6):
    ax.imshow(x.reshape(28, 28), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if border:
        for sp in ax.spines.values(): sp.set_edgecolor(border); sp.set_linewidth(lw)


def fig_generator():
    rng = np.random.default_rng(7)
    X, y = mnist.load("train"); M = X[y == 1].astype(np.float32) / 255
    real = M[rng.choice(len(M), 10, replace=False)]
    P = np.array([[14.5, 14.5, 20.0, w, a] for w, a in
                  [(1.9, -8), (2.6, 0), (3.2, 6), (2.2, 12), (4.4, 15), (3.0, 20), (2.4, 25), (3.8, 28), (2.0, 33), (4.0, -4)]])
    gen = G.render(P)
    FW, FH = 17.0, 8.4
    fig = plt.figure(figsize=(FW, FH))
    for r, (imgs, lab, col) in enumerate([(real, "REAL MNIST 1s\n(the reference)", GREEN),
                                          (gen, "GREY GENERATOR\n(5 knobs, coverage greys)", BLUE)]):
        gs = fig.add_gridspec(1, 10, left=0.17, right=0.985, top=1 - (1.75 + r * 2.15) / FH,
                              bottom=1 - (3.35 + r * 2.15) / FH, wspace=0.08)
        for c in range(10):
            show(fig.add_subplot(gs[0, c]), imgs[c], col, 2.0)
        fig.text(0.155, 1 - (2.55 + r * 2.15) / FH, lab, ha="right", va="center", fontsize=12,
                 weight="bold", color=col, linespacing=1.45)
    rows = [("knob", "range", "from MNIST 1s (5th-95th percentile)"),
            ("centre x, y", "14.0 - 15.0", "centre of ink within +/-0.5 px of the middle (MNIST centres every digit)"),
            ("height", "19 - 20.5 rows", "visible height is 20 rows for 98% (MNIST rescales every digit)"),
            ("stroke width", "1.8 - 4.6 px", "ink per row: 1.79 at the 5th percentile, 4.63 at the 95th"),
            ("lean", "-10 to +35 deg", "real 1s mostly lean right; + = top to the right"),
            ("visible", ">= 0.5", "70% of real ink is >= 0.5; generator gives 68%  (median value 0.86 vs 0.87)"),
            ("smudges", "none floating", "every nonzero pixel joins the stroke: 94.7% of real 1s; generator 100%")]
    for k, (a, b, c) in enumerate(rows):
        yy = 1 - (6.25 + k * 0.29) / FH
        wt = "bold" if k == 0 else "normal"
        fig.text(0.05, yy, a, fontsize=10.5, weight=wt); fig.text(0.19, yy, b, fontsize=10.5, weight=wt)
        fig.text(0.34, yy, c, fontsize=10.5, weight=wt, color="#222" if k == 0 else GREY)
    fig.text(0.5, 1 - 0.42 / FH, "A GREY GENERATOR FOR STRAIGHT 1s, WITH ITS NUMBERS TAKEN FROM MNIST",
             ha="center", fontsize=16, weight="bold")
    fig.text(0.5, 1 - 0.85 / FH, "Grey = coverage: each pixel's value is the fraction of it the stroke "
             "covers -- which is how MNIST's greys arose when LeCun et al. shrank black-and-white scans.",
             ha="center", fontsize=11.5)
    fig.text(0.5, 1 - 1.2 / FH, "MNIST had already removed 'move' and 'size': every real 1 is 20 rows "
             "tall and centred. What really varies is width and lean.", ha="center", fontsize=11,
             style="italic", color=GREY)
    fig.savefig(f"{OUT}/grey_generator.png", dpi=115, facecolor="white"); plt.close(fig)
    print("wrote grey_generator.png")


def fig_dimension():
    rng = np.random.default_rng(0)
    mid = G.RANGES.mean(1); span = G.RANGES[:, 1] - G.RANGES[:, 0]
    anchors = mid + (rng.random((12, 5)) - .5) * .6 * span
    radii = [0.25, 1.0, 4.0]; spec = {}
    for r in radii:
        spec[r] = np.mean([(lambda l: l[:10] / l.sum())(TG.spectrum(TG.local_ball(p, r, 1500, rng)[0]))
                           for p in anchors], 0)
    p0 = np.array([14.5, 14.5, 19.75, 3.0, 12.0])
    X0, _ = TG.local_ball(p0, 0.5, 2000, rng)
    lam0, U0 = TG.spectrum(X0, vectors=True); T5 = U0[:, :5]
    J = TG.jacobian(p0); Jn = J / np.linalg.norm(J, axis=0)
    cos_sub = TG.principal_cosines(T5, J)
    S = G.render(G.sample(60000, rng)).reshape(60000, -1)
    Xm, ym = mnist.load("train"); M = Xm[ym == 1].reshape(-1, 784).astype(np.float32) / 255
    KS = [10, 20, 40, 80, 160, 320, 640]
    def curve(D):
        sq = (D ** 2).sum(1); out = {k: [] for k in KS}
        for a in rng.choice(len(D), 40, replace=False):
            o = np.argsort(np.sqrt(np.maximum(sq + sq[a] - 2 * D @ D[a], 0)))
            for k in KS: out[k].append(TG.gap_dim(TG.spectrum(D[o[:k + 1]]))[0])
        return out
    cs, cm = curve(S), curve(M)
    specM = np.mean([(lambda l: l[:10] / l.sum())(TG.spectrum(M[np.argsort(((M - M[a]) ** 2).sum(1))[:81]]))
                     for a in rng.choice(len(M), 40, replace=False)], 0)
    specS = np.mean([(lambda l: l[:10] / l.sum())(TG.spectrum(S[np.argsort(((S - S[a]) ** 2).sum(1))[:81]]))
                     for a in rng.choice(len(S), 40, replace=False)], 0)

    FW, FH = 17.5, 15.2
    fig = plt.figure(figsize=(FW, FH))
    fy = lambda t: 1 - t / FH
    fig.text(0.5, fy(0.42), "TRACING THE DIMENSION FROM IMAGES ALONE:  a cliff after exactly 5 directions",
             ha="center", fontsize=16.5, weight="bold")
    fig.text(0.5, fy(0.86), "Take the valid 1s near any one 1 and ask how many independent directions "
             "they spread along (local PCA). The knobs are never used to find them -- only to check.",
             ha="center", fontsize=11.8)

    ax = fig.add_axes([0.06, fy(6.0), 0.40, (6.0 - 1.75) / FH])
    cols = {0.25: BLUE, 1.0: PURPLE, 4.0: "#999"}
    wbar = 0.26
    for j, r in enumerate(radii):
        ax.bar(np.arange(1, 11) + (j - 1) * wbar, spec[r], width=wbar, color=cols[r],
               label=f"neighbourhood radius {r:g}", zorder=3)
    ax.axvline(5.5, color=RED, lw=2, ls="--", zorder=4)
    ax.text(5.62, 0.235, "the cliff:\nafter 5", color=RED, fontsize=11, weight="bold", va="top")
    ax.set_xticks(range(1, 11)); ax.set_xlabel("direction (largest first)", fontsize=11.5)
    ax.set_ylabel("share of the local spread", fontsize=11.5); ax.grid(alpha=.3, axis="y")
    ax.legend(fontsize=10.5, loc="upper right"); ax.set_ylim(0, 0.30)
    ax.set_title("'AS MUCH DETAIL AS NEEDED': members near 12 different 1s\n"
                 "small neighbourhoods show 5 directions then nothing; big ones blur the cliff",
                 fontsize=11.5, pad=10)

    ax = fig.add_axes([0.56, fy(6.0), 0.40, (6.0 - 1.75) / FH])
    ax.plot(range(1, 11), specS, "o-", color=BLUE, lw=2.6, ms=7, label="grey generator (60,000 samples)")
    ax.plot(range(1, 11), specM, "s-", color=GREEN, lw=2.6, ms=7, label="real MNIST 1s (6,742)")
    ax.axvline(5.5, color=RED, lw=2, ls="--")
    ax.set_yscale("log"); ax.set_xticks(range(1, 11)); ax.grid(alpha=.3)
    ax.set_xlabel("direction (largest first)", fontsize=11.5)
    ax.set_ylabel("share of the local spread (log scale)", fontsize=11.5)
    ax.legend(fontsize=10.5, loc="lower left")
    ax.set_title("A FINITE SAMPLE, 80 nearest neighbours of 40 points\n"
                 "generator: a cliff after 5.   Real 1s: a smooth slope, no cliff at all",
                 fontsize=11.5, pad=10)

    fig.text(0.5, fy(7.0), "THE 5 TRACED DIRECTIONS, AS IMAGES  (red = pixel gets brighter, blue = "
             "darker, for a small step along that direction)", ha="center", fontsize=12.5, weight="bold")
    gs = fig.add_gridspec(1, 6, left=0.06, right=0.94, top=fy(7.35), bottom=fy(10.25), wspace=0.18)
    show(fig.add_subplot(gs[0, 0]), G.render(p0)[0], GREEN, 2.4)
    fig.axes[-1].set_title("the 1 they are\ntraced around", fontsize=10.5, color=GREEN, pad=6)
    for i in range(5):
        ax = fig.add_subplot(gs[0, i + 1]); v = T5[:, i].reshape(28, 28); m = np.abs(v).max()
        ax.imshow(v, cmap="RdBu_r", vmin=-m, vmax=m, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        c = np.abs(Jn.T @ T5[:, i]); j = int(np.argmax(c))
        ax.set_title(f"direction {i + 1}\n{lam0[i] / lam0.sum() * 100:.0f}% of the spread", fontsize=10.5, pad=6)
        ax.text(0.5, -0.06, f"closest knob: {G.KNOBS[j]}\n(|cos| {c[j]:.2f})", transform=ax.transAxes,
                ha="center", va="top", fontsize=10, color=GREY, linespacing=1.35)
    fig.text(0.5, fy(11.1), f"Together they span the true knob directions almost exactly: cosines "
             f"{', '.join(f'{v:.3f}' for v in cos_sub)} (1 = identical).\nBut the axes inside that 5-D "
             "space are not fixed by the data: any rotation of them is an equally good set of knobs.",
             ha="center", va="top", fontsize=11, style="italic", color=GREY, linespacing=1.55)

    ax = fig.add_axes([0.14, fy(14.6), 0.72, (14.6 - 11.95) / FH])
    for data, col, mk, lab in [(cs, BLUE, "o", "grey generator (60,000 samples)"),
                               (cm, GREEN, "s", "real MNIST 1s (6,742)")]:
        med = [np.median(data[k]) for k in KS]
        lo = [np.percentile(data[k], 25) for k in KS]; hi = [np.percentile(data[k], 75) for k in KS]
        ax.fill_between(KS, lo, hi, color=col, alpha=.15)
        ax.plot(KS, med, mk + "-", color=col, lw=2.6, ms=8, label=lab + ": typical answer, band = middle half")
    ax.axhline(5, color=RED, lw=1.6, ls="--"); ax.text(690, 5.35, "5 = the generator's\nknobs", color=RED, fontsize=10, va="bottom", ha="center")
    ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlim(8.5, 1000); ax.set_ylim(0, 12); ax.grid(alpha=.3)
    ax.set_xlabel("neighbourhood size (k nearest samples)", fontsize=11.5)
    ax.set_ylabel("dimension found\n(position of the cliff)", fontsize=11.5)
    ax.legend(fontsize=10.5, loc="upper right")
    ax.set_title("WITH A FINITE SAMPLE: the generator answers 5 at every size; real 1s wander, because "
                 "there is no cliff to find", fontsize=11.5, pad=8)
    fig.savefig(f"{OUT}/grey_dimension.png", dpi=110, facecolor="white"); plt.close(fig)
    print("wrote grey_dimension.png  | subspace cosines", np.round(cos_sub, 4))


if __name__ == "__main__":
    fig_generator()
    fig_dimension()
