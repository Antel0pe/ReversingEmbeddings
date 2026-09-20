"""Three figures describing the state space the grey-1 generator carves out.

  statespace_dimension.png  -- what 95%-of-variance hides: the worst-case residual
  statespace_sweeps.png     -- the five knobs, swept across their full range
  statespace_atlas.png      -- how many flat tiles cover it, and which knob is curved

Run: python make_statespace_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt, time
from matplotlib.patches import Rectangle
import grey_ones as G

OUT = "figures"
INK, GOOD, BAD, REF = "#1a1a1a", "#1a7f37", "#c62828", "#1565c0"
EPS8 = 0.11          # 1/255 per pixel over 784 px -- invisible at 8-bit
KN = ["cx", "cy", "height", "width", "lean"]
UNITS = ["px", "px", "px", "px", "deg"]
MID = G.RANGES.mean(1)


def flat(P):
    return G.render(np.atleast_2d(P)).reshape(len(np.atleast_2d(P)), -1).astype(np.float64)


# ===========================================================================
# 1. the dimension figure
# ===========================================================================
def fig_dimension():
    rng = np.random.default_rng(0)
    N = 20000
    P = G.sample(N, rng); X = flat(P)
    mu = X.mean(0); Z = X - mu
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    C = U * S; tot = (Z ** 2).sum(1)
    cum = np.cumsum(S ** 2) / (S ** 2).sum()
    k95 = int(np.searchsorted(cum, 0.95)) + 1

    K = np.arange(1, 301)
    run = np.cumsum(C[:, :300] ** 2, axis=1)
    R = np.sqrt(np.maximum(tot[:, None] - run, 0))
    mx, mn = R.max(0), R.mean(0)
    k_eps = int(np.argmax(mx < EPS8)) + 1

    fig = plt.figure(figsize=(13.5, 5.6))
    fig.text(.5, .955, "95% of the variance leaves out most of the manifold",
             ha="center", va="center", fontsize=15, fontweight="bold")
    fig.text(.5, .898, "20,000 points from the 5-knob generator. Left: reconstruction error after keeping k principal directions.  "
                       "Right: the single worst point at k=21.",
             ha="center", va="center", fontsize=9.5, color="#555")

    ax = fig.add_axes([.055, .135, .50, .70])
    ax.semilogy(K, mx, color=BAD, lw=2.2, label="WORST point (covers everything)")
    ax.semilogy(K, mn, color=REF, lw=2.2, ls="--", label="mean point (the flattering number)")
    ax.axhline(EPS8, color=GOOD, lw=1.4, ls=":")
    ax.text(298, EPS8 * 1.18, "invisible at 8-bit (1/255 per pixel)", ha="right",
            va="bottom", fontsize=8.5, color=GOOD)
    ax.axvline(k95, color="#888", lw=1.2)
    ax.text(k95 + 5, mx.max() * .62, f"k={k95}\n95% of variance", fontsize=9, color="#444")
    ax.axvline(k_eps, color=INK, lw=1.2)
    ax.text(k_eps + 5, mx.max() * .62, f"k={k_eps}\nnothing left out", fontsize=9, color=INK)
    ax.plot([k95], [mx[k95 - 1]], "o", color=BAD, ms=7)
    ax.annotate(f"worst point still off by {mx[k95-1]:.2f}", (k95, mx[k95 - 1]),
                xytext=(k95 + 40, mx[k95 - 1] * 2.6), fontsize=9, color=BAD,
                arrowprops=dict(arrowstyle="->", color=BAD, lw=1.1))
    ax.set_xlabel("k — number of principal directions kept", fontsize=10)
    ax.set_ylabel("distance from the real image  (L2 over 784 pixels)", fontsize=10)
    ax.set_xlim(0, 300); ax.legend(fontsize=9, loc="lower left", framealpha=.95)
    ax.grid(alpha=.25, which="both")

    # the worst point at k95
    run95 = (C[:, :k95] ** 2).sum(1)
    w = int(np.argmax(tot - run95))
    true = X[w]
    rec = mu + C[w, :k95] @ Vt[:k95]
    err = true - rec
    m = np.abs(err).max()
    panels = [(true, "the real 1", "gray_r", 0, 1),
              (rec, f"rebuilt from k={k95}", "gray_r", 0, 1),
              (err, f"what is missing  (L2 = {np.linalg.norm(err):.2f})", "RdBu_r", -m, m)]
    for i, (img, lab, cm, v0, v1) in enumerate(panels):
        a = fig.add_axes([.615 + i * .128, .30, .118, .40])
        a.imshow(img.reshape(28, 28), cmap=cm, vmin=v0, vmax=v1, interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
        a.text(.5, -.09, lab, transform=a.transAxes, ha="center", va="top",
               fontsize=8.6, color=INK if i < 2 else BAD, wrap=True)
    fig.text(.745, .855, "worst-reconstructed point of the 20,000",
             ha="center", va="center", fontsize=9.5, fontweight="bold")
    knobtxt = "  ".join(f"{n}={P[w,j]:.1f}{u}" for j, (n, u) in enumerate(zip(KN, UNITS)))
    fig.text(.745, .155, f"its knobs:  {knobtxt}", ha="center", va="center",
             fontsize=8.4, color="#555")
    fig.text(.745, .095, "it sits on the EDGE of the knob box — which is where\n"
                         "the points 95%-of-variance discards always live",
             ha="center", va="center", fontsize=8.6, color="#555", style="italic")
    fig.savefig(f"{OUT}/statespace_dimension.png", dpi=155)
    plt.close(fig)
    return k95, k_eps


# ===========================================================================
# 2. the sweeps
# ===========================================================================
def fig_sweeps(ncol=9):
    fig = plt.figure(figsize=(13.2, 8.3))
    fig.text(.5, .962, "The five knobs, each swept across its whole range",
             ha="center", va="center", fontsize=15, fontweight="bold")
    fig.text(.5, .916, "Other four knobs held at the middle of their range. The boxed centre column is the common reference 1; "
                       "the number under each panel is its\ndistance from that reference, in the same L2 units as "
                       "θ = 1.89 (MNIST's mean nearest-neighbour distance), drawn as the dashed line.",
             ha="center", va="center", fontsize=9.3, color="#555", linespacing=1.5)

    L, R, TOP, BOT = .085, .995, .855, .085
    rowh = (TOP - BOT) / 5
    ref = flat(MID)[0]
    for r, (name, unit) in enumerate(zip(KN, UNITS)):
        lo, hi = G.RANGES[r]
        vals = np.linspace(lo, hi, ncol)
        Pr = np.tile(MID, (ncol, 1)); Pr[:, r] = vals
        Xr = flat(Pr)
        d = np.linalg.norm(Xr - ref, axis=1)
        y0 = TOP - (r + 1) * rowh
        for c in range(ncol):
            a = fig.add_axes([L + c * (R - L - .195) / ncol, y0 + .040,
                              (R - L - .195) / ncol * .86, rowh * .60])
            a.imshow(Xr[c].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1,
                     interpolation="nearest")
            a.set_xticks([]); a.set_yticks([])
            mid_c = (ncol - 1) // 2
            for s in a.spines.values():
                s.set_edgecolor(REF if c == mid_c else "#ccc")
                s.set_linewidth(2.2 if c == mid_c else .6)
            a.text(.5, -.13, f"{vals[c]:.1f}", transform=a.transAxes, ha="center",
                   va="top", fontsize=7.6, color="#444")
            a.text(.5, -.40, f"{d[c]:.2f}", transform=a.transAxes, ha="center",
                   va="top", fontsize=7.8, fontweight="bold",
                   color=GOOD if d[c] < 1.89 else BAD)
        fig.text(L - .012, y0 + .040 + rowh * .30, f"{name}\n({unit})", ha="right",
                 va="center", fontsize=10.5, fontweight="bold", color=INK,
                 linespacing=1.4)
        # the distance profile for this knob
        a2 = fig.add_axes([R - .165, y0 + .045, .155, rowh * .55])
        a2.plot(vals, d, "-o", color=INK, ms=3, lw=1.3)
        a2.axhline(1.89, color=BAD, ls="--", lw=1.1)
        a2.set_xlim(lo, hi); a2.set_ylim(0, max(7.2, d.max() * 1.12))
        a2.tick_params(labelsize=7); a2.grid(alpha=.22)
        if r == 0:
            a2.text(.5, 1.10, "distance from reference", transform=a2.transAxes,
                    ha="center", va="bottom", fontsize=8, color="#444")
            a2.text(.99, .995, "θ=1.89", transform=a2.transAxes, ha="right",
                    va="top", fontsize=7, color=BAD)
    fig.text(.5, .030, "Across its OWN range each knob moves a 1 by:  lean 8.1   width 3.8   cx 2.8   cy 1.3   height 0.9.\n"
                       "That ranking — not the per-unit stiffness, which the ranges invert — is the shape of the state space.",
             ha="center", va="center", fontsize=9.4, color="#333", style="italic", linespacing=1.6)
    fig.savefig(f"{OUT}/statespace_sweeps.png", dpi=150)
    plt.close(fig)


# ===========================================================================
# 3. the atlas
# ===========================================================================
def atlas(eps, rng):
    LO, HI = G.RANGES[:, 0].copy(), G.RANGES[:, 1].copy()
    stack = [(LO, HI)]; leaves = []; splits = np.zeros(5, int)

    def err(lo, hi, n=24):
        A = lo + rng.random((n, 5)) * (hi - lo); B = lo + rng.random((n, 5)) * (hi - lo)
        W = np.linalg.lstsq(np.c_[A, np.ones(n)], flat(A), rcond=None)[0]
        return np.linalg.norm(flat(B) - np.c_[B, np.ones(n)] @ W, axis=1).max()

    while stack:
        lo, hi = stack.pop()
        if err(lo, hi) < eps:
            leaves.append((lo, hi)); continue
        c = (lo + hi) / 2; ext = np.empty(5)
        for k in range(5):
            a, b = c.copy(), c.copy(); a[k], b[k] = lo[k], hi[k]
            Xk = flat(np.array([a, b])); ext[k] = np.linalg.norm(Xk[1] - Xk[0])
        k = int(np.argmax(ext)); splits[k] += 1
        m = (lo[k] + hi[k]) / 2
        h1 = hi.copy(); h1[k] = m; l2 = lo.copy(); l2[k] = m
        stack += [(lo, h1), (l2, hi)]
    return leaves, splits


def fig_atlas():
    rng = np.random.default_rng(1)
    epss = [4.0, 2.0, 1.0, 0.5]
    counts, splits = [], None
    for e in epss:
        L, s = atlas(e, rng); counts.append(len(L))
        if e == 0.5: splits = s
    counts = np.array(counts)

    fig = plt.figure(figsize=(12.4, 5.0))
    fig.text(.5, .955, "The manifold is 5-dimensional but not flat — so it takes many flat pieces to cover",
             ha="center", va="center", fontsize=14.5, fontweight="bold")
    fig.text(.5, .893, "A piece is kept when one affine (straight) model of the renderer is accurate to ε everywhere inside it. "
                       "Every point is covered — nothing is left out.",
             ha="center", va="center", fontsize=9.3, color="#555")

    ax = fig.add_axes([.07, .155, .38, .655])
    ax.loglog(epss, counts, "-o", color=INK, lw=2, ms=7)
    for e, c in zip(epss, counts):
        ax.annotate(f"{c}", (e, c), xytext=(6, -12), textcoords="offset points", fontsize=9.5)
    p = np.polyfit(np.log(epss), np.log(counts), 1)[0]
    ax.set_xlabel("ε — how accurate each flat piece must be  (L2)", fontsize=10)
    ax.set_ylabel("number of flat pieces needed", fontsize=10)
    ax.invert_xaxis(); ax.grid(alpha=.25, which="both")
    ax.text(.04, .93, f"slope = {p:.2f}\na flat 5-d box would give slope 0\na curved 5-d manifold predicts −2.5",
            transform=ax.transAxes, va="top", fontsize=9, color="#333",
            bbox=dict(fc="white", ec="#ccc", boxstyle="round,pad=.45"), linespacing=1.6)

    ax2 = fig.add_axes([.575, .155, .385, .655])
    order = np.argsort(splits)[::-1]
    ax2.barh(np.arange(5), splits[order], color=[BAD if i == order[0] else "#7a7a7a" for i in order],
             height=.62)
    ax2.set_yticks(np.arange(5)); ax2.set_yticklabels([KN[i] for i in order], fontsize=10.5)
    ax2.invert_yaxis()
    for i, v in enumerate(splits[order]):
        ax2.text(v + splits.max() * .015, i, f"{v}", va="center", fontsize=9.5)
    ax2.set_xlim(0, splits.max() * 1.16)
    ax2.set_xlabel("times this knob had to be cut, at ε = 0.5", fontsize=10)
    ax2.text(.5, 1.06, "where the curvature lives", transform=ax2.transAxes,
             ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax2.grid(alpha=.25, axis="x")
    fig.text(.5, .043, f"The knob that gets cut most is the one the renderer bends along most — the state space is far from "
                       f"straight in that direction.",
             ha="center", va="center", fontsize=9.2, color="#333", style="italic")
    fig.savefig(f"{OUT}/statespace_atlas.png", dpi=150)
    plt.close(fig)
    return counts, splits


if __name__ == "__main__":
    t = time.time()
    k95, keps = fig_dimension(); print(f"dimension fig: 95%->{k95}, exact->{keps}  ({time.time()-t:.0f}s)")
    fig_sweeps(); print("sweeps fig done")
    c, s = fig_atlas(); print(f"atlas fig: counts={list(c)}  splits per knob={dict(zip(KN,s))}")
    print(f"total {time.time()-t:.0f}s")
