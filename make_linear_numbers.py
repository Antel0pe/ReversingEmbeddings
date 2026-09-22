"""Two figures.

  knob_patterns.png    -- knobs do not own pixels; they own PATTERNS on shared pixels
  lean_in_pcs.png      -- the lean axis written in linear-dimension numbers (PC coordinates):
                          a harmonic series of cosines, and the horseshoe it makes

Run: python make_linear_numbers.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import grey_ones as G

OUT = "figures"
INK, GOOD, BAD, REF = "#1a1a1a", "#1a7f37", "#c62828", "#1565c0"
KN = G.KNOBS
C0 = np.array([14.5, 14.5, 19.75, 3.2, 0.0])
H = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-2])
WAVE = ["#1565c0", "#c62828", "#d08a00", "#6a3d9a"]


def jac(t):
    J = np.empty((784, 5))
    for k in range(5):
        p = np.tile(t, (2, 1)); p[0, k] -= H[k]; p[1, k] += H[k]
        im = G.render(p).reshape(2, -1).astype(np.float64); J[:, k] = (im[1] - im[0]) / (2 * H[k])
    return J


def fig_patterns():
    fig = plt.figure(figsize=(13.4, 7.4))
    fig.text(.5, .958, "The knobs don't own pixels — they own patterns on shared pixels",
             ha="center", va="center", fontsize=15, fontweight="bold")
    fig.text(.5, .905, "Each image is what a tiny nudge of one knob does to every pixel (red = gains ink, blue = loses ink). "
                       "cx and width change the SAME edge columns; cy and height the SAME\nend rows. "
                       "At the upright 1 they are still perfectly independent, because the patterns are perpendicular: cx pushes the edges opposite ways, width the same way.",
             ha="center", va="center", fontsize=9.3, color="#555", linespacing=1.55)
    rows = [("upright 1\n(lean 0°)", C0), ("leaning 1\n(lean 30°)", np.r_[C0[:4], 30.0])]
    for r, (lab, t) in enumerate(rows):
        J = jac(t); S = np.abs(J) > 1e-6
        y0 = .50 - r * .40
        for k in range(5):
            ax = fig.add_axes([.14 + k * .168, y0, .115, .115 * 13.4 / 7.4])
            v = J[:, k] / np.abs(J[:, k]).max()
            ax.imshow(v.reshape(28, 28), cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            ax.contour(G.render(t)[0], levels=[.5], colors="#999", linewidths=.6)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.text(.5, 1.06, KN[k], transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=12, fontweight="bold")
            ax.text(.5, -.05, f"{S[:, k].sum()} pixels", transform=ax.transAxes, ha="center",
                    va="top", fontsize=8.8)
        fig.text(.12, y0 + .104, lab, ha="right", va="center", fontsize=10.5, fontweight="bold",
                 linespacing=1.4)
    fig.text(.5, .045, "Grey outline = the stroke itself. Once the stroke leans, every pattern moves to different pixels, and cy's pattern starts to look like cx's "
                       "(a leaning stroke moved up\nalso moves sideways) — at 30° those two are no longer perpendicular. Across the whole box no pixel belongs to just one knob.",
             ha="center", va="center", fontsize=9.2, color="#333", style="italic", linespacing=1.55)
    fig.savefig(f"{OUT}/knob_patterns.png", dpi=140)
    plt.close(fig)


def fig_lean():
    rng = np.random.default_rng(0)
    P = G.sample(20000, rng)
    X = G.render(P).reshape(20000, -1).astype(np.float64); mu = X.mean(0)
    Vt = np.linalg.svd(X - mu, full_matrices=False)[2][:40]
    L = np.linspace(-10, 35, 451); PP = np.tile(C0, (451, 1)); PP[:, 4] = L
    Z = (G.render(PP).reshape(451, -1).astype(np.float64) - mu) @ Vt.T
    s = (L + 10) / 45
    ticks = np.arange(-10, 36, 5)

    fig = plt.figure(figsize=(14.6, 8.4))
    fig.text(.5, .962, "The lean axis written in linear-dimension numbers: a harmonic series",
             ha="center", va="center", fontsize=15.5, fontweight="bold")
    fig.text(.5, .915, "Walk the plain upright 1 along the lean axis from −10° to 35° and record its principal-axis coordinates "
                       "zₖ = PCₖ · (image − mean image).\nSolid = measured. Dashed = a single cosine fitted to each. "
                       "Like a guitar string: z1 does half a wave, z2 a whole wave, z4 one and a half.",
             ha="center", va="center", fontsize=9.5, color="#555", linespacing=1.55)

    # tick images along the top
    for i, t in enumerate(ticks):
        p = C0.copy(); p[4] = t
        ax = fig.add_axes([.063 + (t + 10) / 45 * .52 - .019, .775, .038, .038 * 14.6 / 8.4])
        ax.imshow(G.render(p)[0], cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor("#bbb"); sp.set_linewidth(.5)

    ax = fig.add_axes([.063, .12, .52, .60])
    fits = []
    for i, k in enumerate([0, 1, 3, 6]):
        z = Z[:, k]; best = None
        for m in [1, 2, 3, 4]:
            M = np.c_[np.ones_like(s), np.cos(m * np.pi * s), np.sin(m * np.pi * s)]
            co = np.linalg.lstsq(M, z, rcond=None)[0]; r = z - M @ co
            R2 = 1 - (r ** 2).sum() / ((z - z.mean()) ** 2).sum()
            if best is None or R2 > best[0]: best = (R2, m, co, M @ co)
        R2, m, co, fit = best
        amp = np.hypot(co[1], co[2])
        fits.append((k + 1, m, amp, R2))
        ax.plot(L, z, color=WAVE[i], lw=2.4, label=f"z{k+1}: {m} half-wave{'s' if m > 1 else ''}, R² {R2:.2f}")
        ax.plot(L, fit, color=WAVE[i], lw=1.2, ls="--")
    for t in ticks: ax.axvline(t, color="#eee", lw=.8, zorder=0)
    ax.axhline(0, color="#ccc", lw=.8)
    ax.set_xlim(-10, 35); ax.set_xlabel("lean (degrees) — the intrinsic coordinate", fontsize=10.5)
    ax.set_ylabel("linear-dimension number zₖ", fontsize=10.5)
    ax.legend(fontsize=9, loc="upper center", ncol=2, framealpha=.95)
    ax.tick_params(labelsize=9)

    ax2 = fig.add_axes([.655, .12, .31, .60])
    sc = ax2.scatter(Z[:, 0], Z[:, 1], c=L, cmap="viridis", s=7, zorder=3)
    co = np.polyfit(Z[:, 0], Z[:, 1], 2)
    xx = np.linspace(Z[:, 0].min(), Z[:, 0].max(), 100)
    ax2.plot(xx, np.polyval(co, xx), color=BAD, ls="--", lw=1.3, zorder=2)
    for t in ticks:
        j = np.argmin(np.abs(L - t))
        ax2.annotate(f"{t}°", (Z[j, 0], Z[j, 1]), xytext=(5, 3), textcoords="offset points",
                     fontsize=8, color=INK)
    ax2.set_xlabel("z1", fontsize=10.5); ax2.set_ylabel("z2", fontsize=10.5)
    ax2.set_title("z1 against z2: the horseshoe", fontsize=11.5, fontweight="bold", pad=8)
    ax2.text(.5, .04, f"red dashed: z2 ≈ {co[0]:.2f}·z1² {co[2]:+.2f}", transform=ax2.transAxes,
             ha="center", va="bottom", fontsize=9, color=BAD)
    ax2.set_aspect("equal", adjustable="datalim"); ax2.grid(alpha=.2)
    fig.text(.5, .030, "Half a wave against a whole wave traces a U — that is the horseshoe every lean sheet showed. "
                       "Three numbers (z1, z2, z4) carry 90% of this axis; the other 10% is spread over ~20 more, each a faster, smaller wave.",
             ha="center", va="center", fontsize=9.2, color="#333", style="italic")
    fig.savefig(f"{OUT}/lean_in_pcs.png", dpi=135)
    plt.close(fig)
    return fits


if __name__ == "__main__":
    fig_patterns()
    for f in fig_lean(): print("z%d  %d half-waves  amplitude %.2f  R2 %.3f" % f)
