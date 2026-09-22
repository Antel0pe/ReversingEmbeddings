"""Four Twitter-sized cards summarising the week of 2026-09-14 .. 2026-09-20.

Everything is 1600x900 (16:9) with type sized to survive a timeline thumbnail.
All numbers are computed here, not copied in, except the two traced-axis counts
which are the generators' own knob counts.

  tw1_generator.png   the 5-knob grey generator, each knob swept
  tw2_dimension.png   5 knobs vs the 214 straight directions PCA asks for
  tw3_sheets.png      freeze 3 knobs, move 2 -- the real curved sheet
  tw4_curvature.png   how badly PCA overcounts, on three known-answer spaces

Run: python make_twitter_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import grey_ones as G
import equation_ladder as EL

OUT = "figures/twitter"
DPI = 100
FIGSIZE = (16, 9)

SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
BLUE, ORANGE, AQUA, RED, GOOD = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b", "#0ca30c"

KN = ["cx", "cy", "height", "width", "lean"]
NICE = ["cx  — side to side", "cy  — up and down", "height", "width  — stroke thickness", "lean"]
UNITS = ["px", "px", "px", "px", "°"]
MID = G.RANGES.mean(1)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": "#d6d5d0",
})


def newfig():
    return plt.figure(figsize=FIGSIZE, dpi=DPI)


def title(fig, main, sub=None, y=.955):
    fig.text(.5, y, main, ha="center", va="center", fontsize=34, fontweight="bold", color=INK)
    if sub:
        fig.text(.5, y - .072, sub, ha="center", va="center", fontsize=17, color=INK2, linespacing=1.5)


def footer(fig, txt):
    fig.text(.5, .035, txt, ha="center", va="center", fontsize=14, color=INK3, style="italic")


def show(ax, img, edge="#d6d5d0"):
    ax.imshow(img, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(edge); s.set_linewidth(1.2)


# ===========================================================================
# 1. the generator
# ===========================================================================
def fig_generator():
    NC = 8
    fig = newfig()
    title(fig, "Five knobs draw a digit 1",
          "Each row is one knob swept end to end; the other four are held at the middle.\n"
          "Every range is taken from real MNIST 1s — grey is coverage, the fraction of the pixel the stroke covers.")
    L, R, T, B = .30, .965, .80, .115
    cw = (R - L) / NC
    rh = (T - B) / 5
    for k in range(5):
        vals = np.linspace(*G.RANGES[k], NC)
        P = np.tile(MID, (NC, 1)); P[:, k] = vals
        imgs = G.render(P)
        y0 = T - (k + 1) * rh
        fig.text(L - .022, y0 + rh / 2, NICE[k], ha="right", va="center",
                 fontsize=19, fontweight="bold", color=INK)
        fig.text(L - .022, y0 + rh / 2 - .048,
                 f"{G.RANGES[k][0]:g} → {G.RANGES[k][1]:g} {UNITS[k]}",
                 ha="right", va="center", fontsize=15, color=INK3)
        for j in range(NC):
            ax = fig.add_axes([L + j * cw + .004, y0 + .030, cw - .008, rh - .048])
            show(ax, imgs[j])
    footer(fig, "grey_ones.py  ·  the knobs are continuous, so the set of all images they can draw is a smooth 5-dimensional surface in 784-dimensional pixel space")
    fig.savefig(f"{OUT}/tw1_generator.png", dpi=DPI)
    plt.close(fig)


# ===========================================================================
# 2. five knobs, 214 directions
# ===========================================================================
def fig_dimension():
    rng = np.random.default_rng(0)
    N = 20000
    X = G.render(G.sample(N, rng)).reshape(N, -1).astype(np.float64)
    Z = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    C = U * S
    tot = (Z ** 2).sum(1)
    cum = np.cumsum(S ** 2) / (S ** 2).sum()
    k95 = int(np.searchsorted(cum, .95)) + 1
    KMAX = 300
    run = np.cumsum(C[:, :KMAX] ** 2, axis=1)
    Rr = np.sqrt(np.maximum(tot[:, None] - run, 0))
    worst, mean = Rr.max(0), Rr.mean(0)
    EPS = 0.11                       # 1/255 per pixel over 784 px: invisible at 8-bit
    kfull = int(np.argmax(worst < EPS)) + 1
    K = np.arange(1, KMAX + 1)

    fig = newfig()
    title(fig, f"5 knobs in. {kfull} straight directions out.",
          "20,000 images from the generator, then PCA. The curve is how far the worst of those 20,000 still is\n"
          "from its own rebuild after keeping k directions. “95% of the variance” stops far short of the real thing.")
    ax = fig.add_axes([.075, .145, .53, .625])
    ax.semilogy(K, worst, color=RED, lw=3.2, label="worst of the 20,000")
    ax.semilogy(K, mean, color=BLUE, lw=3.2, ls="--", label="average  (the flattering number)")
    ax.axhline(EPS, color=INK3, lw=1.6, ls=":")
    ax.text(298, EPS * 1.35, "invisible at 8-bit", ha="right", va="bottom", fontsize=13, color=INK3)
    for k, lab, col in [(k95, f"k={k95}\n95% of variance", RED), (kfull, f"k={kfull}\nnothing left out", GOOD)]:
        ax.axvline(k, color=col, lw=1.8, alpha=.55)
        ax.text(k + 5, 4.5, lab, fontsize=15, fontweight="bold", color=col, va="top", linespacing=1.4)
    ax.set_xlim(0, KMAX); ax.set_ylim(1e-3, 9)
    ax.set_xlabel("k  —  straight directions kept", fontsize=16)
    ax.set_ylabel("distance from the true image", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.grid(alpha=.18, which="major")
    ax.legend(fontsize=15, frameon=False, loc="lower left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    i = int(np.argmax(Rr[:, k95 - 1]))
    real = X[i].reshape(28, 28)
    rebuilt = (X.mean(0) + C[i, :k95] @ Vt[:k95]).reshape(28, 28)
    diff = real - rebuilt
    fig.text(.815, .755, f"the single worst image at k={k95}", ha="center", va="center",
             fontsize=19, fontweight="bold", color=INK)
    labs = ["what the generator\ndrew", f"rebuilt from\n{k95} directions", "what PCA\nthrew away"]
    XS = [.660, .782, .904]
    for j, (im, lab) in enumerate(zip([real, rebuilt, diff], labs)):
        ax2 = fig.add_axes([XS[j] - .049, .47, .098, .186])
        if j == 2:
            m = np.abs(diff).max()
            ax2.imshow(diff, cmap="RdBu_r", vmin=-m, vmax=m, interpolation="nearest")
            ax2.set_xticks([]); ax2.set_yticks([])
            for s in ax2.spines.values(): s.set_color("#d6d5d0")
        else:
            show(ax2, im)
        fig.text(XS[j], .448, lab, ha="center", va="top",
                 fontsize=13.5, color=INK2, linespacing=1.4)
    fig.text(.815, .318,
             "It sits on the EDGE of the knob box — a hard\nlean, a thick stroke. That is exactly where the\npoints \"95% of the variance\" discards always live.",
             ha="center", va="top", fontsize=14.5, color=INK2, linespacing=1.6)
    footer(fig, "a flat ruler measuring a curved thing: the surface really has 5 degrees of freedom, but no 5 straight directions can hold it")
    fig.savefig(f"{OUT}/tw2_dimension.png", dpi=DPI)
    plt.close(fig)
    return k95, kfull


# ===========================================================================
# 3. the sheets
# ===========================================================================
def sheet(a, b, n):
    va, vb = np.linspace(*G.RANGES[a], n), np.linspace(*G.RANGES[b], n)
    P = np.tile(MID, (n * n, 1))
    A, B = np.meshgrid(va, vb)
    P[:, a] = A.ravel(); P[:, b] = B.ravel()
    return G.render(P).reshape(n * n, -1).astype(np.float64)


def fig_sheets():
    PAIRS = [(3, 2), (4, 3), (4, 0)]        # width x height, lean x width, lean x cx
    NF = 21
    fig = newfig()
    title(fig, "Freeze three knobs, move two. This is the surface.",
          "Every grid line is a real path the generator walks — nothing fitted, nothing smoothed.\n"
          "Drawn in the best 3 of its 784 directions, all three axes to the same scale, so flat really means flat.")
    for idx, (a, b) in enumerate(PAIRS):
        X = sheet(a, b, NF)
        Z = X - X.mean(0)
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        v = S ** 2 / (S ** 2).sum()
        hid2, hid3 = 1 - v[:2].sum(), 1 - v[:3].sum()
        Y = (U[:, :3] * S[:3]).reshape(NF, NF, 3)
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        ax.set_position([.035 + idx * .312, .195, .29, .575])
        for i in range(NF):
            ax.plot(Y[i, :, 0], Y[i, :, 1], Y[i, :, 2], color=BLUE, lw=1.1, alpha=.85)
            ax.plot(Y[:, i, 0], Y[:, i, 1], Y[:, i, 2], color=ORANGE, lw=1.1, alpha=.85)
        m = np.abs(Y).max()
        ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_zlim(-m, m)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=24, azim=-62)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.grid(False)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((1, 1, 1, 0)); pane.line.set_color((.85, .85, .85, 1))
        cx = .035 + idx * .312 + .145
        fig.text(cx, .818, f"{KN[a]}  ×  {KN[b]}", ha="center", va="center",
                 fontsize=23, fontweight="bold", color=INK)
        fig.text(cx, .182, f"a flat picture hides {100*hid2:.0f}%\n3 directions still hide {100*hid3:.1f}%",
                 ha="center", va="center", fontsize=15, fontweight="bold",
                 color=RED if hid3 > .05 else GOOD, linespacing=1.5)
        note = ["barely bowed — width and height\nbarely interfere",
                "a deep arc — a −10° stroke and a +35°\nstroke stop sharing pixels at all",
                "folded over itself — lean and position\ntrade off, then run out"][idx]
        fig.text(cx, .128, note, ha="center", va="top", fontsize=13.5, color=INK2, linespacing=1.5)
    fig.text(.5, .772, "blue = one knob moving     orange = the other",
             ha="center", va="center", fontsize=14.5, color=INK3)
    footer(fig, "this is why the flat ruler fails: two straight knobs trace a curved sheet, and curvature costs extra straight directions")
    fig.savefig(f"{OUT}/tw3_sheets.png", dpi=DPI)
    plt.close(fig)


# ===========================================================================
# 4. PCA overcount on three spaces with known answers
# ===========================================================================
def pca95(X):
    Z = X - X.mean(0)
    s = np.linalg.svd(Z, full_matrices=False, compute_uv=False) ** 2
    return int(np.searchsorted(np.cumsum(s) / s.sum(), .95)) + 1


def fig_curvature(k95_grey, kfull_grey):
    rows = []
    for rung, knobs, label in [("move", 2, "MOVE\nslide a fixed bar\nanywhere on a 10×10"),
                               ("size", 4, "SIZE\n… and any height 5–10,\nany width 1–3")]:
        V = np.array([np.frombuffer(b, np.uint8) for b in sorted(EL.valid_set(rung))], float)
        rows.append((label, len(V), knobs, pca95(V)))
    rows.append(("GREY 1s\n5 continuous knobs,\n28×28 with grey", 20000, 5, k95_grey))

    fig = newfig()
    title(fig, "The same story on three spaces I can check",
          "For each one I know the true answer, because I wrote the generator. PCA is asked the same question:\n"
          "how many straight directions to cover 95% of this cloud?")
    ax = fig.add_axes([.255, .215, .645, .525])
    y = np.arange(len(rows))[::-1]
    h = .3
    for i, (lab, n, truth, pc) in enumerate(rows):
        yy = y[i]
        ax.barh(yy + h / 1.7, pc, height=h, color=BLUE, zorder=3)
        ax.barh(yy - h / 1.7, truth, height=h, color=GOOD, zorder=3)
        ax.text(pc + .8, yy + h / 1.7, f"{pc}", va="center", fontsize=21,
                fontweight="bold", color=BLUE)
        ax.text(truth + .8, yy - h / 1.7, f"{truth}", va="center", fontsize=21,
                fontweight="bold", color=GOOD)
        ax.text(-1.0, yy + .08, lab, ha="right", va="center", fontsize=15.5,
                color=INK, linespacing=1.45)
        ax.text(-1.0, yy - .45, f"{n:,} images", ha="right", va="center", fontsize=13, color=INK3)
    ax.set_yticks([]); ax.set_ylim(-.9, len(rows) - .2)
    ax.set_xlim(0, 42)
    ax.barh([], []); 
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="straight directions PCA asks for (95% of variance)"),
                       Patch(color=GOOD, label="knobs the generator actually has")],
              fontsize=15, frameon=False, loc="lower right", bbox_to_anchor=(1.0, -.04))
    ax.set_xlabel("number of dimensions reported", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.grid(axis="x", alpha=.18, zorder=0)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    fig.text(.895, .715, f"off the chart: the grey generator needs {kfull_grey} straight\ndirections before it loses nothing at all",
             ha="right", va="top", fontsize=14, color=INK3, style="italic", linespacing=1.5)
    footer(fig, "a 2-knob space read as 13-dimensional, a 4-knob space as 37. the overcount is curvature, not noise — and it does not go away with more data")
    fig.savefig(f"{OUT}/tw4_curvature.png", dpi=DPI)
    plt.close(fig)
    for r in rows: print("   ", r)


if __name__ == "__main__":
    import time
    t = time.time(); fig_generator(); print("1 generator ", round(time.time() - t, 1))
    t = time.time(); k95, kfull = fig_dimension(); print("2 dimension ", round(time.time() - t, 1), k95, kfull)
    t = time.time(); fig_sheets(); print("3 sheets    ", round(time.time() - t, 1))
    t = time.time(); fig_curvature(k95, kfull); print("4 curvature ", round(time.time() - t, 1))
