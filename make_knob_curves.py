"""The curve each knob traces, drawn with nothing thrown away.

A knob varied alone traces a 1-d curve through 784-d pixel space. That curve is
fully described by its 784 coordinate functions: pixel value vs knob. Only the
pixels that change are shown (the rest are constant), so the heatmap IS the curve.

Upright stroke (lean = 0) for the four knobs whose curves are exact polygons;
lean itself about the mid-range stroke. cx and cy are swept wider than their
MNIST range so the full on/ramp/off shape of each pixel is visible; the MNIST
range is shaded.

Run: python make_knob_curves.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import grey_ones as G

OUT = "figures"
INK, GOOD, BAD, REF = "#1a1a1a", "#1a7f37", "#c62828", "#1565c0"
TRACE = ["#1565c0", "#c62828", "#d08a00", "#1a7f37"]
N = 1200


def sweep(k, lo, hi, lean0):
    base = G.RANGES.mean(1).copy(); base[4] = lean0
    P = np.tile(base, (N, 1)); P[:, k] = np.linspace(lo, hi, N)
    return np.linspace(lo, hi, N), G.render(P).reshape(N, -1).astype(np.float64)


def corners(t, X):
    bend = np.linalg.norm(np.diff(X, 2, axis=0), axis=1) > 1e-5
    idx = np.where(bend)[0]
    if not len(idx): return []
    return [t[g[len(g) // 2] + 1] for g in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)]


KNOBS = [  # knob, sweep lo, hi, lean at, unit, shape word
    (0, 10.0, 19.0, 0.0, "px", "polygon"),
    (1, 10.0, 19.0, 0.0, "px", "polygon"),
    (2, 19.0, 20.5, 0.0, "px", "polygon"),
    (3, 1.8, 4.6, 0.0, "px", "polygon"),
    (4, -10.0, 35.0, None, "°", "curve"),
]


def main():
    fig = plt.figure(figsize=(15.6, 10.4))
    fig.text(.5, .972, "The curve each knob traces through 784-d pixel space — drawn in full",
             ha="center", va="center", fontsize=16, fontweight="bold")
    fig.text(.5, .935, "Top: four individual pixels, value vs knob — four of the curve's coordinates. "
                       "Middle: EVERY pixel that changes, one row each (sorted by where it changes) — this heatmap is the whole curve,\n"
                       "nothing projected or dropped. Bottom: where those pixels sit on the canvas. Dashed ticks mark corners, "
                       "where the curve turns to a new straight direction. Shaded band = the MNIST range.",
             ha="center", va="center", fontsize=9.3, color="#555", linespacing=1.55)
    L, R = .045, .99
    cw = (R - L) / 5
    summary = []
    for c, (k, lo, hi, lean0, unit, shape) in enumerate(KNOBS):
        lean0 = G.RANGES.mean(1)[4] if lean0 is None else lean0
        t, X = sweep(k, lo, hi, lean0)
        ch = np.abs(X - X[0]).max(0) > 1e-6
        ch |= np.abs(X - X[-1]).max(0) > 1e-6
        cols = np.where(ch)[0]
        Xc = X[:, cols]
        crn = corners(t, X)
        x0 = L + c * cw + .018
        name = G.KNOBS[k]
        mlo, mhi = G.RANGES[k]

        # --- top: 4 pixel traces
        ax = fig.add_axes([x0, .655, cw * .80, .19])
        swing = Xc.max(0) - Xc.min(0)
        where = np.argmax(np.abs(np.diff(Xc, axis=0)), axis=0)
        pick = []
        for q in np.argsort(where[np.argsort(-swing)[:40]]):
            j = np.argsort(-swing)[:40][q]
            if all(abs(where[j] - where[p]) > N // 9 for p in pick): pick.append(j)
            if len(pick) == 4: break
        for i, j in enumerate(pick):
            p = cols[j]
            ax.plot(t, Xc[:, j], color=TRACE[i], lw=1.8, label=f"pixel ({p//28},{p%28})")
        for cc in crn[:60]:
            ax.axvline(cc, color="#999", lw=.5, ls=(0, (2, 2)), zorder=0)
        ax.axvspan(mlo, mhi, color="#1a7f37", alpha=.08, zorder=0)
        ax.set_xlim(lo, hi); ax.set_ylim(-.04, 1.08)
        ax.tick_params(labelsize=7.5); ax.set_xticklabels([])
        if c == 0: ax.set_ylabel("pixel value", fontsize=9)
        ax.set_title(f"{name}", fontsize=13, fontweight="bold", pad=18)
        n_c = len(crn)
        sub = ("genuinely curved" if shape != "polygon"
               else "a straight line: no corners" if n_c == 0
               else f"polygon: {n_c} corner{'s' if n_c != 1 else ''}")
        ax.text(.5, 1.03, sub, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.8, color=REF if shape == "polygon" else BAD)
        ax.legend(fontsize=6.4, loc="upper right", framealpha=.9, handlelength=1.2)

        # --- middle: every changing pixel
        order = np.argsort(np.argmax(np.abs(np.diff(Xc, axis=0)), axis=0))
        ax2 = fig.add_axes([x0, .285, cw * .80, .345])
        ax2.imshow(Xc[:, order].T, aspect="auto", cmap="gray_r", vmin=0, vmax=1,
                   extent=[lo, hi, len(cols), 0], interpolation="nearest")
        ax2.axvline(mlo, color=GOOD, lw=1.1); ax2.axvline(mhi, color=GOOD, lw=1.1)
        ax2.set_xlabel(f"{name} ({unit})", fontsize=9.5)
        ax2.tick_params(labelsize=7.5)
        if c == 0: ax2.set_ylabel("each row = one pixel that changes", fontsize=9)
        ax2.text(.5, -.20, f"{len(cols)} of 784 pixels move", transform=ax2.transAxes,
                 ha="center", va="top", fontsize=8.8, color=INK, fontweight="bold")

        # --- bottom: where those pixels are, coloured by how much they move
        ax3 = fig.add_axes([x0 + cw * .22, .045, cw * .36, .15])
        M = np.full(784, np.nan); M[cols] = swing
        ax3.imshow(np.ma.masked_invalid(M.reshape(28, 28)), cmap="magma_r", vmin=0, vmax=1,
                   interpolation="nearest")
        ax3.imshow(X[N // 2].reshape(28, 28), cmap="Greys", alpha=.0)
        ax3.set_xticks([]); ax3.set_yticks([])
        for s_ in ax3.spines.values(): s_.set_edgecolor("#bbb")
        summary.append((name, len(cols), n_c, shape))
    fig.text(.012, .12, "which\npixels\nmove", ha="left", va="center", fontsize=8.6, color="#555",
             linespacing=1.3)
    fig.text(.5, .012, "Bottom maps: dark = the pixel swings all the way between 0 and 1 over the sweep, light = it only changes a little, blank = it never changes.",
             ha="center", va="center", fontsize=8.6, color="#777")
    fig.savefig(f"{OUT}/knob_curves.png", dpi=135)
    plt.close(fig)
    for s in summary: print(s)


if __name__ == "__main__":
    main()
