"""One knob, many straight axes: why intrinsic dimension and linear dimension differ.

A single stroke slides sideways across the canvas. One number (cx) says where it
is, so the intrinsic dimension is 1. But to hold every position with STRAIGHT
axes you need dozens. Drawn in pairs of principal directions, the path turns out
to be a coil: it goes round the first pair of axes once, the next pair twice, the
next three times ... like a spring wound through many dimensions at once.

Run: python make_coil_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import grey_ones as G

OUT = "figures"
INK, GOOD, BAD, REF = "#1a1a1a", "#1a7f37", "#c62828", "#1565c0"
EPS8 = 0.11
N = 1500
LO, HI = 4.0, 24.0


def main():
    P = np.tile(G.RANGES.mean(1), (N, 1))
    P[:, 4] = 0.0                              # upright, so the picture is simple
    P[:, 0] = np.linspace(LO, HI, N)
    X = G.render(P).reshape(N, -1).astype(np.float64)
    Z = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    C = U * S
    tot = (Z ** 2).sum(1)
    R = np.sqrt(np.maximum(tot[:, None] - np.cumsum(C[:, :120] ** 2, 1), 0)).max(0)
    k_all = int(np.argmax(R < EPS8)) + 1
    t = P[:, 0]

    fig = plt.figure(figsize=(14.4, 9.0))
    fig.text(.5, .968, f"One knob, {k_all} straight axes: a sliding stroke is a coil",
             ha="center", va="center", fontsize=16, fontweight="bold")
    fig.text(.5, .927, "A single upright stroke slides from left to right. ONE number says where it is — intrinsic dimension 1. "
                       f"Yet {k_all} straight axes are needed\nto hold every position to 8-bit accuracy. Below, the same path drawn in pairs "
                       "of those axes. Colour is the knob: dark = left of canvas, yellow = right.",
             ha="center", va="center", fontsize=9.6, color="#555", linespacing=1.55)

    # --- row 1: the stroke at a few positions
    pos = np.linspace(0, N - 1, 7).astype(int)
    for i, j in enumerate(pos):
        ax = fig.add_axes([.165 + i * .098, .735, .078, .135])
        ax.imshow(X[j].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        col = plt.cm.viridis((t[j] - LO) / (HI - LO))
        for s_ in ax.spines.values(): s_.set_edgecolor(col); s_.set_linewidth(2.4)
        ax.text(.5, -.09, f"cx = {t[j]:.0f}", transform=ax.transAxes, ha="center",
                va="top", fontsize=8.6, color=INK)

    # --- row 2: the path in pairs of principal directions
    pairs = [(0, 1), (2, 3), (4, 5), (8, 9), (18, 19)]
    for i, (a, b) in enumerate(pairs):
        ax = fig.add_axes([.045 + i * .192, .235, .165, .40])
        pts = C[:, [a, b]]
        seg = np.stack([pts[:-1], pts[1:]], 1)
        lc = LineCollection(seg, cmap="viridis", linewidth=2.0)
        lc.set_array((t[:-1] - LO) / (HI - LO)); ax.add_collection(lc)
        m = np.abs(C[:, :2]).max() * 1.08
        ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_aspect("equal")
        ax.axhline(0, color="#ddd", lw=.7, zorder=0); ax.axvline(0, color="#ddd", lw=.7, zorder=0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"axes {a+1} & {b+1}", fontsize=11.5, fontweight="bold", pad=6)
        share = (S[a] ** 2 + S[b] ** 2) / (S ** 2).sum()
        ax.text(.5, -.05, f"{100*share:.1f}% of the spread", transform=ax.transAxes,
                ha="center", va="top", fontsize=8.8, color="#555")
    fig.text(.5, .655, "↓  same path, same scale in every panel  ↓", ha="center", va="center",
             fontsize=9, color="#777", style="italic")

    fig.text(.5, .105, "Each pair of axes is a circle, and later pairs go round faster. Axes 1&2: the path goes round about once. "
                       "3&4: about twice.\n5&6: about three times. By axes 19&20 it is winding tightly with a tiny radius — "
                       "small, but not zero, and still needed to get every position exactly right.",
             ha="center", va="center", fontsize=9.4, color="#333", style="italic", linespacing=1.6)
    fig.text(.5, .040, "This is your circle-on-paper idea from ideas.md, all the way down: a 1-dimensional thing that needs 2 axes when it "
                       "bends once, and many axes when it bends in many\ndirections. The renderer bends the knob into pixel space; "
                       "straight axes can only follow that bend by stacking up more of them.",
             ha="center", va="center", fontsize=9.2, color="#333", linespacing=1.6)
    fig.savefig(f"{OUT}/coil_one_knob.png", dpi=140)
    plt.close(fig)
    print(f"one knob -> {k_all} linear dims")


if __name__ == "__main__":
    main()
