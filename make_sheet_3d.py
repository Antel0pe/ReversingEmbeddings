"""The six two-knob sheets as 3-D surfaces, with how much is still being hidden.

The 2-D version draws each sheet in its own best flat PLANE, which throws away
whatever sticks out of that plane -- up to 28.8% for lean x width. This adds the
third direction and reports what is still missing after three.

Run: python make_sheet_3d.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import grey_ones as G

OUT = "figures"
INK, GOOD, BAD, REF = "#1a1a1a", "#1a7f37", "#c62828", "#1565c0"
KN = ["cx", "cy", "height", "width", "lean"]
MID = G.RANGES.mean(1)
PAIRS = [(4, 3), (4, 2), (4, 0), (3, 2), (0, 1), (3, 1)]
NF = 21


def sheet(a, b, n):
    va, vb = np.linspace(*G.RANGES[a], n), np.linspace(*G.RANGES[b], n)
    P = np.tile(MID, (n * n, 1))
    A, B = np.meshgrid(va, vb)
    P[:, a] = A.ravel(); P[:, b] = B.ravel()
    return G.render(P).reshape(n * n, -1).astype(np.float64)


def main():
    fig = plt.figure(figsize=(14.6, 9.2))
    fig.text(.5, .968, "The same six sheets in three directions instead of two",
             ha="center", va="center", fontsize=16, fontweight="bold")
    fig.text(.5, .910, "Each surface is the real 2-d sheet, drawn in the best 3 of its 784 directions. "
                       "The grid lines ARE the paths the two knobs trace —\nnothing is smoothed or fitted. "
                       "“still hidden” is the share of the sheet's spread that even three directions cannot show.\n"
                       "All three axes are drawn to the SAME scale, so a sheet that looks flat really is flat.",
             ha="center", va="center", fontsize=9.4, color="#555", linespacing=1.55)
    L, T, B = .015, .885, .075
    cw, ch = (1 - 2 * L) / 3, (T - B) / 2
    rows = []
    for idx, (a, b) in enumerate(PAIRS):
        r, c = divmod(idx, 3)
        X = sheet(a, b, NF)
        Z = X - X.mean(0)
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        v = S ** 2 / (S ** 2).sum()
        hid2, hid3 = 1 - v[:2].sum(), 1 - v[:3].sum()
        Y = (U[:, :3] * S[:3]).reshape(NF, NF, 3)
        ax = fig.add_subplot(2, 3, idx + 1, projection="3d")
        ax.set_position([L + c * cw + .012, T - (r + 1) * ch + .045, cw * .95, ch * .82])
        for i in range(NF):
            ax.plot(Y[i, :, 0], Y[i, :, 1], Y[i, :, 2], color=REF, lw=.8, alpha=.8)
            ax.plot(Y[:, i, 0], Y[:, i, 1], Y[:, i, 2], color="#d08a00", lw=.8, alpha=.8)
        ax.set_title(f"{KN[a]} × {KN[b]}", fontsize=12.5, fontweight="bold", pad=-2)
        # EQUAL scaling on all three axes -- matplotlib autoscales each one
        # independently by default, which inflates the third direction and makes
        # a nearly flat sheet look like a dramatic trough.
        m = np.abs(Y).max()
        ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_zlim(-m, m)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=24, azim=-62)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.grid(False)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((1, 1, 1, 0)); pane.line.set_color((.8, .8, .8, 1))
        ax.text2D(.5, -.055, f"hidden by 2 directions {100*hid2:4.1f}%   →   by 3 directions {100*hid3:4.1f}%",
                  transform=ax.transAxes, ha="center", va="top", fontsize=8.8,
                  color=BAD if hid3 > .05 else GOOD)
        rows.append((KN[a], KN[b], 100 * hid2, 100 * hid3))
    fig.text(.5, .036, "Blue lines = one knob varying, orange = the other. Every sheet with LEAN in it is an arc, because a stroke at −10° and one at +35° "
                       "barely overlap,\nso their distance stops growing once they stop sharing pixels. The third direction recovers most, but not all, of what the flat "
                       "picture hid.",
             ha="center", va="center", fontsize=9.2, color="#333", style="italic", linespacing=1.6)
    fig.savefig(f"{OUT}/statespace_sheets_3d.png", dpi=140)
    plt.close(fig)
    print(f"  {'pair':22s} {'hidden by 2':>12s} {'hidden by 3':>12s}")
    for na, nb, h2, h3 in rows:
        print(f"  {na+' x '+nb:22s} {h2:11.1f}% {h3:11.1f}%")


if __name__ == "__main__":
    main()
