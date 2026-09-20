"""Two-knob sheets: the 1s a pair of knobs produces, and the shape of the patch they cover.

  statespace_sheets.png        -- six 2-knob sheets, rendered
  statespace_sheet_shapes.png  -- the same six sheets as SURFACES: area, and how much
                                  they bend out of any flat plane

Run: python make_sheet_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import grey_ones as G

OUT = "figures"
INK, GOOD, BAD, REF = "#1a1a1a", "#1a7f37", "#c62828", "#1565c0"
KN = ["cx", "cy", "height", "width", "lean"]
UNITS = ["px", "px", "px", "px", "°"]
MID = G.RANGES.mean(1)
PAIRS = [(4, 3), (4, 2), (4, 0), (3, 2), (0, 1), (3, 1)]     # (knob across, knob down)
NG = 7           # montage is NG x NG images
NF = 25          # geometry is measured on a finer NF x NF grid


def flat(P):
    P = np.atleast_2d(P)
    return G.render(P).reshape(len(P), -1).astype(np.float64)


def sheet(a, b, n):
    """n x n grid over knobs a (across) and b (down); returns params and images."""
    va = np.linspace(*G.RANGES[a], n)
    vb = np.linspace(*G.RANGES[b], n)
    P = np.tile(MID, (n * n, 1))
    A, B = np.meshgrid(va, vb)                    # row index = b, col index = a
    P[:, a] = A.ravel(); P[:, b] = B.ravel()
    return va, vb, P, flat(P)


def geometry(a, b):
    """Surface area of the sheet, and how far it is from flat."""
    va, vb, P, X = sheet(a, b, NF)
    Xg = X.reshape(NF, NF, 784)
    # area: sum over cells of the parallelogram spanned by the two edge vectors
    e_a = Xg[:-1, 1:, :] - Xg[:-1, :-1, :]        # step in knob a
    e_b = Xg[1:, :-1, :] - Xg[:-1, :-1, :]        # step in knob b
    na, nb = (e_a ** 2).sum(-1), (e_b ** 2).sum(-1)
    dot = (e_a * e_b).sum(-1)
    area = np.sqrt(np.maximum(na * nb - dot ** 2, 0)).sum()
    # flatness: variance left after the best 2-d plane
    Z = X - X.mean(0)
    s = np.linalg.svd(Z, compute_uv=False) ** 2
    curl = 1 - s[:2].sum() / s.sum()
    # the flat quadrilateral through the four corners, for an area reference
    C = Xg[[0, 0, -1, -1], [0, -1, 0, -1], :]
    u, v = C[1] - C[0], C[2] - C[0]
    flat_area = np.sqrt(max(np.dot(u, u) * np.dot(v, v) - np.dot(u, v) ** 2, 0))
    return X, Z, area, curl, flat_area, va, vb


def montage(X, n, pad=2):
    """n x n images -> one array, with white gutters."""
    c = 28 + pad
    M = np.full((n * c - pad, n * c - pad), np.nan)
    for i in range(n):
        for j in range(n):
            M[i * c:i * c + 28, j * c:j * c + 28] = X[i * n + j].reshape(28, 28)
    return M


# ===========================================================================
def fig_sheets():
    fig = plt.figure(figsize=(14.2, 9.9))
    fig.text(.5, .968, "Six two-knob sheets — every 1 the pair can draw",
             ha="center", va="center", fontsize=16, fontweight="bold")
    fig.text(.5, .930, "The other three knobs are held at the middle of their range. "
                       "Each panel is a 7×7 sweep across both knobs' full ranges.",
             ha="center", va="center", fontsize=9.6, color="#555")
    L, R, T, B = .055, .985, .885, .075
    cw, ch = (R - L) / 3, (T - B) / 2
    for idx, (a, b) in enumerate(PAIRS):
        r, c = divmod(idx, 3)
        va, vb, P, X = sheet(a, b, NG)
        _, _, area, curl, flat_area, _, _ = geometry(a, b)
        ax_y0, ax_h = T - (r + 1) * ch + .085, ch * .70
        ax = fig.add_axes([L + c * cw + .038, ax_y0, cw * .80, ax_h])
        ax.imshow(montage(X, NG), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_edgecolor("#bbb"); s_.set_linewidth(.8)
        ax.set_title(f"{KN[a]} × {KN[b]}", fontsize=12.5, fontweight="bold",
                     color=INK, pad=7)
        # axis annotations, outside the image
        ax.annotate("", xy=(1.0, 1.045), xytext=(0.0, 1.045), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=REF, lw=1.3))
        ax.text(.5, 1.075, f"{KN[a]}  {va[0]:g} → {va[-1]:g} {UNITS[a]}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8.6, color=REF)
        ax.annotate("", xy=(-.048, 0.0), xytext=(-.048, 1.0), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=REF, lw=1.3))
        ax.text(-.075, .5, f"{KN[b]}  {vb[0]:g} → {vb[-1]:g} {UNITS[b]}",
                transform=ax.transAxes, ha="center", va="center", fontsize=8.6,
                color=REF, rotation=90)
        crumple = area / flat_area
        ax.text(.5, -.085, f"surface area {area:7.1f}     bends out of plane {100*curl:4.1f}%",
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                color=BAD if curl > .05 else GOOD)
        ax.text(.5, -.155, f"crumple {crumple:.2f}×  (area ÷ flat quadrilateral through its corners)",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.2, color="#666")
    fig.text(.5, .030, "Surface area is in squared L2 pixel units — the same ruler as θ = 1.89. "
                       "“Bends out of plane” is the share of the sheet's spread\nthat no single flat plane can hold: "
                       "0% would mean the pair sweeps out an ordinary flat parallelogram.",
             ha="center", va="center", fontsize=9.2, color="#333", style="italic",
             linespacing=1.6)
    fig.savefig(f"{OUT}/statespace_sheets.png", dpi=140)
    plt.close(fig)


# ===========================================================================
def fig_shapes():
    fig = plt.figure(figsize=(14.2, 9.4))
    fig.text(.5, .968, "The same six sheets as surfaces — the patch each pair carves out",
             ha="center", va="center", fontsize=16, fontweight="bold")
    fig.text(.5, .928, "Each sheet is a 2-d surface living in 784 dimensions. Here it is drawn in its OWN best flat plane "
                       "(top 2 principal directions of that\nsheet alone), with the knob grid on it. A flat sheet would draw "
                       "a perfect parallelogram; the bending you see is what no flat plane can hold.",
             ha="center", va="center", fontsize=9.4, color="#555", linespacing=1.55)
    L, R, T, B = .05, .985, .875, .075
    cw, ch = (R - L) / 3, (T - B) / 2
    for idx, (a, b) in enumerate(PAIRS):
        r, c = divmod(idx, 3)
        X, Z, area, curl, flat_area, va, vb = geometry(a, b)
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        Y = (U[:, :2] * S[:2]).reshape(NF, NF, 2)
        ax = fig.add_axes([L + c * cw + .045, T - (r + 1) * ch + .130, cw * .78, ch * .60])
        for i in range(NF):
            ax.plot(Y[i, :, 0], Y[i, :, 1], color=REF, lw=.85, alpha=.75)
            ax.plot(Y[:, i, 0], Y[:, i, 1], color="#d08a00", lw=.85, alpha=.75)
        ax.plot(Y[[0, 0, -1, -1, 0], [0, -1, -1, 0, 0], 0],
                Y[[0, 0, -1, -1, 0], [0, -1, -1, 0, 0], 1], color=INK, lw=1.6, ls="--")
        for (i, j), lab, off in [((0, 0), "start", (7, -11)), ((0, -1), f"{KN[a]} max", (7, 7)),
                                 ((-1, 0), f"{KN[b]} max", (-7, -11))]:
            ax.plot(Y[i, j, 0], Y[i, j, 1], "o", color=INK, ms=5, zorder=5)
            ax.annotate(lab, (Y[i, j, 0], Y[i, j, 1]), xytext=off, ha="right" if off[0] < 0 else "left",
                        textcoords="offset points", fontsize=7.6, color=INK, zorder=6,
                        bbox=dict(fc="white", ec="none", alpha=.8, pad=.8))
        ax.set_aspect("equal"); ax.tick_params(labelsize=7); ax.grid(alpha=.18)
        ax.set_title(f"{KN[a]} × {KN[b]}", fontsize=12.5, fontweight="bold", pad=6)
        ax.text(.5, -.30, f"area {area:.1f}    out of plane {100*curl:.1f}%    crumple {area/flat_area:.2f}×",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.8,
                color=BAD if curl > .05 else GOOD)
    fig.text(.29, .033, "blue lines = " + "one knob varying", ha="right", va="center",
             fontsize=9, color=REF)
    fig.text(.30, .033, "|", ha="center", va="center", fontsize=9, color="#bbb")
    fig.text(.31, .033, "orange lines = the other", ha="left", va="center",
             fontsize=9, color="#d08a00")
    fig.text(.72, .033, "dashed black = the flat quadrilateral through the four corners",
             ha="center", va="center", fontsize=9, color=INK)
    fig.savefig(f"{OUT}/statespace_sheet_shapes.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    fig_sheets(); print("sheets done")
    fig_shapes(); print("shapes done")
    print("\n  pair                 area    out-of-plane   crumple")
    for a, b in PAIRS:
        _, _, ar, cu, fa, _, _ = geometry(a, b)
        print(f"  {KN[a]:>6s} x {KN[b]:<8s}  {ar:8.1f}   {100*cu:8.1f}%   {ar/fa:7.2f}x")
