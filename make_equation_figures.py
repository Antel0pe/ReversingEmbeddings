"""An exact equation for a toy "1", term by term -- and why it needs pairs of pixels.

Toy valid set: a binary image whose only ink is a vertical bar, `w` pixels wide,
filling the middle 80% of the rows, at ANY horizontal position that fits.

    E(x) = A + B + C + D          x is valid  <=>  E(x) = 0

    A  nothing lit outside the band of rows
    B  every band row is identical to the one below it
    C  the top band row has exactly w lit pixels
    D  no two of those lit pixels are w or more columns apart (so they touch)

x is the image being tested -- one number per pixel. There is no template in
the equation; the horizontal freedom exists because no term pins the column.

The second figure answers: can it be done looking at one pixel at a time?
No. Any sum of per-pixel terms on binary pixels is linear, and
valid0 - valid1 + valid2 is itself a (split, invalid) binary image, so every
linear equation that accepts the valid ones must accept it too. Pairs of pixels
are enough: replace D by "no two lit pixels in the row are w or more apart".

The third figure asks whether any term can be dropped: delete each one in turn
and enumerate every 5x5 binary image to see what sneaks through. Each term's
job depends on the others -- B is what lets C and D look at a single row.

Writes figures/equation_terms.png, equation_per_pixel.png, equation_necessity.png.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = "figures"
GREEN, RED, ORANGE, GREY, BAND = "#1a7f37", "#c62828", "#ef6c00", "#444", "#fff3c4"
H = W = 10
ROWS = list(range(1, 9))        # middle 80% of 10 rows
R0 = ROWS[0]
WIDTH = 2


def bar(c, width=WIDTH):
    im = np.zeros((H, W), int); im[ROWS[0]:ROWS[-1] + 1, c:c + width] = 1; return im


def terms(x):
    out = [r for r in range(H) if r not in ROWS]
    A = int(x[out].sum())
    B = int(((x[ROWS[:-1]] - x[ROWS[1:]]) ** 2).sum())
    row = x[R0]
    C = int((row.sum() - WIDTH) ** 2)
    far = [(i, j) for i in range(W) for j in range(i + WIDTH, W) if row[i] and row[j]]
    D = len(far)                                  # sum of x_i * x_j over pairs >= WIDTH apart
    return A, B, C, D, far


def base(ax, x, grid=True):
    ax.set_facecolor("white")
    ax.imshow(np.ma.masked_where(x == 0, x), cmap="gray_r", vmin=0, vmax=1.6,
              interpolation="nearest", zorder=2)
    if grid:
        ax.set_xticks(np.arange(-.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-.5, H, 1), minor=True)
        ax.grid(which="minor", color="#bbb", lw=0.6)
    ax.set_xticks([])
    ax.set_yticks(range(H)); ax.set_yticklabels([str(r) for r in range(H)], fontsize=7.5,
                                                 color="#777")
    ax.tick_params(which="both", length=0, pad=2)
    ax.set_xlim(-.5, W - .5); ax.set_ylim(H - .5, -.5)


def mark(ax, r, c, col=RED):
    ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, fc=col, ec=col, alpha=.85, lw=1.5, zorder=3))


def value(ax, v, text):
    ax.text(0.5, -0.07, text, transform=ax.transAxes, ha="center", va="top", fontsize=11,
            weight="bold", color=GREEN if v == 0 else RED)


# ===========================================================================
# FIGURE 1 -- what each term looks at
# ===========================================================================
def fig_terms():
    ex = [("VALID\nbar at column 4", bar(4), True),
          ("VALID\nbar at column 0\n(position is free)", bar(0), True)]
    x = bar(4); x[9, 2] = 1
    ex.append(("stray dot below\nthe band", x, False))
    x = bar(4); x[5] = 0; x[5, 5:7] = 1
    ex.append(("one row slid\nsideways", x, False))
    ex.append(("too wide\n(3 pixels)", bar(4, 3), False))
    x = np.zeros((H, W), int); x[1:9, 2] = 1; x[1:9, 6] = 1
    ex.append(("split into\ntwo thin strokes", x, False))

    heads = ["the image x\n(the input being tested)",
             "A   count lit pixels\nin rows 0 and 9",
             "B   compare row 1 to 2, 2 to 3,\n... 7 to 8: count differences",
             "C   count lit pixels in\nrow 1, subtract 2, square",
             "D   count lit pairs in row 1\nthat are 2+ columns apart",
             "E(x) = A + B + C + D\n0 means a valid 1"]
    forms = [None,
             r"$\sum_{r\in\{0,9\}}\ \sum_c\ x_{r,c}$",
             r"$\sum_{r=1}^{7}\ \sum_c\ (x_{r,c} - x_{r+1,c})^2$",
             r"$(\ \sum_c\ x_{1,c}\ -\ 2\ )^2$",
             r"$\sum_{j - i\ \geq\ 2}\ x_{1,i}\cdot x_{1,j}$",
             None]
    n = len(ex)
    FW, FH = 19.5, 3.05 * n + 3.6
    fig = plt.figure(figsize=(FW, FH))
    gs = fig.add_gridspec(n, 6, left=0.115, right=0.99, top=1 - 2.85 / FH, bottom=0.55 / FH,
                          wspace=0.28, hspace=0.42)
    for r, (name, x, valid) in enumerate(ex):
        A, B, C, D, far = terms(x)
        E = A + B + C + D
        axs = [fig.add_subplot(gs[r, k]) for k in range(6)]

        base(axs[0], x)
        axs[0].text(-0.12, 0.5, name, transform=axs[0].transAxes, ha="right", va="center",
                    fontsize=11.5, color=GREEN if valid else "#222",
                    weight="bold" if valid else "normal", linespacing=1.45)

        base(axs[1], x)                                  # A
        for rr in range(H):
            if rr not in ROWS:
                axs[1].axhspan(rr - .5, rr + .5, color="#9e9e9e", alpha=.35, zorder=1)
                for c in range(W):
                    if x[rr, c]: mark(axs[1], rr, c)
        value(axs[1], A, f"A = {A}")

        base(axs[2], x)                                  # B
        for rr in ROWS[:-1]:
            for c in range(W):
                if x[rr, c] != x[rr + 1, c]:
                    axs[2].add_patch(Rectangle((c - .45, rr + .5 - .28), .9, .56, fc=RED,
                                               ec="none", alpha=.9, zorder=4))
        value(axs[2], B, f"B = {B}")

        base(axs[3], x)                                  # C
        axs[3].axhspan(R0 - .5, R0 + .5, color=BAND, alpha=1, zorder=1)
        axs[3].add_patch(Rectangle((-.5, R0 - .5), W, 1, fc="none", ec=ORANGE, lw=2.4, zorder=5))
        lit = int(x[R0].sum())
        value(axs[3], C, f"{lit} lit  ->  ({lit} - 2)² = {C}")

        base(axs[4], x)                                  # D
        axs[4].axhspan(R0 - .5, R0 + .5, color=BAND, alpha=1, zorder=1)
        axs[4].add_patch(Rectangle((-.5, R0 - .5), W, 1, fc="none", ec=ORANGE, lw=2.4, zorder=5))
        lit = [c for c in range(W) if x[R0, c]]
        for a_ in range(len(lit)):
            for b_ in range(a_ + 1, len(lit)):
                bad = lit[b_] - lit[a_] >= WIDTH
                axs[4].add_patch(FancyArrowPatch((lit[a_], R0 - .45), (lit[b_], R0 - .45),
                                 arrowstyle="-", connectionstyle="arc3,rad=-0.5",
                                 color=RED if bad else GREEN, lw=3.2, zorder=6))
        value(axs[4], D, f"{D} far pair{'s' if D != 1 else ''}  ->  D = {D}")

        ax = axs[5]; ax.axis("off")
        ax.add_patch(Rectangle((0.08, 0.12), 0.84, 0.76, transform=ax.transAxes, fc="#f7f7f7",
                               ec=GREEN if E == 0 else RED, lw=3))
        ax.text(0.5, 0.60, f"E = {E}", transform=ax.transAxes, ha="center", va="center",
                fontsize=24, weight="bold", color=GREEN if E == 0 else RED)
        ax.text(0.5, 0.30, "valid 1" if E == 0 else f"invalid\n({E} violation{'s' if E != 1 else ''})",
                transform=ax.transAxes, ha="center", va="center", fontsize=11,
                color=GREEN if E == 0 else RED)
        for k, a in enumerate(axs[:5]):
            for sp in a.spines.values():
                sp.set_edgecolor(GREEN if valid else "#888"); sp.set_linewidth(2.2 if valid else 1)
        if r == 0:
            for k, a in enumerate(axs):
                a.set_title(heads[k], fontsize=11.5, pad=52 if forms[k] else 12, linespacing=1.5,
                            weight="bold")
                if forms[k]:
                    a.text(0.5, 1.035, forms[k], transform=a.transAxes, ha="center", va="bottom",
                           fontsize=13, color="#333")

    fig.text(0.5, 1 - 0.38 / FH, "E(x), TERM BY TERM:  four checks, each 0 when passed, added together",
             ha="center", fontsize=17, weight="bold")
    fig.text(0.5, 1 - 0.85 / FH,
             "x is the image you hand the equation: one number per pixel, 0 = off, 1 = on "
             "(100 numbers here, 784 for MNIST). E eats the image and returns one number.",
             ha="center", fontsize=12)
    fig.text(0.5, 1 - 1.22 / FH,
             "Red marks show exactly which pixels each term is complaining about. Every term is a "
             "count or a square, so none can go negative -- the total is 0 only if all four are.",
             ha="center", fontsize=11.5, style="italic", color=GREY)
    fig.text(0.5, 1 - 1.58 / FH,
             "There is no reference image anywhere in the equation. Nothing mentions WHICH column -- "
             "that missing term is why the bar may sit anywhere.",
             ha="center", fontsize=11.5, style="italic", color=GREY)
    fig.savefig(f"{OUT}/equation_terms.png", dpi=120, facecolor="white")
    plt.close(fig)
    print("wrote equation_terms.png")


# ===========================================================================
# FIGURE 2 -- one pixel at a time is impossible; pairs are enough
# ===========================================================================
def enumerate_5x5():
    """Every binary 5x5 image: which ones does the best per-pixel equation accept,
    and which does the pairwise one accept?"""
    h = w = 5; rows = [1, 2, 3]; wd = 2; nb = 25
    V = []
    for c in range(w - wd + 1):
        im = np.zeros((h, w), int); im[1:4, c:c + wd] = 1; V.append(im)
    V = np.array(V)
    M = np.hstack([np.ones((len(V), 1)), V.reshape(len(V), -1)])
    null = np.linalg.svd(M)[2][len(V):]
    coef = np.random.default_rng(0).normal(size=len(null)) @ null
    lin, pair = [], []
    for st in range(0, 1 << nb, 1 << 20):
        n = np.arange(st, min(st + (1 << 20), 1 << nb), dtype=np.uint32)
        X = np.unpackbits(n.view(np.uint8).reshape(-1, 4)[:, ::-1], axis=1)[:, -nb:]
        X = X.reshape(-1, h, w).astype(np.int16)
        L = coef[0] + X.reshape(len(X), -1) @ coef[1:]
        lin += list(X[np.abs(L) < 1e-9])
        out = [r for r in range(h) if r not in rows]
        E2 = (X[:, out].sum((1, 2)) + ((X[:, rows[:-1]] - X[:, rows[1:]]) ** 2).sum((1, 2))
              + (X[:, 1].sum(1) - wd) ** 2
              + sum(X[:, 1, i] * X[:, 1, j] for i in range(w) for j in range(i + wd, w)))
        pair += list(X[E2 == 0])
    isvalid = lambda x: any((x == v).all() for v in V)
    return V, lin, pair, isvalid


def fig_per_pixel():
    V5, lin, pair, isvalid = enumerate_5x5()
    print(f"  5x5: best per-pixel equation accepts {len(lin)}, pairwise accepts {len(pair)}, "
          f"valid {len(V5)}")

    FW, FH = 17.0, 20.3
    fig = plt.figure(figsize=(FW, FH))
    fy = lambda inches_from_top: 1 - inches_from_top / FH

    fig.text(0.5, fy(0.40), "ONE PIXEL AT A TIME IS PROVABLY IMPOSSIBLE.  PAIRS OF PIXELS ARE ENOUGH.",
             ha="center", fontsize=17, weight="bold")
    fig.text(0.5, fy(0.85), "\"One pixel at a time\" means E is a sum of knobs: one number chosen per "
             "pixel, depending only on whether THAT pixel is on.", ha="center", fontsize=12)
    fig.text(0.5, fy(1.20), "With on/off pixels, every such equation is secretly a weighted sum:  "
             "E(x) = c + w1·x1 + w2·x2 + ... + w100·x100.", ha="center", fontsize=12,
             style="italic", color=GREY)

    # ---- section 1: the proof --------------------------------------------
    fig.text(0.04, fy(1.85), "1.  THE PROOF, AS A PICTURE", fontsize=14, weight="bold")
    gs = fig.add_gridspec(1, 7, left=0.10, right=0.90, top=fy(2.55), bottom=fy(5.35),
                          width_ratios=[3, 1, 3, 1, 3, 1, 3], wspace=0.05)
    parts = [(bar(2), "VALID\nbar at column 2", GREEN), ("−", None, None),
             (bar(3), "VALID\nbar at column 3", GREEN), ("+", None, None),
             (bar(4), "VALID\nbar at column 4", GREEN), ("=", None, None)]
    split = bar(2) - bar(3) + bar(4)
    parts.append((split, "INVALID\nsplit stroke", RED))
    for k, (obj, lbl, col) in enumerate(parts):
        ax = fig.add_subplot(gs[0, k])
        if lbl is None:
            ax.axis("off")
            ax.text(0.5, 0.5, obj, ha="center", va="center", fontsize=46, weight="bold",
                    transform=ax.transAxes)
            continue
        base(ax, obj)
        ax.set_title(lbl, fontsize=12, color=col, weight="bold", pad=10, linespacing=1.4)
        for sp in ax.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(3)
        ax.text(0.5, -0.06, "E = 0 required" if col == GREEN else "so E = 0 − 0 + 0 = 0 (!)",
                transform=ax.transAxes, ha="center", va="top", fontsize=11.5, weight="bold",
                color=col)
    fig.text(0.5, fy(5.95), "Pixel by pixel, this is true arithmetic: column 3 is on in bars 2 and 3 "
             "and cancels; column 4 in bars 3 and 4 and cancels. What survives is columns 2 and 5.",
             ha="center", fontsize=11.5)
    fig.text(0.5, fy(6.30), "A weighted sum treats the pictures the same way: E(split) = E(bar2) − E(bar3) "
             "+ E(bar4) = 0. It is forced to call the split stroke valid -- whatever weights you choose.",
             ha="center", fontsize=11.5)
    fig.text(0.5, fy(6.65), "This is your own objection, made exact: two far-apart pixels can "
             "\"sum together and act like a middle\" -- and a per-pixel equation cannot tell.",
             ha="center", fontsize=11.5, style="italic", color=GREY)

    # ---- section 2: the evidence -----------------------------------------
    fig.text(0.04, fy(7.35), "2.  THE EVIDENCE:  every one of the 33,554,432 binary 5×5 images, checked",
             fontsize=14, weight="bold")
    for i, (label, acc, top) in enumerate([
            ("best possible per-pixel equation\naccepts these " + str(len(lin)), lin, 8.05),
            ("pairwise equation\naccepts these " + str(len(pair)), pair, 9.95)]):
        gsr = fig.add_gridspec(1, 7, left=0.27, right=0.93, top=fy(top), bottom=fy(top + 1.25),
                               wspace=0.25)
        fig.text(0.25, fy(top + 0.62), label, ha="right", va="center", fontsize=12,
                 color=RED if len(acc) > len(V5) else GREEN, weight="bold", linespacing=1.4)
        acc = sorted(acc, key=lambda x: not isvalid(x))
        for k, x in enumerate(acc):
            ax = fig.add_subplot(gsr[0, k])
            ax.imshow(x, cmap="gray_r", vmin=0, vmax=1.6, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ok = isvalid(x)
            for sp in ax.spines.values():
                sp.set_edgecolor(GREEN if ok else RED); sp.set_linewidth(2.2 if ok else 3.4)
            ax.text(0.5, -0.1, "valid" if ok else "WRONG", transform=ax.transAxes, ha="center",
                    va="top", fontsize=10.5, weight="bold", color=GREEN if ok else RED)
    fig.text(0.5, fy(11.75), "(A small 5×5 canvas so the whole state space can be enumerated. "
             "Same rule: bar 2 wide in the middle rows, any column. Truth: 4 images.)",
             ha="center", fontsize=10.5, style="italic", color=GREY)

    # ---- section 3: the fix ----------------------------------------------
    fig.text(0.04, fy(12.45), "3.  THE FIX:  look at pixels in pairs", fontsize=14, weight="bold")
    fig.text(0.04, fy(12.85), "Replace term D with:  count the pairs of lit pixels in the row that "
             "are w = 2 or more columns apart. Each pair is one product  x_i · x_j , which is 1 only "
             "if BOTH are on.", fontsize=11.5)
    rows_demo = [("VALID\ncolumns 4, 5", [4, 5], GREEN), ("SPLIT\ncolumns 2, 5", [2, 5], RED),
                 ("SPREAD\ncolumns 1, 5, 8", [1, 5, 8], RED)]
    SW, YSPAN = 0.46, 2.1                        # strip width (fig fraction), data height
    SH_IN = SW * FW * YSPAN / W                  # inches tall for square pixels
    for i, (lbl, lit, col) in enumerate(rows_demo):
        top = 13.35 + i * (SH_IN + 0.30)
        ax = fig.add_axes([0.27, fy(top + SH_IN), SW, SH_IN / FH])
        row = np.zeros((1, W), int); row[0, lit] = 1
        ax.set_facecolor("white")
        ax.imshow(np.ma.masked_where(row == 0, row), cmap="gray_r", vmin=0, vmax=1.6,
                  interpolation="nearest", aspect="equal", zorder=2)
        ax.set_xticks(range(W)); ax.set_xticklabels([str(c) for c in range(W)], fontsize=10)
        ax.set_yticks([]); ax.tick_params(length=0)
        ax.set_xticks(np.arange(-.5, W, 1), minor=True); ax.grid(which="minor", color="#bbb", lw=.6)
        ax.set_ylim(0.5, 0.5 - YSPAN); ax.set_xlim(-.5, W - .5)
        bad = 0
        for a in range(len(lit)):
            for b in range(a + 1, len(lit)):
                far = lit[b] - lit[a] >= WIDTH
                bad += far
                ax.add_patch(FancyArrowPatch((lit[a], -.55), (lit[b], -.55), arrowstyle="-",
                                             connectionstyle="arc3,rad=-0.25",
                                             color=RED if far else GREEN, lw=3, zorder=4))
        for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
        ymid = fy(top + SH_IN * 0.55)
        fig.text(0.25, ymid, lbl, ha="right", va="center", fontsize=12, color=col,
                 weight="bold", linespacing=1.4)
        fig.text(0.75, ymid, f"{bad} far pair{'s' if bad != 1 else ''}  ->  D' = {bad}",
                 ha="left", va="center", fontsize=13, weight="bold", color=GREEN if bad == 0 else RED)
    foot = 13.35 + 3 * (SH_IN + 0.30) + 0.15
    fig.text(0.5, fy(foot), "Green arc: a lit pair that is allowed (adjacent). Red arc: a lit pair "
             "too far apart to be one stroke.\nWith the count fixed at 2 by term C, no red arcs means "
             "the two lit pixels are side by side -- one block.", ha="center", va="top",
             fontsize=11.5, style="italic", color=GREY, linespacing=1.6)
    fig.savefig(f"{OUT}/equation_per_pixel.png", dpi=120, facecolor="white")
    plt.close(fig)
    print("wrote equation_per_pixel.png")


# ===========================================================================
# FIGURE 3 -- is every term needed?  delete one, see what gets in
# ===========================================================================
def fig_necessity():
    h = w = 5; band = [1, 2, 3]; wd = 2; nb = 25
    out = [r for r in range(h) if r not in band]
    keep = {k: [] for k in ["all", "A", "B", "C", "D"]}
    count = {k: 0 for k in keep}
    for st in range(0, 1 << nb, 1 << 20):
        n = np.arange(st, min(st + (1 << 20), 1 << nb), dtype=np.uint32)
        X = np.unpackbits(n.view(np.uint8).reshape(-1, 4)[:, ::-1], axis=1)[:, -nb:]
        X = X.reshape(-1, h, w).astype(np.int16)
        T = {"A": X[:, out].sum((1, 2)),
             "B": ((X[:, band[:-1]] - X[:, band[1:]]) ** 2).sum((1, 2)),
             "C": (X[:, band[0]].sum(1) - wd) ** 2,
             "D": sum(X[:, band[0], i] * X[:, band[0], j] for i in range(w) for j in range(i + wd, w))}
        for k in keep:
            ok = sum(v for t, v in T.items() if t != k) == 0
            count[k] += int(ok.sum())
            keep[k] += list(X[ok][:4000])
    V = keep["all"]
    isvalid = lambda x: any((x == v).all() for v in V)
    for k in "ABCD":
        bad = [x for x in keep[k] if not isvalid(x)]
        keep[k] = sorted(bad, key=lambda x: (x.sum(), x.tobytes()))
    print("  necessity counts:", count)

    rows = [("ALL FOUR TERMS\n(the baseline)", "all", V, count["all"], GREEN,
             "exactly the 4 bars -- nothing else"),
            ("drop A\n(rows 0 and 4 unchecked)", "A", keep["A"], count["A"], RED,
             "ink anywhere in the unchecked rows"),
            ("drop B\n(rows no longer copy)", "B", keep["B"], count["B"], RED,
             "C and D only ever look at row 1 --\nrows 2 and 3 can be anything"),
            ("drop C\n(count no longer fixed)", "C", keep["C"], count["C"], RED,
             "blank image, and 1-pixel-wide bars:\nno far pairs, so D is happy"),
            ("drop D\n(no contiguity check)", "D", keep["D"], count["D"], RED,
             "any 2 pixels in the row, even far apart")]
    NCOL = 6
    FW, FH = 17.5, 2.05 * len(rows) + 2.3
    fig = plt.figure(figsize=(FW, FH))
    gs = fig.add_gridspec(len(rows), NCOL, left=0.245, right=0.80, top=1 - 1.85 / FH,
                          bottom=0.35 / FH, wspace=0.22, hspace=0.55)
    for r, (lbl, k, imgs, cnt, col, why) in enumerate(rows):
        for c in range(NCOL):
            ax = fig.add_subplot(gs[r, c])
            if c >= len(imgs):
                ax.axis("off"); continue
            x = imgs[c]
            ax.set_facecolor("white")
            for rr in out:
                ax.axhspan(rr - .5, rr + .5, color="#9e9e9e", alpha=.35, zorder=1)
            ax.imshow(np.ma.masked_where(x == 0, x), cmap="gray_r", vmin=0, vmax=1.6,
                      interpolation="nearest", zorder=2)
            ax.set_xticks(np.arange(-.5, w, 1), minor=True)
            ax.set_yticks(np.arange(-.5, h, 1), minor=True)
            ax.grid(which="minor", color="#bbb", lw=.6)
            ax.set_xticks([]); ax.set_yticks([]); ax.tick_params(which="minor", length=0)
            ax.set_xlim(-.5, w - .5); ax.set_ylim(h - .5, -.5)
            for sp in ax.spines.values():
                sp.set_edgecolor(GREEN if k == "all" else RED)
                sp.set_linewidth(2.4 if k == "all" else 2.0)
        fig.text(0.225, 0.0 + gs[r, 0].get_position(fig).y0 + gs[r, 0].get_position(fig).height / 2,
                 lbl, ha="right", va="center", fontsize=12.5, color=col, weight="bold",
                 linespacing=1.45)
        ymid = gs[r, 0].get_position(fig).y0 + gs[r, 0].get_position(fig).height / 2
        fig.text(0.815, ymid + 0.012, f"{cnt:,} accepted" + ("" if k == "all" else
                 f"\n({cnt - 4:,} are not 1s)"), ha="left", va="bottom", fontsize=12,
                 weight="bold", color=col, linespacing=1.4)
        fig.text(0.815, ymid - 0.008, why, ha="left", va="top", fontsize=10.5, color=GREY,
                 linespacing=1.4)
    fig.text(0.5, 1 - 0.42 / FH, "EVERY TERM IS NEEDED:  delete any one, and things that are not 1s "
             "get through", ha="center", fontsize=16.5, weight="bold")
    fig.text(0.5, 1 - 0.88 / FH, "All 33,554,432 on/off 5×5 images checked. Grey rows are outside the "
             "band. Rows 2+ show the simplest intruders each deletion lets in.",
             ha="center", fontsize=11.5)
    fig.text(0.5, 1 - 1.25 / FH, "The terms lean on each other: B makes every band row a copy of row 1, "
             "which is the only reason C and D can get away with checking row 1 alone.",
             ha="center", fontsize=11, style="italic", color=GREY)
    fig.savefig(f"{OUT}/equation_necessity.png", dpi=120, facecolor="white")
    plt.close(fig)
    print("wrote equation_necessity.png")


if __name__ == "__main__":
    fig_terms()
    fig_per_pixel()
    fig_necessity()
