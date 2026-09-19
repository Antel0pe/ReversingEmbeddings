"""Figures for equation_ladder.py: explainable equations for moving, sized and tilted 1s.

Writes figures/ladder_rungs.png, ladder_checks.png, ladder_lean.png,
readable_equations.png.
"""

import itertools
import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import equation_ladder as L

OUT = "figures"
GREEN, RED, GREY, BLUE = "#1a7f37", "#c62828", "#444", "#1565c0"
H, W = L.H, L.W

PLAIN = {
    "one_block_of_rows": "inked rows form one unbroken block",
    "height_8_to_8": "height is exactly 8 rows",
    "height_5_to_10": "height is 5 to 10 rows",
    "one_run_per_row": "every inked row is one solid run",
    "same_width_every_row": "every row has the same width",
    "width_2_to_2": "width is exactly 2",
    "width_1_to_3": "width is 1 to 3",
    "no_sideways_steps": "no row shifts sideways",
    "leans_one_way": "never steps both left and right",
    "leans_evenly": "equal stretches step equally (±1)",
    "lean_at_most_23deg": "leans no more than 23 degrees",
}


def pix(ax, x, border=None, lw=2.2):
    ax.set_facecolor("white")
    ax.imshow(np.ma.masked_where(x == 0, x), cmap="gray_r", vmin=0, vmax=1.6,
              interpolation="nearest", zorder=2)
    ax.set_xticks(np.arange(-.5, x.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, x.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#c8c8c8", lw=.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.tick_params(which="minor", length=0)
    ax.set_xlim(-.5, x.shape[1] - .5); ax.set_ylim(x.shape[0] - .5, -.5)
    if border:
        for sp in ax.spines.values(): sp.set_edgecolor(border); sp.set_linewidth(lw)


def line(top, h, w, a, b=0.5, left=None):
    rel = np.floor(a * np.arange(h) + b).astype(int); rel -= rel.min()
    s0 = (W - w - rel.max()) // 2 if left is None else left
    return L.draw(top, h, w, s0 + rel)


def word_img(word, w=2, top=None):
    rel = np.concatenate([[0], np.cumsum(word)]); rel -= rel.min()
    h = len(word) + 1
    return L.draw((H - h) // 2 if top is None else top, h, w, (W - w - rel.max()) // 2 + rel)


# ===========================================================================
def fig_rungs(counts):
    ex = {
        "move": [L.draw(0, 8, 2, [4] * 8), L.draw(2, 8, 2, [0] * 8), L.draw(1, 8, 2, [8] * 8),
                 L.draw(2, 8, 2, [5] * 8), L.draw(0, 8, 2, [1] * 8)],
        "size": [L.draw(0, 10, 1, [4] * 10), L.draw(3, 5, 3, [1] * 5), L.draw(1, 7, 2, [6] * 7),
                 L.draw(4, 6, 1, [8] * 6), L.draw(0, 9, 3, [3] * 9)],
        "tilt": [line(0, 10, 1, 0.40), line(0, 10, 2, -0.40), line(2, 7, 3, 0.22),
                 line(1, 8, 1, -0.18, 0.2), line(3, 5, 2, 0.30)],
    }
    info = {
        "move": ("RUNG 1  -  MOVE", "generator knobs: top row, left column\n(height 8, width 2 fixed)"),
        "size": ("RUNG 2  -  SIZE", "+ height 5 to 10\n+ width 1 to 3"),
        "tilt": ("RUNG 3  -  TILT", "+ lean up to 23 degrees\n   either way, still straight"),
    }
    prev = []
    FW, RH = 19.0, 3.35
    FH = RH * 3 + 2.4
    fig = plt.figure(figsize=(FW, FH))
    for r, rung in enumerate(["move", "size", "tilt"]):
        top_in = 1.95 + r * RH
        y1, y0 = 1 - top_in / FH, 1 - (top_in + 2.05) / FH
        gs = fig.add_gridspec(1, 5, left=0.215, right=0.585, top=y1, bottom=y0, wspace=0.12)
        for k, x in enumerate(ex[rung]):
            ax = fig.add_subplot(gs[0, k]); pix(ax, x, GREEN)
            assert L.E(x, rung) == 0
        ym = (y0 + y1) / 2
        name, knobs = info[rung]
        fig.text(0.015, ym + 0.028, name, fontsize=14, weight="bold", va="bottom",
                 color=GREEN if rung == "move" else "#222")
        fig.text(0.015, ym + 0.018, knobs, fontsize=11, va="top", color=GREY, linespacing=1.5)
        fig.text(0.40, y0 - 0.012, f"{counts[rung]:,} valid images on the 10×10 grid "
                 "-- E = 0 on every one, and on nothing else", ha="center", va="top",
                 fontsize=11, weight="bold", color=GREEN)
        names = [c.__name__ for c in L.checks(rung)]
        lines = []
        for i, nm in enumerate(names):
            new = nm not in prev
            lines.append((f"{i + 1}.  {PLAIN[nm]}", new and r > 0))
        yy = y1 + 0.004
        for txt, new in lines:
            fig.text(0.615, yy, txt + ("   <- new" if new else ""), fontsize=11.2, va="top",
                     color=BLUE if new else "#222", weight="bold" if new else "normal")
            yy -= 0.235 / FH * 1.0
        prev = names
        if r < 2:
            fig.add_artist(plt.Line2D([0.015, 0.985], [y0 - 0.034] * 2, color="#ddd", lw=1.2))
    fig.text(0.5, 1 - 0.45 / FH, "THE LADDER:  each new freedom swaps checks in or out, and every "
             "check is a plain sentence", ha="center", fontsize=17, weight="bold")
    fig.text(0.5, 1 - 0.92 / FH, "Left: what the generator can vary.  Middle: examples it draws.  "
             "Right: the checks whose violation counts add up to E(x).",
             ha="center", fontsize=12)
    fig.text(0.5, 1 - 1.27 / FH, "Blue = changed from the rung above. Tilt costs the most: one check "
             "('no row shifts sideways') becomes three.", ha="center", fontsize=11.5,
             style="italic", color=GREY)
    fig.savefig(f"{OUT}/ladder_rungs.png", dpi=115, facecolor="white"); plt.close(fig)
    print("wrote ladder_rungs.png")


# ===========================================================================
def only_fails(x, target, rung="tilt"):
    b = L.breakdown(x, rung)
    return b[target] > 0 and all(v == 0 for k, v in b.items() if k != target)


def find_word(target, n=7):
    best = None
    for word in itertools.product((-1, 0, 1), repeat=n):
        x = word_img(word)
        if only_fails(x, target):
            key = (sum(map(abs, word)), word)
            if best is None or key < best[0]: best = (key, x)
    return best[1]


def fig_checks():
    x1 = L.draw(0, 10, 2, [4] * 10); x1[4] = 0
    x3 = L.draw(1, 8, 2, [4] * 8); x3[4] = 0; x3[4, 4] = x3[4, 6] = 1
    x4 = L.draw(1, 8, 2, [4] * 8); x4[5, 6] = 1
    cases = [("VALID - leaning 1", line(0, 10, 2, 0.35), None),
             ("gap in the middle", x1, "one_block_of_rows"),
             ("only 4 rows tall", L.draw(3, 4, 2, [4] * 4), "height_5_to_10"),
             ("one row broken in two", x3, "one_run_per_row"),
             ("one row too wide", x4, "same_width_every_row"),
             ("too thick (4 wide)", L.draw(1, 8, 4, [3] * 8), "width_1_to_3"),
             ("bends back: < shape", find_word("leans_one_way"), "leans_one_way"),
             ("upright, then leans:\na bend, not a line", find_word("leans_evenly", 9), "leans_evenly"),
             ("straight, but 27°", word_img((1, 0, 1, 0, 1, 0, 1, 0, 1)), "lean_at_most_23deg")]
    FW, FH = 17.0, 17.6
    fig = plt.figure(figsize=(FW, FH))
    gs = fig.add_gridspec(3, 3, left=0.04, right=0.96, top=1 - 1.75 / FH, bottom=0.2 / FH,
                          wspace=0.10, hspace=0.30)
    names = [c.__name__ for c in L.checks("tilt")]
    for i, (title, x, target) in enumerate(cases):
        if target: assert only_fails(x, target), (title, L.breakdown(x, "tilt"))
        else: assert L.E(x, "tilt") == 0
        sub = gs[i // 3, i % 3].subgridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.06)
        ax = fig.add_subplot(sub[0, 0]); pix(ax, x, GREEN if target is None else RED, 2.6)
        ax.set_title(title, fontsize=12, weight="bold", color=GREEN if target is None else RED,
                     pad=8, linespacing=1.3)
        tx = fig.add_subplot(sub[0, 1]); tx.axis("off")
        b = L.breakdown(x, "tilt")
        for j, nm in enumerate(names):
            v = b[nm]; y = 0.97 - j * 0.105
            tx.text(0.02, y, str(v), fontsize=10, va="top", ha="left",
                    color=RED if v else GREEN, weight="bold", transform=tx.transAxes)
            tx.text(0.13, y, PLAIN[nm], fontsize=9.2, va="top",
                    color=RED if v else "#555", weight="bold" if v else "normal",
                    transform=tx.transAxes)
        E = sum(b.values())
        tx.text(0.02, 0.97 - len(names) * 0.105 - 0.03, f"E = {E}", fontsize=15, va="top",
                weight="bold", color=GREEN if E == 0 else RED, transform=tx.transAxes)
    fig.text(0.5, 1 - 0.45 / FH, "EVERY CHECK, CAUGHT IN THE ACT:  the valid baseline, then one image "
             "failing each check - and only that check", ha="center", fontsize=16, weight="bold")
    fig.text(0.5, 1 - 0.92 / FH, "Rung 3 (move + size + tilt). Beside each image: its score on all "
             "eight checks. A check scores 0 when satisfied; E is the total.",
             ha="center", fontsize=12)
    fig.text(0.5, 1 - 1.27 / FH, "The last three are the tilt checks. Each catches a way a staircase "
             "can fail to be a straight line that the other two cannot see.",
             ha="center", fontsize=11.5, style="italic", color=GREY)
    fig.savefig(f"{OUT}/ladder_checks.png", dpi=115, facecolor="white"); plt.close(fig)
    print("wrote ladder_checks.png")


# ===========================================================================
def fig_lean(stats):
    cases = [("STRAIGHT, 20°", (0, 1, 0, 0, 1, 0, 0, 1, 0), GREEN),
             ("BENT", (0, 0, 0, 0, 1, 0, 1, 0, 1), RED),
             ("ZIG-ZAG", (1, 0, 0, 0, -1, 0, 0, 0, 1), RED),
             ("STRAIGHT, 27°", (1, 0, 1, 0, 1, 0, 1, 0, 1), RED)]
    lean = [L.leans_one_way, L.leans_evenly, L.lean_at_most()]
    FW, FH = 17.0, 15.8
    fig = plt.figure(figsize=(FW, FH))
    fy = lambda t: 1 - t / FH
    fig.text(0.5, fy(0.45), "WHY TILT NEEDS THREE CHECKS:  what makes a pixel staircase a straight line",
             ha="center", fontsize=16.5, weight="bold")
    fig.text(0.5, fy(0.92), "A leaning stroke on a grid is a staircase. Read it as a list of steps: "
             "0 = next row starts in the same column, 1 = one column over.",
             ha="center", fontsize=12)
    gs = fig.add_gridspec(1, 4, left=0.05, right=0.95, top=fy(1.75), bottom=fy(6.3), wspace=0.35)
    for k, (nm, word, col) in enumerate(cases):
        x = word_img(word, w=1, top=0)
        ax = fig.add_subplot(gs[0, k]); pix(ax, x, col, 2.6)
        ax.set_title(nm, fontsize=13, weight="bold", color=col, pad=8)
        ax.text(0.5, -0.05, "steps:  " + " ".join(str(abs(s)) if s >= 0 else "-1" for s in word),
                transform=ax.transAxes, ha="center", va="top", fontsize=11, family="monospace")
        for j, c in enumerate(lean):
            v = c(x)
            ax.text(0.5, -0.15 - j * 0.075, f"{PLAIN[c.__name__]}: {v}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=10, color=RED if v else GREEN,
                    weight="bold" if v else "normal")
    fig.text(0.5, fy(8.0), "BENT: its steps bunch up at the bottom -- 3 rows of 0 0 0 at the top, "
             "1 0 1 at the bottom. A real line spreads its steps evenly, so any two stretches of the "
             "same length\nmay differ by at most one step. That is 'leans evenly', and it is the "
             "one check that has to compare distant parts of the stroke.",
             ha="center", va="top", fontsize=11.2, linespacing=1.55)

    ax = fig.add_axes([0.08, fy(14.7), 0.50, (14.7 - 10.1) / FH])
    Ls = np.arange(1, 10)
    allowed = np.floor(L.AMAX * Ls).astype(int) + 1
    ax.bar(Ls, allowed, color="#90caf9", edgecolor=BLUE, zorder=3, label="steps allowed")
    ax.plot(Ls, L.AMAX * Ls, "o-", color=RED, lw=2.4, zorder=4,
            label="an exact 23° line (tan 23° = 0.42 per row)")
    for l_, a_ in zip(Ls, allowed):
        ax.text(l_, a_ + 0.08, str(a_), ha="center", fontsize=11, weight="bold", color=BLUE)
    ax.set_xticks(Ls); ax.set_xlabel("length of stretch (rows)", fontsize=11.5)
    ax.set_ylabel("sideways steps", fontsize=11.5); ax.set_ylim(0, 5.3)
    ax.grid(alpha=.3, axis="y"); ax.legend(loc="upper left", fontsize=10.5)
    ax.set_title("WHERE 23° ENDS ON A GRID: max steps = floor(0.42 × L) + 1\n"
                 "(the +1 is rounding a real line onto whole pixels)", fontsize=12, pad=10)

    tx = fig.add_axes([0.64, fy(14.7), 0.33, (14.7 - 10.1) / FH]); tx.axis("off")
    rows = [("EXHAUSTIVE CHECK", None),
            (f"{stats['words']:,} staircases (every step list", None),
            ("of length 4-9 using -1, 0, 1)", None),
            ("compared with an exact solver that", None),
            ("asks 'does a 23° line draw this?'", None),
            ("", None),
            ("disagreements: 0", GREEN),
            ("", None),
            ("DELETE ONE CHECK, and this many", None),
            ("non-straight staircases get in:", None)]
    rows += [(f"  {PLAIN[k]}:  {v}", RED) for k, v in stats["leak"].items()]
    for j, (t, c) in enumerate(rows):
        tx.text(0.0, 1 - j * 0.074, t, fontsize=11.5, va="top", color=c or "#222",
                weight="bold" if c or t.isupper() else "normal", transform=tx.transAxes)
    fig.savefig(f"{OUT}/ladder_lean.png", dpi=115, facecolor="white"); plt.close(fig)
    print("wrote ladder_lean.png")


# ===========================================================================
def fig_readable():
    V = []
    for c in range(W - 1):
        im = np.zeros((H, W), int); im[1:9, c:c + 2] = 1; V.append(im.ravel())
    V = np.array(V); M = np.hstack([np.ones((len(V), 1)), V])
    rank = np.linalg.matrix_rank(M); N = np.linalg.svd(M)[2][rank:]
    nz = np.array([(np.abs(n[1:]) > 1e-6).sum() for n in N])
    dense = N[int(np.argmax(nz))][1:].reshape(H, W)
    A = np.zeros((H, W)); A[0, 3] = 1
    B = np.zeros((H, W)); B[4, 5] = 1; B[5, 5] = -1
    C = np.zeros((H, W)); C[1] = 1
    alt = np.zeros((H, W)); alt[1] = [(-1) ** c for c in range(W)]
    FW, FH = 18.0, 10.2
    fig = plt.figure(figsize=(FW, FH))
    fy = lambda t: 1 - t / FH
    def heat(rect, m, title, sub, col="#222"):
        ax = fig.add_axes(rect); v = np.abs(m).max()
        ax.imshow(m, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        ax.set_xticks(np.arange(-.5, W, 1), minor=True); ax.set_yticks(np.arange(-.5, H, 1), minor=True)
        ax.grid(which="minor", color="#ccc", lw=.5); ax.set_xticks([]); ax.set_yticks([])
        ax.tick_params(which="minor", length=0)
        ax.set_title(title, fontsize=12, weight="bold", color=col, pad=8, linespacing=1.35)
        ax.text(0.5, -0.05, sub, transform=ax.transAxes, ha="center", va="top", fontsize=10.5,
                color=GREY, linespacing=1.45)
    fig.text(0.5, fy(0.45), "ARE PIXEL-WEIGHT EQUATIONS UNREADABLE?  Only in a bad basis.",
             ha="center", fontsize=17, weight="bold")
    fig.text(0.5, fy(0.9), "For the sliding bar, 92 independent weighted-sum equations are true on "
             "every valid image. Red = +weight, blue = -weight, white = 0. Both panels below are "
             "the SAME set of equations.", ha="center", fontsize=11.5)
    heat([0.03, fy(5.6), 0.22, 0.22 * FW / FH * 1.0], dense,
         "WHAT A SOLVER HANDS YOU\none of its 92 equations",
         f"weights on {nz.max()} of 100 pixels\n(median across the 92: {int(np.median(nz))})", RED)
    fig.text(0.285, fy(3.6), "=", fontsize=40, ha="center", va="center", weight="bold")
    fig.text(0.285, fy(4.2), "same\nspace", fontsize=10.5, ha="center", va="top", color=GREY)
    for k, (m, t, s) in enumerate([(A, "A: pixel = 0", "1 weight\n(20 of these)"),
                                   (B, "B: pixel = pixel below", "2 weights\n(70 of these)"),
                                   (C, "C: row 1 sums to 2", "10 weights, constant -2\n(1 of these)")]):
        heat([0.33 + k * 0.155, fy(5.0), 0.13, 0.13 * FW / FH], m, t, s, GREEN)
    fig.text(0.56, fy(1.6), "THE SAME EQUATIONS, WRITTEN SPARSELY: our hand-written checks", ha="center",
             fontsize=12.5, weight="bold", color=GREEN)
    fig.text(0.56, fy(6.25), "Together these give 91 of the 92.  The computer found the last one:",
             ha="center", fontsize=12, weight="bold")
    heat([0.82, fy(5.0), 0.13, 0.13 * FW / FH], alt, "the one we missed:\n+ - + - along row 1",
         "'as many lit pixels in even\ncolumns as in odd columns'", BLUE)
    tests = [("adjacent pair\n(valid)", [4, 5], True), ("gap of 2", [3, 5], False),
             ("gap of 3", [2, 5], False), ("gap of 4", [2, 6], False)]
    for k, (t, lit, ok) in enumerate(tests):
        ax = fig.add_axes([0.12 + k * 0.205, fy(8.15), 0.16, 0.16 * FW / FH * 0.1])
        row = np.zeros((1, W)); row[0, lit] = 1
        ax.imshow(np.ma.masked_where(row == 0, row), cmap="gray_r", vmin=0, vmax=1.6, aspect="equal")
        ax.set_xticks(np.arange(-.5, W, 1), minor=True); ax.grid(which="minor", color="#bbb", lw=.5)
        ax.set_xticks(range(W)); ax.set_xticklabels(["+" if c % 2 == 0 else "-" for c in range(W)],
                                                    fontsize=10, color=BLUE, weight="bold")
        ax.set_yticks([]); ax.tick_params(length=0)
        score = sum((-1) ** c for c in lit)
        caught = score != 0
        verdict = ("0  ->  passes (correct)" if ok else
                   (f"{score:+d}  ->  caught" if caught else "0  ->  MISSED"))
        ax.set_title(t, fontsize=11.5, pad=6, color=GREEN if ok else "#222", weight="bold")
        ax.text(0.5, -1.6, f"alternating sum = {verdict}", transform=ax.transAxes, ha="center",
                va="top", fontsize=11, weight="bold",
                color=GREEN if ok else (GREEN if caught else RED))
    fig.text(0.5, fy(9.35), "A weighted sum gets HALF of 'the two pixels touch': it catches every even gap "
             "and misses every odd one.\nThe gap of 3 it misses is exactly the split stroke the earlier "
             "proof said no weighted sum could ever reject.", ha="center", va="top", fontsize=11.5,
             style="italic", color=GREY, linespacing=1.55)
    fig.savefig(f"{OUT}/readable_equations.png", dpi=115, facecolor="white"); plt.close(fig)
    print("wrote readable_equations.png")


if __name__ == "__main__":
    counts = {r: len(L.valid_set(r)) for r in L.RUNGS}
    print("valid images per rung:", counts)
    fig_rungs(counts)
    fig_checks()
    fig_lean({"words": 29484, "leak": {"leans_one_way": 512, "leans_evenly": 120,
                                        "lean_at_most_23deg": 310}})
    fig_readable()
