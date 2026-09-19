"""Describing a many-knob family with 1-D curves.

curves_graph.png  -- the 5-knob grey family as a graph (join 1s within 1.5):
                     seeds needed, how flat, how many curves, what curves cost
curves_sheet.png  -- a 2-knob slice (lean x width): one curve through all of it,
                     and the slice rebuilt from products of 1-D curves
curves_sheet.gif  -- the whole slice as a single movie, walking the Hilbert curve
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.optimize import least_squares

import grey_ones as G

OUT = "figures"
GREEN, RED, GREY, BLUE, PURPLE, ORANGE = "#1a7f37", "#c62828", "#444", "#1565c0", "#6a1b9a", "#ef6c00"
EPS = 1.5


def dists(X):
    sq = (X ** 2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0)); np.fill_diagonal(D, np.inf)
    return D


def graph_stats():
    rng = np.random.default_rng(0)
    perc = []
    for n in [100, 200, 400, 800, 1200, 1600, 2400, 3200, 4800, 6400]:
        X = G.render(G.sample(n, rng)).reshape(n, -1); D = dists(X)
        k, lab = connected_components(csr_matrix(D <= EPS), directed=False)
        perc.append((n, k, np.bincount(lab).max() / n))
    n = 3200; P = G.sample(n, rng); X = G.render(P).reshape(n, -1); D = dists(X)
    flat = []
    iu = np.triu(np.ones_like(D, bool), 1)
    for lo, hi in [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0)]:
        I, J = np.where((D > lo) & (D <= hi) & iu)
        for t in rng.choice(len(I), min(40, len(I)), replace=False):
            i, j = I[t], J[t]; mid = (X[i] + X[j]) / 2
            f = lambda q: G.render(np.clip(q, G.RANGES[:, 0], G.RANGES[:, 1]))[0].ravel() - mid
            r = least_squares(f, (P[i] + P[j]) / 2, diff_step=1e-3, max_nfev=200)
            flat.append((D[i, j], np.linalg.norm(f(r.x))))
    n = 6400; X = G.render(G.sample(n, rng)).reshape(n, -1); D = dists(X)
    best = None
    for s in rng.choice(n, 5, replace=False):
        seen = np.zeros(n, bool); curves = [[s]]; seen[s] = True; cur = s
        for _ in range(n - 1):
            d = np.where(seen, np.inf, D[cur]); j = int(np.argmin(d))
            if d[j] > EPS: curves.append([])
            curves[-1].append(j); seen[j] = True; cur = j
        if best is None or len(curves) < len(best): best = curves
    chain = np.concatenate(best); pos = np.empty(n, int); pos[chain] = np.arange(n)
    I, J = np.where(np.triu(D <= EPS, 1)); gap = np.abs(pos[I] - pos[J])
    deg = int(np.median((D <= EPS).sum(1)))
    return dict(perc=perc, flat=np.array(flat), lens=sorted(map(len, best), reverse=True),
                gap=gap, n=n, deg=deg)


def fig_graph(st):
    FW, FH = 17.0, 12.8
    fig = plt.figure(figsize=(FW, FH))
    fig.text(0.5, 1 - 0.42 / FH, "THE 5-KNOB FAMILY AS A GRAPH:  join any two 1s within 1.5, then try "
             "to describe it with curves", ha="center", fontsize=16, weight="bold")
    fig.text(0.5, 1 - 0.86 / FH, "Grey generator (5 knobs), sampled at random. 1.5 is the 'close enough "
             "to interpolate' radius. All distances are in pixel space.", ha="center", fontsize=11.8)
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.97, top=1 - 1.75 / FH, bottom=0.6 / FH,
                          hspace=0.42, wspace=0.22)

    ax = fig.add_subplot(gs[0, 0])
    n, k, g = np.array(st["perc"]).T
    ax.plot(n, 100 * g, "o-", color=BLUE, lw=2.6, ms=8, label="share in the largest piece")
    for x_, k_, g_ in st["perc"]:
        ax.text(x_, 100 * g_ + 4, f"{int(k_)}", ha="center", fontsize=9.5, color=GREY)
    ax.set_xscale("log"); ax.set_ylim(0, 112); ax.grid(alpha=.3)
    ax.set_xlabel("number of random seeds", fontsize=11.5); ax.set_ylabel("% of seeds in the largest piece", fontsize=11.5)
    ax.set_title("1. SEEDS NEEDED FOR ONE PIECE  (grey numbers = pieces)\n"
                 "a sudden jump between 800 and 1,600 seeds; one piece by 6,400", fontsize=11.5, pad=8)

    ax = fig.add_subplot(gs[0, 1])
    L_, s_ = st["flat"].T
    ax.scatter(L_, s_, s=16, color=PURPLE, alpha=.55, zorder=3, label="one joined pair")
    c = np.sum(s_ * L_ ** 2) / np.sum(L_ ** 4)
    xs = np.linspace(0, 4, 100)
    ax.plot(xs, c * xs ** 2, color=RED, lw=2.4, label=f"sagitta rule: {c:.3f} x length²  (curvature radius {1/(8*c):.1f})")
    ax.axvline(EPS, color=GREEN, lw=2, ls="--"); ax.text(EPS + 0.05, ax.get_ylim()[1] * 0.9 if False else 1.05, "1.5",
                                                          color=GREEN, fontsize=11, weight="bold")
    ax.set_xlim(0, 4); ax.set_ylim(0, 1.25); ax.grid(alpha=.3); ax.legend(fontsize=10, loc="upper left")
    ax.set_xlabel("distance between the two joined 1s", fontsize=11.5)
    ax.set_ylabel("their 50/50 blend: distance\nto the nearest valid 1", fontsize=11.5)
    ax.set_title(f"2. IS 'WITHIN 1.5' LOCALLY FLAT?  mostly: at 1.5 the blend is ~{c*2.25:.2f} off\n"
                 "the error grows with the SQUARE of the distance, as a curved sheet's should", fontsize=11.5, pad=8)

    ax = fig.add_subplot(gs[1, 0])
    lens = np.array(st["lens"])
    ax.bar(np.arange(1, len(lens) + 1), lens, color=BLUE, width=1.0, zorder=3)
    ax.set_yscale("log"); ax.grid(alpha=.3, axis="y")
    ax.set_xlabel("curve, longest first", fontsize=11.5); ax.set_ylabel("1s on this curve", fontsize=11.5)
    ax.set_title(f"3. HOW MANY CURVES?  {len(lens)} curves that never revisit a point cover all "
                 f"{st['n']:,} 1s\n(or exactly ONE curve, if it may retrace -- a walk around a spanning "
                 f"tree, {2*(st['n']-1):,} steps)", fontsize=11.5, pad=8)

    ax = fig.add_subplot(gs[1, 1])
    gap = st["gap"]; bins = np.logspace(0, np.log10(st["n"]), 30)
    ax.hist(gap, bins=bins, color=ORANGE, edgecolor="#b35400", zorder=3)
    ax.set_xscale("log"); ax.grid(alpha=.3, axis="y")
    ax.axvline(np.median(gap), color=RED, lw=2.2, ls="--")
    ax.text(np.median(gap) * 1.1, ax.get_ylim()[1] * 0.9, f"median {int(np.median(gap))}", color=RED,
            fontsize=11, weight="bold")
    ax.set_xlabel("steps apart along the curve", fontsize=11.5)
    ax.set_ylabel("pairs of 1s within 1.5\nof each other", fontsize=11.5)
    ax.set_title(f"4. THE PRICE: each 1 has ~{st['deg']} neighbours, a curve can keep only 2 beside it\n"
                 f"{100*np.mean(gap <= 1):.0f}% of neighbour pairs end up adjacent on the curve; "
                 f"half are {int(np.median(gap))}+ steps apart", fontsize=11.5, pad=8)
    fig.savefig(f"{OUT}/curves_graph.png", dpi=110, facecolor="white"); plt.close(fig)
    print("wrote curves_graph.png")


# ---------------------------------------------------------------------------
def hilbert(order):
    """Cell order of the Hilbert curve on a 2^order x 2^order grid."""
    n = 2 ** order; pts = []
    for d in range(n * n):
        x = y = 0; t = d; s = 1
        while s < n:
            rx = 1 & (t // 2); ry = 1 & (t ^ rx)
            if ry == 0:
                if rx == 1: x, y = s - 1 - x, s - 1 - y
                x, y = y, x
            x += s * rx; y += s * ry; t //= 4; s *= 2
        pts.append((x, y))
    return pts


def lawnmower(n):
    return [(c if r % 2 == 0 else n - 1 - c, r) for r in range(n) for c in range(n)]


def sheet(n):
    leans = np.linspace(-10, 35, n); widths = np.linspace(1.8, 4.6, n)
    P = np.array([[14.5, 14.5, 20.0, w, a] for w in widths for a in leans])
    return leans, widths, G.render(P).reshape(n, n, 784)


def neighbour_gaps(path, n):
    pos = {p: i for i, p in enumerate(path)}; gaps = []
    for x in range(n):
        for y in range(n):
            for dx, dy in [(1, 0), (0, 1)]:
                if x + dx < n and y + dy < n:
                    gaps.append(abs(pos[(x, y)] - pos[(x + dx, y + dy)]))
    return np.array(gaps)


def fig_sheet():
    n = 8; leans, widths, S = sheet(n)
    FW, FH = 17.5, 16.3
    fig = plt.figure(figsize=(FW, FH))
    fy = lambda t: 1 - t / FH
    fig.text(0.5, fy(0.42), "A 2-KNOB SHEET OF 1s, DESCRIBED BY 1-D CURVES:  two different ways",
             ha="center", fontsize=16.5, weight="bold")
    fig.text(0.5, fy(0.86), "The slice of the grey family where only lean (left to right) and stroke "
             "width (top to bottom) vary. 64 images; the real sheet is continuous.",
             ha="center", fontsize=11.8)
    for k, (path, name, col) in enumerate([(hilbert(3), "HILBERT CURVE", BLUE),
                                           (lawnmower(n), "LAWNMOWER", ORANGE)]):
        x0 = 0.04 + k * 0.49; W_ = 0.43; H_ = W_ * FW / FH
        ax = fig.add_axes([x0, fy(1.95) - H_, W_, H_])
        big = np.ones((n * 29, n * 29))
        for (x, y) in [(i, j) for i in range(n) for j in range(n)]:
            big[y * 29:y * 29 + 28, x * 29:x * 29 + 28] = 1 - S[y, x].reshape(28, 28)
        ax.imshow(big, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        xs = [p[0] * 29 + 13.5 for p in path]; ys = [p[1] * 29 + 13.5 for p in path]
        ax.plot(xs, ys, color=col, lw=2.6, alpha=.85)
        ax.plot(xs[0], ys[0], "o", color=GREEN, ms=11); ax.plot(xs[-1], ys[-1], "s", color=RED, ms=11)
        ax.set_xticks([p * 29 + 13.5 for p in (0, n - 1)]); ax.set_xticklabels([f"lean {leans[0]:.0f}°", f"{leans[-1]:.0f}°"])
        ax.set_yticks([p * 29 + 13.5 for p in (0, n - 1)]); ax.set_yticklabels([f"width {widths[0]:.1f}", f"{widths[-1]:.1f}"])
        ax.tick_params(length=0, labelsize=10)
        g = neighbour_gaps(path, n); Pp = np.array(path)
        span8 = np.mean([max(np.ptp(Pp[i:i + 8, 0]), np.ptp(Pp[i:i + 8, 1])) + 1 for i in range(64 - 7)])
        ax.set_title(f"{name}: ONE line through all 64\n"
                     f"side-by-side 1s: {g.mean():.1f} steps apart on the line on average, worst {g.max()}\n"
                     f"any 8 frames in a row span {span8:.1f} cells across",
                     fontsize=11.5, color=col, weight="bold", pad=8, linespacing=1.4)
    fig.text(0.5, fy(9.85), "Both are 1-D descriptions of a 2-D sheet, and both must fold -- they just pay "
             "in different currencies. The lawnmower keeps side-by-side 1s closer on the line;\nthe Hilbert "
             "curve keeps every stretch of the line inside a compact patch (why it makes the better movie). "
             "With more knobs, both costs grow.", ha="center", va="top", fontsize=11, style="italic",
             color=GREY, linespacing=1.55)

    # second way: products of 1-D curves (separable / Tucker form)
    m = 24; leans, widths, S = sheet(m)
    T = S - S.mean((0, 1), keepdims=True)
    def curves_needed(unfold):
        lam = np.linalg.eigvalsh(unfold @ unfold.T)[::-1].clip(0); c = np.cumsum(lam) / lam.sum()
        return lam, c
    lam_a, c_a = curves_needed(np.transpose(T, (1, 0, 2)).reshape(m, -1))     # functions of lean
    lam_w, c_w = curves_needed(T.reshape(m, -1))                              # functions of width
    w_, V = np.linalg.eigh(np.transpose(T, (1, 0, 2)).reshape(m, -1) @ np.transpose(T, (1, 0, 2)).reshape(m, -1).T)
    V = V[:, ::-1]
    ax = fig.add_axes([0.07, fy(15.4), 0.38, (15.4 - 11.35) / FH])
    for r, cc in zip(range(3), [BLUE, PURPLE, ORANGE]):
        ax.plot(leans, V[:, r] * np.sign(V[m // 2, r] + 1e-9), "-", lw=2.6, color=cc, label=f"lean-curve {r + 1}")
    ax.set_xlabel("lean (degrees)", fontsize=11.5); ax.set_ylabel("value", fontsize=11.5); ax.grid(alpha=.3)
    ax.legend(fontsize=10.5)
    ax.set_title("THE OTHER WAY: sums of products of 1-D curves\nthe first three curves in lean the sheet is "
                 "built from", fontsize=12, pad=8)
    ax = fig.add_axes([0.56, fy(15.4), 0.39, (15.4 - 11.35) / FH])
    k = np.arange(1, m + 1)
    ax.plot(k, 100 * (1 - c_a), "o-", color=BLUE, lw=2.4, label="curves in LEAN")
    ax.plot(k, 100 * (1 - c_w), "s-", color=GREEN, lw=2.4, label="curves in WIDTH")
    ax.set_yscale("log"); ax.set_xlim(0.5, 16.5); ax.grid(alpha=.3); ax.legend(fontsize=10.5)
    need_a = int(np.searchsorted(c_a, .999) + 1); need_w = int(np.searchsorted(c_w, .999) + 1)
    ax.axhline(0.1, color=RED, lw=1.8, ls="--"); ax.text(16.3, 0.12, "0.1% left", color=RED, ha="right", fontsize=10)
    ax.set_xlabel("number of 1-D curves used", fontsize=11.5)
    ax.set_ylabel("% of the sheet's variation\nNOT yet captured", fontsize=11.5)
    ax.set_title(f"HOW MANY CURVES?  to capture 99.9%: {need_a} in lean, {need_w} in width\n"
                 "width is nearly 'straight'; lean curls, so it needs more", fontsize=12, pad=8)
    fig.savefig(f"{OUT}/curves_sheet.png", dpi=110, facecolor="white"); plt.close(fig)
    print(f"wrote curves_sheet.png  (curves for 99.9%: lean {need_a}, width {need_w})")

    # the GIF: the whole 8x8 sheet as one movie along the Hilbert curve
    n = 8; leans, widths, S = sheet(n); path = hilbert(3)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 4.4), gridspec_kw=dict(width_ratios=[1, 1.15]))
    big = np.ones((n * 29, n * 29))
    for x in range(n):
        for y in range(n):
            big[y * 29:y * 29 + 28, x * 29:x * 29 + 28] = 1 - S[y, x].reshape(28, 28)
    a2.imshow(big, cmap="gray", vmin=0, vmax=1); a2.set_xticks([]); a2.set_yticks([])
    xs = [p[0] * 29 + 13.5 for p in path]; ys = [p[1] * 29 + 13.5 for p in path]
    line, = a2.plot([], [], color=BLUE, lw=2.4); dot, = a2.plot([], [], "o", color=RED, ms=9)
    im = a1.imshow(S[0, 0].reshape(28, 28), cmap="gray", vmin=0, vmax=1); a1.set_xticks([]); a1.set_yticks([])
    ttl = a1.set_title("", fontsize=11)
    a2.set_title("one line through the whole lean x width sheet", fontsize=10.5)
    def upd(i):
        x, y = path[i]; im.set_data(S[y, x].reshape(28, 28))
        line.set_data(xs[:i + 1], ys[:i + 1]); dot.set_data([xs[i]], [ys[i]])
        ttl.set_text(f"frame {i + 1}/64\nlean {leans[x]:.0f}°, width {widths[y]:.1f}")
        return im, line, dot, ttl
    fig.tight_layout()
    FuncAnimation(fig, upd, frames=64, interval=180).save(f"{OUT}/curves_sheet.gif", writer=PillowWriter(fps=6))
    plt.close(fig); print("wrote curves_sheet.gif")


if __name__ == "__main__":
    fig_graph(graph_stats())
    fig_sheet()
