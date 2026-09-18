"""The smallest exact equation for the toy 1, found by optimisation.

"Fewest terms" can be gamed -- any sum can be bundled into one term -- so size is
measured in MONOMIALS. On on/off pixels every function has exactly one expansion
as a sum of monomials: a constant, single pixels x_i, pairs x_i*x_j, triples, ...
(x^2 = x, so nothing else can appear). The size of an equation is how many of
those have a nonzero coefficient.

The search: fewest monomials such that the equation is exactly 0 on every valid
image and at least 1 on every other image -- the same shape as the hand-built
E(x), which is a count of violations. That is a mixed-integer program: one
continuous coefficient per candidate monomial, one yes/no switch saying whether
it is used, minimise the number of switches that are on.

On a 4x4 grid all 65,536 images go in as constraints directly. On 5x5 (33.5M
images) the search solves on a subset, checks the answer against every image,
adds whatever it got wrong, and repeats until nothing is wrong.

Run:  python sparsest_equation.py          (4x4 search + figure, ~2 min)
      python sparsest_equation.py --big    (also the 5x5 search, can take an hour)
Writes figures/equation_sparsest.png and figures/sparsest_solutions.json.
"""

import matplotlib
matplotlib.use("Agg")
import itertools, json, sys, time, os
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix, vstack, hstack, identity

OUT = "figures"
CACHE = f"{OUT}/sparsest_solutions.json"
GREEN, RED, BLUE, GREY = "#1a7f37", "#c62828", "#1565c0", "#444"


# ---------------------------------------------------------------------------
# the toy
# ---------------------------------------------------------------------------
def valid_set(H, W, band, wd):
    V = []
    for c in range(W - wd + 1):
        im = np.zeros((H, W), np.int8); im[band[0]:band[-1] + 1, c:c + wd] = 1; V.append(im.ravel())
    return np.array(V)


def images(N, s, e):
    return ((np.arange(s, e, dtype=np.int64)[:, None] >> np.arange(N)[::-1]) & 1).astype(np.int8)


def monomials(N, deg):
    return [()] + [m for d in range(1, deg + 1) for m in itertools.combinations(range(N), d)]


def features(X, mons):
    F = np.ones((len(X), len(mons)), np.int8)
    for k, m in enumerate(mons):
        for i in m:
            F[:, k] &= X[:, i]
    return F


def evaluate(terms, X):
    X = np.atleast_2d(X)
    return sum(c * np.prod(X[:, list(m)], axis=1) if m else c * np.ones(len(X)) for m, c in terms)


def hand_built(H, W, band, wd):
    """Our E = A + B + C + D, expanded into monomials (exact, by solving on enough images)."""
    N = H * W
    out = [r for r in range(H) if r not in band]

    def E(X):
        I = X.reshape(-1, H, W).astype(int)
        return (I[:, out].sum((1, 2)) + ((I[:, band[:-1]] - I[:, band[1:]]) ** 2).sum((1, 2))
                + (I[:, band[0]].sum(1) - wd) ** 2
                + sum(I[:, band[0], i] * I[:, band[0], j] for i in range(W) for j in range(i + wd, W)))
    mons = monomials(N, 2)
    X = images(N, 0, 1 << N) if N <= 16 else \
        (np.random.default_rng(0).random((80000, N)) < .4).astype(np.int8)
    c = np.linalg.lstsq(features(X, mons).astype(float), E(X).astype(float), rcond=None)[0]
    return [(mons[k], round(float(c[k]), 6)) for k in range(len(mons)) if abs(c[k]) > 1e-6]


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------
def _milp(Fv, Fo, K, M, tlim, max_used=None):
    """Variables [c | z | t]: coefficients, used-switches, |c|.
    max_used=None: minimise the number of monomials used.
    max_used=n:    among equations with at most n monomials, minimise the total |coefficient|."""
    I = identity(K, format="csr"); Z = csr_matrix((K, K))
    pad = lambda A: hstack([csr_matrix(A), csr_matrix((A.shape[0], 2 * K))])
    cons = [LinearConstraint(pad(Fv), 0, 0), LinearConstraint(pad(Fo), 1, np.inf),
            LinearConstraint(vstack([hstack([I, -M * I, Z]), hstack([-I, -M * I, Z]),
                                     hstack([I, Z, -I]), hstack([-I, Z, -I])]), -np.inf, 0)]
    if max_used is None:
        obj = np.r_[np.zeros(K), np.ones(K), np.zeros(K)]
    else:
        obj = np.r_[np.zeros(2 * K), np.ones(K)]
        cons.append(LinearConstraint(np.r_[np.zeros(K), np.ones(K), np.zeros(K)][None], 0, max_used))
    return milp(obj, constraints=cons, integrality=np.r_[np.zeros(K), np.ones(K), np.zeros(K)],
                bounds=Bounds(np.r_[-M * np.ones(K), np.zeros(2 * K)],
                              np.r_[M * np.ones(K), np.ones(K), M * np.ones(K)]),
                options={"time_limit": tlim})


def sparsest(H, W, band, wd, deg=2, M=64.0, tlim=1200, seed=0, verbose=True):
    """Fewest monomials: 0 on valid images, >= 1 on all others. Returns (terms, proven_optimal)."""
    N = H * W
    V = valid_set(H, W, band, wd)
    mons = monomials(N, deg); K = len(mons)
    Fv = features(V, mons).astype(float)
    rng = np.random.default_rng(seed)
    if N <= 16:
        S = images(N, 0, 1 << N)
    else:
        S = np.array([v ^ np.eye(N, dtype=np.int8)[i] for v in V for i in range(N)]
                     + list((rng.random((3000, N)) < 0.3).astype(np.int8)))
    S = np.unique(S, axis=0)
    S = S[~(S[:, None, :] == V[None]).all(2).any(1)]
    t0 = time.time()
    for it in range(60):
        Fo = features(S, mons).astype(float)
        r = _milp(Fv, Fo, K, M, tlim)                         # stage 1: fewest monomials
        if r.x is None:
            return None, False
        n_min = int(round(r.x[K:2 * K].sum()))
        r2 = _milp(Fv, Fo, K, M, tlim, max_used=n_min)        # stage 2: tidiest among them
        if r2.x is not None:
            r = r2
        c = r.x[:K]; c[np.abs(c) < 1e-7] = 0
        terms = [(mons[k], float(c[k])) for k in np.nonzero(c)[0]]
        bad = []
        for s in range(0, 1 << N, 1 << 20):
            Xc = images(N, s, min(s + (1 << 20), 1 << N))
            isv = (Xc[:, None, :] == V[None]).all(2).any(1)
            val = evaluate(terms, Xc)
            bad.append(Xc[(~isv) & (val < 1 - 1e-4)])
            assert not isv.any() or np.abs(val[isv]).max() < 1e-6
            if sum(len(b) for b in bad) > 4000:
                break
        bad = np.concatenate(bad)
        if verbose:
            print(f"  {H}x{W} iter {it}: {len(S):6d} constraints -> {len(terms)} monomials, "
                  f"{len(bad)} images wrong ({time.time() - t0:.0f}s)", flush=True)
        if len(bad) == 0:
            return terms, r.status == 0
        S = np.unique(np.vstack([S, bad[rng.permutation(len(bad))[:1500]]]), axis=0)
    return None, False


# ---------------------------------------------------------------------------
# drawing an equation on the pixel grid
# ---------------------------------------------------------------------------
def _fmt(c):
    return f"{round(c, 3):+g}"


def draw_equation(ax, H, W, band, terms, title, colour):
    ax.set_xlim(-0.55, W - 0.45); ax.set_ylim(H - 0.45, -0.55); ax.set_aspect("equal"); ax.axis("off")
    lin = {m[0]: c for m, c in terms if len(m) == 1}
    const = sum(c for m, c in terms if not m)
    for r in range(H):
        for c in range(W):
            i = r * W + c
            fc = "#fff7d6" if r in band else "#e6e6e6"
            if i in lin:
                fc = "#f6c7c7" if lin[i] > 0 else "#c8daf5"
            ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, fc=fc, ec="#999", lw=1, zorder=1))
            if i in lin:                                   # single-pixel coefficient: corner label
                ax.text(c + .44, r + .44, _fmt(lin[i]), ha="right", va="bottom", fontsize=10,
                        color=RED if lin[i] > 0 else BLUE, weight="bold", zorder=6)
    pairs = [(m, c) for m, c in terms if len(m) == 2]
    # only label lines where a colour carries more than one value
    label_sign = {sg for sg in (1, -1)
                  if len({round(c, 3) for _, c in pairs if np.sign(c) == sg}) > 1}
    for (i, j), c in pairs:
        p1 = np.array(divmod(i, W)[::-1], float); p2 = np.array(divmod(j, W)[::-1], float)
        col = RED if c > 0 else BLUE
        d = p2 - p1; L = np.linalg.norm(d)
        if L <= 1.0 + 1e-9:                                # neighbours: straight line
            ctrl = (p1 + p2) / 2
        else:                                              # bow away from the pixels in between
            perp = np.array([d[1], -d[0]]) / L
            if perp[1] > 0 or (perp[1] == 0 and perp[0] < 0):
                perp = -perp
            ctrl = (p1 + p2) / 2 + perp * 0.30 * L
        t = np.linspace(0, 1, 40)[:, None]
        curve = (1 - t) ** 2 * p1 + 2 * t * (1 - t) * ctrl + t ** 2 * p2
        ax.plot(curve[:, 0], curve[:, 1], color=col, lw=1.5 + 0.3 * min(abs(c), 8), alpha=.85,
                zorder=3, solid_capstyle="round")
        ax.plot(*np.c_[p1, p2], "o", ms=5, color=col, zorder=4)
        mid = 0.25 * p1 + 0.5 * ctrl + 0.25 * p2
        if np.sign(c) not in label_sign:
            continue
        ax.text(mid[0], mid[1], _fmt(c), ha="center", va="center", fontsize=9, color=col,
                weight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col, lw=0.8, alpha=.95))
    higher = [(m, c) for m, c in terms if len(m) > 2]
    ax.set_title(title, fontsize=13, weight="bold", color=colour, pad=14, linespacing=1.4)
    stats = (f"constant {_fmt(const)}   ·   {len(lin)} single-pixel terms   ·   {len(pairs)} pair terms"
             + (f"   ·   {len(higher)} triples" if higher else ""))
    pv = {sg: sorted({round(c, 3) for _, c in pairs if np.sign(c) == sg}) for sg in (1, -1)}
    uniform = "   ·   ".join(f"every {nm} line is {_fmt(pv[sg][0])}"
                              for sg, nm in ((1, "red"), (-1, "blue")) if len(pv[sg]) == 1)
    return stats, uniform


def main():
    big = "--big" in sys.argv
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    runs = [(4, 4, [1, 2], 2)] + ([(5, 5, [1, 2, 3], 2)] if big or "5x5" in cache else [])
    res = {}
    for H, W, band, wd in runs:
        key = f"{H}x{W}"
        hb = hand_built(H, W, band, wd)
        if key in cache and not (big and key != "4x4" and "--rerun" in sys.argv):
            sp = [(tuple(m), c) for m, c in cache[key]["sparsest"]]; opt = cache[key]["optimal"]
        else:
            sp, opt = sparsest(H, W, band, wd)
        res[key] = (H, W, band, wd, hb, sp, opt)
        cache[key] = {"sparsest": [(list(m), c) for m, c in sp], "optimal": bool(opt),
                      "hand_built": [(list(m), c) for m, c in hb]}
        print(f"{key}: hand-built {len(hb)} monomials, sparsest {len(sp)} "
              f"({'proven optimal' if opt else 'best found'})")
    json.dump(cache, open(CACHE, "w"), indent=1)

    # ---- figure: one row per grid size, hand-built vs sparsest, then examples ----
    H, W, band, wd, hb, sp, opt = res["4x4"]
    ex = []
    V = valid_set(H, W, band, wd)
    ex.append(("VALID", V[0])); ex.append(("VALID", V[2]))
    z = np.zeros((H, W), np.int8); ex.append(("blank", z.ravel()))
    z = np.zeros((H, W), np.int8); z[1:3, 1] = 1; ex.append(("1 column", z.ravel()))
    z = np.zeros((H, W), np.int8); z[1:3, 0:3] = 1; ex.append(("3 columns", z.ravel()))
    z = np.zeros((H, W), np.int8); z[1:3, [0, 2]] = 1; ex.append(("split", z.ravel()))
    z = np.zeros((H, W), np.int8); z[1, 0:2] = 1; z[2, 1:3] = 1; ex.append(("rows slid", z.ravel()))
    z = V[1].reshape(H, W).copy(); z[3, 0] = 1; ex.append(("stray dot", z.ravel()))

    grids = list(res.keys())
    FW, GRID, BLOCK = 17.0, 5.0, 8.3          # inches: figure width, grid side, one grid-size block
    FH = 1.9 + BLOCK * len(grids) + 3.1
    fig = plt.figure(figsize=(FW, FH))
    fy = lambda i: 1 - i / FH
    fig.text(0.5, fy(0.42), "THE SMALLEST EXACT EQUATION:  what an optimiser does with the same job",
             ha="center", fontsize=17, weight="bold")
    fig.text(0.5, fy(0.85), "Size = number of monomials: a constant, single pixels x, and pairs x·x "
             "(on/off pixels have exactly one such expansion, so this can't be gamed).",
             ha="center", fontsize=11.5)
    fig.text(0.5, fy(1.20), "Both equations are exactly 0 on every valid bar and at least 1 on every "
             "other image -- checked against every possible image on the grid.",
             ha="center", fontsize=11.5, style="italic", color=GREY)
    fig.text(0.5, fy(1.55), "Red = adds when lit (a penalty).   Blue = subtracts when lit (a reward).   "
             "A line joins the two pixels of a pair term; it only counts when BOTH are on.",
             ha="center", fontsize=11, color=GREY)
    readings = [("Thinks in ROWS: rows copy each other (B), then one row\nis counted (C) and checked "
                 "for gaps (D). Squaring C and B\nis what multiplies the number of terms."),
                ("Thinks in COLUMNS:  start at 2,  -1 for every complete column,\n+1 for any ink "
                 "outside the band,  +1 for a top pixel and a\nbottom pixel lit 2+ columns apart "
                 "(a diagonal too long to be one bar).")]
    y = 1.9
    for key in grids:
        H, W, band, wd, hb, sp, opt = res[key]
        gtop = y + 0.85
        for k, (terms, ttl, col) in enumerate([
                (hb, f"OUR HAND-BUILT E  ({key})\n{len(hb)} monomials", GREY),
                (sp, f"THE SPARSEST  ({key})\n{len(sp)} monomials"
                     + ("  (proven minimum)" if opt else "  (best found)"), GREEN)]):
            cx = 0.27 + 0.46 * k
            ax = fig.add_axes([cx - GRID / FW / 2, fy(gtop + GRID), GRID / FW, GRID / FH])
            stats, uniform = draw_equation(ax, H, W, band, terms, ttl, col)
            fig.text(cx, fy(gtop + GRID + 0.25), stats, ha="center", va="top", fontsize=10.5, color=GREY)
            if uniform:
                fig.text(cx, fy(gtop + GRID + 0.60), uniform, ha="center", va="top", fontsize=10.5,
                         color=GREY)
            fig.text(cx, fy(gtop + GRID + 1.05), readings[k], ha="center", va="top", fontsize=11.5,
                     color="#222", linespacing=1.55)
        y += BLOCK

    H, W, band, wd, hb, sp, opt = res["4x4"]
    fig.text(0.5, fy(y + 0.05), "WORKED EXAMPLES (4×4):  what each equation says about the same images",
             ha="center", fontsize=13.5, weight="bold")
    gs = fig.add_gridspec(1, len(ex), left=0.07, right=0.97, top=fy(y + 0.75), bottom=fy(y + 2.35),
                          wspace=0.35)
    for k, (lbl, x) in enumerate(ex):
        a = fig.add_subplot(gs[0, k])
        a.set_facecolor("white")
        img = x.reshape(H, W)
        for r in range(H):
            if r not in band:
                a.axhspan(r - .5, r + .5, color="#9e9e9e", alpha=.35, zorder=1)
        a.imshow(np.ma.masked_where(img == 0, img), cmap="gray_r", vmin=0, vmax=1.6,
                 interpolation="nearest", zorder=2)
        a.set_xticks(np.arange(-.5, W, 1), minor=True); a.set_yticks(np.arange(-.5, H, 1), minor=True)
        a.grid(which="minor", color="#bbb", lw=.6); a.set_xticks([]); a.set_yticks([])
        a.tick_params(which="minor", length=0)
        ok = lbl == "VALID"
        for sp_ in a.spines.values():
            sp_.set_edgecolor(GREEN if ok else RED); sp_.set_linewidth(2.4)
        a.set_title(lbl, fontsize=11, color=GREEN if ok else RED, weight="bold", pad=6)
        e1, e2 = (round(float(evaluate(t_, x)[0]), 6) + 0.0 for t_ in (hb, sp))
        a.text(0.5, -0.10, f"ours: {e1:g}\nsparsest: {e2:g}", transform=a.transAxes, ha="center",
               va="top", fontsize=10.5, color=GREEN if ok else RED, linespacing=1.5)
    fig.savefig(f"{OUT}/equation_sparsest.png", dpi=120, facecolor="white")
    plt.close(fig)
    print("wrote equation_sparsest.png")


if __name__ == "__main__":
    main()
