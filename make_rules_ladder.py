"""A ladder of rules: add one at a time, watch the distance to a real 1 fall.

  rules_ladder.png   -- the measurement
  rules_gallery.png  -- what each rung actually produces

Rules, in the order they are added (each rung keeps all the ones above it):
  R1 CONNECTED  one 8-connected piece of ink
  R2 SOLID      >= 40% of inked pixels fully black
  R3 STROKE     the skeleton is a simple path: two ends, no junctions, no loops
  R4 WIDTH      the stroke averages at least 2 px thick (area / skeleton length)
  R5 SMOOTH     the path does not wander -- a bias toward carrying straight on

Run: python make_rules_ladder.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from scipy.ndimage import label, shift as ndshift
import stroke_rules as SR, speckle_free as SF, grey_ones as G, mnist

OUT = "figures"
INK, GOOD, BAD, REF, MID = "#1a1a1a", "#1a7f37", "#c62828", "#1565c0", "#9a9a9a"
NPER = 200
SOLID = 0.40
WMIN = 2.0
TARGET_INK = 83


def centre(Q):
    """Slide each image so its ink centroid sits at the middle of the canvas.
    Removes WHERE the mark is, leaving only what shape it is."""
    out = np.empty_like(Q)
    yy, xx = np.mgrid[0:28, 0:28]
    for i, q in enumerate(Q):
        I = q.reshape(28, 28); m = I.sum()
        if m <= 0: out[i] = q; continue
        cy, cx = (yy * I).sum() / m, (xx * I).sum() / m
        out[i] = np.clip(ndshift(I, (13.5 - cy, 13.5 - cx), order=1, mode="constant"), 0, 1).ravel()
    return out


def unit(Q):
    n = np.linalg.norm(Q, axis=1, keepdims=True)
    return Q / np.maximum(n, 1e-9)


def nearest(Q, R):
    Rsq = (R ** 2).sum(1); out = np.empty(len(Q))
    for s in range(0, len(Q), 400):
        q = Q[s:s + 400]
        d2 = (q ** 2).sum(1)[:, None] + Rsq[None] - 2 * q @ R.T
        out[s:s + 400] = np.sqrt(np.maximum(d2.min(1), 0))
    return out


def gen_blob(n, rng, solid):
    return np.array([SF.shade(SF.grow(TARGET_INK, 1.0, rng), solid, rng).ravel()
                     for _ in range(n)])


def gen_walk(n, rng, length, r_core, r_out, persistence):
    """Strokes by construction: a self-avoiding path, thickened, with a grey halo.
    Rejected if thickening made the skeleton branch or close a loop."""
    out = []
    while len(out) < n:
        p = SR.long_walk(length, rng, persistence=persistence)
        X = SR.core_and_halo(p, r_core, r_out, rng)
        if not SR.is_stroke(X >= 0.5): continue
        out.append(X.ravel())
    return np.array(out)


def audit(Q):
    """Fraction of a set passing each rule, plus mean ink and width."""
    o = np.zeros(4); w = []
    for q in Q:
        im = q.reshape(28, 28); I = im > 0; Iv = im >= 0.5
        o[0] += label(I, np.ones((3, 3)))[1] == 1
        o[1] += ((im[I] >= .95).mean() >= SOLID - 1e-9) if I.any() else 0
        o[2] += SR.is_stroke(Iv)
        ww = SR.mean_width(Iv); w.append(ww)
        o[3] += ww >= WMIN
    return o / len(Q), float(np.mean(w))


def main():
    rng = np.random.default_rng(7)
    Xa, ya = mnist.load("train")
    ONES = (Xa[ya == 1].reshape(-1, 784) / 255.0).astype(np.float64)

    sets = {}
    sets["uniform noise"] = rng.random((NPER, 784))
    sets["sparse noise"] = (rng.random((NPER, 784)) < TARGET_INK / 784) * rng.random((NPER, 784))
    sets["+R1 connected"] = gen_blob(NPER, rng, 0.0)
    sets["+R2 40% black"] = gen_blob(NPER, rng, SOLID)
    sets["+R3 stroke"] = gen_walk(NPER, rng, length=55, r_core=0.0, r_out=0.0, persistence=0.0)
    sets["+R4 width\u22652"] = gen_walk(NPER, rng, length=11, r_core=1.0, r_out=2.2, persistence=2.0)
    sets["+R5 smooth"] = gen_walk(NPER, rng, length=11, r_core=1.0, r_out=2.2, persistence=8.0)
    sets["your generator"] = G.render(G.sample(NPER, rng)).reshape(NPER, -1).astype(np.float64)
    idx = rng.choice(len(ONES), NPER, replace=False)
    sets["a real 1"] = ONES[idx]

    order = list(sets)
    d = {}
    for k in order:
        if k == "a real 1":
            Rsq = (ONES ** 2).sum(1); out = np.empty(NPER)
            for i, j in enumerate(idx):
                dd = np.sqrt(np.maximum((ONES[j] ** 2).sum() + Rsq - 2 * ONES @ ONES[j], 0))
                dd[j] = np.inf; out[i] = dd.min()
            d[k] = out
        else:
            d[k] = nearest(sets[k], ONES)
    # placement-free comparison: centre every image, then compare directions only
    ONES_c = unit(centre(ONES))
    dS = {}
    for k in order:
        Qc = unit(centre(sets[k]))
        if k == "a real 1":
            out = np.empty(NPER)
            for i, j in enumerate(idx):
                dd = np.sqrt(np.maximum(2 - 2 * (ONES_c @ ONES_c[j]), 0)); dd[j] = np.inf
                out[i] = dd.min()
            dS[k] = out
        else:
            dS[k] = nearest(Qc, ONES_c)

    aud = {k: audit(sets[k]) for k in order}
    ink = {k: (sets[k] > 0).sum(1).mean() for k in order}

    print(f"{'set':18s} {'raw':>9s} {'shape':>8s} {'ink':>5s} {'width':>6s}    R1    R2    R3    R4")
    for k in order:
        a, w = aud[k]
        print(f"{k:18s} {np.median(d[k]):9.2f} {np.median(dS[k]):8.3f} {ink[k]:5.0f} {w:6.2f}   " +
              "  ".join(f"{100*x:3.0f}%" for x in a))

    # ---------------- ladder figure ----------------
    fig = plt.figure(figsize=(14.2, 6.4))
    fig.text(.5, .960, "Four rules close about a quarter of the gap to a real 1",
             ha="center", va="center", fontsize=15.5, fontweight="bold")
    fig.text(.5, .906, "Each rung keeps every rule above it. Left: raw distance, which is dominated by WHERE the mark sits. "
                       "Right: the same images slid to a\ncommon centre and compared by direction only, so only the SHAPE counts. "
                       "Bars are medians, whiskers the 10th\u201390th percentile.",
             ha="center", va="center", fontsize=9.3, color="#555", linespacing=1.55)
    cols = [GOOD if k == "a real 1" else REF if k == "your generator"
            else BAD if "noise" in k else MID for k in order]
    x = np.arange(len(order))
    for pi, (D, ttl, ylab, top) in enumerate([
            (d, "Raw distance — placement included", "distance to nearest real 1", None),
            (dS, "Shape only — every image re-centred", "shape distance (0 = identical, 1.41 = unrelated)", 1.45)]):
        ax = fig.add_axes([.055 + pi * .495, .225, .40, .60])
        med = np.array([np.median(D[k]) for k in order])
        lo = np.array([np.percentile(D[k], 10) for k in order])
        hi = np.array([np.percentile(D[k], 90) for k in order])
        ax.bar(x, med, color=cols, width=.66)
        ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="none", ecolor="#333",
                    capsize=2.5, lw=1.0)
        for i, m in enumerate(med):
            ax.text(i, m + (top or max(hi)) * .015, f"{m:.2f}" if top else f"{m:.1f}",
                    ha="center", va="bottom", fontsize=8.6, fontweight="bold")
        ax.axhline(med[-1], color=GOOD, ls="--", lw=1.3)
        ax.text(-.35, med[-1] + (top or max(hi)) * .018, "a real 1", ha="left", va="bottom",
                fontsize=8.5, color=GOOD,
                bbox=dict(fc="white", ec="none", alpha=.9, pad=1.2))
        ax.set_xticks(x); ax.set_xticklabels(order, fontsize=8.4, rotation=30, ha="right")
        ax.set_ylabel(ylab, fontsize=9.2)
        ax.set_ylim(0, top or max(hi) * 1.14)
        ax.grid(alpha=.22, axis="y")
        ax.set_title(ttl, fontsize=11, fontweight="bold", pad=8)
    fig.text(.5, .072, "The right-hand panel is the honest one. The four rules take a random image from 1.01 to 0.77; a real 1 sits at 0.18. "
                       "R3 ALONE MAKES IT WORSE\n(1.01) — a branch-free hairline is not more digit-like. R3 only pays once R4 gives it a "
                       "thickness. The rules are worth having and they are not a description.",
             ha="center", va="center", fontsize=9.2, color="#333", style="italic", linespacing=1.6)
    fig.text(.5, .018, "Share of REAL 1s passing each rule:  R1 94%   R2 58%   R3 51%   R4 90%.  "
                       "R3's failures are junctions and loops — the feet and flags on ornate 1s.",
             ha="center", va="center", fontsize=8.5, color="#777")
    fig.savefig(f"{OUT}/rules_ladder.png", dpi=150)
    plt.close(fig)

    # ---------------- gallery ----------------
    rows = [k for k in order if k not in ("uniform noise", "sparse noise")]
    ncol = 8
    fig = plt.figure(figsize=(12.6, 8.6))
    fig.text(.5, .965, "What each rung actually draws",
             ha="center", va="center", fontsize=15.5, fontweight="bold")
    fig.text(.5, .925, "Same eight samples per rung, untouched. The number under each is its SHAPE distance to the nearest real MNIST 1 "
                       "(re-centred; 0 = identical, 1.41 = unrelated).",
             ha="center", va="center", fontsize=9.2, color="#555")
    L, T, B = .155, .885, .065
    rh = (T - B) / len(rows)
    for r, k in enumerate(rows):
        y0 = T - (r + 1) * rh
        for c in range(ncol):
            ax = fig.add_axes([L + c * .092, y0 + rh * .28, .077, rh * .60])
            ax.imshow(sets[k][c].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            em = k in ("a real 1", "your generator")
            for s_ in ax.spines.values():
                s_.set_edgecolor(GOOD if k == "a real 1" else REF if k == "your generator" else "#ccc")
                s_.set_linewidth(1.8 if em else .6)
            ax.text(.5, -.10, f"{dS[k][c]:.2f}", transform=ax.transAxes, ha="center",
                    va="top", fontsize=8, fontweight="bold",
                    color=GOOD if dS[k][c] < .35 else BAD)
        fig.text(L - .014, y0 + rh * .57, k, ha="right", va="center", fontsize=10,
                 fontweight="bold",
                 color=GOOD if k == "a real 1" else REF if k == "your generator" else INK)
    fig.text(.5, .026, "R5 produces smooth thick strokes that look hand-drawn, and they still sit 4× further from a 1 than a 1 does.\n"
                       "And look at R3: a branch-free hairline comes out as an OUTLINE, not a stroke — which is why it scores worse than a blob.",
             ha="center", va="center", fontsize=9.3, color="#333", style="italic", linespacing=1.6)
    fig.savefig(f"{OUT}/rules_gallery.png", dpi=145)
    plt.close(fig)


if __name__ == "__main__":
    main()
