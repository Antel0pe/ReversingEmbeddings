"""Two figures for analyze_generated_one_line_interpolation.

Fig 1: do the five one-knob lines from the root add up to a joint change?
Fig 2: how many lines does it take to reach the whole family?
"""
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import map_coordinates

from coverage_grids import render_coverage
from grey_ones import KNOBS, RANGES, render

OUTDIR = "figures/generated_one_line_interpolation"
ROOT = np.array([14.0, 14.0, 19.75, 3.2, 0.0])
UNIT = np.eye(5)
SPAN = RANGES[:, 1] - RANGES[:, 0]
UNITS = ["px", "px", "px", "px", "deg"]
INK = "#2b2b2b"


def img(p):
    return render(np.asarray(p, float))[0].astype(np.float64)


def flat(P):
    return render(np.asarray(P, float)).reshape(len(P), -1).astype(np.float64)


I0 = img(ROOT)
LINES = np.array([img(ROOT + UNIT[k]) - I0 for k in range(5)])


def show_img(ax, a, title, sub=None, diverging=False, vmax=None):
    if diverging:
        m = vmax or np.abs(a).max()
        ax.imshow(a, cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -m, m), interpolation="nearest")
    else:
        ax.imshow(a, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10.5, pad=4)
    if sub:
        ax.text(0.5, -0.09, sub, transform=ax.transAxes, ha="center", va="top", fontsize=9.5)
    for sp in ax.spines.values():
        sp.set_color("#999"); sp.set_linewidth(0.8)


# ====================================================================== fig 1
def fig1():
    fig = plt.figure(figsize=(14.5, 10.2))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.965, "Five one-knob lines from the root 1: adding two of them "
             "does not land on the joint change", ha="center", fontsize=16, weight="bold")
    fig.text(0.5, 0.932, "An image is 28x28 ink-coverage numbers (0 = blank, 1 = full ink). "
             "A \"line\" is one image minus another, so it is itself a 28x28 picture.",
             ha="center", fontsize=11, color="#444")

    # ---- row 1: the root and the five lines
    fig.text(0.035, 0.885, "A.  the starting picture, and what +1 on each dial does to it",
             fontsize=12.5, weight="bold")
    vm = np.abs(LINES).max()
    ax = fig.add_axes([0.035, 0.615, 0.125, 0.215])
    show_img(ax, I0, "root 1", "cx 14, cy 14, height 19.75,\nwidth 3.2, lean 0")
    for k in range(5):
        ax = fig.add_axes([0.195 + 0.148 * k, 0.615, 0.125, 0.215])
        show_img(ax, LINES[k].reshape(28, 28), f"line {k + 1}:  {KNOBS[k]} +1 {UNITS[k]}",
                 f"length {np.linalg.norm(LINES[k]):.2f}\n"
                 f"(+1 = {100 / SPAN[k]:.0f}% of its range)", diverging=True, vmax=vm)
    cax = fig.add_axes([0.937, 0.635, 0.011, 0.175])
    plt.colorbar(plt.cm.ScalarMappable(TwoSlopeNorm(0, -vm, vm), "RdBu_r"), cax=cax)
    cax.set_ylabel("ink gained (red) / lost (blue)", fontsize=8.5, labelpad=2)
    cax.tick_params(labelsize=8)

    # ---- row 2 left: the corner test on the worst pair
    i, j = 0, 3                                   # cx + width, the worst pair
    true = img(ROOT + UNIT[i] + UNIT[j])
    pred = I0 + LINES[i] + LINES[j]
    miss = pred - true
    err = 100 * np.linalg.norm(miss) / np.linalg.norm(true - I0)
    fig.text(0.035, 0.535, f"B.  line 1 + line 4 added together, against the real "
             f"{KNOBS[i]}+1 {KNOBS[j]}+1 image", fontsize=12.5, weight="bold")
    for n, (a, t, sb, dv) in enumerate([
            (true, "the real thing\nrender(cx 15, width 4.2)", "what we are aiming at", False),
            (pred.reshape(28, 28), "root + line 1 + line 4", "the two lines added", False),
            (miss.reshape(28, 28), "what the sum got wrong", f"miss = {err:.1f}% of the move", True)]):
        ax = fig.add_axes([0.035 + 0.148 * n, 0.235, 0.125, 0.215])
        show_img(ax, a.reshape(28, 28), t, sb, diverging=dv)
        if dv:
            ax.set_title(t, fontsize=10.5, pad=4, color="#b2182b")
    ax = fig.add_axes([0.035, 0.115, 0.42, 0.075]); ax.axis("off")
    rows = []
    for a, b in itertools.combinations(range(5), 2):
        tc = flat([ROOT + UNIT[a] + UNIT[b]])[0]
        pc = (I0 + LINES[a] + LINES[b]).ravel()
        rows.append((f"{KNOBS[a]}+{KNOBS[b]}",
                     100 * np.linalg.norm(pc - tc) / np.linalg.norm(tc - I0.ravel())))
    rows.sort(key=lambda r: -r[1])
    ax.text(0, 1.0, "all ten pairs, at +1 on each dial, miss by:", fontsize=10,
            weight="bold", va="top")
    ax.text(0, 0.62, "   ".join(f"{n} {e:.0f}%" for n, e in rows[:5]), fontsize=9.5, va="top")
    ax.text(0, 0.30, "   ".join(f"{n} {e:.0f}%" for n, e in rows[5:]), fontsize=9.5, va="top")
    t5 = flat([ROOT + UNIT.sum(0)])[0]
    e5 = 100 * np.linalg.norm((I0 + LINES.sum(0)).ravel() - t5) / np.linalg.norm(t5 - I0.ravel())
    ax.text(0, -0.02, f"all five dials at +1 together: miss = {e5:.0f}% of the move",
            fontsize=10, va="top", weight="bold", color="#b2182b")

    # ---- row 2 right: error vs step size
    ax = fig.add_axes([0.60, 0.145, 0.345, 0.37])
    ts = np.array([2, 1, .5, .25, .1, .05, .02, .01, .005, .002])
    ys = []
    for t in ts:
        ln = np.array([flat([ROOT + t * UNIT[k]])[0] - I0.ravel() for k in range(5)])
        tt = flat([ROOT + t * UNIT.sum(0)])[0]
        ys.append(100 * np.linalg.norm(I0.ravel() + ln.sum(0) - tt) / np.linalg.norm(tt - I0.ravel()))
    ys = np.array(ys)
    ax.loglog(ts, ys, "o-", color=INK, lw=1.8, ms=6, label="measured miss")
    g = np.array([0.002, 0.3])
    ax.loglog(g, ys[-1] * (g / ts[-1]), "--", color="#2166ac", lw=1.5,
              label="straight line = miss proportional to step\n(second order, since the move itself grows like the step)")
    for (t, y), off in (((1.0, ys[1]), (-18, -42)), ((0.05, ys[5]), (-72, 18))):
        ax.annotate(f"step = {t:g}\nmiss {y:.1f}%", (t, y), textcoords="offset points",
                    xytext=off, fontsize=9.5, ha="center",
                    arrowprops=dict(arrowstyle="-", color="#777", lw=0.8))
    ax.set_ylim(0.03, 400)
    ax.set_xlabel("step on every dial, as a multiple of +1 unit", fontsize=11)
    ax.set_ylabel("miss, as % of how far the image moved", fontsize=11)
    ax.set_title("C.  the miss is second order: it is not a fixed tax,\n"
                 "it shrinks in proportion to the step you take",
                 fontsize=12.5, weight="bold", loc="left", pad=8)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8.2, loc="lower right")
    ax.tick_params(labelsize=9.5)

    fig.text(0.035, 0.082, "Fixed: every move starts from the same root 1. The five lines are "
             "measured once, with a +1 step, and reused. \"% of the move\" divides the mistake by "
             "the distance\nthe real image travelled from the root, so 0% means the sum landed "
             "exactly and 100% means the sum was as wrong as not moving at all. Panel C moves all "
             "five dials at once and\nremeasures the lines at each step size; the straight dashed "
             "line is what 'proportional to the step' looks like on these axes (the measured "
             "log-log slope over steps 0.002-0.05 is 2.00).",
             fontsize=9, color="#444", va="top", linespacing=1.6)
    fig.savefig(f"{OUTDIR}/lines_do_not_add.png", dpi=135, facecolor="white")
    plt.close(fig)
    print("wrote lines_do_not_add.png")


# ====================================================================== fig 2
def fig2():
    rng = np.random.default_rng(1)
    Ptr = RANGES[:, 0] + rng.random((4000, 5)) * SPAN
    Pte = RANGES[:, 0] + rng.random((2000, 5)) * SPAN
    Itr, Ite = flat(Ptr), flat(Pte)
    nte = np.linalg.norm(Ite, axis=1)

    def design(P, deg):
        u = 2 * (P - RANGES[:, 0]) / SPAN - 1
        c = [np.prod(u ** np.array(e), axis=1)
             for e in itertools.product(*[range(deg + 1)] * 5) if sum(e) <= deg]
        return np.column_stack(c)

    poly = []
    for deg in (1, 2, 3, 4):
        A, B = design(Ptr, deg), design(Pte, deg)
        W = np.linalg.lstsq(A, Itr, rcond=None)[0]
        poly.append((A.shape[1], np.median(100 * np.linalg.norm(B @ W - Ite, axis=1) / nte)))
    mu = Itr.mean(0)
    V = np.linalg.svd(Itr - mu, full_matrices=False)[2]
    ks = np.arange(1, 161)
    D = Ite - mu
    co = D @ V[:160].T
    tot = (D ** 2).sum(1)
    pcacurve = [np.median(100 * np.sqrt(np.maximum(tot - (co[:, :k] ** 2).sum(1), 0)) / nte)
                for k in ks]

    fig = plt.figure(figsize=(16.5, 7.6))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.955, "How many lines does it take to reach the whole family?",
             ha="center", fontsize=16, weight="bold")
    fig.text(0.5, 0.916, "Error below is always |predicted image - true image| / |true image|, "
             "in percent, on held-out settings drawn from the full default range box.",
             ha="center", fontsize=11, color="#444")

    # --- A: number of lines
    ax = fig.add_axes([0.045, 0.285, 0.265, 0.555])
    ax.semilogy(ks, pcacurve, "-", color="#2166ac", lw=2,
                label="best possible fixed images\n(coefficients read off the target)")
    ax.semilogy([p[0] for p in poly], [p[1] for p in poly], "s-", color="#b2182b", lw=2, ms=7,
                label="lines and products of the 5 dials\n(coefficients computed from the knobs)")
    for (n, e), lab in zip(poly, ["5 lines\n+ mean", "+ pairs", "+ triples", "+ quads"]):
        ax.annotate(f"{lab}: {n} lines, {e:.0f}%", (n, e), textcoords="offset points",
                    xytext=(9, 11) if n < 100 else (8, -17), fontsize=9, color="#b2182b",
                    ha="left")
    ax.axhline(1.0, color="#444", ls=":", lw=1.4)
    ax.text(178, 1.1, "1% target", fontsize=9.5, ha="right", color="#444")
    k1 = int(ks[np.argmax(np.array(pcacurve) < 1.0)])
    ax.plot([k1], [1.0], "o", color="#2166ac", ms=9, mfc="white", mew=2)
    ax.annotate(f"{k1} fixed images\nreach 1%", (k1, 1.0), textcoords="offset points",
                xytext=(12, 26), fontsize=9.5, color="#2166ac")
    ax.set_xlabel("number of fixed 28x28 pictures used (\"lines\")", fontsize=11)
    ax.set_ylabel("held-out error, % of the image", fontsize=11)
    ax.set_title("A.  no small set of lines spans it", fontsize=13, weight="bold", loc="left", pad=8)
    ax.set_xlim(0, 185); ax.set_ylim(0.3, 70)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.2, loc="lower left")
    ax.tick_params(labelsize=9.5)

    # --- B: patches of five lines
    ax = fig.add_axes([0.385, 0.285, 0.265, 0.555])
    rng2 = np.random.default_rng(2)
    centres = [ROOT] + list(RANGES[:, 0] + rng2.random((4, 5)) * SPAN)
    hs = 0.5 ** np.arange(1, 11)
    curves = []
    for c in centres:
        row = []
        ic = flat([c])[0]
        for h in hs:
            ln = np.array([flat([c + h * SPAN * UNIT[k]])[0] - ic for k in range(5)])
            du = rng2.uniform(-h, h, (300, 5)) * SPAN
            T = flat(c + du)
            pr = ic + (du / (h * SPAN)) @ ln
            row.append(np.percentile(100 * np.linalg.norm(pr - T, axis=1)
                                     / np.linalg.norm(T, axis=1), 95))
        curves.append(row)
    for n, row in enumerate(curves):
        ax.loglog(hs, row, "o-", lw=1.6, ms=4.5,
                  color=INK if n == 0 else "#7fa8cf",
                  label="the root 1 (an unusually flat spot)" if n == 0
                  else ("4 other starting 1s" if n == 1 else None), zorder=3 if n == 0 else 2)
    pool = np.array(curves)[1:].mean(0)
    m = pool < 5
    a, b = np.polyfit(np.log(hs[m]), np.log(pool[m]), 1)
    hstar = float(np.exp((np.log(1.0) - b) / a))
    ax.axhline(1.0, color="#444", ls=":", lw=1.4)
    ax.axvline(hstar, color="#b2182b", ls="--", lw=1.4)
    ax.text(0.03, 0.80, f"patch must be +-{100 * hstar:.1f}% of each\n"
            f"range (red dashes) to stay under 1%", fontsize=9.5, color="#b2182b",
            transform=ax.transAxes, va="top")
    ax.set_xlabel("patch half-width, as a fraction of each dial's range", fontsize=11)
    ax.set_ylabel("error at the patch edge (95th pct), % of image", fontsize=11)
    ax.set_title("B.  five fresh lines work, but only nearby", fontsize=13,
                 weight="bold", loc="left", pad=8)
    n_patch = (1 / (2 * hstar)) ** 5
    ax.text(0.50, 0.30, f"tiling the whole 5-dial box\nwith such patches:\n"
            f"{1 / (2 * hstar):.0f} per dial\n{n_patch:,.0f} patches\n"
            f"{5 * n_patch:,.0f} lines", transform=ax.transAxes, fontsize=9.5, va="top",
            bbox=dict(fc="#fdeaec", ec="#b2182b", lw=0.8, pad=4))
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9, loc="upper left")
    ax.tick_params(labelsize=9.5)

    # --- C: warp space
    ax = fig.add_axes([0.715, 0.615, 0.265, 0.225])
    cx0, cy0, h0, w0 = ROOT[:4]
    P = Pte[:300]
    cxv, cyv, hv, wv, lv = P.T
    sx, sy = wv / w0, hv / h0
    a12 = -np.tan(np.radians(lv)) * sy
    C = np.column_stack([sx, a12, sy, cxv - cx0 * sx - cy0 * a12, cyv - cy0 * sy])
    res = []
    for U in (P, np.column_stack([cxv, cyv, hv, wv, hv * np.tan(np.radians(lv))])):
        A = np.column_stack([U, np.ones(len(U))])
        R = C - A @ np.linalg.lstsq(A, C, rcond=None)[0]
        res.append(max(100 * np.linalg.norm(R) / np.linalg.norm(C - C.mean(0)), 1e-13))
    I0f = I0
    gy, gx = np.mgrid[0:28, 0:28]
    tpl = {}
    for n in (28, 56, 112, 224):
        f = n // 28
        rootf = render_coverage(ROOT[None], n_pix=n)[0]
        pit = 28.0 / n
        g2y, g2x = np.mgrid[0:n, 0:n]
        xq, yq = (g2x + 0.5) * pit, (g2y + 0.5) * pit
        e = []
        for c, q in zip(C[:80], P[:80]):
            s1, s2, s3, bx, by = c
            Y = (yq - by) / s3
            X = (xq - bx - s2 * Y) / s1
            wf = map_coordinates(rootf, [Y / pit - 0.5, X / pit - 0.5], order=1, cval=0.0)
            t = img(q).ravel()
            e.append(100 * np.linalg.norm(wf.reshape(28, f, 28, f).mean((1, 3)).ravel() - t)
                     / np.linalg.norm(t))
        tpl[n] = np.median(e)
    labels = ["PIXEL space\n(panel A)", "WARP space\ndials as given",
              "WARP space\nheight x tan(lean)"]
    vals = [poly[0][1], res[0], res[1]]
    cols = ["#b2182b", "#d6a000", "#1a7f37"]
    ax.bar(range(3), vals, color=cols, width=0.6)
    ax.set_yscale("log"); ax.set_ylim(1e-13, 300)
    for n, v in enumerate(vals):
        ax.text(n, v * 1.7, f"{v:.3g}%" if v > 1e-6 else f"{v:.0e}%", ha="center",
                fontsize=10, weight="bold", color=cols[n])
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("miss (%, log scale)", fontsize=10)
    ax.set_title("C.  five lines that DO add up: warp space", fontsize=13,
                 weight="bold", loc="left", pad=8)
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.tick_params(labelsize=9.5)
    ax2 = fig.add_axes([0.762, 0.305, 0.218, 0.145])
    pit = np.array([28 / n for n in tpl])
    ax2.loglog(pit, list(tpl.values()), "o-", color="#1a7f37", lw=1.8, ms=5.5)
    for n, (q, v) in zip(tpl, zip(pit, tpl.values())):
        ax2.annotate(f"{n}x{n}", (q, v), textcoords="offset points", xytext=(0, 7),
                     fontsize=7.5, ha="center", color="#1a7f37")
    ax2.set_xlabel("pitch of the root template (px)", fontsize=8.5, labelpad=1)
    ax2.set_ylabel("% of image", fontsize=8.5, labelpad=1)
    ax2.set_title("cost of turning those five exact warp numbers\n"
                  "back into 28x28 pixels, by warping a template",
                  fontsize=8.5, color="#1a7f37", pad=3)
    ax2.tick_params(labelsize=7.5); ax2.grid(alpha=0.3, which="both")
    ax2.set_ylim(0.06, 60); ax2.set_xlim(0.085, 1.5)

    fig.text(0.045, 0.155, "Panel A: 4000 training / 2000 held-out settings drawn uniformly from "
             "the default range box; red fits a polynomial in the five dials by least squares "
             "(6, 21, 56, 126 terms), blue is the\nbest any k fixed pictures can do (PCA of the "
             "same training images), which is an unbeatable ceiling, and its coefficients cannot "
             "be computed from the dials.  Panel B: each patch re-measures five fresh\nlines at "
             "its own centre; points are drawn uniformly inside the patch; the dashed red line is "
             "where the fitted power law (slope 1.9, close to the second-order slope 2) crosses "
             "1%.  Panel C: warp space means\nthe five numbers of the affine canvas map "
             "(x = bx + sx X + a12 Y, y = by + sy Y) that carries the root stroke onto the target "
             "stroke; the two right-hand bars add lines in that space instead of in pixels.",
             fontsize=8.8, color="#444", va="top", linespacing=1.6)
    fig.savefig(f"{OUTDIR}/how_many_lines.png", dpi=130, facecolor="white")
    plt.close(fig)
    print("wrote how_many_lines.png")


if __name__ == "__main__":
    fig1()
    fig2()
