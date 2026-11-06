"""Figures for `analyze_pixel_metric_locality.py`.

Figure 1  why the pixel basis has no notion of "nearby", and what restores it
Figure 2  whether the derivative and the tangent plane come back
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedFormatter, FixedLocator

from analyze_pixel_metric_locality import (
    CANVAS, KNOBS, METRICS, ROOT, cov, hard, reweight, cosines)

OUT = "figures/pixel_metric_locality"
SHOW = ["pixel", "blur 1px", "blur 4px", "H^-1/2", "H^-1", "sharpen"]
COLOR = {"pixel": "#111111", "blur 1px": "#1f77b4", "blur 4px": "#6baed6",
         "H^-1/2": "#d95f02", "H^-1": "#e7a33e", "sharpen": "#8c564b"}
STYLE = {"pixel": dict(lw=3.0, zorder=5), "sharpen": dict(lw=1.4, ls=":")}


def ttl(ax, txt):
    ax.set_title(txt, fontsize=10, loc="left", pad=8)


def logx(ax, hs, fmt="{:g}"):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(hs))
    ax.xaxis.set_major_formatter(FixedFormatter([fmt.format(h) for h in hs]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.invert_xaxis()


# ============================================================ figure 1

def figure1():
    n = 28
    fig = plt.figure(figsize=(14.0, 10.3))
    gs = fig.add_gridspec(2, 3, hspace=0.72, wspace=0.42,
                          left=0.06, right=0.975, top=0.815, bottom=0.135)

    fig.suptitle("Pixel space does not know that two nearby pixels are nearby — and what restores it",
                 fontsize=15, y=0.982)
    fig.text(0.5, 0.945,
             "Each candidate fix is the same move: re-encode the picture, then use ORDINARY Euclidean distance on the encoding.\n"
             "Every linear, isotropic re-encoding is a reweighting of spatial frequencies: blur s keeps detail coarser than s; H^-k damps every scale at once;\n"
             "sharpen is the control that goes the wrong way.   cos = 1.000 means \"the same direction\"; cos = 0.000 means \"perpendicular, i.e. unrelated\".",
             ha="center", va="top", fontsize=9.4, color="#444444", linespacing=1.6)

    # ---- (a) the object
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(cov(ROOT, n)[0], cmap="gray_r", vmin=0, vmax=1)
    ttl(ax, "a. The object\nroot 1: a 28x28 picture = one point in 784-D")
    ax.set_xlabel("column (px)", fontsize=8.5); ax.set_ylabel("row (px)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("ink coverage (0 = none, 1 = full)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    # ---- (b) the probe
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(np.zeros((n, n)), cmap="gray_r", vmin=0, vmax=1)
    ax.add_patch(Rectangle((7.5, 13.5), 1, 1, color="#c62828"))
    ax.annotate("A", (6.3, 14.0), color="#c62828", fontsize=11, ha="right",
                va="center", fontweight="bold")
    for d, c, yy in [(1, "#1565c0", 12.6), (12, "#2e7d32", 12.6)]:
        ax.add_patch(Rectangle((7.5 + d, 13.5), 1, 1, color=c))
        ax.annotate(f"d={d}", (8 + d, yy), color=c, fontsize=9.5,
                    ha="center", va="bottom", fontweight="bold")
    cA = np.zeros((n, n)); cA[14, 8] = 1
    c1 = np.zeros((n, n)); c1[14, 9] = 1
    c12 = np.zeros((n, n)); c12[14, 20] = 1
    g1, g12 = cosines(cA[None], c1[None], n), cosines(cA[None], c12[None], n)
    ttl(ax, "b. The probe\none pixel of ink, vs the same pixel d columns right")
    ax.set_xlabel("column (px)", fontsize=8.5); ax.set_ylabel("row (px)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.text(0.5, 0.055,
            f"in the PIXEL metric:\n"
            f"cos(A, d=1)  = {g1['pixel']:.3f}\n"
            f"cos(A, d=12) = {g12['pixel']:.3f}\n"
            "one pixel over and half the canvas\nover score IDENTICALLY",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9,
            family="monospace", linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.45", fc="#fff3e0", ec="#e65100"))

    # ---- (c) proximity curve
    ax = fig.add_subplot(gs[0, 2])
    ds = np.arange(0, 20)
    for m in SHOW:
        w = METRICS[m]
        a = reweight(cA[None], w, n)[0].ravel()
        ys = []
        for d in ds:
            o = np.zeros((n, n)); o[14, 8 + d] = 1
            b = reweight(o[None], w, n)[0].ravel()
            ys.append(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        ax.plot(ds, ys, color=COLOR[m], marker="o", ms=3.2, label=m,
                **STYLE.get(m, dict(lw=1.8)))
    ax.axhline(0, color="#999999", lw=0.8)
    ax.axhline(0.1, color="#bbbbbb", lw=0.9, ls="--")
    ax.text(19.3, 0.115, "0.1 = \"now unrelated\"", fontsize=7.5, color="#777777",
            ha="right", va="bottom", zorder=9,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    ttl(ax, "c. Does the metric see spatial distance?\nthe same two pixels, scored by each encoding")
    ax.set_xlabel("offset d between the two pixels (px)", fontsize=8.5)
    ax.set_ylabel("cos between the two one-pixel pictures", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.set_ylim(-0.62, 1.06); ax.set_xlim(-0.5, 19.5)
    ax.legend(fontsize=7.6, loc="lower left", framealpha=0.96, ncol=3)
    ax.annotate("pixel metric: a STEP.\n1 at d=0, 0 for every d>0.\nSpatial distance does not exist.",
                xy=(3.0, 0.0), xytext=(6.2, 0.70), fontsize=8.4, color="#111111",
                arrowprops=dict(arrowstyle="->", color="#111111", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.38", fc="white", ec="#111111", lw=1.0))

    # ---- (d,e) the gesture and its continuation
    N = 224
    k = KNOBS.index("width")
    I0 = hard(ROOT, N)[0]
    p1 = ROOT.copy(); p1[k] += 1.0
    p2 = ROOT.copy(); p2[k] += 2.0
    I1, I2 = hard(p1, N)[0], hard(p2, N)[0]
    d1, d2 = I1 - I0, I2 - I1
    far = np.roll(d1, int(10 * N / CANVAS), axis=1)
    lo, hi = 10.0, 18.0
    sl = slice(int(lo * N / CANVAS), int(hi * N / CANVAS))
    ext = [lo, hi, 22.0, 6.0]
    rows = slice(int(6 * N / CANVAS), int(22 * N / CANVAS))

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(d1[rows, sl], cmap="RdBu_r", vmin=-1, vmax=1, extent=ext, aspect="auto")
    ttl(ax, "d. One gesture: widen by 1 px\nchange in ink,  R(w+1) - R(w)")
    ax.set_xlabel("column (px)", fontsize=8.5); ax.set_ylabel("row (px)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("change in ink\nred = gains, blue = loses", fontsize=7.2)
    cb.ax.tick_params(labelsize=7)
    ax.text(0.5, -0.30, "Binary ink on a 224x224 grid (pitch 0.125 px),\nso the stroke edges are true cliffs.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.3, color="#444444")

    ax = fig.add_subplot(gs[1, 1])
    sup = np.zeros_like(d1)
    sup[d1 != 0] = 1
    sup[d2 != 0] = 2
    cmap = ListedColormap(["#f5f5f5", "#2e7d32", "#7b1fa2"])
    ax.imshow(sup[rows, sl], cmap=cmap, vmin=-0.5, vmax=2.5, extent=ext, aspect="auto")
    ttl(ax, "e. The SAME gesture continued: widen by 1 px again\nwhich pixels each step touches")
    ax.set_xlabel("column (px)", fontsize=8.5); ax.set_ylabel("row (px)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    for c, lab, yy in [("#2e7d32", "step 1:  w -> w+1", 0.95),
                       ("#7b1fa2", "step 2:  w+1 -> w+2", 0.88)]:
        ax.text(0.03, yy, lab, transform=ax.transAxes, fontsize=8.6, color="white",
                va="top", bbox=dict(boxstyle="round,pad=0.25", fc=c, ec="none"))
    shared = int(((d1 != 0) & (d2 != 0)).sum())
    ax.text(0.5, -0.30,
            f"The two bands abut — they are the same motion, continued.\n"
            f"Pixels shared between them: {shared}.  So in the pixel metric their cos is exactly 0.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.3, color="#444444")

    # ---- (f) separation
    ax = fig.add_subplot(gs[1, 2])
    same = cosines(d1[None], d2[None], N)
    away = cosines(d1[None], far[None], N)
    xs = np.arange(len(SHOW))
    ax.bar(xs - 0.20, [same[m] for m in SHOW], 0.38, color="#2e7d32",
           label="panel e's continuation  (should be HIGH)")
    ax.bar(xs + 0.20, [away[m] for m in SHOW], 0.38, color="#c62828",
           label="the same bands 10 px away  (should be LOW)")
    ax.axhline(0, color="#444444", lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(SHOW, fontsize=8.2, rotation=18, ha="right")
    ax.set_ylabel("cos with the gesture in panel d", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.set_ylim(-0.80, 1.72)
    ttl(ax, "f. Can the encoding tell those two apart?\nthe gap between the bars is the whole point")
    ax.legend(fontsize=7.8, loc="upper center", framealpha=0.96)
    for i, m in enumerate(SHOW):
        if abs(same[m]) < 0.03 and abs(away[m]) < 0.03:
            ax.annotate("both 0.00", (i, 0.0), fontsize=7.2, ha="center",
                        va="bottom", xytext=(0, 4), textcoords="offset points")
            continue
        for off, v in [(-0.20, same[m]), (0.20, away[m])]:
            ax.annotate(f"{v:+.2f}", (i + off, v), fontsize=7.2, ha="center",
                        va="bottom" if v >= 0 else "top",
                        xytext=(0, 3 if v >= 0 else -3), textcoords="offset points")
    ax.annotate("pixel metric: both exactly 0.\nThe gesture continued and an\nunrelated blob score the same.",
                xy=(0.0, 0.02), xytext=(0.45, -0.62), fontsize=8.3, color="#111111",
                arrowprops=dict(arrowstyle="->", color="#111111", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#111111", lw=1.0))

    fig.savefig(f"{OUT}/fig1_locality.png", dpi=130)
    plt.close(fig)
    print(f"{OUT}/fig1_locality.png")


# ============================================================ figure 2

def _ot_curves(cache=f"{OUT}/ot_cache.npz"):
    hs = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    if os.path.exists(cache):
        z = np.load(cache)
        return hs, {int(n): z[str(n)] for n in z.files}
    out = {}
    for n in [28, 56, 112]:
        pitch = CANVAS / n
        x = np.arange(float(n)) * pitch
        C1 = (x[:, None] - x[None, :]) ** 2
        K = np.exp(-C1 / 0.02); Kc = K * C1

        def ot(a, b, iters=4000):
            a = np.maximum(a, 1e-11); a = a / a.sum()
            b = np.maximum(b, 1e-11); b = b / b.sum()
            u = np.ones((n, n)); v = np.ones((n, n))
            for _ in range(iters):
                u = a / np.maximum(K @ (K @ v.T).T, 1e-300)
                v = b / np.maximum(K @ (K @ u.T).T, 1e-300)
            return float((u * (Kc @ v @ K.T + K @ v @ Kc.T)).sum())

        A = cov(ROOT, n)[0]; s0 = ot(A, A)
        ys = []
        for h in hs:
            p = ROOT.copy(); p[0] += h
            B = cov(p, n)[0]
            d = ot(A, B) - 0.5 * (s0 + ot(B, B))
            ys.append(np.sqrt(d) / h if d > 0 else np.nan)
        out[n] = np.array(ys)
        print(f"  OT {n}x{n}: " + " ".join(f"{v:.4f}" for v in out[n]))
    np.savez(cache, **{str(k): v for k, v in out.items()})
    return hs, out


def figure2():
    fig = plt.figure(figsize=(15.0, 6.4))
    gs = fig.add_gridspec(1, 3, wspace=0.44, left=0.065, right=0.985,
                          top=0.705, bottom=0.265)
    fig.suptitle("Does the derivative come back?  Three tests, all at root 1", fontsize=15, y=0.965)
    fig.text(0.5, 0.875,
             "A genuine derivative means a FLAT line in panel a and an angle falling to ZERO in panel b.  The pixel metric fails both; re-encoding fixes both.\n"
             "In every panel the knob step shrinks to the right, so these read left-to-right as \"zoom in and see whether it settles down\".",
             ha="center", va="top", fontsize=9.4, color="#444444", linespacing=1.6)

    N = 224
    hs = np.array([4.0, 2.0, 1.0, 0.5, 0.25])

    # ---- (a) speed
    ax = fig.add_subplot(gs[0, 0])
    k = KNOBS.index("width")
    I0 = hard(ROOT, N)[0]
    for m in SHOW:
        w = METRICS[m]
        e0 = reweight(I0[None], w, N)[0]; sc = np.linalg.norm(e0)
        sp = []
        for h in hs:
            p = ROOT.copy(); p[k] += h
            sp.append(np.linalg.norm(reweight(hard(p, N)[0][None], w, N)[0] - e0) / sc / h)
        sl = np.polyfit(np.log(hs), np.log(sp), 1)[0]
        ax.plot(hs, sp, color=COLOR[m], marker="o", ms=4,
                label=f"{m}  (slope {sl:+.2f})", **STYLE.get(m, dict(lw=1.8)))
    ax.plot(hs, 0.40 * hs ** -0.5, color="#c62828", lw=1.3, ls="--")
    ax.annotate("slope -1/2:\nthe paper's blow-up", (0.35, 0.40 * 0.35 ** -0.5),
                xytext=(0.52, 0.172), fontsize=8.2, color="#c62828", ha="center",
                arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.1))
    logx(ax, hs)
    ax.set_yscale("log")
    ttl(ax, "a. Speed: how far the picture moves\nper pixel of knob turned")
    ax.set_xlabel("width step h (px)  —  shrinking to the right", fontsize=8.8)
    ax.set_ylabel("|| R(w+h) - R(w) || / h\n(fractions of the image, per px)", fontsize=8.8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.96)
    ax.grid(alpha=0.25, which="major")
    ax.set_ylim(0.15, 14)
    ax.text(0.5, -0.195, "Binary ink, 224x224, width knob.  Flat = the derivative exists.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.3, color="#444444")

    # ---- (b) twist
    ax = fig.add_subplot(gs[0, 1])
    N2 = 448
    box = np.array([1.0, 1.0, 1.5, 2.8, 45.0])
    hn = [0.5, 0.25, 0.125, 0.0625]
    I0b = hard(ROOT, N2)[0].ravel()
    P = {h: np.array([hard(np.where(np.arange(5) == j, ROOT + h * box, ROOT), N2)[0].ravel() - I0b
                      for j in range(5)]) for h in hn}
    mids = [np.sqrt(a * b) for a, b in zip(hn[:-1], hn[1:])]
    for m in SHOW:
        w = METRICS[m]
        Q = {h: np.linalg.qr(reweight(P[h].reshape(-1, N2, N2), w, N2).reshape(5, -1).T)[0]
             for h in hn}
        ang = [np.degrees(np.arccos(np.clip(
            np.linalg.svd(Q[a].T @ Q[b], compute_uv=False).min(), -1, 1)))
            for a, b in zip(hn[:-1], hn[1:])]
        ax.plot(mids, ang, color=COLOR[m], marker="o", ms=4.5, label=m,
                **STYLE.get(m, dict(lw=1.8)))
    logx(ax, mids, "{:.3f}")
    ax.set_yscale("log")
    ttl(ax, "b. Twist: does the 5-knob plane settle down as you zoom in?\nsame six encodings and colours as panel a")
    ax.set_xlabel("geometric mean of the step pair h, h/2\n(fraction of each knob's range)  —  shrinking to the right", fontsize=8.8)
    ax.set_ylabel("largest principal angle between the\n5-knob plane at h and at h/2  (degrees)", fontsize=8.8)
    ax.tick_params(labelsize=8)
    ax.set_ylim(1.7, 900)

    ax.grid(alpha=0.25, which="major")
    ax.annotate("flat: the plane keeps rotating by the\nsame angle forever = NO tangent plane",
                xy=(mids[-1], 53.4), xytext=(0.200, 330), fontsize=8.4, color="#111111", ha="center",
                arrowprops=dict(arrowstyle="->", color="#111111", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#111111", lw=1.0))
    ax.annotate("halving each time: the plane\nconverges = a tangent plane EXISTS",
                xy=(mids[-1], 2.7), xytext=(0.255, 2.15), fontsize=8.4, color="#1f77b4", ha="center",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#1f77b4", lw=1.0))
    ax.text(0.5, -0.195, "Binary ink, 448x448.  All five knob increments nonzero at every h shown.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.3, color="#444444")

    # ---- (c) OT
    ax = fig.add_subplot(gs[0, 2])
    hsot, curves = _ot_curves()
    cols = {28: "#4a148c", 56: "#9c27b0", 112: "#ce93d8"}
    for n in [28, 56, 112]:
        pitch = CANVAS / n
        ax.plot(hsot, curves[n], color=cols[n], marker="s", ms=5, lw=2.0,
                label=f"{n}x{n} grid,  pitch {pitch:g} px  (dotted line)")
        ax.axvline(pitch, color=cols[n], lw=1.1, ls=":")

    ax.plot(hsot, [1 / np.sqrt(h) for h in hsot], color="#c62828", lw=1.3, ls="--",
            label="1/sqrt(h), the blow-up")
    ax.axhline(1.0, color="#2e7d32", lw=1.5)
    ax.annotate("1.000 = exactly right:\nmoving the 1 by h costs h,\nwith nothing blurred away",
                xy=(1.41, 1.0), xytext=(0.108, 0.52), fontsize=8.2, color="#2e7d32", ha="center",
                arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#2e7d32", lw=1.0))
    logx(ax, hsot)
    ax.set_yscale("log")
    ax.set_ylim(0.255, 6.4)
    ttl(ax, "c. Optimal transport, and where it stops working\n(not a re-encoding: a different distance)")
    ax.set_xlabel("cx step h (px)  —  shrinking to the right", fontsize=8.8)
    ax.set_ylabel("ink-walking distance / h\n(px travelled per px of cx)", fontsize=8.8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.96)
    ax.grid(alpha=0.25, which="major")
    ax.text(0.5, -0.195,
            "Dotted vertical line = that grid's pixel pitch.\n"
            "Each curve leaves 1.000 exactly at its OWN pitch, so the sub-pixel\n"
            "blow-up is the SAMPLING GRID, not the sharp edges.\n"
            "(Debiased Sinkhorn, eps = 0.02 px^2, coverage render.)",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.3, color="#444444")

    fig.savefig(f"{OUT}/fig2_derivative.png", dpi=130)
    plt.close(fig)
    print(f"{OUT}/fig2_derivative.png")


if __name__ == "__main__":
    figure1()
    figure2()
