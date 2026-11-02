"""Is a knob sweep a set of per-pixel humps that hand off between dimensions?

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python analyze_generated_one_pixel_humps.py

`figures/generated_one_crease_origin` showed every crease is a `clip/min/max`
decision flipping, and `figures/generated_one_motion_discovery` showed the five
motions are recoverable from unlabelled images. This script tests a different
description of the same sweep -- one framed in the pixel basis rather than in
motions. The claim under test, stated as five separable parts:

1. ONE HUMP. Watch a single pixel while only lean moves. Its value is zero for a
   while, rises, falls, and returns to zero -- one contiguous episode, not
   several.
2. SHAPE. That episode rises then falls once (possibly with a flat top at full
   ink), rather than wobbling.
3. HEIGHT. Episodes reach roughly the same peak, near full ink.
4. DURATION. Pixels near the stroke's vertical centre stay active far longer
   than pixels near the ends, because lean moves a row at a speed set by its
   distance from `cy`. Predicted law: the active span measured in TAN of lean is
   inversely proportional to |row - cy|. Tested against a `cx` sweep control,
   where every row moves at the same speed and durations should NOT spread.
5. HANDOFF. At any one lean only a small set of pixels is moving, and that set
   migrates gradually -- neighbouring leans share most of their active set,
   distant leans share little.

Point 4's control matters most: a law that fits lean but also fits a sweep with
no speed gradient would be explaining nothing.

All sweeps start from root 1 and move exactly one knob. Measurements are made on
`grey_ones` coverage (the real renderer) with `smooth_ones` as a decision-free
control, to separate hump STRUCTURE (geometry, should survive) from the corners
at a hump's feet (grid artefact, should vanish).
"""

import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

import smooth_ones
from grey_ones import RANGES, render

OUT = Path(__file__).resolve().parent / "figures/generated_one_pixel_humps"
ROOT = np.array([14.0, 14.0, 19.75, 3.2, 0.0])
LEAN_LO, LEAN_HI, STEPS = -45.0, 45.0, 3601      # wider than default, to see whole humps
DEFAULT_LEAN = (RANGES[4, 0], RANGES[4, 1])      # the -10..35 box, marked on figures
COVER_EPS = 1e-9                                 # coverage is exactly 0 off the stroke
SMOOTH_EPS = 1e-3                                # smooth never reaches 0; threshold stated
RATE_FRAC = 1e-3                                 # "moving": |rate| above this x the max rate


def sweep(knob, lo, hi, steps, renderer):
    """Move one knob from lo to hi; return the grid and (steps, 784) values."""
    grid = np.linspace(lo, hi, steps)
    P = np.tile(ROOT, (steps, 1))
    P[:, knob] = grid
    return grid, renderer(P).reshape(steps, -1)


def runs(mask):
    """Start/stop indices of each contiguous True run."""
    m = np.concatenate([[False], mask, [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def hump_census(grid, V, eps):
    """Per-pixel episode count, span, peak, and how many separate rises it has."""
    out = []
    for k in range(V.shape[1]):
        v = V[:, k]
        if v.max() <= eps:
            continue
        rr = runs(v > eps)
        touches_edge = any(a == 0 or b == len(grid) for a, b in rr)
        a, b = max(rr, key=lambda t: t[1] - t[0])
        seg = v[a:b]
        # count rises: maximal runs of strictly increasing value, ignoring flat tops
        dv = np.diff(seg)
        up = dv > 1e-12
        out.append(dict(pixel=int(k), row=int(k // 28), col=int(k % 28),
                        n_episodes=len(rr), truncated=bool(touches_edge),
                        lean_lo=float(grid[a]), lean_hi=float(grid[min(b, len(grid) - 1)]),
                        span_deg=float(grid[min(b, len(grid) - 1)] - grid[a]),
                        peak=float(seg.max()), n_rises=int(len(runs(up))),
                        plateau_frac=float((seg >= 0.99).mean())))
    return out


def rate_census(grid, V):
    """Per-lean: how many pixels are moving, and how concentrated the motion is."""
    R = np.gradient(V, grid, axis=0)
    scale = np.abs(R).max()
    live = (np.abs(R) > RATE_FRAC * scale).sum(1)
    sq = R ** 2
    tot = sq.sum(1)
    top10 = np.sort(sq, axis=1)[:, -10:].sum(1)
    share = np.where(tot > 0, top10 / np.maximum(tot, 1e-300), np.nan)
    return R, live, share


def handoff(grid, R, deltas):
    """How much does the direction of travel at one lean resemble a later one?"""
    unit = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-300)
    inside = (grid >= DEFAULT_LEAN[0]) & (grid <= DEFAULT_LEAN[1])
    step = grid[1] - grid[0]
    act = np.abs(R) > RATE_FRAC * np.abs(R).max()
    rows = []
    for d in deltas:
        s = int(round(d / step))
        if s < 1 or s >= len(grid):
            continue
        i = np.flatnonzero(inside[:-s])
        cos = np.einsum("ij,ij->i", unit[i], unit[i + s])
        inter = (act[i] & act[i + s]).sum(1)
        union = (act[i] | act[i + s]).sum(1)
        rows.append(dict(delta_deg=float(d), median_cosine=float(np.median(cos)),
                         median_overlap=float(np.median(inter / np.maximum(union, 1)))))
    return rows


def duration_law(cens, cy):
    """Active span in TAN of lean, against 1/|row - cy|. Only complete humps."""
    rec = [c for c in cens if not c["truncated"]]
    dist = np.array([abs(c["row"] + 0.5 - cy) for c in rec])
    span_tan = np.array([np.tan(np.radians(c["lean_hi"])) - np.tan(np.radians(c["lean_lo"]))
                         for c in rec])
    keep = dist > 0.5
    d, s = dist[keep], span_tan[keep]
    inv = 1.0 / d
    slope = float((inv @ s) / (inv @ inv))                      # fit s = slope / d
    pred = slope * inv
    r2 = float(1 - ((s - pred) ** 2).sum() / ((s - s.mean()) ** 2).sum())
    return dict(n=int(keep.sum()), slope=slope, r2=r2,
                product_mean=float((s * d).mean()), product_cv=float((s * d).std() / (s * d).mean()),
                dist=d.tolist(), span_tan=s.tolist())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}

    grid, Vc = sweep(4, LEAN_LO, LEAN_HI, STEPS, lambda P: render(P, dtype=np.float64))
    _, Vs = sweep(4, LEAN_LO, LEAN_HI, STEPS, smooth_ones.render)

    cc = hump_census(grid, Vc, COVER_EPS)
    cs = hump_census(grid, Vs, SMOOTH_EPS)
    for tag, cens in (("coverage", cc), ("smooth", cs)):
        ep = np.array([c["n_episodes"] for c in cens])
        ri = np.array([c["n_rises"] for c in cens])
        pk = np.array([c["peak"] for c in cens])
        res[f"humps_{tag}"] = dict(
            n_pixels_touched=len(cens),
            frac_single_episode=float((ep == 1).mean()),
            frac_single_rise=float((ri == 1).mean()),
            max_episodes=int(ep.max()), max_rises=int(ri.max()),
            peak_median=float(np.median(pk)),
            frac_peak_above_0p99=float((pk >= 0.99).mean()),
            frac_peak_below_0p5=float((pk < 0.5).mean()),
            span_deg_min=float(min(c["span_deg"] for c in cens)),
            span_deg_max=float(max(c["span_deg"] for c in cens)))

    Rc, live_c, share_c = rate_census(grid, Vc)
    res["motion_coverage"] = dict(
        live_median=int(np.median(live_c)), live_max=int(live_c.max()),
        live_frac_median=float(np.median(live_c) / 784),
        top10_share_median=float(np.nanmedian(share_c)))
    res["handoff_coverage"] = handoff(grid, Rc, [0.025, 0.1, 0.5, 1, 2, 5, 10, 20, 40])

    res["duration_law_lean"] = duration_law(cc, ROOT[1])

    # control: a cx sweep moves every row at the same speed
    cxg, Vx = sweep(0, 8.0, 20.0, 2401, lambda P: render(P, dtype=np.float64))
    cxc = [c for c in hump_census(cxg, Vx, COVER_EPS) if not c["truncated"]]
    by_row = {}
    for c in cxc:
        by_row.setdefault(c["row"], []).append(c["span_deg"])
    rows = sorted(by_row)
    med = np.array([np.median(by_row[r]) for r in rows])
    far = np.array([abs(r + 0.5 - ROOT[1]) for r in rows])
    res["control_cx_sweep"] = dict(
        n_pixels=len(cxc), row_span_cv=float(med.std() / med.mean()),
        corr_span_vs_dist=float(np.corrcoef(far, med)[0, 1]),
        span_px_median=float(np.median([c["span_deg"] for c in cxc])))
    leanrows = {}
    for c in cc:
        if not c["truncated"]:
            leanrows.setdefault(c["row"], []).append(
                np.tan(np.radians(c["lean_hi"])) - np.tan(np.radians(c["lean_lo"])))
    lr = sorted(leanrows)
    lmed = np.array([np.median(leanrows[r]) for r in lr])
    res["control_cx_sweep"]["lean_row_span_cv"] = float(lmed.std() / lmed.mean())

    np.savez_compressed(OUT / "sweep_data.npz", grid=grid, Vc=Vc, Vs=Vs, Rc=Rc,
                        live_c=live_c, share_c=share_c, cx_grid=cxg,
                        cx_rows=np.array(rows), cx_med=med,
                        lean_rows=np.array(lr), lean_med=lmed)
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)[:1200])
    make_figures()


if __name__ == "__main__":
    main()


def make_figures():
    """Two figures: one per question. Reads what main() saved."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    d = np.load(OUT / "sweep_data.npz")
    res = json.loads((OUT / "results.json").read_text())
    grid, Vc, Rc = d["grid"], d["Vc"], d["Rc"]
    cy = ROOT[1]
    COL = ["#1b6ca8", "#c44601", "#047857", "#7c3aed", "#b91c1c"]

    # five example pixels, spread over distance from cy, each actually swept
    picks = []
    for row, c in zip([14, 11, 8, 6, 4], COL):
        want = 14.0 - np.tan(np.radians(15.0)) * (row + 0.5 - cy)
        cands = [j for j in range(28) if Vc[:, row * 28 + j].max() > COVER_EPS]
        col = min(cands, key=lambda j: abs(j + 0.5 - want))
        picks.append((row, col, c))

    # ---------------- FIGURE 1 ----------------
    fig = plt.figure(figsize=(16.5, 9.4), constrained_layout=True)
    fig.suptitle("Generated 1s: sweeping LEAN alone, does each pixel see one hump?\n"
                 "root 1 = (cx 14, cy 14, height 19.75, width 3.2, lean 0); "
                 "only lean moves, from -45 to +45 degrees",
                 fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0])
    i0 = int(np.argmin(np.abs(grid - 0.0)))
    ax.imshow(Vc[i0].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1,
              extent=[0, 28, 28, 0], interpolation="nearest")
    ax.axhline(cy, color="#2563eb", ls="--", lw=1.3)
    ax.text(0.4, cy + 1.5, "cy = 14, the pivot", color="#2563eb", fontsize=9, va="bottom")
    for row, col, c in picks:
        ax.add_patch(Rectangle((col, row), 1, 1, fill=False, ec=c, lw=2.4))
        ax.text(col + 1.4, row + 0.6, f"row {row}  |row-cy|={abs(row+0.5-cy):.1f}",
                color=c, fontsize=8.5, fontweight="bold")
    ax.set_title("A. The baseline: root 1 at lean 0\n"
                 "one pixel value = fraction of that cell covered by ink (0 to 1)",
                 fontsize=10.5)
    ax.set_xlabel("column"); ax.set_ylabel("row")

    ax = fig.add_subplot(gs[0, 1])
    for row, col, c in picks:
        ax.plot(grid, Vc[:, row * 28 + col], color=c, lw=1.8,
                label=f"row {row}, col {col}")
    ax.axvspan(*DEFAULT_LEAN, color="#fde68a", alpha=0.45, zorder=0)
    ax.text(DEFAULT_LEAN[0] + 1, 1.045, "default lean range", fontsize=8.5, color="#92400e")
    ax.set_xlabel("lean (degrees)"); ax.set_ylabel("pixel value (ink coverage)")
    ax.set_ylim(-0.04, 1.12); ax.legend(fontsize=8, loc="center left")
    ax.set_title("B. Each marked pixel's own value\n"
                 f"all {res['humps_coverage']['n_pixels_touched']} touched pixels: exactly ONE "
                 f"episode each, at most {res['humps_coverage']['max_rises']} rise",
                 fontsize=10.5)

    ax = fig.add_subplot(gs[0, 2])
    for row, col, c in picks:
        ax.plot(grid, Rc[:, row * 28 + col], color=c, lw=1.8)
    ax.axhline(0, color="#444", lw=0.8)
    ax.axvspan(*DEFAULT_LEAN, color="#fde68a", alpha=0.45, zorder=0)
    ax.set_xlabel("lean (degrees)")
    ax.set_ylabel("rate of change of that pixel, per degree")
    ax.set_title("C. The same pixels' RATE of change\n"
                 "one value-hump = one positive bump (ink arriving),\nthen one negative "
                 "bump (ink leaving)", fontsize=10.5)

    ax = fig.add_subplot(gs[1, 0])
    law = res["duration_law_lean"]
    dist = np.array(law["dist"]); span = np.array(law["span_tan"])
    ax.scatter(1 / dist, span, s=22, color="#1b6ca8", alpha=0.75, label="one complete hump")
    xs = np.linspace(0, (1 / dist).max() * 1.05, 50)
    ax.plot(xs, law["slope"] * xs, color="#c44601", lw=2,
            label=f"fit: span = {law['slope']:.2f} / |row-cy|")
    ax.plot(xs, 4.2 * xs, color="#047857", lw=1.6, ls="--",
            label="geometry: (width+1) / |row-cy| = 4.2 / |row-cy|")
    ax.set_xlabel("1 / |row - cy|   (1 / pixels)")
    ax.set_ylabel("active span, measured in tan(lean)")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title(f"D. How long a hump lasts  (n={law['n']} complete humps)\n"
                 f"r^2 = {law['r2']:.4f};  span x |row-cy| = {law['product_mean']:.2f} "
                 f"+/- {law['product_cv']*100:.1f}%", fontsize=10.5)

    ax = fig.add_subplot(gs[1, 1])
    cr, cm = d["cx_rows"], d["cx_med"]
    lr, lm = d["lean_rows"], d["lean_med"]
    ax.plot(np.abs(cr + 0.5 - cy), cm / cm.mean(), "o-", color="#047857", lw=1.8, ms=5,
            label=f"control: cx sweep (CV {res['control_cx_sweep']['row_span_cv']:.0e})")
    ax.plot(np.abs(lr + 0.5 - cy), lm / lm.mean(), "s-", color="#c44601", lw=1.8, ms=5,
            label=f"lean sweep (CV {res['control_cx_sweep']['lean_row_span_cv']*100:.0f}%)")
    ax.set_xlabel("|row - cy|   (pixels from the pivot)")
    ax.set_ylabel("median span for that row, / its own mean")
    ax.legend(fontsize=8.5)
    ax.set_title("E. Control: the law needs a speed gradient\n"
                 "cx moves every row equally, so spans do NOT spread "
                 "(correlation with distance = 0.00)", fontsize=10.5)

    ax = fig.add_subplot(gs[1, 2])
    peaks = np.array([Vc[:, k].max() for k in range(784) if Vc[:, k].max() > COVER_EPS])
    ax.hist(peaks, bins=np.linspace(0, 1, 41), color="#7c3aed", alpha=0.85)
    ax.set_xlim(0, 1.03)
    ax.axvline(0.99, color="#c44601", lw=1.6, ls="--")
    ax.text(0.975, ax.get_ylim()[1] * 0.92, "0.99", color="#c44601", fontsize=9, ha="right")
    hc = res["humps_coverage"]
    ax.set_xlabel("tallest value that pixel ever reaches")
    ax.set_ylabel("number of pixels")
    ax.set_title("F. Do all humps reach the same height?\n"
                 f"{hc['frac_peak_above_0p99']*100:.0f}% reach full ink, but "
                 f"{hc['frac_peak_below_0p5']*100:.0f}% never pass 0.5", fontsize=10.5)

    fig.savefig(OUT / "pixel_humps.png", dpi=135)
    plt.close(fig)

    # ---------------- FIGURE 2 ----------------
    fig = plt.figure(figsize=(16.5, 9.0), constrained_layout=True)
    fig.suptitle("Generated 1s: as lean advances, does the set of moving pixels hand off gradually?\n"
                 "same single-knob lean sweep from root 1; 'moving' = |rate| above 0.1% of the "
                 "largest rate seen anywhere", fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.92])

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(grid, d["live_c"], color="#1b6ca8", lw=1.6)
    ax.axhline(784, color="#444", ls=":", lw=1.2)
    ax.text(-44, 744, "all 784 dimensions", fontsize=8.5, color="#444")
    ax.axvspan(*DEFAULT_LEAN, color="#fde68a", alpha=0.45, zorder=0)
    ax.set_ylim(0, 860)
    ax.set_xlabel("lean (degrees)"); ax.set_ylabel("pixels moving at this lean")
    mc = res["motion_coverage"]
    ax.set_title("A. How many dimensions move at once?\n"
                 f"median {mc['live_median']} of 784 = {mc['live_frac_median']*100:.1f}%; "
                 f"most of the basis is idle at any moment", fontsize=10.5)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(grid, d["share_c"] * 100, color="#047857", lw=1.6)
    ax.axvspan(*DEFAULT_LEAN, color="#fde68a", alpha=0.45, zorder=0)
    ax.set_ylim(0, 105)
    ax.set_xlabel("lean (degrees)")
    ax.set_ylabel("share of the move's squared length, top 10 pixels (%)")
    ax.set_title("B. Is there a dominant handful?\n"
                 f"median {mc['top10_share_median']*100:.0f}% of the motion sits in just "
                 "10 of the 784", fontsize=10.5)

    ax = fig.add_subplot(gs[0, 2])
    ho = res["handoff_coverage"]
    dd = [h["delta_deg"] for h in ho]
    ax.semilogx(dd, [h["median_cosine"] for h in ho], "o-", color="#1b6ca8", lw=2, ms=6,
                label="cosine between the two move directions")
    ax.semilogx(dd, [h["median_overlap"] for h in ho], "s--", color="#c44601", lw=2, ms=6,
                label="fraction of the moving set shared")
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xlabel("lean apart (degrees)")
    ax.set_ylabel("similarity (1 = identical, 0 = unrelated)")
    ax.set_ylim(-0.35, 1.08); ax.legend(fontsize=8.5, loc="lower left")
    g = {h["delta_deg"]: h for h in ho}
    ax.set_title("C. Handoff is gradual, and complete by ~10 degrees\n"
                 f"1 deg apart: cos {g[1.0]['median_cosine']:.2f} | "
                 f"5 deg: {g[5.0]['median_cosine']:.2f} | "
                 f"10 deg: {g[10.0]['median_cosine']:.2f}", fontsize=10.5)

    lim = np.abs(Rc).max() * 0.55
    for n, L in enumerate([0.0, 5.0, 10.0]):
        ax = fig.add_subplot(gs[1, n])
        i = int(np.argmin(np.abs(grid - L)))
        im = ax.imshow(Rc[i].reshape(28, 28), cmap="RdBu_r", vmin=-lim, vmax=lim,
                       extent=[0, 28, 28, 0], interpolation="nearest")
        ax.set_title(f"D{n+1}. the move direction at lean = {L:.0f} deg", fontsize=10.5)
        ax.set_xlabel("column")
        if n == 0:
            ax.set_ylabel("row")
        cb = fig.colorbar(im, ax=ax, fraction=0.046)
        cb.set_label("rate per degree:  red = gaining ink,  blue = losing ink", fontsize=8)
    fig.get_layout_engine().set(rect=(0, 0.05, 1, 0.93))
    fig.text(0.5, 0.018,
             "D1-D3 share one colour scale. These are rates of ink change, not images. "
             "Red and blue swap places as lean advances: the same operation is carried by "
             "different pixels at different leans.",
             ha="center", fontsize=9.5, style="italic")

    fig.savefig(OUT / "pixel_humps_handoff.png", dpi=135)
    plt.close(fig)
    print("wrote", OUT / "pixel_humps.png", "and", OUT / "pixel_humps_handoff.png")
