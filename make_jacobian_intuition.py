"""Explanatory figures: the Jacobian of the generated-1 renderer as pictures.

Three questions, one figure each.

  1. What are the columns of the Jacobian, as images?
  2. Do those columns stay the same as you move along the manifold?
  3. Can the change in one column be predicted exactly?

Everything here is an analytic/finite-difference diagnostic of the KNOWN
five-knob renderer. No fitting, no shell sampling, no learned basis.
Units matter: cx, cy, height, width derivatives are per pixel of knob;
the lean derivative is per degree. Raw-unit panels therefore carry their
own colour scale; range-normalised panels share one.

Run from the repository root:

    OPENBLAS_NUM_THREADS=1 .venv/bin/python make_jacobian_intuition.py
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from grey_ones import KNOBS, RANGES, render

OUT = os.path.join("figures", "jacobian_intuition")
ROOT = np.array([14.0, 14.0, 19.75, 3.2, 0.0])
SPAN = RANGES[:, 1] - RANGES[:, 0]
PRETTY = ["cx (horizontal centre)", "cy (vertical centre)", "height", "width", "lean"]
UNITS = ["per pixel", "per pixel", "per pixel", "per pixel", "per degree"]
STEP = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-3])
INK = "red = that pixel gains ink, blue = it loses ink"
NL = "\n"


def img(p):
    return render(np.asarray(p, float)[None, :], dtype=np.float64)[0]


def jac_column(p, k, step):
    """Central difference of render in knob k. Returns a (28, 28) pixel-change rate."""
    hi, lo = np.array(p, float), np.array(p, float)
    hi[k] += step
    lo[k] -= step
    return (img(hi) - img(lo)) / (2 * step)


def jacobian(p, steps=STEP):
    return np.stack([jac_column(p, k, steps[k]) for k in range(5)])


def mixed_second(p, j, k, sj, sk):
    """d2F / dknob_j dknob_k by a four-corner central difference."""
    acc = np.zeros((28, 28))
    for a in (+1, -1):
        for b in (+1, -1):
            q = np.array(p, float)
            q[j] += a * sj
            q[k] += b * sk
            acc += a * b * img(q)
    return acc / (4 * sj * sk)


def diverging(ax, v, vmax, title, sub=None):
    ax.imshow(v, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8.5, pad=4)
    if sub:
        ax.set_xlabel(sub, fontsize=7.2, labelpad=3)


def plane_view(ax, u, v, label_u, label_v, cosine):
    """Exact 2-D picture of the plane spanned by two 784-d vectors.

    u is drawn along +x with its true length; v is placed by its true length
    and its true angle to u. Nothing is projected away: the plane is complete.
    """
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    ang = np.arccos(np.clip(cosine, -1, 1))
    pts = np.array([[nu, 0.0], [nv * np.cos(ang), nv * np.sin(ang)]])
    lim = 1.25 * max(nu, nv)
    ax.axhline(0, color="0.85", lw=0.7, zorder=0)
    ax.axvline(0, color="0.85", lw=0.7, zorder=0)
    for (x, y), c, lab in zip(pts, ["#b2182b", "#2166ac"], [label_u, label_v]):
        ax.annotate("", (x, y), (0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2.1))
        ax.text(x * 1.06, y * 1.06 + 0.02 * lim, lab, color=c, fontsize=8, ha="left", va="bottom")
    th = np.linspace(0, ang, 80)
    r = 0.33 * min(nu, nv)
    ax.plot(r * np.cos(th), r * np.sin(th), color="0.25", lw=1.1)
    ax.text(1.95 * r * np.cos(ang / 2), 1.95 * r * np.sin(ang / 2),
            "%.1f deg" % np.degrees(ang), fontsize=8.8, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.75", lw=0.6))
    ax.set_xlim(-0.2 * lim, lim)
    ax.set_ylim(-0.2 * lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def figure_one(res):
    j = jacobian(ROOT)
    root = img(ROOT)
    norms = np.linalg.norm(j.reshape(5, -1), axis=1)
    jn = j * SPAN[:, None, None]
    norms_n = np.linalg.norm(jn.reshape(5, -1), axis=1)
    panel_max = np.abs(jn).max(axis=(1, 2))
    # Share a scale set by the second-largest panel; lean runs past it and is
    # marked. Using lean's own scale would flatten the other four to blank.
    shared = float(np.sort(panel_max)[-2])
    clipped = [int((np.abs(jn[k]) > shared * (1 + 1e-9)).sum()) for k in range(5)]
    rank = int(np.linalg.matrix_rank(j.reshape(5, -1).T, tol=1e-10))

    fig = plt.figure(figsize=(15.0, 7.8))
    gs = GridSpec(2, 6, figure=fig, hspace=0.5, wspace=0.2,
                  left=0.035, right=0.9, top=0.805, bottom=0.235)
    fig.suptitle("The Jacobian of the generated-1 renderer at the root 1, column by column",
                 fontsize=14.5, y=0.975)
    fig.text(0.5, 0.925,
             "One column per knob. A column is NOT an image of a 1 - it is the rate at which every pixel's ink "
             "coverage changes when that one knob moves and the other four are held fixed.",
             ha="center", fontsize=9.5)
    fig.text(0.5, 0.888, "Colour in all ten change panels: " + INK + ". Pale grey = no change.",
             ha="center", fontsize=8.8, color="0.3")

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(root, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("BASELINE: the root 1", fontsize=9.8, fontweight="bold", pad=4)
    ax.set_xlabel("28x28; one value = ink coverage" + NL
                  + "0 = no ink (white), 1 = full ink (black)" + NL
                  + "cx 14, cy 14, height 19.75, width 3.2, lean 0",
                  fontsize=7.2, labelpad=3)

    for k in range(5):
        vmax = float(np.abs(j[k]).max())
        diverging(fig.add_subplot(gs[0, k + 1]), j[k], vmax,
                  "ROW 1:  d(image) / d(%s)" % KNOBS[k],
                  "%s%sits own scale: +/-%.3f %s%slength %.4f %s"
                  % (PRETTY[k], NL, vmax, UNITS[k], NL, norms[k], UNITS[k]))

    axb = fig.add_subplot(gs[1, 0])
    order = np.arange(5)
    axb.barh(order, norms_n, color="#4393c3", height=0.62)
    for i, v in enumerate(norms_n):
        axb.text(v + 0.6, i, "%.2f" % v, va="center", fontsize=8)
    axb.set_yticks(order)
    axb.set_yticklabels(KNOBS, fontsize=8.5)
    axb.invert_yaxis()
    axb.set_xlim(0, norms_n.max() * 1.25)
    axb.set_xlabel("ink-coverage change per full knob range", fontsize=7.4)
    axb.set_title("ROW 2 measured:" + NL + "comparable knob speeds",
                  fontsize=9.0, fontweight="bold", pad=4)
    axb.tick_params(axis="x", labelsize=7.5)
    axb.grid(axis="x", alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)

    for k in range(5):
        axk = fig.add_subplot(gs[1, k + 1])
        if clipped[k]:
            note = ("CLIPPED: %d px run past the scale;" % clipped[k]
                    + NL + "true extreme +/-%.3f" % panel_max[k])
        else:
            note = NL + "(nothing clipped; peak +/-%.3f)" % panel_max[k]
        diverging(axk, jn[k], shared,
                  "ROW 2:  d/d(%s)  x  range %g" % (KNOBS[k], SPAN[k]),
                  "length %.3f    shared scale +/-%.2f%s%s" % (norms_n[k], shared, NL, note))
        if clipped[k]:
            axk.contour(np.abs(jn[k]) > shared * (1 + 1e-9), levels=[0.5],
                        colors="#00a000", linewidths=0.9)

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-shared, shared))
    cb = fig.colorbar(sm, ax=[fig.axes[i] for i in range(7, 12)],
                      fraction=0.023, pad=0.015, extend="both")
    cb.set_label("row 2 only: ink-coverage change per full knob range", fontsize=7.8)
    cb.ax.tick_params(labelsize=7)

    fig.text(0.035, 0.15,
             "ROW 1 - raw knob units. cx, cy, height and width move in pixels; lean moves in degrees. Lengths in "
             "different units are not comparable, so each panel carries its own colour scale.",
             fontsize=8.6)
    fig.text(0.035, 0.113,
             "ROW 2 - each column multiplied by its own allowed knob range (%s), which puts all five in one unit, so a "
             "single colour scale and the bar chart apply to all."
             % ", ".join("%s %g" % (k, s) for k, s in zip(KNOBS, SPAN)),
             fontsize=8.6)
    fig.text(0.035, 0.071,
             "Row 2 is a linear extrapolation of the local rate across the range, not an actual full-range move. Its "
             "shared scale +/-%.2f is set by the second-largest panel; lean's %d clipped pixels are outlined in green."
             % (shared, clipped[4]),
             fontsize=8.6, color="#006000")
    fig.text(0.035, 0.026,
             "MEASURED  |  the five columns have rank %d of 5, so no combination of knob nudges leaves the image "
             "unchanged - the knobs are independent. Per full range, lean moves the image %.1fx further than height."
             % (rank, norms_n[4] / norms_n[2]),
             fontsize=8.8, fontweight="bold")
    fig.savefig(os.path.join(OUT, "01_jacobian_columns_as_pictures.png"), dpi=155)
    plt.close(fig)

    res["figure_1"] = {
        "root_settings": ROOT.tolist(),
        "raw_column_norms": dict(zip(KNOBS, norms.round(6).tolist())),
        "range_normalised_column_norms": dict(zip(KNOBS, norms_n.round(6).tolist())),
        "row2_shared_colour_scale": shared,
        "row2_clipped_pixels": dict(zip(KNOBS, clipped)),
        "column_rank": rank,
    }


def figure_two(res):
    starts = [("root (lean 0 degrees)", 0.0), ("same 1, leaned 30 degrees", 30.0)]
    sweep = np.linspace(RANGES[4, 0], RANGES[4, 1], 91)
    cos_sweep, ranks = [], []
    for ln in sweep:
        p = ROOT.copy()
        p[4] = ln
        j = jacobian(p)
        a, b = j[0].ravel(), j[1].ravel()
        cos_sweep.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
        ranks.append(int(np.linalg.matrix_rank(j.reshape(5, -1).T, tol=1e-10)))
    cos_sweep = np.array(cos_sweep)
    min_rank = int(min(ranks))

    fig = plt.figure(figsize=(14.6, 9.0))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.0, 1.0, 0.8],
                  hspace=0.52, wspace=0.26, left=0.09, right=0.985, top=0.85, bottom=0.115)
    fig.suptitle("The same two Jacobian columns at two places on the manifold",
                 fontsize=14.5, y=0.978)
    fig.text(0.5, 0.912,
             "Only the lean knob differs between the rows; the cx and cy columns are measured the same way in both. "
             "A linear map would have one Jacobian everywhere,",
             ha="center", fontsize=9.4)
    fig.text(0.5, 0.882,
             "so these columns - and the angle between them - could not change. Colour: " + INK + ".",
             ha="center", fontsize=9.4)

    per_row = []
    for r, (name, lean) in enumerate(starts):
        p = ROOT.copy()
        p[4] = lean
        j = jacobian(p)
        a, b = j[0].ravel(), j[1].ravel()
        c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        per_row.append((name, lean, c, float(np.linalg.norm(a)), float(np.linalg.norm(b))))
        vmax = float(max(np.abs(j[0]).max(), np.abs(j[1]).max()))

        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(img(p), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(("BASELINE: " if r == 0 else "VARIATION: ") + name,
                     fontsize=9.5, fontweight="bold", pad=4)
        ax.set_xlabel("lean = %g deg, other four knobs at root" % lean + NL
                      + "ink coverage 0 (white) to 1 (black)", fontsize=7.3, labelpad=3)

        diverging(fig.add_subplot(gs[r, 1]), j[0], vmax, "d(image) / d(cx)",
                  "scale shared with the panel at right:" + NL
                  + "+/-%.3f per pixel     length %.4f" % (vmax, np.linalg.norm(a)))
        diverging(fig.add_subplot(gs[r, 2]), j[1], vmax, "d(image) / d(cy)",
                  "scale shared with the panel at left:" + NL
                  + "+/-%.3f per pixel     length %.4f" % (vmax, np.linalg.norm(b)))

        axp = fig.add_subplot(gs[r, 3])
        plane_view(axp, a, b, "d/d(cx)", "d/d(cy)", c)
        axp.set_title("exact view of the plane they span" + NL + "cosine = %.6f" % c,
                      fontsize=8.8, pad=4)
        axp.set_xlabel("both columns at their true lengths and" + NL
                       + "true angle; nothing projected away", fontsize=7.3, labelpad=3)

    axs = fig.add_subplot(gs[2, :])
    axs.plot(sweep, cos_sweep, color="#2166ac", lw=2.0, zorder=3)
    offsets = [(3.4, -0.3), (-5.5, -0.34)]
    for (name, lean, c, _, _), (dx, dy) in zip(per_row, offsets):
        axs.plot([lean], [c], "o", ms=7.5, color="#b2182b", zorder=5)
        axs.annotate(name + NL + "cosine %.4f" % c, (lean, c), (lean + dx, c + dy),
                     fontsize=8.2, color="#b2182b", ha="center",
                     arrowprops=dict(arrowstyle="-", color="#b2182b", lw=0.9))
    axs.axhline(0, color="0.6", lw=0.9, ls="--", zorder=1)
    axs.text(RANGES[4, 0] + 0.6, 0.06, "cosine 0 = the two knob directions are perpendicular",
             fontsize=8.0, color="0.45", ha="left")
    axs.set_xlim(RANGES[4, 0], RANGES[4, 1])
    axs.set_ylim(-0.66, 1.06)
    axs.set_xlabel("lean knob (degrees) - swept across its whole allowed range, all four other knobs held at the root",
                   fontsize=8.8)
    axs.set_ylabel("cosine between" + NL + "d/d(cx) and d/d(cy)", fontsize=8.8)
    axs.set_title("SWEEP: the angle between two fixed knob directions depends on where you stand",
                  fontsize=10.2, pad=6)
    axs.tick_params(labelsize=8)
    axs.grid(alpha=0.25, lw=0.6)

    fig.text(0.09, 0.052,
             "MEASURED  |  across the allowed lean range (91 sampled leans) the cosine between the cx and cy columns "
             "runs from %.4f to %.4f; it is 0 at lean 0 only." % (cos_sweep.min(), cos_sweep.max()),
             fontsize=8.8, fontweight="bold")
    fig.text(0.09, 0.02,
             "The five columns keep rank %d throughout the sweep, so the knobs stay independent everywhere while "
             "ceasing to be perpendicular. The small steps in the curve are the renderer's 16-subrow sampling."
             % min_rank,
             fontsize=8.6)
    fig.savefig(os.path.join(OUT, "02_columns_change_as_you_move.png"), dpi=155)
    plt.close(fig)

    res["figure_2"] = {
        "starts": [{"name": n, "lean_deg": l, "cos_cx_cy": c, "norm_dcx": na, "norm_dcy": nb}
                   for n, l, c, na, nb in per_row],
        "min_jacobian_rank_over_sweep": min_rank,
        "lean_sweep_deg": sweep.round(4).tolist(),
        "cos_cx_cy_sweep": cos_sweep.round(8).tolist(),
    }


def figure_three(res):
    w0, dw = 3.2, 0.05
    p0, p1 = ROOT.copy(), ROOT.copy()
    p1[3] = w0 + dw
    jh0 = jac_column(p0, 2, STEP[2])
    jh1 = jac_column(p1, 2, STEP[2])
    change = jh1 - jh0
    D = mixed_second(ROOT.copy(), 2, 3, 1e-3, 1e-3)
    pred = dw * D
    resid = float(np.linalg.norm((change - pred).ravel()))

    ws = np.arange(0.0, 0.401, 0.02)
    meas = []
    for d in ws:
        q = ROOT.copy()
        q[3] = w0 + d
        meas.append(float(np.linalg.norm((jac_column(q, 2, STEP[2]) - jh0).ravel())))
    meas = np.array(meas)
    slope = float(np.linalg.norm(D.ravel()))
    maxdev = float(np.abs(meas - slope * ws).max())

    vmax = float(max(np.abs(jh0).max(), np.abs(jh1).max()))
    cmax = float(max(np.abs(change).max(), np.abs(pred).max()))

    fig = plt.figure(figsize=(14.6, 8.8))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.05],
                  hspace=0.42, wspace=0.24, left=0.085, right=0.985, top=0.83, bottom=0.06)
    fig.suptitle("How one Jacobian column changes, predicted exactly", fontsize=14.5, y=0.975)
    fig.text(0.5, 0.918,
             "Panels 1-2 are the height column at two widths on one shared scale. They look identical - that is the "
             "point, the drift is small. Panel 3 is their difference,",
             ha="center", fontsize=9.4)
    fig.text(0.5, 0.888,
             "drawn at a %.0fx finer colour scale. Panels 3-4 are a CHANGE IN A DIRECTION VECTOR: their colours say how "
             "the pixel-change pattern itself shifted, not ink gained by an image." % (vmax / cmax),
             ha="center", fontsize=9.4)

    diverging(fig.add_subplot(gs[0, 0]), jh0, vmax,
              "BASELINE   d(image)/d(height)  at width 3.2",
              "scale shared with panel 2: +/-%.3f per pixel" % vmax + NL
              + "length %.4f" % np.linalg.norm(jh0))
    diverging(fig.add_subplot(gs[0, 1]), jh1, vmax,
              "VARIATION   same column at width %g" % (w0 + dw),
              "scale shared with panel 1: +/-%.3f per pixel" % vmax + NL
              + "length %.4f" % np.linalg.norm(jh1))
    diverging(fig.add_subplot(gs[0, 2]), change, cmax,
              "MEASURED change in the column" + NL + "(panel 2 minus panel 1)",
              "second difference; scale shared with panel 4:" + NL
              + "+/-%.4f     length %.6f" % (cmax, np.linalg.norm(change)))
    diverging(fig.add_subplot(gs[0, 3]), pred, cmax,
              "PREDICTED change:  dw x D" + NL + "D = mixed derivative d2F/d(height)d(width)",
              "second difference; scale shared with panel 3:" + NL
              + "+/-%.4f     length %.6f" % (cmax, np.linalg.norm(pred)))

    axl = fig.add_subplot(gs[1, :2])
    axl.plot(ws, meas, "o", ms=5.5, color="#b2182b", zorder=4,
             label="measured: length of the change in the height column")
    axl.plot(ws, slope * ws, "-", lw=1.8, color="#2166ac", zorder=3,
             label="straight line of slope |D| = %.4f per pixel of width" % slope)
    axl.axvline(dw, color="0.55", ls="--", lw=1.0, zorder=2)
    axl.annotate("the four panels above" + NL + "use this step, dw = 0.05",
                 (dw, slope * dw), (0.185, 0.028), fontsize=8.0, color="0.3",
                 arrowprops=dict(arrowstyle="->", color="0.45", lw=0.9))
    axl.set_xlabel("dw: how far the width knob is moved from 3.2 (pixels); height, cx, cy and lean held at the root",
                   fontsize=8.6)
    axl.set_ylabel("L2 length of" + NL + "[ d/d(height) at width 3.2+dw ]  minus" + NL
                   + "[ d/d(height) at width 3.2 ]", fontsize=8.4)
    axl.set_title("SWEEP: the height column drifts exactly linearly in the width knob",
                  fontsize=10.2, pad=6)
    axl.legend(fontsize=8, loc="upper left", framealpha=0.95)
    axl.tick_params(labelsize=8)
    axl.grid(alpha=0.25, lw=0.6)

    axt = fig.add_subplot(gs[1, 2:])
    axt.axis("off")
    lines = [
        "WHY IT IS EXACT HERE, AND ONLY HERE",
        "",
        "At the root's centre and zero lean, with height in (18, 20)",
        "and width in (2, 4), the coverage image is an outer product",
        "of two affine profiles:",
        "",
        "      F(h, w) = C + h*U + w*V + h*w*D",
        "",
        "with C, U, V, D fixed 28x28 arrays. So  d/dh = U + w*D: the",
        "height column is a straight line in the width knob, and a",
        "width step dw changes it by exactly  dw*D.",
        "",
        "MEASURED    measured change, dw=%g   L2 %.6f" % (dw, np.linalg.norm(change)),
        "            predicted dw x D         L2 %.6f" % np.linalg.norm(pred),
        "            residual between the two L2 %.3e" % resid,
        "            worst gap from the line  L2 %.3e" % maxdev,
        "",
        "SCOPE  A fixed-knob slice: cx, cy and lean stay at the root.",
        "       Crossing into another pixel-formula region changes D,",
        "       so this line must not be extrapolated past the patch.",
    ]
    axt.text(0.0, 1.02, NL.join(lines), fontsize=7.4, va="top",
             family="monospace", linespacing=1.32)

    fig.savefig(os.path.join(OUT, "03_exact_column_drift.png"), dpi=155)
    plt.close(fig)

    res["figure_3"] = {
        "width_base": w0, "width_step": dw,
        "measured_change_l2": float(np.linalg.norm(change)),
        "predicted_change_l2": float(np.linalg.norm(pred)),
        "residual_l2": resid,
        "slope_norm_D": slope,
        "worst_deviation_from_line": maxdev,
        "sweep_dw": ws.round(4).tolist(),
        "sweep_measured_l2": meas.round(10).tolist(),
    }


def step_check(res):
    """Confirm the finite-difference columns are converged, not step-size artefacts."""
    a = jacobian(ROOT, STEP)
    b = jacobian(ROOT, STEP / 2)
    rel = float(np.abs(a - b).max() / np.abs(a).max())
    res["finite_difference_check"] = {
        "steps": dict(zip(KNOBS, STEP.tolist())),
        "max_relative_change_on_halving_steps": rel,
    }
    return rel


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    res = {}
    print("finite-difference stability on halving steps: %.3e" % step_check(res))
    figure_one(res)
    figure_two(res)
    figure_three(res)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res["figure_1"], indent=2))
    print(json.dumps(res["figure_2"]["starts"], indent=2))
    print(json.dumps({k: v for k, v in res["figure_3"].items() if not k.startswith("sweep")}, indent=2))
