"""One figure: what two widenings genuinely share, and whether it pays off."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grey_ones import render
from analyze_what_is_the_same import (ROOT, SHORT, STEP, NAMES, one, ang, arrows,
                                      arrows_warp, left_total, _describe, _recover,
                                      _two_steps, _basis)

OUT = "figures/what_is_the_same/what_is_the_same.png"

fig = plt.figure(figsize=(15.0, 7.2))
gs = fig.add_gridspec(1, 3, wspace=0.38, left=0.055, right=0.965, top=0.715, bottom=0.235)
fig.suptitle("Two widenings share nothing in pixels. They share everything in WHERE THE INK MOVED.",
             fontsize=15.5, y=0.965)
fig.text(0.5, 0.875,
         "Describing a change by where ink slid, instead of by which pixels lit up, is the one description that works on every knob\n"
         "and on a different dataset — and it needs 6 numbers instead of 784.",
         ha="center", va="top", fontsize=10.5, color="#444444", linespacing=1.6)

# ---------------- a: the arrows
ax = fig.add_subplot(gs[0, 0])
I0 = one(ROOT)
p1 = ROOT.copy(); p1[3] += 2.0
p2 = ROOT.copy(); p2[3] += 4.0
I1, I2 = one(p1), one(p2)
D1, D2 = I1 - I0, I2 - I1
c1 = arrows_warp(I0, I1, 1)[0]; c2 = arrows_warp(I1, I2, 1)[0]
ax.imshow(I0, cmap="gray_r", vmin=0, vmax=1)


def quiver(c, color, ax, scale, off=0.0):
    n = 28
    g = (np.arange(n) - (n - 1) / 2) / ((n - 1) / 2)
    pts = np.arange(3, 26, 4)
    Y, X = np.meshgrid(g[pts], g[pts], indexing="ij")
    vx = c[0] + c[1] * X + c[2] * Y
    vy = c[3] + c[4] * X + c[5] * Y
    ax.quiver(pts[None, :] + 0 * X + off, pts[:, None] + 0 * Y, vx, -vy,
              color=color, angles="xy", scale_units="xy", scale=scale,
              width=0.016 if color.startswith("#ef") else 0.007,
              headwidth=3.2, zorder=3 if color.startswith("#ef") else 4)


quiver(c1 / np.linalg.norm(c1), "#ef5350", ax, 0.055)
quiver(c2 / np.linalg.norm(c2), "#1b5e20", ax, 0.075)
ax.set_title("a.  Arrows for the 1st widening (thick pale red) and\n"
             "the 2nd (thin dark green) sit on top of each other",
             fontsize=12, loc="left", pad=10)
ax.set_xticks([]); ax.set_yticks([])
ax.text(0.5, -0.045,
        f"which pixels changed:  {ang(D1, D2):.0f}° apart  (nothing in common)\n"
        f"where the ink moved :  {ang(c1, c2):.0f}° apart  (the same instruction)",
        transform=ax.transAxes, ha="center", va="top", fontsize=11, family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", fc="#f1f8e9", ec="#2e7d32", lw=1.3))

# ---------------- b: the overfit test
ax = fig.add_subplot(gs[0, 1])
meth = [("which pixels\nchanged", lambda k: ang(*_two_steps(k)[3:5]), "#9e9e9e"),
        ("ink to the LEFT\nof here", lambda k: ang(left_total(_two_steps(k)[1]) - left_total(_two_steps(k)[0]),
                                                   left_total(_two_steps(k)[2]) - left_total(_two_steps(k)[1])), "#1565c0"),
        ("where the ink\nMOVED", lambda k: ang(arrows_warp(_two_steps(k)[0], _two_steps(k)[1], 1)[0],
                                               arrows_warp(_two_steps(k)[1], _two_steps(k)[2], 1)[0]), "#2e7d32")]
w = 0.26
xs = np.arange(5)
for i, (nm, f, col) in enumerate(meth):
    v = [f(k) for k in range(5)]
    ax.bar(xs + (i - 1) * w, v, w, color=col, label=nm)
    for j, val in enumerate(v):
        ax.annotate(f"{val:.0f}", (xs[j] + (i - 1) * w, val), fontsize=7.5,
                    ha="center", va="bottom", xytext=(0, 2), textcoords="offset points")
ax.axhline(90, color="#c62828", lw=1.3, ls="--")
ax.text(4.45, 91, "90° = nothing in common", fontsize=8.5, color="#c62828", ha="right")
ax.set_xticks(xs)
ax.set_xticklabels(["slide\nsideways", "slide\nup-down", "make\ntaller", "widen", "lean\nover"], fontsize=9.5)
ax.set_ylabel("angle between a knob's 1st and 2nd step\n(small = the method sees them as the same thing)", fontsize=10)
ax.set_ylim(0, 118)
ax.set_title("b.  The overfitting test: run the SAME recipe\non every knob, not just the one it suits",
             fontsize=12, loc="left", pad=10)
ax.legend(fontsize=9, loc="upper left", ncol=3, columnspacing=0.9, handlelength=1.2)
ax.grid(alpha=0.2, axis="y")
ax.text(0.5, -0.20,
        "Ink-to-the-left is a left-right trick: it fixes widening but leaves the two up-down\n"
        "knobs exactly where it found them (82°, unchanged). The arrows work on all five.",
        transform=ax.transAxes, ha="center", va="top", fontsize=10, color="#444444")

# ---------------- c: the payoff
ax = fig.add_subplot(gs[0, 2])
rng = np.random.default_rng(1)
box = np.array([1.0, 1.0, 1.5, 2.8, 45.0])
fracs = [0.05, 0.20, 0.50]
series = {"all 784 pixels": ("pixels", "#9e9e9e"),
          "6 numbers: ink sliding": ("arrows1", "#1565c0"),
          "6 numbers: ink MOVED": ("warp1", "#2e7d32")}
res = {k: [] for k in series}
for frac in fracs:
    TH = (rng.random((150, 5)) - 0.5) * 2 * frac
    ims = render(ROOT + TH * box, dtype=np.float64)
    for nm, (how, _) in series.items():
        res[nm].append(100 * _recover(_describe(ims, I0, how), TH).max())
xs = np.arange(3)
for i, (nm, (_, col)) in enumerate(series.items()):
    ax.bar(xs + (i - 1) * w, res[nm], w, color=col, label=nm)
    for j, v in enumerate(res[nm]):
        ax.annotate(f"{v:.1f}%", (xs[j] + (i - 1) * w, v), fontsize=8,
                    ha="center", va="bottom", xytext=(0, 2), textcoords="offset points")
ax.set_xticks(xs)
ax.set_xticklabels([f"small\n(up to {int(100*f)}% of each\nknob's range)" if i == 0 else
                    f"medium\n(up to {int(100*f)}%)" if i == 1 else f"large\n(up to {int(100*f)}%)"
                    for i, f in enumerate(fracs)], fontsize=9.5)
ax.set_ylabel("worst knob's error when reading the knob\nvalues back  (% of that knob's range)", fontsize=10)
ax.set_title("c.  The payoff: can you read the knob\nvalues back? (held-out half of 150)",
             fontsize=12, loc="left", pad=10)
ax.legend(fontsize=9.5, loc="upper left")
ax.grid(alpha=0.2, axis="y")
ax.set_ylim(0, 27)
ax.text(0.5, -0.20,
        "On small changes 6 numbers match all 784.\n"
        "On large ones they beat them nearly two to one.",
        transform=ax.transAxes, ha="center", va="top", fontsize=10, color="#444444")

fig.savefig(OUT, dpi=130)
print(OUT)
