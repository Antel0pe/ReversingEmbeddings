"""One figure: the direction of travel turns by a fixed amount at fixed places."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import smooth_ones
from analyze_turning_pattern import sweep, turns

OUT = "figures/turning_pattern/turning.png"

fig = plt.figure(figsize=(15.0, 6.6))
gs = fig.add_gridspec(1, 3, wspace=0.34, left=0.06, right=0.955, top=0.70, bottom=0.235)
fig.suptitle("Walking a knob, the direction of travel turns by a FIXED amount at FIXED places",
             fontsize=15.5, y=0.965)
fig.text(0.5, 0.878,
         "Each knob is swept in thousands of tiny steps.  The height of the line is how much the direction of travel swerved on that step:\n"
         "0° = still going the same way,  90° = swerved completely.  The journey is a POLYGON — long straight runs joined by sharp corners.",
         ha="center", va="top", fontsize=10.5, color="#444444", linespacing=1.6)

# ---- a: width
ax = fig.add_subplot(gs[0, 0])
t, ims = sweep(3, 1.0, 9.0, 8001)
a, _ = turns(ims)
ax.plot(t[1:-1], a, color="#c62828", lw=1.4)
for c in [2.0, 4.0, 6.0, 8.0]:
    ax.axvline(c, color="#1565c0", lw=1.0, ls=":")
ax.axhline(90, color="#2e7d32", lw=1.2, ls="--")
ax.text(8.9, 91.5, "90°", color="#2e7d32", fontsize=10, ha="right")
ax.set_title("a.  WIDEN the 1\na corner every 2.000, each turning 90.0°", fontsize=12, loc="left", pad=10)
ax.set_xlabel("width of the stroke (px)", fontsize=10.5)
ax.set_ylabel("how far the direction swerved on this step (degrees)", fontsize=10.5)
ax.set_ylim(-4, 108); ax.set_xlim(1, 9)
ax.grid(alpha=0.2)
ax.text(0.5, -0.195,
        "Dotted blue lines: where a stroke edge crosses a whole-number\n"
        "pixel boundary, worked out in advance. Every corner lands on one.\n"
        "Between corners the turn is 0.0000015° — dead straight.",
        transform=ax.transAxes, ha="center", va="top", fontsize=9.8, color="#1565c0")

# ---- b: cx
ax = fig.add_subplot(gs[0, 1])
t, ims = sweep(0, 11.0, 15.0, 8001)
a, _ = turns(ims)
ax.plot(t[1:-1], a, color="#c62828", lw=1.4)
for w in range(11, 15):
    for d in (0.4, 0.6):
        ax.axvline(w + d, color="#1565c0", lw=1.0, ls=":")
ax.axhline(60, color="#2e7d32", lw=1.2, ls="--")
ax.text(14.95, 61.5, "60°", color="#2e7d32", fontsize=10, ha="right")
ax.set_title("b.  SLIDE the 1 sideways\ntwo corners per pixel, each turning 60.0°", fontsize=12, loc="left", pad=10)
ax.set_xlabel("centre of the stroke, left-right (px)", fontsize=10.5)
ax.set_ylabel("how far the direction swerved on this step (degrees)", fontsize=10.5)
ax.set_ylim(-4, 108); ax.set_xlim(11, 15)
ax.grid(alpha=0.2)
ax.text(0.5, -0.195,
        "60° not 90° because sliding takes the two stroke edges over\n"
        "their boundaries at different moments, so only half of the\n"
        "change hands over instead of all of it.",
        transform=ax.transAxes, ha="center", va="top", fontsize=9.8, color="#444444")

# ---- c: lean
ax = fig.add_subplot(gs[0, 2])
t, ims = sweep(4, 10.0, 10.05, 6000)
a, _ = turns(ims)
ax.plot(t[1:-1], a, color="#c62828", lw=1.2, label="lean, zoomed into 1/900th of its range")
t2, im2 = sweep(4, 10.0, 10.05, 2000,
                renderer=lambda p: smooth_ones.render(np.asarray(p, float)[None])[0])
a2, _ = turns(im2)
ax.plot(t2[1:-1], a2 * 1000, color="#1565c0", lw=1.6,
        label="the same, soft edges, x1000 to be visible")
ax.set_title("c.  LEAN the 1 over\nsame mechanism, corners far too close to see", fontsize=12, loc="left", pad=10)
ax.set_xlabel("lean (degrees)", fontsize=10.5)
ax.set_ylabel("how far the direction swerved on this step (degrees)", fontsize=10.5)
ax.set_xlim(10.0, 10.05)
ax.grid(alpha=0.2)
ax.legend(fontsize=9.3, loc="upper right")
ax.text(0.5, -0.195,
        "Over its whole range lean looks like a smooth curve.\n"
        "Zoom in 900x and the corners are there; between them the\n"
        "turn is exactly 0.0000°. Soft edges (blue): no corners at all.",
        transform=ax.transAxes, ha="center", va="top", fontsize=9.8, color="#444444")

fig.savefig(OUT, dpi=130)
print(OUT)
