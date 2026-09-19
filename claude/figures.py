"""Figures for the ladder experiments. Everything drawn is a 784-vector shown as 28x28."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
X, y, ones, P, E = load()
R = np.load("claude/renders.npz")
n = len(ones); rng = np.random.default_rng(3)
OUT = "claude/figures"
GREEN, RED, GREY = "#1a7f37", "#c62828", "#555"

def show(ax, v, cmap="gray", vmin=0, vmax=1):
    ax.imshow(np.asarray(v).reshape(28, 28), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])

def diff(ax, v, scale=1.0):
    ax.imshow(np.asarray(v).reshape(28, 28), cmap="RdBu_r", vmin=-scale, vmax=scale, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])

# ---------------------------------------------------------------- fig 1: the ladder
# columns: six real 1s chosen by their STRAIGHT residual (easy -> hard); rows: real, then each rung's render + leftover
picks = [int(np.argsort(np.abs(E["straight"] - v))[0]) for v in (1.0, 1.6, 2.2, 3.0, 4.0, 5.5)]
fig = plt.figure(figsize=(13.5, 12.4))
fig.text(0.5, 0.975, "A LADDER OF GENERATORS FOR REAL 1s: the best drawing each rung can make, and what is left over",
         ha="center", fontsize=14, weight="bold")
fig.text(0.5, 0.953, "Columns: real 1s from easy to hard for the straight rung.  Each rung row: its best render (top) and real minus render (bottom, red = ink the render lacks, blue = ink it has that the real 1 lacks).",
         ha="center", fontsize=10, color=GREY)
fig.text(0.5, 0.936, f"Number under each render = leftover distance, same units as THETA = {THETA}.  Green if under 2.0 (the straight-1 cutoff in StraightOnes).",
         ha="center", fontsize=10, color=GREY)
nrow = 1 + 2 * len(RUNGS); ncol = len(picks)
gs = fig.add_gridspec(nrow, ncol, left=0.11, right=0.98, top=0.915, bottom=0.03, hspace=0.35, wspace=0.08,
                      height_ratios=[1] + [1, 1] * len(RUNGS))
for c, i in enumerate(picks):
    ax = fig.add_subplot(gs[0, c]); show(ax, ones[i])
    if c == 0: ax.text(-0.15, 0.5, "REAL", transform=ax.transAxes, ha="right", va="center", fontsize=11, weight="bold")
    ax.set_title(f"1 #{i}", fontsize=9, color=GREY)
for rr, r in enumerate(RUNGS):
    for c, i in enumerate(picks):
        ax = fig.add_subplot(gs[1 + 2 * rr, c]); show(ax, R[r][i])
        e = E[r][i]
        ax.text(0.5, -0.12, f"{e:.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=9,
                color=GREEN if e < 2.0 else RED, weight="bold")
        if c == 0:
            ax.text(-0.15, 0.5, f"{r.upper()}\n{len(ladder.KNOBS[r])} knobs", transform=ax.transAxes, ha="right", va="center", fontsize=10, weight="bold")
        ax2 = fig.add_subplot(gs[2 + 2 * rr, c]); diff(ax2, ones[i] - R[r][i])
        if c == 0: ax2.text(-0.15, 0.5, "leftover", transform=ax2.transAxes, ha="right", va="center", fontsize=9, color=GREY)
fig.savefig(f"{OUT}/ladder.png", dpi=110, facecolor="white"); plt.close(fig)

# ---------------------------------------------------------------- fig 2: residual distributions per rung
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
bins = np.linspace(0, 6, 61)
for r, col in zip(RUNGS, ["#bbbbbb", "#7fa7d1", "#3b6ea5", "#1b3b5f"]):
    ax[0].hist(E[r], bins, histtype="step", lw=2, color=col, label=f"{r} ({len(ladder.KNOBS[r])} knobs): median {np.median(E[r]):.2f}")
ax[0].axvline(THETA, color=RED, ls="--", lw=1.2); ax[0].text(THETA + 0.05, ax[0].get_ylim()[1] * 0.95, "THETA", color=RED, fontsize=9, va="top")
ax[0].set_xlabel("leftover distance ||real - best render||"); ax[0].set_ylabel("number of 1s"); ax[0].legend(fontsize=9)
ax[0].set_title("How thick is the tube around each rung?", fontsize=11)
cuts = np.linspace(0.5, 4, 36)
for r, col in zip(RUNGS, ["#bbbbbb", "#7fa7d1", "#3b6ea5", "#1b3b5f"]):
    ax[1].plot(cuts, [100 * (E[r] < c).mean() for c in cuts], color=col, lw=2, label=r)
ax[1].axvline(THETA, color=RED, ls="--", lw=1.2); ax[1].axhline(90, color=GREY, ls=":", lw=1)
ax[1].set_xlabel("tube radius"); ax[1].set_ylabel("% of real 1s inside the tube"); ax[1].legend(fontsize=9)
ax[1].set_title("Share of the 6742 real 1s within a given distance of the surface", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/tube.png", dpi=110, facecolor="white"); plt.close(fig)

# ---------------------------------------------------------------- fig 3: the knobs as images at two different 1s
r = "foot"; names = ladder.KNOBS[r]
thin = int(np.argsort(P[r][:, 4])[len(ones) // 20]); thick = int(np.argsort(P[r][:, 4])[-len(ones) // 20])
fig = plt.figure(figsize=(14, 6.4))
fig.text(0.5, 0.965, "WHERE THE INTRINSIC DIRECTIONS ARE, in the original 784 coordinates", ha="center", fontsize=14, weight="bold")
fig.text(0.5, 0.925, "Each panel is the 784-vector 'turn this one knob a little' at that 1, i.e. one column of the generator's Jacobian, drawn as an image (red = ink appears, blue = ink vanishes).\n"
         "Two 1s, one thin and one thick: the SAME knob is a DIFFERENT direction in pixel space at the two points. That is what a curved surface means here.",
         ha="center", va="top", fontsize=10, color=GREY)
gs = fig.add_gridspec(2, len(names) + 1, left=0.03, right=0.99, top=0.83, bottom=0.06, wspace=0.08, hspace=0.35)
for row, i, lbl in [(0, thin, "thin 1"), (1, thick, "thick 1")]:
    ax = fig.add_subplot(gs[row, 0]); show(ax, ones[i]); ax.set_title(f"{lbl} #{i}", fontsize=10, weight="bold")
    J = ladder.jacobian(P[r][i], r)
    for j, nm in enumerate(names):
        ax = fig.add_subplot(gs[row, j + 1]); col = J[:, j]
        diff(ax, col / max(np.abs(col).max(), 1e-9))
        if row == 0: ax.set_title(nm, fontsize=10)
        ax.text(0.5, -0.08, f"|d/d{nm}| = {np.linalg.norm(col):.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=8, color=GREY)
fig.savefig(f"{OUT}/knob_directions.png", dpi=110, facecolor="white"); plt.close(fig)

# ---------------------------------------------------------------- fig 4: slider board through one real 1
i = int(np.argsort(np.abs(E["foot"] - np.median(E["foot"])))[0])
p = P[r][i].copy(); lo = np.percentile(P[r], 5, axis=0); hi = np.percentile(P[r], 95, axis=0)
steps = 7
fig = plt.figure(figsize=(13, 2.0 * len(names) + 1.6))
fig.text(0.5, 1 - 0.35 / fig.get_figheight(), f"WALKING EACH KNOB THROUGH A REAL 1 (#{i}): the surface, and how close real data stays to it",
         ha="center", fontsize=14, weight="bold")
fig.text(0.5, 1 - 0.65 / fig.get_figheight(), "Top of each pair: the render as one knob sweeps its 5th-95th population percentile (others fixed at this 1's values).  Bottom: the nearest real 1 to that render.\n"
         "Number = distance from render to that nearest real 1: green if under THETA = 1.89.  Middle column is this 1's own fitted knobs.",
         ha="center", va="top", fontsize=10, color=GREY)
gs = fig.add_gridspec(2 * len(names), steps, left=0.10, right=0.99, top=1 - 1.25 / fig.get_figheight(), bottom=0.015, wspace=0.06, hspace=0.32)
for j, nm in enumerate(names):
    vals = np.linspace(lo[j], hi[j], steps); vals[steps // 2] = p[j]
    for c, v in enumerate(vals):
        q = p.copy(); q[j] = v; img = ladder.render(q, r)
        d, k = nearest(img[None], ones); d, k = d[0], k[0]
        ax = fig.add_subplot(gs[2 * j, c]); show(ax, img)
        ax.text(0.5, -0.1, f"{d:.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=9, weight="bold", color=GREEN if d < THETA else RED)
        if c == 0: ax.text(-0.12, 0.5, nm, transform=ax.transAxes, ha="right", va="center", fontsize=10, weight="bold")
        if j == 0: ax.set_title("this 1" if c == steps // 2 else f"{'low' if c < steps//2 else 'high'}", fontsize=9, color=GREY)
        ax2 = fig.add_subplot(gs[2 * j + 1, c]); show(ax2, ones[k])
        if c == 0: ax2.text(-0.12, 0.5, "nearest\nreal", transform=ax2.transAxes, ha="right", va="center", fontsize=8, color=GREY)
        ax.text(0.03, 0.97, f"{v:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=7, color="#ffcc00")
fig.savefig(f"{OUT}/sliders.png", dpi=105, facecolor="white"); plt.close(fig)

# ---------------------------------------------------------------- fig 5: the knob population = the manifold in nameable coordinates
Pr = P[r]; ok = E[r] < 2.5
fig, axes = plt.subplots(2, 5, figsize=(14, 5.6))
fig.suptitle(f"THE 1s IN KNOB COORDINATES: the fitted value of each knob over the {ok.sum()} real 1s whose leftover is under 2.5", fontsize=13, weight="bold")
units = {"cx": "px", "cy": "px", "lean": "rad", "halflen": "px", "halfwidth": "px", "softness": "px", "bend": "px per px^2", "flaglen": "px", "flagdir": "rad", "footlen": "px"}
for j, (ax, nm) in enumerate(zip(axes.ravel(), names)):
    ax.hist(Pr[ok, j], 40, color="#3b6ea5"); ax.set_title(f"{nm} ({units[nm]})", fontsize=10)
    ax.text(0.97, 0.95, f"median {np.median(Pr[ok, j]):.2f}\n5-95%: {np.percentile(Pr[ok,j],5):.2f} to {np.percentile(Pr[ok,j],95):.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(f"{OUT}/knob_population.png", dpi=110, facecolor="white"); plt.close(fig)
print("figures written")
