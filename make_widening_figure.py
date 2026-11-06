"""One figure: why two widenings look unrelated, and one way to fix it."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coverage_grids import render_coverage

OUT = "figures/widening_as_a_matrix/widening_steps.png"
ROOT = np.array([14.0, 14.0, 19.75, 3.2, 0.0])
WS = [3.2, 5.2, 7.2]
C = ["#1565c0", "#c62828", "#2e7d32"]


def hard(w):
    p = ROOT.copy(); p[3] = w
    return (render_coverage(p[None], n_pix=28) >= 0.5).astype(float)[0]


def angle(a, b):
    a, b = a.ravel(), b.ravel()
    return np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))


ims = [hard(w) for w in WS]
row = 14
x = np.arange(28)
s1, s2 = ims[1] - ims[0], ims[2] - ims[1]
cum = [np.cumsum(m, axis=1) for m in ims]
c1, c2 = cum[1] - cum[0], cum[2] - cum[1]
a_plain, a_cum = angle(s1, s2), angle(c1, c2)
shared = int(((s1 != 0) & (s2 != 0)).sum())

fig = plt.figure(figsize=(13.5, 10.2))
gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.26,
                      left=0.075, right=0.975, top=0.805, bottom=0.235)

fig.suptitle("Two widenings of the same 1 look completely unrelated to a computer.\n"
             "Writing the picture down differently fixes that.", fontsize=16, y=0.975)
fig.text(0.5, 0.875,
         "Pictures are 28x28. A pixel is either fully inked (black) or empty (white).",
         ha="center", va="top", fontsize=10.5, color="#555555")

# ---------- a
ax = fig.add_subplot(gs[0, 0])
strip = np.concatenate([np.pad(m, ((0, 0), (1, 1)), constant_values=0.12) for m in ims], axis=1)
ax.imshow(strip, cmap="gray_r", vmin=0, vmax=1)
for i, w in enumerate(WS):
    ax.text(30 * i + 15, -1.5, f"width {w} px", ha="center", va="bottom",
            fontsize=11, color=C[i], fontweight="bold")
ax.set_title("a.  Start with a 1, then widen it twice", fontsize=12.5, loc="left", pad=26)
ax.axhline(row, color="#ff8f00", lw=2.0, ls="--")
ax.text(0.5, -0.10, "the orange line is the middle row — panels b, c and d all read along it",
        transform=ax.transAxes, ha="center", va="top", fontsize=10, color="#e65100")
ax.set_xticks([]); ax.set_yticks([])

# ---------- b
ax = fig.add_subplot(gs[0, 1])
for i, (m, w) in enumerate(zip(ims, WS)):
    ax.plot(x, m[row], drawstyle="steps-mid", color=C[i], lw=2.4, label=f"width {w} px")
ax.set_title("b.  The same three pictures, as a graph\n"
             "ink along that middle row", fontsize=12.5, loc="left", pad=10)
ax.set_xlabel("which column of pixels, left to right", fontsize=10.5)
ax.set_ylabel("ink in that pixel\n(0 = empty, 1 = full)", fontsize=10.5)
ax.set_ylim(-0.12, 1.45); ax.set_xlim(6, 22)
ax.legend(fontsize=10, loc="upper left")
ax.grid(alpha=0.22)

# ---------- c
ax = fig.add_subplot(gs[1, 0])
ax.fill_between(x, 0, s1[row], step="mid", color="#c62828", alpha=0.45)
ax.fill_between(x, 0, s2[row], step="mid", color="#2e7d32", alpha=0.45)
ax.plot(x, s1[row], drawstyle="steps-mid", color="#c62828", lw=2.6, label="what the 1st widening changed")
ax.plot(x, s2[row], drawstyle="steps-mid", color="#2e7d32", lw=2.6, label="what the 2nd widening changed")
ax.axhline(0, color="#444444", lw=1)
ax.set_title("c.  THE PROBLEM — the usual way of writing a picture:\n"
             "how much ink is in each pixel", fontsize=12.5, loc="left", pad=10)
ax.set_xlabel("which column of pixels, left to right", fontsize=10.5)
ax.set_ylabel("extra ink gained in that pixel", fontsize=10.5)
ax.set_ylim(-0.12, 1.45); ax.set_xlim(6, 22)
ax.legend(fontsize=10, loc="upper left")
ax.grid(alpha=0.22)
ax.text(0.5, -0.30,
        f"The two humps never touch — {shared} pixels shared.\n"
        f"So the computer scores them {a_plain:.0f}° apart: the same\n"
        "score it would give two pictures with nothing\n"
        "whatever to do with each other.",
        transform=ax.transAxes, ha="center", va="top", fontsize=11, linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.5", fc="#ffebee", ec="#c62828", lw=1.4))

# ---------- d
ax = fig.add_subplot(gs[1, 1])
ax.fill_between(x, 0, c1[row], step="mid", color="#c62828", alpha=0.35)
ax.fill_between(x, 0, c2[row], step="mid", color="#2e7d32", alpha=0.35)
ax.plot(x, c1[row], drawstyle="steps-mid", color="#c62828", lw=2.6, label="what the 1st widening changed")
ax.plot(x, c2[row], drawstyle="steps-mid", color="#2e7d32", lw=2.6, label="what the 2nd widening changed")
ax.axhline(0, color="#444444", lw=1)
ax.set_title("d.  ONE FIX — write the same picture as:\n"
             "how much ink is to the LEFT of here", fontsize=12.5, loc="left", pad=10)
ax.set_xlabel("which column of pixels, left to right", fontsize=10.5)
ax.set_ylabel("extra ink gained to the left of here", fontsize=10.5)
ax.set_ylim(-0.12, 2.9); ax.set_xlim(6, 22)
ax.legend(fontsize=10, loc="upper left")
ax.grid(alpha=0.22)
ax.text(0.5, -0.30,
        f"Now the two lie almost on top of each other: {a_cum:.0f}° apart.\n"
        "Nearly the same change, done twice.\n"
        "Nothing blurred, nothing thrown away: subtracting\n"
        "neighbouring columns gives panel c back exactly.",
        transform=ax.transAxes, ha="center", va="top", fontsize=11, linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.5", fc="#e8f5e9", ec="#2e7d32", lw=1.4))

fig.text(0.5, 0.035,
         "0° = \"the very same change again\".   90° = \"two changes with nothing in common\".",
         ha="center", fontsize=11, color="#333333")

fig.savefig(OUT, dpi=130)
print(OUT)
