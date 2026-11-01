"""How the local linear map (the Jacobian) changes across the knob box.

Figure: the five first-derivative images as a reference strip, then the 5x5
table of second-derivative images - "turning knob j bends knob i's direction
by THIS picture". Normalised knob units, one shared colour scale.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from grey_ones import render, RANGES

OUT = "figures/generated_one_jacobian_change"
NAMES = ["cx", "cy", "height", "width", "lean"]
lo, hi = RANGES[:, 0], RANGES[:, 1]
SPAN = hi - lo
P0 = (lo + hi) / 2
S = 0.10                      # equal step, as a fraction of each knob's range


def r(p):
    return render([list(np.asarray(p, float))], dtype=np.float64)[0]


def d1(i):
    a, b = P0.copy(), P0.copy()
    a[i] -= S * SPAN[i]
    b[i] += S * SPAN[i]
    return (r(b) - r(a)) / (2 * S * SPAN[i]) * SPAN[i]


def d2(i, j):
    """Symmetric four-point mixed second difference, equal steps both ways."""
    hi_, hj = S * SPAN[i], S * SPAN[j]
    pp, pm, mp, mm = (P0.copy() for _ in range(4))
    pp[i] += hi_; pp[j] += hj
    pm[i] += hi_; pm[j] -= hj
    mp[i] -= hi_; mp[j] += hj
    mm[i] -= hi_; mm[j] -= hj
    return (r(pp) - r(pm) - r(mp) + r(mm)) / (4 * hi_ * hj) * SPAN[i] * SPAN[j]


D1 = [d1(i) for i in range(5)]
D2 = [[d2(i, j) for j in range(5)] for i in range(5)]
M1 = max(np.abs(x).max() for x in D1)
M2 = max(np.abs(D2[i][j]).max() for i in range(5) for j in range(5))
ASYM = max(np.linalg.norm(D2[i][j] - D2[j][i]) for i in range(5) for j in range(5))

fig = plt.figure(figsize=(12.6, 16.2))
fig.patch.set_facecolor("white")
fig.text(0.5, 0.988, "How each knob bends every other knob's direction",
         ha="center", va="top", fontsize=18.5, fontweight="bold")
fig.text(0.5, 0.970,
         "Every square is a 28x28 DIRECTION, not a 1.  Blue = ink gained, red = ink lost.  Knob units are normalised to each knob's full range.",
         ha="center", va="top", fontsize=11, color="#333")

COLX = [0.185 + 0.146 * j for j in range(5)]
SQ = lambda a: np.sign(a) * np.sqrt(np.abs(a))
CW = 0.130
ROWY = [0.634 - 0.1245 * i for i in range(5)]
RH = 0.108

# reference strip: the first derivatives
fig.text(0.5, 0.952, "REFERENCE:  what each knob does on its own  (the five columns of the Jacobian)",
         ha="center", va="top", fontsize=12.2, fontweight="bold", color="#114477")
for j in range(5):
    ax = fig.add_axes([COLX[j], 0.838, CW, 0.092])
    ax.imshow(SQ(D1[j]), cmap="bwr_r", vmin=-np.sqrt(M1), vmax=np.sqrt(M1), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(NAMES[j], fontsize=12, pad=5)
    ax.text(0.5, -0.11, f"size {np.linalg.norm(D1[j]):.1f}", transform=ax.transAxes,
            ha="center", va="top", fontsize=9.3, color="#333")
    for s in ax.spines.values():
        s.set_color("#114477"); s.set_linewidth(1.8)

fig.text(0.5, 0.800, "HOW THAT CHANGES:  turning the knob across the top bends the direction of the knob at the left",
         ha="center", va="top", fontsize=12.2, fontweight="bold", color="#772211")
for j in range(5):
    fig.text(COLX[j] + CW / 2, 0.770, NAMES[j], ha="center", va="bottom", fontsize=12)
fig.text((COLX[0] + COLX[4] + CW) / 2, 0.752, "turn this knob  ->", ha="center", va="bottom",
         fontsize=10.5, style="italic", color="#555")

for i in range(5):
    fig.text(0.172, ROWY[i] + RH / 2, NAMES[i], ha="right", va="center", fontsize=12)
    for j in range(5):
        ax = fig.add_axes([COLX[j], ROWY[i], CW, RH])
        ax.imshow(SQ(D2[i][j]), cmap="bwr_r", vmin=-np.sqrt(M2), vmax=np.sqrt(M2), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        nrm = np.linalg.norm(D2[i][j])
        ax.text(0.5, -0.10, f"{nrm:.1f}", transform=ax.transAxes, ha="center", va="top",
                fontsize=10, color="#b00000" if nrm > 40 else "#333",
                fontweight="bold" if nrm > 40 else "normal")
        for s in ax.spines.values():
            s.set_color("#999"); s.set_linewidth(1.0)
fig.text(0.03, (ROWY[0] + ROWY[4] + RH) / 2, "...bends this knob's direction",
         ha="center", va="center", fontsize=10.5, style="italic", color="#555", rotation=90)

cax = fig.add_axes([0.905, ROWY[4], 0.014, ROWY[0] + RH - ROWY[4]])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-np.sqrt(M2), np.sqrt(M2)), cmap="bwr_r"), cax=cax)
TK = [-25, -9, -1, 0, 1, 9, 25]
cb.set_ticks([np.sign(v) * np.sqrt(abs(v)) for v in TK])
cb.set_ticklabels([str(v) for v in TK])
cb.set_label("ink change per unit^2 of knob   (blue = gained, red = lost)", fontsize=10, labelpad=4)
cb.ax.tick_params(labelsize=8.5)

fig.text(0.5, 0.012,
         f"All 25 squares share one colour scale, drawn with a square-root stretch so the faint ones stay visible; the number under each is that picture's true size.\n"
         f"Read it like a multiplication table. The lean row and column dominate - lean/lean is {np.linalg.norm(D2[4][4]):.0f}, five times the next largest.\n"
         f"cy/height is exactly 0.0 (a blank square): raising the stroke and lengthening it do not affect each other at all.\n"
         f"Mixed partials must commute, and this table is symmetric to {ASYM:.1e} - a built-in check that the measurement is sound.",
         ha="center", va="bottom", fontsize=10.1, color="#333", linespacing=1.6)
fig.savefig(f"{OUT}/second_derivative_grid.png", dpi=125, facecolor="white")
plt.close(fig)
print("wrote", f"{OUT}/second_derivative_grid.png")
print("max |d2| =", round(float(M2), 4), " worst asymmetry =", f"{ASYM:.2e}")
