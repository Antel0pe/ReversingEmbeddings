"""Shareable map of the measured bumps and dips, with attention marks that sit beside
the features (a circle around the square, arrows next to the paths), never on them.

Usage: python -I lidar/make_social_relief.py SURFACES.npz TRACES.json OUT.png CX CY
"""
import json
import sys

import matplotlib
import numpy as np
from scipy import ndimage as ndi

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

surf, traces_path, out = sys.argv[1:4]
CX, CY = float(sys.argv[4]), float(sys.argv[5])
X_RANGE, Y_RANGE = (-330, 330), (-420, 405)  # metres from the square's centre (4:5)
LIM = 0.3  # colour scale, metres

s = np.load(surf)
dtm = s["dtm"].astype(float)
X0, Y0 = s["origin"]
CELL = float(s["cell"])
filled = np.where(np.isnan(dtm), np.nanmedian(dtm), dtm)
rel = ndi.gaussian_filter(filled - ndi.gaussian_filter(filled, 10 / CELL), 1.5 / CELL)

c0, c1 = int((CX + X_RANGE[0] - X0) / CELL), int((CX + X_RANGE[1] - X0) / CELL)
r0, r1 = int((Y0 - (CY + Y_RANGE[1])) / CELL), int((Y0 - (CY + Y_RANGE[0])) / CELL)
img = rel[r0:r1, c0:c1]

fig = plt.figure(figsize=(8, 10), dpi=200)
ax = fig.add_axes([0.02, 0.1, 0.96, 0.8])
im = ax.imshow(np.clip(img, -LIM, LIM), cmap="RdBu_r", vmin=-LIM, vmax=LIM,
               extent=(*X_RANGE, *Y_RANGE), interpolation="bilinear")

INK = "#111111"
BOX = dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.92)
traces = json.load(open(traces_path))

# circle around the square, well outside its ditch
ax.add_patch(Circle((3, 2), 78, fill=False, ec=INK, lw=2.2))
ax.annotate("square earthwork, about 75 m across", (55, 58), xytext=(70, 175), fontsize=12.5, bbox=BOX,
            arrowprops=dict(arrowstyle="-", color=INK, lw=1.2))


def arrow_beside(name, along_m, side=1, gap=9, length=45):
    """An arrow whose tip stops `gap` metres to one side of the traced path."""
    p = np.array(traces[name]["path"]) - [CX, CY]
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    i = int(np.argmin(np.abs(d - along_m)))
    j = min(i + 3, len(p) - 1)
    t = (p[j] - p[max(i - 3, 0)])
    t /= np.linalg.norm(t)
    n = np.array([-t[1], t[0]]) * side
    tip = p[i] + gap * n
    ax.annotate("", tip, xytext=tip + length * n,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2, mutation_scale=16))
    return tip + (length + 14) * n


lab = arrow_beside("nw_out", 130, side=1)
ax.text(*lab, "path", fontsize=12.5, ha="center", va="center", bbox=BOX)
arrow_beside("nw_back", 60, side=1)
lab = arrow_beside("east", 160, side=-1)
ax.text(*lab, "path", fontsize=12.5, ha="center", va="center", bbox=BOX)
lab = arrow_beside("sw_back", 90, side=1, length=70)
ax.text(*lab, "path, across\nthe stream", fontsize=12.5, ha="center", va="center", bbox=BOX)

# scale bar and north arrow
x_s, y_s = X_RANGE[1] - 135, Y_RANGE[0] + 28
ax.add_patch(plt.Rectangle((x_s - 12, y_s - 14), 124, 46, fc="white", ec="none", alpha=0.92))
ax.plot([x_s, x_s + 100], [y_s, y_s], color=INK, lw=4, solid_capstyle="butt")
ax.text(x_s + 50, y_s + 8, "100 m", ha="center", va="bottom", fontsize=11)
ax.annotate("N", (X_RANGE[1] - 28, Y_RANGE[0] + 140), xytext=(X_RANGE[1] - 28, Y_RANGE[0] + 92),
            ha="center", va="top", fontsize=12, fontweight="bold", bbox=BOX,
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
ax.set_xlim(*X_RANGE)
ax.set_ylim(*Y_RANGE)
ax.axis("off")

cax = fig.add_axes([0.25, 0.062, 0.5, 0.013])
cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-LIM, 0, LIM])
cb.ax.set_xticklabels([f"{-LIM:.1f} m lower", "level", f"{LIM:.1f} m higher"], fontsize=10)
cb.outline.set_visible(False)
fig.text(0.5, 0.083, "Ground height compared with the ground around it (within ~10 m)", ha="center", fontsize=10.5)

fig.text(0.03, 0.955, "Under the rainforest in Acre, Brazil", fontsize=19, fontweight="bold")
fig.text(0.03, 0.925, "Airborne lidar with the trees removed: blue is lower ground, red is higher.", fontsize=12.5)
fig.text(0.03, 0.025, "Lidar: Sustainable Landscapes Brazil (Embrapa, US Forest Service), 2013, CC BY-NC 4.0.\n"
         "Square published as site ACE-05 (Peripato et al. 2023, Science).", fontsize=9, va="center", color="#444444")
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out)
