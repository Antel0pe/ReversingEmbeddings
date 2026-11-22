"""One clean map for sharing: shaded bare ground with the square and its paths.

Usage: python -I lidar/make_social_map.py SURFACES.npz TRACES.json OUT.png CX CY
"""
import json
import sys

import matplotlib
import numpy as np
from scipy import ndimage as ndi

matplotlib.use("Agg")
import matplotlib.pyplot as plt

surf, traces_path, out = sys.argv[1:4]
CX, CY = float(sys.argv[4]), float(sys.argv[5])
X_RANGE, Y_RANGE = (-520, 560), (-900, 560)  # metres from the square's centre

s = np.load(surf)
dtm = s["dtm"].astype(float)
X0, Y0 = s["origin"]
CELL = float(s["cell"])
NY, NX = dtm.shape

c0, c1 = int((CX + X_RANGE[0] - X0) / CELL), int((CX + X_RANGE[1] - X0) / CELL)
r0, r1 = int((Y0 - (CY + Y_RANGE[1])) / CELL), int((Y0 - (CY + Y_RANGE[0])) / CELL)
z = dtm[r0:r1, c0:c1]
zz = ndi.gaussian_filter(np.where(np.isnan(z), np.nanmedian(z), z), 2.5) * 3.0  # heights x3

gy, gx = np.gradient(zz, CELL)  # gy increases southward, as in the standard hillshade formula
slope = np.arctan(np.hypot(gx, gy))
aspect = np.arctan2(gy, -gx)
alt, az = np.radians(40), np.radians(360 - 315 + 90)
shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
shade[np.isnan(z)] = np.nan

cmap = plt.get_cmap("gray").copy()
cmap.set_bad("white")
fig = plt.figure(figsize=(8, 10), dpi=200)
ax = fig.add_axes([0.02, 0.07, 0.96, 0.84])
ax.imshow(shade, cmap=cmap, vmin=0.15, vmax=0.95, extent=(*X_RANGE, *Y_RANGE), interpolation="bilinear")

ORANGE = "#ff6a00"
BOX = dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.92)
traces = json.load(open(traces_path))
square = []
for name, t in traces.items():
    if name.startswith("unstable") or not t["path"]:
        continue
    p = np.array(t["path"]) - [CX, CY]
    if name.startswith("square"):
        square.append(p)
        continue
    ax.plot(p[:, 0], p[:, 1], color=ORANGE, lw=2.4, alpha=0.9, solid_capstyle="round")
# the four traced sides (north, east, south, west order in the file): fit a straight line to
# each, and use where neighbouring lines cross as the corners
def fit(p):
    c = p.mean(axis=0)
    d = np.linalg.svd(p - c)[2][0]
    return c, d


def cross(a, b):
    (c1, d1), (c2, d2) = a, b
    t = np.linalg.solve(np.c_[d1, -d2], c2 - c1)
    return c1 + t[0] * d1


sides = [fit(p) for p in square]
corners = [cross(sides[i], sides[(i + 1) % 4]) for i in range(4)]
ring = np.vstack(corners + corners[:1])
ax.plot(ring[:, 0], ring[:, 1], color=ORANGE, lw=3.0)

ax.annotate("square earthwork, about 75 m across", (50, 45), xytext=(110, 200), fontsize=12, bbox=BOX,
            arrowprops=dict(arrowstyle="-", color="k", lw=1.2))
ax.annotate("paths leading from it", (-290, 290), xytext=(-500, 420), fontsize=12, bbox=BOX,
            arrowprops=dict(arrowstyle="-", color="k", lw=1.2))
ax.annotate("stream", (-200, -120), xytext=(-470, -60), fontsize=12, bbox=BOX,
            arrowprops=dict(arrowstyle="-", color="k", lw=1.2))

# scale bar and north arrow
x_s, y_s = X_RANGE[1] - 250, Y_RANGE[0] + 50
ax.add_patch(plt.Rectangle((x_s - 20, y_s - 25), 240, 75, fc="white", ec="none", alpha=0.92))
ax.plot([x_s, x_s + 200], [y_s, y_s], color="k", lw=4, solid_capstyle="butt")
ax.text(x_s + 100, y_s + 14, "200 m", ha="center", va="bottom", fontsize=11)
ax.annotate("N", (X_RANGE[1] - 45, Y_RANGE[0] + 230), xytext=(X_RANGE[1] - 45, Y_RANGE[0] + 150),
            ha="center", va="top", fontsize=12, fontweight="bold", bbox=BOX,
            arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))
ax.set_xlim(*X_RANGE)
ax.set_ylim(*Y_RANGE)
ax.axis("off")
fig.text(0.03, 0.955, "Under the rainforest in Acre, Brazil", fontsize=19, fontweight="bold")
fig.text(0.03, 0.925, "Airborne lidar with the trees removed. Shading shows the bare ground.", fontsize=12.5)
fig.text(0.03, 0.035, "Lidar: Sustainable Landscapes Brazil (Embrapa, US Forest Service), 2013, CC BY-NC 4.0.\n"
         "Square published as site ACE-05 (Peripato et al. 2023, Science). Paths traced from the same data.",
         fontsize=9, color="#444444", va="center")
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out)
