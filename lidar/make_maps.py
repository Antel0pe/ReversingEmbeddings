"""Draw the maps from process.py's surfaces.npz.

Usage: python -I lidar/make_maps.py OUT_DIR TILE.laz CX CY [HALF_WIDTH_M]
  CX, CY: UTM (zone 19S) centre of the feature to zoom on.

Figure 1  whole tile: canopy top | bare ground | bumps and dips
Figure 2  zoom on the feature: canopy top | bumps and dips, plus a side view of
          the raw laser returns along one line through it (tree vs ground).
"""
import os
import sys

import laspy
import matplotlib
import numpy as np
from scipy import ndimage as ndi

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

out_dir, tile = sys.argv[1], sys.argv[2]
CX, CY = float(sys.argv[3]), float(sys.argv[4])
HALF = float(sys.argv[5]) if len(sys.argv) > 5 else 110.0

s = np.load(os.path.join(out_dir, "surfaces.npz"))
dtm, dsm, covered = s["dtm"].astype(float), s["dsm"].astype(float), s["covered"]
X0, Y0 = s["origin"]
CELL = float(s["cell"])
NY, NX = dtm.shape


def shade(z, exaggerate=1.0, cell=CELL):
    """Shaded relief: a light low in the north-west sky, heights stretched by `exaggerate`."""
    zz = np.where(np.isnan(z), np.nanmedian(z), z) * exaggerate
    gy, gx = np.gradient(zz, cell)  # gy increases southward, as in the standard (ESRI) formula
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(gy, -gx)
    alt, az = np.radians(35), np.radians(360 - 315 + 90)
    h = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    h[np.isnan(z)] = np.nan
    return h


def relief(z, radius_m=10.0):
    """Height above or below the average ground within about radius_m."""
    zz = np.where(np.isnan(z), np.nanmedian(z), z)
    r = zz - ndi.gaussian_filter(zz, radius_m / CELL)
    r = ndi.gaussian_filter(r, 1.0)
    r[~inner] = np.nan
    return r


inner = ndi.binary_erosion(covered, iterations=int(15 / CELL))  # hide the ragged survey edge
ext = (0, NX * CELL, 0, NY * CELL)  # metres from the tile's south-west corner
fx, fy = CX - X0, CY - (Y0 - NY * CELL)  # feature in those metres
rel = relief(dtm)

# ---------------- Figure 1: whole tile ------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(21, 7.9), constrained_layout=True)


def coarse(z, k=4, how=np.nanmean):
    """Average (or max) over k x k blocks: 0.5 m cells -> 2 m cells for the whole-tile view."""
    ny, nx = (z.shape[0] // k) * k, (z.shape[1] // k) * k
    return how(z[:ny, :nx].reshape(ny // k, k, nx // k, k), axis=(1, 3))


panels = [
    (shade(coarse(dsm, how=np.nanmax), cell=4 * CELL), "gray", (0.0, 1.0), "1. What the plane saw: the tops of the trees"),
    (shade(coarse(dtm), 3.0, cell=4 * CELL), "gray", (0.3, 1.0), "2. After removing the trees: bare ground (heights x3)"),
    (rel, "RdBu_r", (-0.5, 0.5), "3. Bumps and dips: ground minus its 10 m surroundings"),
]
for a, (img, cmap, (lo, hi), title) in zip(ax, panels):
    im = a.imshow(img, cmap=cmap, vmin=lo, vmax=hi, extent=ext, interpolation="nearest")
    a.set_title(title, fontsize=15, loc="left")
    a.set_xlabel("metres east", fontsize=12)
    a.add_patch(Rectangle((fx - HALF, fy - HALF), 2 * HALF, 2 * HALF, fill=False, ec="#e8a000", lw=2.5))
ax[0].set_ylabel("metres north", fontsize=12)
cb = fig.colorbar(im, ax=ax[2], shrink=0.8, pad=0.01)
cb.set_label("metres above (red) or below (blue) the nearby ground", fontsize=11)
fig.suptitle(
    "Bonal, Acre, Brazil: one 1 km lidar tile (BONL6870C8908, flown 16 Sep 2013). Orange box = the square, zoomed in figure 2.",
    fontsize=14,
)
fig.savefig(os.path.join(out_dir, "fig1_whole_tile.png"), dpi=90)
plt.close(fig)

# ---------------- Figure 2: zoom + side view ------------------------------------------
c0, c1 = int((fx - HALF) / CELL), int((fx + HALF) / CELL)
r0, r1 = int((NY * CELL - (fy + HALF)) / CELL), int((NY * CELL - (fy - HALF)) / CELL)
zext = (-HALF, HALF, -HALF, HALF)

# Side view: raw returns within 2 m of an east-west line through the feature centre.
las = laspy.read(tile)
x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
band = (np.abs(y - CY) < 2.0) & (np.abs(x - CX) < HALF)
x, z = x[band] - CX, z[band]
col = np.clip(((x + CX - X0) / CELL).astype(int), 0, NX - 1)
row = np.clip(((Y0 - CY) / CELL).astype(int), 0, NY - 1)
ground_z = dtm[row, col]
is_ground = np.abs(z - ground_z) < 0.3

smooth = ndi.gaussian_filter(np.where(np.isnan(dtm), np.nanmedian(dtm), dtm), 10 / CELL)
ground_rel = z - smooth[row, col]

fig = plt.figure(figsize=(18, 17), constrained_layout=True)
gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 0.8, 0.55])
a0, a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, :])
a0.imshow(shade(dsm)[r0:r1, c0:c1], cmap="gray", vmin=0.0, vmax=1, extent=zext)
a0.set_title("A. Tops of the trees (what the plane saw)", fontsize=15, loc="left")
im = a1.imshow(np.clip(rel[r0:r1, c0:c1], -0.4, 0.4), cmap="RdBu_r", vmin=-0.4, vmax=0.4, extent=zext)
a1.set_title("B. Bare-ground bumps and dips, same area", fontsize=15, loc="left")
cb = fig.colorbar(im, ax=a1, shrink=0.85, pad=0.01)
cb.set_label("metres above (red) / below (blue) nearby ground", fontsize=11)
for a in (a0, a1):
    a.axhline(0, color="#e8a000", lw=1.8, ls="--")
    a.set_xlabel("metres from the square's centre (east)", fontsize=12)
a0.set_ylabel("metres (north)", fontsize=12)
a1.text(-HALF + 4, 4, "side view line", color="k", fontsize=12, va="bottom", bbox=dict(fc="white", ec="#e8a000", alpha=0.9))

a2.scatter(x[~is_ground], z[~is_ground], s=1.2, c="#2f8f3a", alpha=0.35, lw=0, label="tree / bush returns")
a2.scatter(x[is_ground], z[is_ground], s=4, c="#7a3e00", lw=0, label="ground returns (kept)")
order = np.argsort(x)
a2.plot(x[order], ground_z[order], color="k", lw=1.2, label="bare-ground surface (figure B)")
a2.set_xlim(-HALF, HALF)
gz = np.nanmedian(ground_z)
a2.set_ylim(gz - 3, gz + 38)
a2.set_xlabel("metres from the square's centre (east)", fontsize=12)
a2.set_ylabel("height above sea level (m)", fontsize=12)
a2.set_title(
    f"C. Side view along the dashed line: every laser return within 2 m of it ({band.sum():,} returns)",
    fontsize=15,
    loc="left",
)
a2.legend(loc="upper right", fontsize=12, markerscale=6)

# D: ground returns only, height measured from the smoothed local ground, stretched.
gx_, gr_ = x[is_ground], ground_rel[is_ground]
bins = np.arange(-HALF, HALF + 1, 1.0)
which = np.digitize(gx_, bins) - 1
med = np.array([np.median(gr_[which == i]) if np.sum(which == i) >= 6 else np.nan for i in range(len(bins) - 1)])
centres = bins[:-1] + 0.5
a3.axhline(0, color="0.6", lw=1)
a3.scatter(gx_, gr_, s=5, c="#7a3e00", lw=0, alpha=0.6, label="ground returns")
a3.plot(centres, med, color="k", lw=2, label="median per metre")
dips = med < -0.2
for cx_ in centres[dips]:
    a3.axvspan(cx_ - 0.5, cx_ + 0.5, color="#2b6cb0", alpha=0.25, lw=0)
a3.set_xlim(-HALF, HALF)
a3.set_ylim(-1.2, 1.0)
a3.set_xlabel("metres from the square's centre (east)", fontsize=12)
a3.set_ylabel("m above/below\nnearby ground", fontsize=12)
fig.canvas.draw()
bb = a3.get_window_extent()
stretch = (2 * HALF / bb.width) / (2.2 / bb.height)
a3.set_title(f"D. Same line, ground returns only, height stretched {stretch:.0f}x: blue = over 20 cm below the surrounding ground", fontsize=15, loc="left")
a3.legend(loc="upper right", fontsize=12, markerscale=3)
fig.savefig(os.path.join(out_dir, "fig2_square_zoom.png"), dpi=90)
plt.close(fig)
print("wrote figures")
