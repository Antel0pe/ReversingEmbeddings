"""A one-minute video of how the maps were made, drawn from the real Bonal data.

Usage: python -I lidar/make_video.py OUT_DIR TILE.laz OUT.mp4
  OUT_DIR holds process.py's surfaces.npz.

Most scenes are a side view of a 1 m wide slice of laser returns along an
east-west line through the square. The tree-removal steps replay the real
algorithm from process.py on the area around that line (same settings), and
show its decisions for the cells the line crosses.
"""
import os
import sys

import laspy
import matplotlib
import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import griddata

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle

out_dir, tile, out_mp4 = sys.argv[1], sys.argv[2], sys.argv[3]
CX, CY, HALF = 687643.0, 8908140.0, 110.0  # centre of the square
LINE_Y = CY  # the side-view line: east-west through the square's centre, as in figure 2
FPS = 24

# Same settings as process.py
MAX_SLOPE, WINDOWS_M, BAND = 0.35, [1, 2, 3, 4, 6, 8, 12, 16, 24], 0.30

s = np.load(os.path.join(out_dir, "surfaces.npz"))
X0, Y0 = s["origin"]
CELL = float(s["cell"])
lowest_all, dtm_all, dsm_all, returns_all = s["lowest"], s["dtm"], s["dsm"], s["returns"]

# ---- Replay the tree filter on a 400 m block around the line -------------------------
M = 200.0  # half-size of the block, metres (wide margin so edges don't matter)
r0, r1 = int((Y0 - (CY + M)) / CELL), int((Y0 - (CY - M)) / CELL)
c0, c1 = int((CX - M - X0) / CELL), int((CX + M - X0) / CELL)
low = lowest_all[r0:r1, c0:c1].astype(float)


def fill_nearest(a):
    idx = ndi.distance_transform_edt(np.isnan(a), return_distances=False, return_indices=True)
    return a[tuple(idx)]


line_row = int((Y0 - LINE_Y) / CELL) - r0  # row of the block the line runs along
surface = fill_nearest(low)
not_ground = np.isnan(low)
steps = []  # per window: (width m, block-top profile along the line, flagged-so-far along the line)
for w in WINDOWS_M:
    k = 2 * int(round(w / CELL)) + 1
    opened = ndi.grey_opening(surface, size=(k, k))
    not_ground |= (surface - opened) > MAX_SLOPE * w + 0.15
    surface = opened
    steps.append((w, opened[line_row].copy(), not_ground[line_row].copy()))
seed = np.where(not_ground, np.nan, low)
rr, cc = np.nonzero(~np.isnan(seed))
rough = griddata((rr, cc), seed[rr, cc], (np.full(low.shape[1], line_row), np.arange(low.shape[1])), method="linear")
rough = fill_nearest(rough[None, :])[0]  # rough ground along the line, one value per 0.5 m cell

# ---- Points: a 1 m wide slice along the line ----------------------------------------
las = laspy.read(tile)
x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
keep = (np.abs(y - LINE_Y) < 0.5) & (np.abs(x - CX) < HALF)
x, y, z = x[keep], y[keep], z[keep]
col = ((x - X0) / CELL).astype(int) - c0  # cell column inside the block
row = ((Y0 - y) / CELL).astype(int) - r0
xr = x - CX  # metres from the square's centre
print(f"{len(x):,} returns in the slice")

# Lowest return per 0.5 m column of the slice.
cols = np.unique(col)
low_idx = np.array([np.flatnonzero(col == c)[np.argmin(z[col == c])] for c in cols])
clean_idx = low_idx  # this survey arrived already cleaned, so no low strays are removed

is_ground = np.abs(z - rough[col]) < BAND

# Where the line crosses the square's ditch: the lowest point of the final ground
# (relative to its 10 m surroundings) near each side, and the highest point just outside it.
d_line = fill_nearest(dtm_all[r0:r1, c0:c1].astype(float))
rel_line = (d_line - ndi.gaussian_filter(d_line, 10 / CELL))[line_row]
xl = (np.arange(c1 - c0) + c0 + 0.5) * CELL + X0 - CX  # metres from the square's centre


def extreme(lo, hi, fn):
    m = (xl >= lo) & (xl <= hi)
    return float(xl[m][fn(rel_line[m])])


DITCHES = [(extreme(-40, -18, np.argmin), "ditch"), (extreme(32, 50, np.argmin), "ditch")]
BANKS = [extreme(DITCHES[1][0] + 1, DITCHES[1][0] + 10, np.argmax)]
print("ditches at", [round(d[0], 1) for d in DITCHES], "bank at", [round(b, 1) for b in BANKS])
ground_line_x = (np.arange(low.shape[1]) + c0 + 0.5) * CELL + X0 - CX
in_view = np.abs(ground_line_x) <= HALF
gl_x, gl_z = ground_line_x[in_view], rough[in_view]  # step 5: the rough ground line
fin_z = d_line[line_row][in_view]  # steps 6-7: the final ground map (average of ground dots, gaps filled)

# Top views for the last scene (220 m square around the centre)
tr0, tr1 = int((Y0 - (CY + HALF)) / CELL), int((Y0 - (CY - HALF)) / CELL)
tc0, tc1 = int((CX - HALF - X0) / CELL), int((CX + HALF - X0) / CELL)


def shade(zz):
    gy, gx = np.gradient(zz, CELL)
    slope, aspect = np.arctan(np.hypot(gx, gy)), np.arctan2(gy, -gx)
    alt, az = np.radians(35), np.radians(360 - 315 + 90)
    return np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)


canopy_img = shade(fill_nearest(dsm_all[tr0 - 40 : tr1 + 40, tc0 - 40 : tc1 + 40].astype(float)))[40:-40, 40:-40]
d = fill_nearest(dtm_all[tr0 - 80 : tr1 + 80, tc0 - 80 : tc1 + 80].astype(float))
relief_img = ndi.gaussian_filter(d - ndi.gaussian_filter(d, 10 / CELL), 1.0)[80:-80, 80:-80]

# ---- Timeline ----------------------------------------------------------------------
SCENES = [  # (name, seconds)
    ("title", 4),
    ("plane", 7),
    ("lowest", 6),
    ("noise", 6),
    ("blocks_intro", 4),
    ("blocks", 18),
    ("band", 6),
    ("delete", 4),
    ("stretch", 8),
    ("topview", 11),
]
starts = np.cumsum([0] + [d for _, d in SCENES])
TOTAL = starts[-1]

GREEN, BROWN, ORANGE, RED, GREY = "#3f9b4a", "#7a3e00", "#e8890c", "#d62728", "#b8b8b8"
CAPTION = {
    "plane": "1. A plane fires laser pulses at the forest. Each dot is one reflection.",
    "lowest": "2. Cut the ground into 0.5 m squares. Keep the lowest dot in each one.",
    "noise": "3. Check for strays: a dot far under the ground would be an error.",
    "blocks_intro": "4. Many lowest dots are still leaves: the laser never reached the ground there.",
    "blocks": "4. Push a flat block up from underneath. Dots too steep for real ground are trees.",
    "band": "5. Join the surviving dots into a ground line. Dots within 30 cm of it are ground.",
    "delete": "6. Delete everything else: the trees are gone.",
    "topview": "8. Do this everywhere and look from above.",
}

fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
ax = fig.add_axes([0.07, 0.11, 0.9, 0.72])


def ease(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def side_axes(ylim, xlabel=True):
    ax.set_xlim(-HALF, HALF)
    ax.set_ylim(*ylim)
    ax.set_ylabel("height above sea level (m)", fontsize=13)
    if xlabel:
        ax.set_xlabel("metres along the line (east →)", fontsize=13)
    ax.tick_params(labelsize=11)


FULL = (144.5, 187.0)
gz = np.median(z[is_ground])
STRETCH = (gz - 1.6, gz + 2.0)
FLAT = (-1.0, 1.0)  # after flattening: metres above/below the nearby ground
trend_z = ndi.gaussian_filter(d_line, 10 / CELL)[line_row][in_view]  # the gentle lie of the land (10 m smoothing)
_bb = ax.get_position()
STRETCH_X = (2 * HALF / (_bb.width * 1280)) / ((FLAT[1] - FLAT[0]) / (_bb.height * 720))
CAPTION["stretch"] = f"7. Stretch the height {STRETCH_X:.0f} times and flatten the slope: a ditch and bank appear."


def caption(text, sub=None):
    fig.text(0.5, 0.93, text, ha="center", va="center", fontsize=19, weight="bold")
    if sub:
        fig.text(0.5, 0.875, sub, ha="center", va="center", fontsize=13, color="0.3")


def draw(t):
    for extra in fig.axes[1:]:
        extra.remove()
    fig.texts.clear()
    ax.cla()
    ax.set_visible(True)
    i = np.searchsorted(starts, t, side="right") - 1
    name = SCENES[i][0]
    u = (t - starts[i]) / SCENES[i][1]  # 0..1 progress within the scene

    if name == "title":
        ax.set_visible(False)
        fig.text(0.5, 0.6, "Seeing under the forest", ha="center", fontsize=40, weight="bold")
        fig.text(0.5, 0.48, "How the lidar maps were made, step by step", ha="center", fontsize=22)
        fig.text(0.5, 0.36, "Real data: Bonal, Acre, Brazil. Laser survey flown 16 September 2013.", ha="center", fontsize=15, color="0.35")
        return

    if name == "topview":
        ax.set_visible(False)
        caption(CAPTION[name], "Left: the tops of the trees. Right: bare-ground bumps (red) and dips (blue).")
        a1 = fig.add_axes([0.06, 0.06, 0.42, 0.75])
        a2 = fig.add_axes([0.53, 0.06, 0.42, 0.75])
        ext = (-HALF, HALF, -HALF, HALF)
        a1.imshow(canopy_img, cmap="gray", vmin=0, vmax=1, extent=ext)
        a2.imshow(canopy_img, cmap="gray", vmin=0, vmax=1, extent=ext)
        a2.imshow(np.clip(relief_img, -0.4, 0.4), cmap="RdBu_r", vmin=-0.4, vmax=0.4, extent=ext, alpha=ease((u - 0.1) * 3))
        for a in (a1, a2):
            a.axhline(LINE_Y - CY, color=ORANGE, lw=2, ls="--")
            a.set_xticks([])
            a.set_yticks([])
        a1.text(-HALF + 5, LINE_Y - CY + 4, "the line we followed", fontsize=12, bbox=dict(fc="white", ec=ORANGE))
        if u > 0.45:
            a2.text(0, -HALF + 8, "a square ditch, about 70 × 80 m,\nhidden under 20–35 m of forest",
                    ha="center", va="bottom", fontsize=15, weight="bold", alpha=ease((u - 0.45) * 4),
                    bbox=dict(fc="white", ec="0.3", alpha=0.9))
        return

    caption(CAPTION[name])

    if name == "plane":
        sweep = -HALF + 2 * HALF * ease(u / 0.85)
        shown = xr <= sweep
        side_axes(FULL)
        ax.scatter(xr[shown], z[shown], s=2.5, c="0.25", lw=0)
        ax.plot([sweep], [FULL[1] - 1.5], marker=">", ms=18, color="k", clip_on=False)
        for dx in (-6, -2, 2, 6):
            ax.plot([sweep, sweep + dx], [FULL[1] - 2, FULL[0] + 2], color=RED, lw=0.6, alpha=0.5)
        ax.text(HALF - 2, FULL[0] + 1.5, f"{shown.sum():,} dots", ha="right", fontsize=13)
        return

    if name == "lowest":
        side_axes(FULL)
        fade = ease(u * 2)
        ax.scatter(xr, z, s=2.5, c=[(0.25 + 0.5 * fade,) * 3], lw=0)
        upto = -HALF + 2 * HALF * ease(u / 0.6)
        sel = low_idx[xr[low_idx] <= upto]
        ax.scatter(xr[sel], z[sel], s=14, c=ORANGE, lw=0, label="lowest dot in each 0.5 m square")
        ax.legend(loc="upper right", fontsize=13, markerscale=1.5)
        return

    if name == "noise":
        side_axes(FULL)
        ax.scatter(xr, z, s=2.5, c=GREY, lw=0)
        ax.scatter(xr[low_idx], z[low_idx], s=14, c=ORANGE, lw=0)
        ax.axhspan(FULL[0], np.min(z[low_idx]) - 0.2, color=RED, alpha=0.12 * ease(u * 3), lw=0)
        if u > 0.25:
            ax.text(-HALF + 2, FULL[1] - 3,
                    "This survey arrived already cleaned by the company:\n"
                    "only 3 of 25 million dots are more than 2 m underground.\n"
                    "So nothing is removed here (only 420 lone spikes above the treetops).",
                    fontsize=14, va="top", color=RED, alpha=ease((u - 0.25) * 4), bbox=dict(fc="white", ec=RED))
        return

    pts = clean_idx  # lowest dots, strays already replaced
    if name == "blocks_intro":
        side_axes(FULL)
        ax.scatter(xr, z, s=2.5, c=GREY, lw=0)
        ax.scatter(xr[pts], z[pts], s=14, c=ORANGE, lw=0)
        high = pts[z[pts] > gz + 3]
        if u > 0.3:
            ax.scatter(xr[high], z[high], s=120, facecolors="none", edgecolors=GREEN, lw=1.5, alpha=ease((u - 0.3) * 3))
            ax.text(HALF - 2, FULL[1] - 3, f"{len(high)} of {len(pts)} lowest dots are up in the trees",
                    ha="right", va="top", fontsize=14, color=GREEN, bbox=dict(fc="white", ec=GREEN))
        return

    if name == "blocks":
        side_axes(FULL)
        n = len(steps)
        k = min(int(u * n), n - 1)
        v = u * n - k  # progress within this block size
        w, top, flagged_now = steps[k]
        flagged_before = steps[k - 1][2] if k else np.zeros_like(flagged_now)
        ax.scatter(xr, z, s=2.5, c=GREY, lw=0)
        lc = col[pts]
        done = flagged_before[lc] | (flagged_now[lc] & (xr[pts] <= -HALF + 2 * HALF * v))
        ax.scatter(xr[pts][~done], z[pts][~done], s=14, c=ORANGE, lw=0, zorder=3)
        ax.scatter(xr[pts][done], z[pts][done], s=14, c=GREEN, lw=0, zorder=3, alpha=0.6)
        # the block slides left to right; behind it, the highest a block of this width could reach
        bx = -HALF + 2 * HALF * v
        prof_x = ground_line_x
        reached = (prof_x <= bx) & (np.abs(prof_x) <= HALF)
        ax.fill_between(prof_x[reached], FULL[0], top[reached], color="#c9b79c", alpha=0.6, lw=0)
        ci = int(np.clip(np.searchsorted(prof_x, bx), 0, len(top) - 1))
        ax.add_patch(Rectangle((bx - w, FULL[0]), 2 * w, top[ci] - FULL[0], fc="#8c6d46", ec="k", lw=1.2, zorder=4))
        ax.text(-HALF + 2, FULL[1] - 3, f"block {2 * w + CELL:g} m wide", fontsize=15, va="top", weight="bold",
                bbox=dict(fc="white", ec="0.3"))
        ax.text(HALF - 2, FULL[1] - 3, f"marked as trees so far: {int(done.sum())} of {len(pts)}", ha="right",
                va="top", fontsize=14, color=GREEN, bbox=dict(fc="white", ec=GREEN))
        return

    if name in ("band", "delete"):
        side_axes(FULL)
        tree_alpha = 0.55 if name == "band" else 0.55 * (1 - ease(u * 1.5))
        if name == "band":
            upto = -HALF + 2 * HALF * ease(u / 0.6)
            g = is_ground & (xr <= upto)
            ax.scatter(xr[~g], z[~g], s=2.5, c=GREEN if u > 0.6 else GREY, lw=0, alpha=tree_alpha if u > 0.6 else 1)
            sel = gl_x <= upto
            ax.plot(gl_x[sel], gl_z[sel], color="k", lw=1.0, zorder=2)
            ax.scatter(xr[g], z[g], s=16, c=BROWN, lw=0, zorder=3)
        else:
            ax.scatter(xr[~is_ground], z[~is_ground], s=2.5, c=GREEN, lw=0, alpha=tree_alpha)
            ax.plot(gl_x, fin_z, color="k", lw=1.0, zorder=2)
            ax.scatter(xr[is_ground], z[is_ground], s=16, c=BROWN, lw=0, zorder=3)
            ax.text(-HALF + 2, FULL[1] - 3, "black line: the ground map, averaged from the brown dots", va="top",
                    fontsize=13, bbox=dict(fc="white", ec="0.3"))
        ax.text(HALF - 2, FULL[1] - 3, f"ground dots: {is_ground.sum():,} of {len(z):,}", ha="right", va="top",
                fontsize=14, color=BROWN, bbox=dict(fc="white", ec=BROWN))
        return

    if name == "stretch":
        # 0-40%: zoom in on the ground. 40-70%: flatten the gentle slope. Then label.
        e1, e2 = ease(u / 0.4), ease((u - 0.4) / 0.3)
        lo = FULL[0] + (STRETCH[0] - FULL[0]) * e1
        hi = FULL[1] + (STRETCH[1] - FULL[1]) * e1
        lo, hi = lo + (FLAT[0] - lo) * e2, hi + (FLAT[1] - hi) * e2
        side_axes((lo, hi))
        if e2 > 0.5:
            ax.set_ylabel("above (+) / below (\u2212) nearby ground (m)", fontsize=12)
        gzd = z[is_ground] - e2 * np.interp(xr[is_ground], gl_x, trend_z)
        ax.scatter(xr[is_ground], gzd, s=16, c=BROWN, lw=0, zorder=3)
        ax.plot(gl_x, fin_z - e2 * trend_z, color="k", lw=1.2, zorder=2)
        if u > 0.72:
            a = ease((u - 0.72) * 5)
            for xd, label in DITCHES:
                zd = np.interp(xd, gl_x, fin_z - trend_z)
                ax.annotate(label, (xd, zd - 0.03), (xd, -0.85), ha="center", fontsize=16, color="#1f5fa8",
                            weight="bold", alpha=a, arrowprops=dict(arrowstyle="->", color="#1f5fa8", lw=2, alpha=a))
            for xb in BANKS:
                zb = np.interp(xb, gl_x, fin_z - trend_z)
                ax.annotate("bank", (xb, zb + 0.05), (xb + 12, 0.8), ha="center", fontsize=16, color=RED, weight="bold",
                            alpha=a, arrowprops=dict(arrowstyle="->", color=RED, lw=2, alpha=a))
        return


if len(sys.argv) > 4:  # test mode: render the given times (seconds) as PNGs
    for t in map(float, sys.argv[4:]):
        draw(t)
        fig.savefig(out_mp4.replace(".mp4", f"_t{t:05.1f}.png"))
    sys.exit()

writer = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20"])
n_frames = int(TOTAL * FPS)
with writer.saving(fig, out_mp4, dpi=100):
    for f in range(n_frames):
        draw(f / FPS)
        writer.grab_frame()
        if f % 120 == 0:
            print(f"frame {f}/{n_frames}", flush=True)
print("wrote", out_mp4, f"({TOTAL} s)")
