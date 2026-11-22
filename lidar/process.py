"""Turn raw lidar tiles into a canopy surface and a bare-ground surface.

Usage: python -I lidar/process.py OUT_DIR TILE.laz [TILE.laz ...] [--cell 0.5] [--vendor VENDOR_DTM.tif]

Steps
1. Grid: every laser return lands in a square cell. Keep the lowest and the
   highest return per cell (lowest = best guess at ground, highest = treetop).
2. Noise: drop a cell's highest value if it sits more than 5 m above all 8
   neighbours (a lone spike: a bird, a speck in the air, a single branch tip).
   Strays far below the ground are only counted, not dropped, unless
   --drop-low-strays is given: this survey arrived already cleaned by the vendor
   (3 of 25 million returns are over 2 m below its ground), and on it the
   low-stray rule only ever caught real ground in narrow stream channels.
3. Trees: a progressive rolling window (morphological opening). Windows of
   growing size are pushed up from below the lowest-return surface; anything that
   pokes up more steeply than real ground can slope is a tree, bush or log, not
   ground. What survives is joined into a rough surface, and every return within
   30 cm of it is called ground.
4. Bare-earth grid: average the ground returns per cell; fill empty cells by
   linear interpolation between the nearest ground cells.

The vendor's own classification and DTM are ignored except as a check at the end.
"""
import argparse
import json
import os

import laspy
import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import griddata

ap = argparse.ArgumentParser()
ap.add_argument("out_dir")
ap.add_argument("tiles", nargs="+")
ap.add_argument("--cell", type=float, default=0.5)
ap.add_argument("--drop-low-strays", action="store_true", help="drop lowest returns far below everything within 2.5 m")
ap.add_argument("--vendor", default=None, help="vendor DTM GeoTIFF (north-up, for comparison only)")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
CELL = args.cell

LOW_NOISE_M = 2.0  # lowest return this far below every other lowest return within LOW_NOISE_RADIUS_M = noise
LOW_NOISE_RADIUS_M = 2.5  # wide enough that real ground usually has another ground hit nearby
HIGH_NOISE_M = 5.0  # highest return this far above the highest of its 8 neighbours = noise
MAX_SLOPE = 0.35  # rise/run that real ground may have (valley walls here are steep)
WINDOWS_M = [1, 2, 3, 4, 6, 8, 12, 16, 24]  # half-widths; 24 m is wider than any crown
GROUND_BAND_M = 0.30  # returns within this height of the rough ground are ground

# Grid covering all tiles, snapped to whole metres.
mins = np.min([laspy.open(t).header.mins for t in args.tiles], axis=0)
maxs = np.max([laspy.open(t).header.maxs for t in args.tiles], axis=0)
X0, Y0 = np.floor(mins[0]), np.ceil(maxs[1])
NX = int(np.ceil((maxs[0] - X0) / CELL))
NY = int(np.ceil((Y0 - mins[1]) / CELL))
print(f"{len(args.tiles)} tiles, grid {NX} x {NY} cells of {CELL} m")


def read(t):
    las = laspy.read(t)
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    col = np.minimum(((x - X0) / CELL).astype(np.int64), NX - 1)
    row = np.minimum(((Y0 - y) / CELL).astype(np.int64), NY - 1)
    return row * NX + col, z, np.asarray(las.classification)


def fill_nearest(a):
    idx = ndi.distance_transform_edt(np.isnan(a), return_distances=False, return_indices=True)
    return a[tuple(idx)]


def interpolate_gaps(a, region):
    rr, cc = np.nonzero(~np.isnan(a))
    tr, tc = np.nonzero(np.isnan(a) & region)
    out = a.copy()
    out[tr, tc] = griddata((rr, cc), a[rr, cc], (tr, tc), method="linear")
    return out


ring = np.ones((3, 3), bool)  # the 8 touching cells
ring[1, 1] = False
kr = int(round(LOW_NOISE_RADIUS_M / CELL))
yy, xx = np.mgrid[-kr : kr + 1, -kr : kr + 1]
around = (xx**2 + yy**2) <= kr * kr  # every cell within the radius ...
around[kr, kr] = False  # ... except the cell itself

# ---- Pass 1: lowest and highest return per cell ---------------------------------
N = NX * NY
zmin, zmax = np.full(N, np.inf), np.full(N, -np.inf)
count, vendor_ground = np.zeros(N, np.int64), np.zeros(N, np.int64)
for t in args.tiles:
    idx, z, cls = read(t)
    np.minimum.at(zmin, idx, z)
    np.maximum.at(zmax, idx, z)
    count += np.bincount(idx, minlength=N)
    vendor_ground += np.bincount(idx[cls == 2], minlength=N)
    print(f"  {os.path.basename(t)}: {len(z):,} returns")
n_points = int(count.sum())

has = (count > 0).reshape(NY, NX)
covered = ndi.binary_fill_holes(ndi.binary_closing(has, iterations=4)) & ndi.binary_dilation(has, iterations=4)
zmin = np.where(count > 0, zmin, np.nan).reshape(NY, NX)
zmax = np.where(count > 0, zmax, np.nan).reshape(NY, NX)

# ---- Noise ------------------------------------------------------------------------
lowest_neighbour = ndi.grey_erosion(np.where(np.isnan(zmin), np.inf, zmin), footprint=around)
low_noise = zmin < lowest_neighbour - LOW_NOISE_M
highest_neighbour = ndi.grey_dilation(np.where(np.isnan(zmax), -np.inf, zmax), footprint=ring)
high_noise = zmax > highest_neighbour + HIGH_NOISE_M
if args.drop_low_strays:
    zmin[low_noise] = np.nan
zmax[high_noise] = np.nan
print(f"noise: {low_noise.sum():,} low-stray candidates ({'dropped' if args.drop_low_strays else 'kept'}), "
      f"{high_noise.sum():,} high spikes dropped")

# ---- Trees: progressive rolling window ---------------------------------------------
surface = fill_nearest(zmin)
not_ground = np.isnan(zmin)
for w in WINDOWS_M:
    k = 2 * int(round(w / CELL)) + 1
    opened = ndi.grey_opening(surface, size=(k, k))
    not_ground |= (surface - opened) > MAX_SLOPE * w + 0.15
    surface = opened
seed = np.where(not_ground, np.nan, zmin)
print(f"rolling window kept {np.isfinite(seed)[covered].mean():.1%} of cells as ground seeds")
rough = fill_nearest(interpolate_gaps(seed, covered)).ravel()

# ---- Pass 2: ground returns ----------------------------------------------------------
gsum, gcnt = np.zeros(N), np.zeros(N, np.int64)
for t in args.tiles:
    idx, z, _ = read(t)
    g = np.abs(z - rough[idx]) < GROUND_BAND_M
    gsum += np.bincount(idx[g], weights=z[g], minlength=N)
    gcnt += np.bincount(idx[g], minlength=N)
n_ground = int(gcnt.sum())

dtm_raw = np.where(gcnt > 0, gsum / np.maximum(gcnt, 1), np.nan).reshape(NY, NX)
dtm = interpolate_gaps(dtm_raw, covered)
dtm[~covered] = np.nan
dsm = fill_nearest(zmax)
dsm[~covered] = np.nan

np.savez_compressed(
    os.path.join(args.out_dir, "surfaces.npz"),
    dtm=dtm.astype(np.float32),
    dsm=dsm.astype(np.float32),
    lowest=zmin.astype(np.float32),
    ground_hits=gcnt.reshape(NY, NX).astype(np.int32),
    returns=count.reshape(NY, NX).astype(np.int32),
    covered=covered,
    origin=np.array([X0, Y0]),
    cell=CELL,
)

stats = {
    "returns": n_points,
    "returns_per_m2": n_points / (covered.sum() * CELL * CELL),
    "ground_returns": n_ground,
    "ground_share": n_ground / n_points,
    "vendor_ground_returns": int(vendor_ground.sum()),
    "area_ha": float(covered.sum() * CELL * CELL / 1e4),
    "cells_with_ground_hit": float((gcnt.reshape(NY, NX) > 0)[covered].mean()),
    "low_stray_candidates": int(low_noise.sum()),
    "low_strays_dropped": bool(args.drop_low_strays),
    "high_noise_cells": int(high_noise.sum()),
    "canopy_height_median_m": float(np.nanmedian((dsm - dtm)[covered])),
}
if args.vendor:
    import subprocess
    import tifffile

    info = subprocess.run(["gdalinfo", args.vendor], capture_output=True, text=True).stdout
    ox, oy = [float(v) for v in info.split("Origin = (")[1].split(")")[0].split(",")]
    v = tifffile.imread(args.vendor).astype(float)
    v[v < -1e30] = np.nan
    rr, cc = np.mgrid[0:NY, 0:NX]
    vx = ((X0 + (cc + 0.5) * CELL) - ox).astype(int)
    vy = (oy - (Y0 - (rr + 0.5) * CELL)).astype(int)
    ok = (vx >= 0) & (vx < v.shape[1]) & (vy >= 0) & (vy < v.shape[0])
    vs = np.full(dtm.shape, np.nan)
    vs[ok] = v[vy[ok], vx[ok]]
    both = np.isfinite(vs) & np.isfinite(dtm)
    d = dtm[both] - vs[both]
    stats.update(
        vendor_median_diff_m=float(np.median(d)),
        vendor_median_abs_diff_m=float(np.median(np.abs(d))),
        vendor_within_30cm=float((np.abs(d) < 0.3).mean()),
        vendor_within_1m=float((np.abs(d) < 1.0).mean()),
        above_vendor_by_1m=float((d > 1.0).mean()),
        below_vendor_by_1m=float((d < -1.0).mean()),
    )
    np.save(os.path.join(args.out_dir, "vendor_on_grid.npy"), vs.astype(np.float32))
print(json.dumps(stats, indent=1))
json.dump(stats, open(os.path.join(args.out_dir, "stats.json"), "w"), indent=1)
