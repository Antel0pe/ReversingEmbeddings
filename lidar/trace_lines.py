"""Follow a groove in the bare ground, step by step, and measure it.

Usage: python -I lidar/trace_lines.py SURFACES.npz STARTS.json OUT.json [LOOK_M HALF_ALONG]

STARTS.json: {"name": [x, y, heading_x, heading_y, (max_steps)], ...} in UTM metres.

How it follows: from the current point, step 3 m along the current heading,
then look sideways up to 3.5 m either way (averaged over 5 m along the line) for the lowest point of the
bumps-and-dips map. Move there, and turn the heading a little toward the move.
Stop when the groove has faded (less than 6 cm below its sides for 6 steps
in a row), when it runs off the data, or when it falls into a stream valley
(more than 1.5 m below its sides).

What it measures at each step:
  depth  = how far the groove's floor is below the ground 6-9 m to either side
  width  = width of the part that is at least half that depth
Straightness = straight-line distance between the ends / distance walked.
"""
import json
import sys

import numpy as np
from scipy import ndimage as ndi

surf_path, starts_path, out_path = sys.argv[1:4]
# optional: LOOK_M HALF_ALONG_STEPS (defaults 3.5 and 2 = average 5 slices 1 m apart)
LOOK_ARG = float(sys.argv[4]) if len(sys.argv) > 4 else 3.5
HALF_ALONG = int(sys.argv[5]) if len(sys.argv) > 5 else 2
s = np.load(surf_path)
dtm = s["dtm"].astype(float)
X0, Y0 = s["origin"]
CELL = float(s["cell"])
NY, NX = dtm.shape

filled = np.where(np.isnan(dtm), np.nanmedian(dtm), dtm)
relief = filled - ndi.gaussian_filter(filled, 10 / CELL)
relief = ndi.gaussian_filter(relief, 0.75 / CELL)
valid = ndi.binary_erosion(np.isfinite(dtm), iterations=int(12 / CELL))

STEP, LOOK, SIDE = 3.0, LOOK_ARG, (6.0, 9.0)


def sample(px, py):
    """Relief at UTM points (bilinear)."""
    col = (np.asarray(px) - X0) / CELL - 0.5
    row = (Y0 - np.asarray(py)) / CELL - 0.5
    return ndi.map_coordinates(relief, [row, col], order=1, mode="nearest")


def inside(px, py):
    col, row = int((px - X0) / CELL), int((Y0 - py) / CELL)
    return 0 <= col < NX and 0 <= row < NY and valid[row, col]


def cross_section(p, h):
    n = np.array([-h[1], h[0]])
    offs = np.arange(-SIDE[1], SIDE[1] + 0.01, 0.25)
    along = np.array([-1.0, 0.0, 1.0])  # average 3 slices 1 m apart to beat noise
    pts = p[None, None, :] + offs[None, :, None] * n + along[:, None, None] * h
    prof = sample(pts[..., 0], pts[..., 1]).mean(axis=0)
    return offs, prof


def trace(x, y, hx, hy, max_steps=600):
    p, h = np.array([x, y], float), np.array([hx, hy], float)
    h /= np.linalg.norm(h)
    path, depth, width, profiles = [], [], [], []
    faded = 0
    reason = "step limit"
    for _ in range(max_steps):
        q = p + STEP * h
        n = np.array([-h[1], h[0]])
        offs = np.arange(-LOOK, LOOK + 0.01, 0.25)
        cand = q[None, :] + offs[:, None] * n
        prof = np.mean([sample(cand[:, 0] + d * h[0], cand[:, 1] + d * h[1]) for d in range(-HALF_ALONG, HALF_ALONG + 1)], axis=0)
        new = cand[np.argmin(prof)]
        if not inside(*new):
            reason = "ran off the data"
            break
        move = (new - p) / np.linalg.norm(new - p)
        h = 0.6 * h + 0.4 * move
        h /= np.linalg.norm(h)
        p = new
        o, cs = cross_section(p, h)
        sides = np.r_[cs[(o <= -SIDE[0])], cs[(o >= SIDE[0])]]
        centre = cs[np.abs(o) <= 0.5].min()
        d = float(np.median(sides) - centre)
        if d > 1.5:
            reason = "fell into a stream valley"
            break
        if len(path) > 8 and np.min(np.linalg.norm(np.array(path[:-8]) - p, axis=1)) < 4:
            # cut off the loop: keep the path only up to where the loop began
            j = int(np.argmax(np.linalg.norm(np.array(path[:-8]) - p, axis=1) < 4))
            path, depth, width, profiles = path[: j + 1], depth[: j + 1], width[: j + 1], profiles[: j + 1]
            reason = "came back to its own path (loop removed)"
            break
        below = cs <= np.median(sides) - d / 2
        # width of the contiguous below-half-depth run through the centre
        mid = np.argmin(np.abs(o))
        lo = hi = mid
        while lo > 0 and below[lo - 1]:
            lo -= 1
        while hi < len(o) - 1 and below[hi + 1]:
            hi += 1
        path.append(p.tolist())
        profiles.append((cs - np.median(sides)).tolist())
        depth.append(d)
        width.append(float(o[hi] - o[lo]))
        faded = faded + 1 if d < 0.06 else 0
        if faded >= 6:
            path, depth, width, profiles = path[:-6], depth[:-6], width[:-6], profiles[:-6]
            reason = "groove faded"
            break
    path = np.array(path)
    walked = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if len(path) > 1 else 0.0
    straight = float(np.linalg.norm(path[-1] - path[0])) if len(path) > 1 else 0.0
    # local straightness: straight distance / walked distance over every 17-step (~50 m) stretch
    k = 17
    local = [
        np.linalg.norm(path[i + k] - path[i]) / np.sum(np.linalg.norm(np.diff(path[i : i + k + 1], axis=0), axis=1))
        for i in range(0, len(path) - k)
    ]
    return {
        "path": path.tolist(),
        "depth_m": depth,
        "width_m": width,
        "length_m": walked,
        "straightness": straight / walked if walked else None,
        "straightness_50m": float(np.median(local)) if local else None,
        "end": path[-1].tolist() if len(path) else None,
        "stopped_because": reason,
        "median_depth_m": float(np.median(depth)) if depth else None,
        "median_width_m": float(np.median(width)) if width else None,
        "profile_offsets_m": np.arange(-SIDE[1], SIDE[1] + 0.01, 0.25).tolist(),
        "median_profile_m": np.median(np.array(profiles), axis=0).tolist() if profiles else None,
    }


starts = json.load(open(starts_path))
out = {}
for name, (x, y, hx, hy, *rest) in starts.items():
    t = trace(x, y, hx, hy, *(int(v) for v in rest))
    out[name] = t
    print(
        f"{name:>14}: {t['length_m']:6.0f} m, straight {t['straightness'] or 0:.2f} (per 50 m {t['straightness_50m'] or 0:.3f}), "
        f"depth {t['median_depth_m'] or 0:.2f} m, width {t['median_width_m'] or 0:.1f} m; stopped: {t['stopped_because']}"
    )
json.dump(out, open(out_path, "w"))
