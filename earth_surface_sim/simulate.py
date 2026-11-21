"""A young mountain island rising from the sea and being carved by rivers.

A simple landscape-evolution model, the kind geomorphologists use, on a
240 km x 120 km patch of the Earth over 9 million years. Three things happen
in every 10,000-year step:

1. Uplift. Plates squeezing together push rock upward (a few mm per year).
2. Rivers. Rain runs downhill, collects into rivers, and every river cuts
   into its bed. Bigger rivers (more upstream area) on steeper slopes cut
   faster:  erosion rate = K * (upstream area)^0.5 * slope.
3. Hillslopes. Soil creeps slowly downhill, rounding off sharp crests.
4. Rebound. The crust floats on the denser mantle like ice on water. When
   rivers remove rock, the region bobs back up by about 80% of what was
   removed, which is why real ranges take tens of millions of years to wear
   away rather than one million.

The sea is the floor the rivers cut down toward: nothing erodes below sea level.

The story is loosely modelled on Taiwan, a real mountain island that rose from
the sea in roughly the last 5-6 million years as the Philippine Sea plate
rammed into Asia, and where younger folds keep rising in the foothills. Here:

  0.0 - 5.5 Myr  main range rises out of a shallow sea
  3.0 - 5.5 Myr  a young foothill ridge rises across the paths of the rivers
                 flowing south
  5.5 - 9.0 Myr  the squeeze stops; rivers and hillslopes wear the range down

Nature is never perfectly smooth, and a perfectly smooth model makes rivers
run in ruler-straight lines along the grid. So the sea floor starts bumpy, the
uplift varies a little from place to place, and rock hardness varies (soft
rock erodes up to ~2x faster than hard rock).

Run:  .venv/bin/python earth_surface_sim/simulate.py
Writes earth_surface_sim/run.npz (snapshots every 50,000 years).
"""

from __future__ import annotations

import heapq
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent

# ---- grid ------------------------------------------------------------------
NX, NY = 300, 150            # cells east-west, north-south
DX = 800.0                   # metres per cell -> 240 km x 120 km
LX, LY = NX * DX, NY * DX

# ---- time ------------------------------------------------------------------
DT = 10_000.0                # years per step
T_END = 9.0e6
SNAP_EVERY = 50_000.0

# ---- physics ---------------------------------------------------------------
K = 5.0e-6                   # river erodibility, 1/yr (with area in m^2)
M = 0.5                      # area exponent in the river cutting law
D = 0.02                     # hillslope creep, m^2/yr (small at 800 m cells)
U_MAIN = 1.15e-3              # main range surface uplift, m/yr
U_FOLD = 1.4e-3              # foothill ridge surface uplift, m/yr
REBOUND = 0.8                # crust floats: removing 1 m of rock lets the
                             # region rise back 0.8 m (rock / mantle density)
FLEX = 12e3                  # ... spread over ~12 km, as stiff plates bend
SEA_FLOOR = -40.0            # sea depth where the range will rise, m
SEA_DEEP = -1500.0           # sea depth at the edges of the map, m

y = (np.arange(NY) + 0.5) * DX          # 0 = south edge
x = (np.arange(NX) + 0.5) * DX          # 0 = west edge
X, Y = np.meshgrid(x, y)                # shape (NY, NX)


def smoothstep(z):
    z = np.clip(z, 0.0, 1.0)
    return z * z * (3 - 2 * z)


# Main range: lens-shaped belt, steeper on its south flank (the thrust front).
RANGE_Y = 0.56 * LY
across = np.where(Y < RANGE_Y,
                  np.exp(-((Y - RANGE_Y) / 15e3) ** 2),
                  np.exp(-((Y - RANGE_Y) / 24e3) ** 2))
along = smoothstep(X / 30e3) * smoothstep((LX - X) / 30e3)
MAIN_SHAPE = across * along

# Foothill ridge: narrow fold south of the range, shorter along strike.
FOLD_Y = 0.37 * LY
FOLD_SHAPE = (np.exp(-((Y - FOLD_Y) / 4.5e3) ** 2)
              * smoothstep((X - 45e3) / 25e3) * smoothstep((LX - 45e3 - X) / 25e3))


# Natural unevenness (fixed seed, so runs are repeatable).
_rng = np.random.default_rng(11)


def _bumps(scale_m, rng=_rng):
    f = gaussian_filter(rng.standard_normal((NY, NX)), scale_m / DX, mode="wrap")
    return f / f.std()


UPLIFT_VARIES = 1.0 + 0.15 * _bumps(10e3)          # +-15% over ~10 km
K_FIELD = K * np.exp(0.35 * _bumps(4e3))           # rock hardness patches
SEA_BUMPS = 35.0 * _bumps(2.5e3) + 8.0 * _bumps(1e3)


def uplift_rate(t):
    """Uplift rate field (m/yr) at time t (yr)."""
    main_on = 1.0 - smoothstep((t - 5.3e6) / 0.4e6)     # switches off ~5.5 Myr
    start = smoothstep(t / 0.3e6)
    fold_on = smoothstep((t - 3.0e6) / 0.3e6) * main_on
    return UPLIFT_VARIES * (U_MAIN * MAIN_SHAPE * start * main_on
                            + U_FOLD * FOLD_SHAPE * fold_on)


# ---- flow routing ----------------------------------------------------------
OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
N = NX * NY
NBRS = []
for i in range(N):
    r, c = divmod(i, NX)
    NBRS.append(tuple((r + dr) * NX + (c + dc) for dr, dc in OFFS
                      if 0 <= r + dr < NY and 0 <= c + dc < NX))
EDGE = np.zeros((NY, NX), bool)
EDGE[0, :] = EDGE[-1, :] = EDGE[:, 0] = EDGE[:, -1] = True
EDGE = EDGE.ravel()


def route(h):
    """Fill pits (priority flood), then steepest-descent receivers.

    Returns (filled, receiver, order). Water in a closed hollow fills it into
    a lake and spills over the lowest point of the rim, as real water does.
    `order` lists cells from the sea upward, so every cell comes after the cell
    it drains into.
    """
    flat = h.ravel()
    sea = (flat <= 0.0) | EDGE
    filled = flat.tolist()
    closed = bytearray(N)
    heap = []
    for i in np.flatnonzero(sea).tolist():
        closed[i] = 1
        heap.append((filled[i], i))
    heapq.heapify(heap)
    order = []
    push, pop = heapq.heappush, heapq.heappop
    while heap:
        z, i = pop(heap)
        order.append(i)
        for j in NBRS[i]:
            if closed[j]:
                continue
            closed[j] = 1
            zj = filled[j]
            if zj <= z:
                zj = z + 1e-3        # lake surface: tiny slope toward the spill
                filled[j] = zj
            push(heap, (zj, j))
    filled = np.array(filled).reshape(NY, NX)

    # Steepest descent on the filled surface.
    pad = np.pad(filled, 1, constant_values=np.inf)
    best = np.zeros((NY, NX))
    rec = np.full((NY, NX), -1, np.int64)
    dist = np.full((NY, NX), DX)
    rows, cols = np.mgrid[0:NY, 0:NX]
    for dr, dc in OFFS:
        d = DX * (2 ** 0.5 if dr and dc else 1.0)
        nb = pad[1 + dr:1 + dr + NY, 1 + dc:1 + dc + NX]
        s = (filled - nb) / d
        take = s > best
        best = np.where(take, s, best)
        rec = np.where(take, (rows + dr) * NX + (cols + dc), rec)
        dist = np.where(take, d, dist)
    rec = rec.ravel()
    rec[sea] = -1
    return filled, rec, dist.ravel(), order, sea


def drainage_area(rec, order):
    area = [DX * DX] * N
    for i in reversed(order):
        r = rec[i]
        if r >= 0:
            area[r] += area[i]
    return np.array(area)


def river_erosion(h, rec, dist, area, order, sea):
    """Implicit stream-power step (Braun & Willett 2013), n = 1."""
    F = (K_FIELD.ravel() * area ** M * DT / dist).tolist()
    hl = h.ravel().tolist()
    recl = rec.tolist()
    for i in order:
        r = recl[i]
        if r < 0:
            continue
        base = hl[r] if hl[r] > 0.0 else 0.0     # sea level is the floor
        if hl[i] > base:                          # lakes do not cut
            f = F[i]
            hl[i] = (hl[i] + f * base) / (1.0 + f)
    return np.array(hl).reshape(NY, NX)


def hillslope_creep(h, land):
    lap = (np.roll(h, 1, 0) + np.roll(h, -1, 0) + np.roll(h, 1, 1)
           + np.roll(h, -1, 1) - 4 * h) / DX ** 2
    return h + np.where(land, D * DT * lap, 0.0)


def main():
    rng = np.random.default_rng(7)
    shelf = np.exp(-((Y - RANGE_Y) / 32e3) ** 2) * (smoothstep(X / 45e3)
                                                     * smoothstep((LX - X) / 45e3))
    h = SEA_DEEP + (SEA_FLOOR - SEA_DEEP) * shelf + SEA_BUMPS \
        + 0.5 * rng.standard_normal((NY, NX))
    lifted = np.zeros((NY, NX))      # total rock pushed up (m)
    eroded = np.zeros((NY, NX))      # total rock removed (m)

    snaps = {k: [] for k in ("t", "h", "lake", "area", "lifted", "eroded", "uplift")}
    nsteps = int(round(T_END / DT))
    every = int(round(SNAP_EVERY / DT))
    t0 = time.time()
    for step in range(nsteps + 1):
        t = step * DT
        if step % every == 0:
            filled, rec, dist, order, sea = route(h)
            area = drainage_area(rec, order)
            snaps["t"].append(t)
            snaps["h"].append(h.astype(np.float32))
            snaps["lake"].append((filled - h).astype(np.float32))
            snaps["area"].append(area.reshape(NY, NX).astype(np.float32))
            snaps["lifted"].append(lifted.sum() * DX * DX)
            snaps["eroded"].append(eroded.sum() * DX * DX)
            snaps["uplift"].append(uplift_rate(t).astype(np.float32))
            if step % (every * 10) == 0:
                print(f"t={t / 1e6:4.2f} Myr  max {h.max():6.0f} m  "
                      f"land {np.mean(h > 0) * 100:4.1f}%  "
                      f"[{time.time() - t0:5.1f} s]", flush=True)
        if step == nsteps:
            break
        u = uplift_rate(t + DT / 2) * DT
        h = h + u
        lifted += np.where(h > 0, np.minimum(u, h), 0.0)   # only rock above the sea
        filled, rec, dist, order, sea = route(h)
        area = drainage_area(rec, order)
        before = h
        h = river_erosion(h, rec, dist, area, order, sea)
        h = hillslope_creep(h, h > 0)
        cut = np.maximum(before - h, 0.0)
        eroded += cut
        h = h + REBOUND * gaussian_filter(cut, FLEX / DX, mode="constant")
        lifted += REBOUND * gaussian_filter(cut, FLEX / DX, mode="constant")

    out = {k: np.array(v) for k, v in snaps.items()}
    out.update(dx=DX, profile_x=0.62 * LX)
    np.savez_compressed(HERE / "run.npz", **out)
    print(f"saved {len(out['t'])} snapshots to {HERE / 'run.npz'} "
          f"in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
