"""Earth's real surface over the last 65 million years, on a globe.

Where the land heights come from
--------------------------------
Nobody can simulate Earth's whole surface from physics alone for 65 million
years; the inputs (how hard every plate pushed, every climate swing) are not
known well enough. What geologists do instead is RECONSTRUCT it from rocks:
fossils say where the sea was, old lava and sediments say where mountains
stood, magnetised rocks say where each continent sat. The PALEOMAP project
(Scotese & Wright 2018) turned all of that into a world height map every
5 million years. This script uses those maps for 65, 60, ..., 5, 0 million
years ago (data/paleodem).

Moving smoothly between the 5-million-year maps
-----------------------------------------------
The same project has a plate-motion model (data/platemodel): which plate every
spot belongs to and how each plate turned on the globe through time. To draw
the Earth at, say, 47.5 million years ago, every spot is carried back along
its own plate's motion to where it sat at 50 and at 45 million years ago, the
two maps are read there, and the heights are blended. So continents glide
instead of blinking, and mountains grow gradually between maps.

Rivers
------
Rivers are not in the maps. For every frame, rain is let fall on every spot of
land and run downhill (lakes fill up and spill over their lowest edge). Where
enough water gathers, a river is drawn; the more upstream land, the bigger.
This is the same calculation as the island simulation, on the whole globe.

Run:  .venv/bin/python earth_surface_sim/earth_history.py [--check]
"""

from __future__ import annotations

import glob
import heapq
import re
import sys
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pygplates
from scipy.ndimage import distance_transform_edt, map_coordinates

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

MAP_AGES = list(range(0, 70, 5))          # million years ago
ROT = pygplates.RotationModel(str(DATA / "platemodel" / "PALEOMAP_PlateModel.rot"))

# ---- the height maps ---------------------------------------------------------
DEM_STEP = 2                               # 0.1 deg -> 0.2 deg


def _dem_file(age):
    for f in glob.glob(str(DATA / "paleodem" / "Scotese*" / "Map*_6min_*Ma.nc")):
        if int(re.search(r"_(\d+)Ma\.nc$", f).group(1)) == age:
            return f
    raise FileNotFoundError(age)


@lru_cache(maxsize=None)
def dem(age):
    """(lat, lon, height) for one PALEOMAP map; height in metres, lat rising."""
    path = CACHE / f"dem_{age:02d}Ma.npz"
    if not path.exists():
        with h5py.File(_dem_file(age), "r") as f:
            lat = f["latitude"][::DEM_STEP]
            lon = f["longitude"][::DEM_STEP]
            z = f["z"][::DEM_STEP, ::DEM_STEP].astype(np.float32)
        if lat[0] > lat[-1]:
            lat, z = lat[::-1], z[::-1]
        np.savez(path, lat=lat, lon=lon, z=z)
    d = np.load(path)
    return d["lat"], d["lon"], d["z"]


def to_xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def to_latlon(v):
    return (np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1))),
            np.degrees(np.arctan2(v[..., 1], v[..., 0])))


def sample_dem(age, v):
    lat, lon, z = dem(age)
    la, lo = to_latlon(v)
    r = (la - lat[0]) / (lat[1] - lat[0])
    c = ((lo - lon[0]) / (lon[1] - lon[0])) % (len(lon) - 1)
    return map_coordinates(z, [r.ravel(), c.ravel()], order=1, mode="nearest").reshape(r.shape)


# ---- plates -------------------------------------------------------------------
GRID = 0.25                                 # degrees, for plate tags and rivers
G_LAT = np.arange(-90 + GRID / 2, 90, GRID)
G_LON = np.arange(-180 + GRID / 2, 180, GRID)
GNY, GNX = len(G_LAT), len(G_LON)
G_XYZ = to_xyz(*np.meshgrid(G_LAT, G_LON, indexing="ij"))


def present_plate_ids():
    path = CACHE / "plate_ids_0p25.npy"
    if path.exists():
        return np.load(path)
    polys = pygplates.FeatureCollection(str(DATA / "platemodel" / "PALEOMAP_PlatePolygons.gpml"))
    part = pygplates.PlatePartitioner(polys, ROT, 0)
    ids = np.full((GNY, GNX), -1, np.int32)
    for i, la in enumerate(G_LAT):
        for j, lo in enumerate(G_LON):
            p = part.partition_point(pygplates.PointOnSphere(float(la), float(lo)))
            if p is not None:
                ids[i, j] = p.get_feature().get_reconstruction_plate_id()
    miss = ids < 0
    if miss.any():
        idx = distance_transform_edt(miss, return_distances=False, return_indices=True)
        ids = ids[idx[0], idx[1]]
    np.save(path, ids)
    return ids


PLATE0 = present_plate_ids()
PLATES = np.unique(PLATE0)


@lru_cache(maxsize=None)
def rot_matrix(plate, t):
    """3x3 matrix taking a present-day position on `plate` to its place at t Myr ago."""
    fr = ROT.get_rotation(float(t), int(plate))
    cols = [np.array((fr * pygplates.PointOnSphere(tuple(e))).to_xyz())
            for e in np.eye(3)]
    return np.stack(cols, 1)


@lru_cache(maxsize=8)
def plate_ids_at(t):
    """Which plate each 0.25-degree cell belongs to, t Myr ago."""
    ids = np.full((GNY, GNX), -1, np.int32)
    flat0 = PLATE0.ravel()
    xyz0 = G_XYZ.reshape(-1, 3)
    for k in PLATES:
        sel = flat0 == k
        la, lo = to_latlon(xyz0[sel] @ rot_matrix(k, t).T)
        r = np.clip(((la + 90) / GRID).astype(int), 0, GNY - 1)
        c = ((lo + 180) / GRID).astype(int) % GNX
        ids[r, c] = k
    miss = ids < 0
    if miss.any():
        idx = distance_transform_edt(miss, return_distances=False, return_indices=True)
        ids = ids[idx[0], idx[1]]
    return ids


def bracket(t):
    a = min(int(t // 5) * 5, MAP_AGES[-1])
    b = min(a + 5, MAP_AGES[-1])
    w = 0.0 if b == a else (t - a) / (b - a)
    return a, b, w


def height_at(t, v):
    """Height (m) at unit vectors v (paleo-positions at t Myr ago)."""
    a, b, w = bracket(t)
    la, lo = to_latlon(v)
    r = np.clip(((la + 90) / GRID).astype(int), 0, GNY - 1)
    c = ((lo + 180) / GRID).astype(int) % GNX
    pid = plate_ids_at(t)[r, c]
    out = np.zeros(v.shape[:-1], np.float32)
    for k in np.unique(pid):
        sel = pid == k
        back = v[sel] @ rot_matrix(k, t)              # -> present-day position
        ha = sample_dem(a, back @ rot_matrix(k, a).T)
        if w == 0.0:
            out[sel] = ha
        else:
            hb = sample_dem(b, back @ rot_matrix(k, b).T)
            out[sel] = (1 - w) * ha + w * hb
    return out


def track(t, lat0, lon0):
    """Where a present-day spot sat t Myr ago (lat, lon)."""
    v = to_xyz(np.array([lat0]), np.array([lon0]))
    r = int((lat0 + 90) / GRID); c = int((lon0 + 180) / GRID) % GNX
    la, lo = to_latlon(v @ rot_matrix(PLATE0[r, c], t).T)
    return float(la[0]), float(lo[0])


def present_region_height(t, lat_rng, lon_rng, min_today=None):
    """Mean height t Myr ago of the land inside a present-day lat/lon box."""
    a, b, w = bracket(t)
    sl_r = (G_LAT >= lat_rng[0]) & (G_LAT <= lat_rng[1])
    sl_c = (G_LON >= lon_rng[0]) & (G_LON <= lon_rng[1])
    xyz = G_XYZ[np.ix_(sl_r, sl_c)].reshape(-1, 3)
    pid = PLATE0[np.ix_(sl_r, sl_c)].ravel()
    if min_today is not None:
        keep = sample_dem(0, xyz) >= min_today
        xyz, pid = xyz[keep], pid[keep]
    h = np.zeros(len(xyz))
    for k in np.unique(pid):
        sel = pid == k
        ha = sample_dem(a, xyz[sel] @ rot_matrix(k, a).T)
        hb = sample_dem(b, xyz[sel] @ rot_matrix(k, b).T)
        h[sel] = (1 - w) * ha + w * hb
    return float(h.mean())


# ---- rivers ---------------------------------------------------------------------
CELL_KM2 = (111.2 * GRID) ** 2 * np.cos(np.radians(G_LAT))[:, None] * np.ones((1, GNX))


def rivers(z):
    """Steepest-descent drainage on a (GNY, GNX) height grid, longitude wraps.

    Returns receiver index (-1 for sea) and upstream area in km^2.
    """
    n = GNY * GNX
    land = (z > 0).ravel()
    zf = z.ravel().astype(float).tolist()
    closed = bytearray(n)
    heap = []
    # seeds: sea cells touching land
    sea2d = z <= 0
    touch = np.zeros_like(sea2d)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            touch |= np.roll(np.roll(~sea2d, dr, 0), dc, 1)
    for i in np.flatnonzero((sea2d & touch).ravel()).tolist():
        closed[i] = 1
        heap.append((0.0, i))
    for i in np.flatnonzero(sea2d.ravel()).tolist():
        closed[i] = 1
    heapq.heapify(heap)
    order = []
    pop, push = heapq.heappop, heapq.heappush
    nx = GNX
    while heap:
        h, i = pop(heap)
        order.append(i)
        r, c = divmod(i, nx)
        for dr in (-1, 0, 1):
            rr = r + dr
            if rr < 0 or rr >= GNY:
                continue
            base = rr * nx
            for dc in (-1, 0, 1):
                j = base + (c + dc) % nx
                if closed[j]:
                    continue
                closed[j] = 1
                zj = zf[j]
                if zj <= h:
                    zj = h + 1e-3
                    zf[j] = zj
                push(heap, (zj, j))
    filled = np.array(zf).reshape(GNY, GNX)
    filled[sea2d] = 0.0

    # steepest descent on the filled surface
    best = np.zeros((GNY, GNX))
    rec = np.full((GNY, GNX), -1, np.int64)
    rows, cols = np.mgrid[0:GNY, 0:GNX]
    coslat = np.cos(np.radians(G_LAT))[:, None]
    pad = np.pad(filled, ((1, 1), (0, 0)), constant_values=np.inf)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nb = np.roll(pad[1 + dr:1 + dr + GNY], -dc, 1)
            dist = np.sqrt(dr ** 2 + (dc * coslat) ** 2)
            s = (filled - nb) / dist
            take = s > best
            best = np.where(take, s, best)
            rec = np.where(take, ((rows + dr) % GNY) * GNX + (cols + dc) % GNX, rec)
    rec = rec.ravel()
    rec[~land] = -1

    area = CELL_KM2.ravel().tolist()
    recl = rec.tolist()
    for i in reversed(order):
        r = recl[i]
        if r >= 0:
            area[r] += area[i]
    area = np.array(area)
    area[~land] = 0.0
    return rec, area.reshape(GNY, GNX)


def frame_data(t):
    """Height grid, river receivers and areas at t Myr ago (cached)."""
    path = CACHE / f"frame_{t:05.1f}.npz"
    if path.exists():
        d = np.load(path)
        return d["z"], d["rec"], d["area"]
    z = height_at(t, G_XYZ)
    rec, area = rivers(z)
    np.savez_compressed(path, z=z.astype(np.float32), rec=rec.astype(np.int32),
                        area=area.astype(np.float32))
    return z, rec, area


if __name__ == "__main__":
    import time
    t0 = time.time()
    z, rec, area = frame_data(0.0)
    print(f"present-day frame in {time.time() - t0:.1f} s; land {np.mean(z > 0):.3f}")
    # Largest present-day river mouths: where do they come out?
    mouth = (area > 0) & (rec.reshape(area.shape) >= 0)
    flat = area.ravel()
    recf = rec
    is_mouth = np.zeros(flat.shape, bool)
    land = (z > 0).ravel()
    is_mouth[land] = ~land[np.maximum(recf[land], 0)]
    top = np.argsort(np.where(is_mouth, flat, 0))[::-1][:12]
    for i in top:
        r, c = divmod(i, GNX)
        print(f"  mouth at lat {G_LAT[r]:6.1f} lon {G_LON[c]:7.1f}: "
              f"{flat[i] / 1e6:5.2f} million km^2 drained")
