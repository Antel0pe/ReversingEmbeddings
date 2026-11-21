"""Draw Earth over the last 65 million years as a globe GIF.

Run:  .venv/bin/python earth_surface_sim/render_globe.py [--stills-only]
Needs earth_history.py's data (see its docstring).
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import earth_history as E  # noqa: E402

HERE = Path(__file__).resolve().parent
TIMES = np.round(np.arange(65.0, -0.01, -0.5), 1)
LAT0, LON0 = 12.0, 72.0                 # globe faces India / southern Asia
PIX = 600
RIVER_MIN_KM2 = 50_000

BG = "#0d1117"
FG = "#e6e6e6"
DIM = "#9aa4ae"

land_cmap = LinearSegmentedColormap.from_list("land", [
    (0.00, "#4d7a3a"), (0.08, "#7f9e53"), (0.25, "#c2b477"), (0.45, "#a7805a"),
    (0.70, "#86705f"), (0.88, "#c9c3bd"), (1.00, "#ffffff")])
sea_cmap = LinearSegmentedColormap.from_list("sea", [
    (0.00, "#0a1f3d"), (0.55, "#173f6e"), (0.93, "#2c6aa3"), (1.00, "#6aaad6")])
LAND_TOP, SEA_BOTTOM = 6000.0, -6000.0

ERAS = [  # (from Myr ago, to Myr ago, colour, title, caption)
    (65, 47, "#e0a458", "India sails north",
     "India is an island continent crossing the ocean toward Asia at about\n"
     "15 cm a year, four times as fast as fingernails grow, which is very\n"
     "fast for a continent. The land that is Tibet today is low ground."),
    (47, 40, "#e07a5f", "India meets Asia",
     "The last of the sea between them is squeezed out. Continental rock is\n"
     "too light to sink back into the Earth, so instead of sliding under,\n"
     "the two edges jam together and start to crumple and thicken."),
    (40, 18, "#d1495b", "Tibet and the Himalayas rise",
     "India keeps shoving north and the crust of southern Asia piles up to\n"
     "about twice its normal thickness. In these maps Tibet climbs from\n"
     "half a kilometre to four kilometres high. New rivers pour off it."),
    (18, -1, "#8ab6d6", "The roof of the world",
     "Tibet reaches nearly 5 km, the highest large plateau on Earth, and\n"
     "the great rivers of Asia (Indus, Ganges, Brahmaputra, Mekong, Yangtze)\n"
     "all start on it. India is still pushing north, about 4 cm a year."),
]

LABELS = [  # present-day spot, name, shown from (Myr ago) or always
    ((21.0, 78.5), "India", None),
    ((48.0, 95.0), "Asia", None),
    ((8.0, 22.0), "Africa", None),
    ((23.0, 45.0), "Arabia", None),
    ((33.0, 88.0), "Tibet", 33.0),
]


def era_at(t):
    for e in ERAS:
        if t > e[1]:
            return e
    return ERAS[-1]


# ---- globe geometry ------------------------------------------------------------
C = E.to_xyz(np.array(LAT0), np.array(LON0))
EAST = np.array([-np.sin(np.radians(LON0)), np.cos(np.radians(LON0)), 0.0])
NORTH = np.cross(C, EAST)
u = np.linspace(-1, 1, PIX)
GX, GY = np.meshgrid(u, -u)
RHO2 = GX ** 2 + GY ** 2
DISK = RHO2 < 1.0
GZ = np.sqrt(np.clip(1 - RHO2, 0, 1))
V = GX[..., None] * EAST + GY[..., None] * NORTH + GZ[..., None] * C


def project(v):
    """Unit vectors -> screen (x, y, visible)."""
    return v @ EAST, v @ NORTH, v @ C


def globe_image(t):
    h = np.full((PIX, PIX), np.nan, np.float32)
    h[DISK] = E.height_at(t, V[DISK])
    hh = np.nan_to_num(h, nan=0.0)
    land = hh > 0
    rgb = np.where(land[..., None],
                   land_cmap(np.clip(hh / LAND_TOP, 0, 1))[..., :3],
                   sea_cmap(np.clip(1 - hh / SEA_BOTTOM, 0, 1))[..., :3])
    # Light from the upper left; heights stretched so mountains show.
    px_m = 2 * 6371e3 / PIX
    gy, gx = np.gradient(hh * np.where(land, 1.0, 0.25), px_m)
    exag = 60.0
    nx, ny = -gx * exag, gy * exag
    norm = np.sqrt(nx ** 2 + ny ** 2 + 1)
    lx, ly, lz = -0.55, 0.55, 0.63
    shade = (nx * lx + ny * ly + lz) / norm
    rgb = rgb * (0.35 + 0.85 * np.clip(shade, 0, 1))[..., None]
    rgb = rgb * (0.45 + 0.55 * np.sqrt(GZ))[..., None]     # darker toward the edge
    rgba = np.concatenate([np.clip(rgb, 0, 1), DISK[..., None].astype(float)], -1)
    return rgba


def river_segments(t):
    z, rec, area = E.frame_data(t)
    a = area.ravel()
    src = np.flatnonzero((a > RIVER_MIN_KM2) & (rec >= 0))
    dst = rec[src]
    xyz = E.G_XYZ.reshape(-1, 3)
    x0, y0, c0 = project(xyz[src])
    x1, y1, c1 = project(xyz[dst])
    ok = (c0 > 0.05) & (c1 > 0.05)
    segs = np.stack([np.stack([x0, y0], -1), np.stack([x1, y1], -1)], 1)[ok]
    w = 0.35 + 0.75 * np.log10(a[src][ok] / RIVER_MIN_KM2)
    return segs, np.clip(w, 0.35, 2.4)


TIBET_CURVE = None


def tibet_curve():
    global TIBET_CURVE
    if TIBET_CURVE is None:
        path = E.CACHE / "tibet_curve.npy"
        if path.exists():
            TIBET_CURVE = np.load(path)
        else:
            TIBET_CURVE = np.array([E.present_region_height(t, (28, 37), (78, 98), 3000)
                                    for t in TIMES])
            np.save(path, TIBET_CURVE)
    return TIBET_CURVE


def draw(t, fig):
    fig.clf()
    fig.patch.set_facecolor(BG)
    era = era_at(t)
    label = "Today" if t == 0 else f"{t:.1f} million years ago"
    fig.text(0.035, 0.945, label, fontsize=21, weight="bold", color=FG, va="center")
    fig.text(0.035, 0.885, era[3], fontsize=14, weight="bold", color=era[2], va="center")
    fig.text(0.035, 0.80, era[4], fontsize=10.5, color=FG, va="center", linespacing=1.4)

    # --- globe ---------------------------------------------------------------
    ax = fig.add_axes([0.02, 0.03, 0.60, 0.70])
    ax.set_facecolor(BG)
    ax.imshow(globe_image(t), extent=(-1, 1, -1, 1), interpolation="bilinear")
    segs, w = river_segments(t)
    ax.add_collection(LineCollection(segs, linewidths=w, colors="#7fd3ff",
                                     capstyle="round", alpha=0.95))
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="#3a4654", lw=1))
    outline = [pe.withStroke(linewidth=2.6, foreground="black")]
    for (la, lo), name, since in LABELS:
        if since is not None and t > since:
            continue
        pla, plo = E.track(t, la, lo)
        x, y, c = project(E.to_xyz(np.array(pla), np.array(plo)))
        if c > 0.15:
            ax.text(x, y, name, color="white", fontsize=10, weight="bold",
                    ha="center", va="center", path_effects=outline)
    ax.set_xlim(-1.03, 1.03); ax.set_ylim(-1.03, 1.03)
    ax.set_aspect("equal"); ax.axis("off")

    # --- Tibet height chart --------------------------------------------------
    cax = fig.add_axes([0.70, 0.43, 0.27, 0.28])
    cax.set_facecolor(BG)
    curve = tibet_curve() / 1e3
    k = int(np.argmin(abs(TIMES - t)))
    cax.plot(TIMES, curve, color="#4a5561", lw=1.2)
    cax.plot(TIMES[:k + 1], curve[:k + 1], color=era[2], lw=2.4)
    cax.plot(TIMES[k], curve[k], "o", color="white", ms=6)
    cax.set_xlim(65, 0); cax.set_ylim(0, 5.5)
    cax.set_xticks([60, 40, 20, 0]); cax.set_yticks([0, 1, 2, 3, 4, 5])
    cax.tick_params(colors=DIM, labelsize=8.5)
    for s in cax.spines.values():
        s.set_color("#3a4654")
    cax.set_xlabel("million years ago", color=DIM, fontsize=9)
    cax.set_ylabel("km above sea", color=DIM, fontsize=9)
    cax.set_title(f"How high is Tibet?\n{curve[k]:.1f} km on average", color=FG,
                  fontsize=10.5, loc="left")

    # --- colour key -----------------------------------------------------------
    k1 = fig.add_axes([0.70, 0.245, 0.27, 0.022])
    k1.imshow(np.linspace(0, 1, 256)[None], cmap=land_cmap, aspect="auto",
              extent=(0, LAND_TOP / 1e3, 0, 1))
    k1.set_yticks([]); k1.set_xticks([0, 2, 4, 6])
    k1.tick_params(colors=DIM, labelsize=8)
    k1.set_title("land height (km)", color=DIM, fontsize=8.5, loc="left", pad=3)
    k2 = fig.add_axes([0.70, 0.165, 0.27, 0.022])
    k2.imshow(np.linspace(0, 1, 256)[None], cmap=sea_cmap, aspect="auto",
              extent=(SEA_BOTTOM / 1e3, 0, 0, 1))
    k2.set_yticks([]); k2.set_xticks([-6, -4, -2, 0])
    k2.tick_params(colors=DIM, labelsize=8)
    k2.set_title("sea floor depth (km)", color=DIM, fontsize=8.5, loc="left", pad=3)
    for kk in (k1, k2):
        for s in kk.spines.values():
            s.set_visible(False)
    fig.text(0.70, 0.125, "—", color="#7fd3ff", fontsize=16, weight="bold", va="center")
    fig.text(0.735, 0.125, "rivers (thicker = more water)", color=DIM, fontsize=8.5,
             va="center")
    fig.text(0.70, 0.02,
             "Land heights: PALEOMAP reconstructions\n(Scotese & Wright 2018), one map per\n"
             "5 million years, blended along plate motions.\n"
             "Rivers: rain run downhill on each frame.",
             color="#6b7681", fontsize=7, va="bottom", linespacing=1.3)


def frame_image(t, fig):
    draw(t, fig)
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())


def _prep(t):
    E.frame_data(float(t))
    return t


def _render_chunk(ts):
    fig = plt.figure(figsize=(10.0, 7.4), dpi=88)
    out = []
    for t in ts:
        p = E.CACHE / f"gframe_{t:05.1f}.png"
        frame_image(float(t), fig).save(p)
        out.append(p)
    plt.close(fig)
    return out


def main():
    stills = [65.0, 50.0, 42.0, 30.0, 15.0, 0.0]
    with Pool(4) as pool:
        need = stills if "--stills-only" in sys.argv else list(TIMES)
        for _ in pool.imap_unordered(_prep, need):
            pass
        tibet_curve()
        if "--stills-only" in sys.argv:
            for p, t in zip(_render_chunk(stills), stills):
                Image.open(p).save(HERE / f"globe_{int(t):02d}Ma.png")
            return
        chunks = [list(TIMES[i::4]) for i in range(4)]
        pool.map(_render_chunk, chunks)
    for t in stills:
        Image.open(E.CACHE / f"gframe_{t:05.1f}.png").save(HERE / f"globe_{int(t):02d}Ma.png")
    frames = [Image.open(E.CACHE / f"gframe_{t:05.1f}.png").convert("RGB") for t in TIMES]
    # One palette per frame: a shared one bands the colour keys and text.
    q = [f.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
         for f in frames]
    dur = [110] * len(q)
    dur[0], dur[-1] = 1500, 3500
    out = HERE / "earth_65_million_years.gif"
    q[0].save(out, save_all=True, append_images=q[1:], duration=dur, loop=0, optimize=True)
    print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
