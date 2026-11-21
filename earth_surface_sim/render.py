"""Turn earth_surface_sim/run.npz into a GIF (and a few still frames).

Run:  .venv/bin/python earth_surface_sim/render.py [--stills-only]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize
from PIL import Image

HERE = Path(__file__).resolve().parent
run = np.load(HERE / "run.npz")
T, H, LAKE, AREA = run["t"], run["h"], run["lake"], run["area"]
LIFTED, ERODED = run["lifted"], run["eroded"]
DX = float(run["dx"])
NY, NX = H.shape[1:]
EXT_KM = (0, NX * DX / 1e3, 0, NY * DX / 1e3)
PROF_COL = int(run["profile_x"] / DX)
Y_KM = (np.arange(NY) + 0.5) * DX / 1e3

ZMAX = 4200.0
land_cmap = LinearSegmentedColormap.from_list("land", [
    (0.00, "#5f8f4e"), (0.10, "#86a95c"), (0.25, "#c3bd74"),
    (0.45, "#b48a57"), (0.65, "#8a6a52"), (0.85, "#a59c96"), (1.00, "#ffffff")])
sea_cmap = LinearSegmentedColormap.from_list("sea", ["#1d4f7a", "#3f7fb0", "#86b9d8"])
RIVER = np.array([0.12, 0.38, 0.75])
LAKE_RGB = np.array([0.30, 0.55, 0.85])
light = LightSource(azdeg=315, altdeg=40)

PHASES = [  # (start, end, colour, short label, what is happening)
    (0.0, 3.0, "#c98b4a", "Plates collide",
     "Two plates are pushing together. The sea floor buckles up into an\n"
     "island and keeps rising a millimetre or so a year. Rain gathers into\n"
     "rivers, and the rivers cut valleys into the land as it rises."),
    (3.0, 5.5, "#b0574a", "Foothill ridge rises",
     "A second ridge starts rising in the southern foothills, right across\n"
     "the rivers' paths. The rivers were there first and cut down as fast as\n"
     "it rises, so nearly all keep their courses and saw gorges through it."),
    (5.5, 9.0, "#6f8fa8", "Squeeze stops",
     "The plates stop pushing. Nothing lifts the mountains any more, so the\n"
     "rivers grind them down. As rock is removed the crust floats back up a\n"
     "little, so the range shrinks slowly rather than vanishing."),
]


def phase_at(t_myr):
    for p in PHASES:
        if t_myr < p[1]:
            return p
    return PHASES[-1]


def shaded_map(k):
    h = H[k].astype(float)
    land = h > 0
    rgb = np.zeros((NY, NX, 3))
    land_rgb = land_cmap(np.clip(h / ZMAX, 0, 1))[..., :3]
    sea_rgb = sea_cmap(np.clip((h + 1200) / 1200, 0, 1))[..., :3]
    rgb = np.where(land[..., None], land_rgb, sea_rgb)
    # Hillshade the land only, so the sea stays a flat readable colour.
    shade = light.hillshade(np.where(land, h, 0.0), vert_exag=4, dx=DX, dy=DX)
    rgb = np.where(land[..., None],
                   rgb * (0.45 + 0.75 * shade[..., None]), rgb)
    lake = land & (LAKE[k] > 2.0)
    rgb[lake] = LAKE_RGB
    area_km2 = AREA[k] / 1e6
    river = land & ~lake & (area_km2 > 12)
    strength = np.clip((np.log10(np.maximum(area_km2, 1)) - 1.08) / 1.6, 0.35, 1.0)
    a = np.where(river, strength, 0.0)[..., None]
    rgb = rgb * (1 - a) + RIVER * a
    return np.clip(rgb, 0, 1)


def draw(k, fig):
    fig.clf()
    t_myr = T[k] / 1e6
    ph = phase_at(t_myr)
    fig.text(0.04, 0.955, f"{t_myr:.2f} million years", fontsize=20,
             weight="bold", va="center")
    fig.text(0.42, 0.955, ph[3], fontsize=15, color=ph[2], weight="bold",
             va="center")
    fig.text(0.04, 0.885, ph[4], fontsize=10.5, va="center", color="#222",
             linespacing=1.35)

    # --- map -------------------------------------------------------------
    ax = fig.add_axes([0.04, 0.395, 0.80, 0.43])
    ax.imshow(shaded_map(k), origin="lower", extent=EXT_KM, interpolation="bilinear")
    xk = PROF_COL * DX / 1e3
    ax.plot([xk, xk], [3, EXT_KM[3] - 3], ls=(0, (4, 3)), color="white", lw=1.4)
    ax.text(xk + 1.5, 4, "S", color="white", fontsize=10, weight="bold")
    ax.text(xk + 1.5, EXT_KM[3] - 10, "N", color="white", fontsize=10, weight="bold")
    if t_myr >= 3.0:
        ax.annotate("new ridge", xy=(48, 0.37 * EXT_KM[3]), xytext=(4, 0.37 * EXT_KM[3] - 14),
                    color="white", fontsize=9, weight="bold",
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.2))
    ax.plot([10, 60], [8, 8], color="white", lw=3)
    ax.text(35, 11, "50 km", color="white", ha="center", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Map view (north is up).  Blue lines are rivers: the thicker, "
                 "the more water.", fontsize=9.5, loc="left", pad=4)
    # height key
    cax = fig.add_axes([0.865, 0.42, 0.018, 0.38])
    cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(0, ZMAX / 1e3), land_cmap),
                      cax=cax)
    cb.set_label("height above sea (km)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # --- cross-section ---------------------------------------------------
    pax = fig.add_axes([0.08, 0.14, 0.76, 0.20])
    prof = H[k][:, PROF_COL]
    pax.axhspan(-1.5, 0, color="#cfe3f1", zorder=0)
    pax.fill_between(Y_KM, -1.5, prof / 1e3, color="#9b7b5b", zorder=1)
    kk = max(k - 20, 0)
    if k >= 20:
        pax.plot(Y_KM, H[kk][:, PROF_COL] / 1e3, color="#333", lw=1, ls="--",
                 zorder=2, label="1 million years earlier")
        pax.legend(loc="upper left", fontsize=8, frameon=False)
    pax.axhline(0, color="#3f7fb0", lw=0.8, zorder=3)
    pax.set_xlim(0, EXT_KM[3]); pax.set_ylim(-1.2, 4.5)
    pax.set_xlabel("distance along the dashed line, south to north (km)", fontsize=9)
    pax.set_ylabel("height (km)", fontsize=9)
    pax.tick_params(labelsize=8)
    pax.text(2, -1.0, "sea", color="#3f7fb0", fontsize=8)
    w_in, h_in = fig.get_size_inches() * np.array([0.76, 0.20])
    exag = (EXT_KM[3] / w_in) / (5.7 / h_in)
    pax.set_title(f"Side view along the dashed line (heights stretched {exag:.0f}x "
                  "so they show up)", fontsize=9.5, loc="left", pad=4)

    # --- timeline + rock counters ----------------------------------------
    tax = fig.add_axes([0.08, 0.035, 0.76, 0.03])
    for a, b, col, lab, _ in PHASES:
        tax.axvspan(a, b, color=col, alpha=0.35)
        tax.text((a + b) / 2, 0.5, lab, ha="center", va="center", fontsize=8)
    tax.axvline(t_myr, color="black", lw=2.5)
    tax.set_xlim(0, 9); tax.set_yticks([])
    tax.set_xticks(range(10)); tax.tick_params(labelsize=7, pad=1)
    for s in tax.spines.values():
        s.set_visible(False)
    gone = ERODED[k] / 1e9
    up = LIFTED[k] / 1e9
    pct = 100 * gone / up if up > 0 else 0
    fig.text(0.865, 0.30, "Rock lifted\nabove the sea", fontsize=8.5, va="top")
    fig.text(0.865, 0.255, f"{up:,.0f} km³", fontsize=11, weight="bold", va="top")
    fig.text(0.865, 0.20, "Rock carried\naway by rivers", fontsize=8.5, va="top")
    fig.text(0.865, 0.155, f"{gone:,.0f} km³", fontsize=11, weight="bold", va="top",
             color="#1f5fa8")
    fig.text(0.865, 0.12, f"({pct:.0f}% of it)", fontsize=8.5, va="top")


def frame_image(k, fig):
    draw(k, fig)
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())


def main():
    fig = plt.figure(figsize=(9.6, 8.4), dpi=90)
    stills = [int(np.argmin(abs(T - t * 1e6))) for t in (0.5, 2.0, 4.0, 5.5, 7.0, 9.0)]
    for k in stills:
        frame_image(k, fig).save(HERE / f"still_{T[k] / 1e6:.1f}Myr.png")
    if "--stills-only" in sys.argv:
        return
    frames = [frame_image(k, fig) for k in range(len(T))]
    frames += [frames[-1]] * 15                       # pause on the final state
    pal = frames[len(frames) // 2].quantize(colors=255, method=Image.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    durations = [80] * len(T) + [120] * 15
    durations[0] = 1200
    q[0].save(HERE / "mountain_island.gif", save_all=True, append_images=q[1:],
              duration=durations, loop=0, optimize=True)
    print("wrote", HERE / "mountain_island.gif",
          f"{(HERE / 'mountain_island.gif').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
