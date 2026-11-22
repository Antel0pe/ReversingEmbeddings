"""Map the traced lines and compare their cross-sections with the square's ditch.

Usage: python -I lidar/map_lines.py SURFACES.npz TRACES.json OUT.png CX CY

Left: bumps-and-dips map of all processed tiles, traced lines drawn on top.
Right: the median cross-section of each line (ground height across the line,
relative to the ground 6-9 m to either side).
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
s = np.load(surf)
dtm = s["dtm"].astype(float)
X0, Y0 = s["origin"]
CELL = float(s["cell"])
NY, NX = dtm.shape
filled = np.where(np.isnan(dtm), np.nanmedian(dtm), dtm)
rel = ndi.gaussian_filter(filled - ndi.gaussian_filter(filled, 10 / CELL), 1.0)
rel[~ndi.binary_erosion(np.isfinite(dtm), iterations=int(12 / CELL))] = np.nan
ext = (X0 - CX, X0 + NX * CELL - CX, Y0 - NY * CELL - CY, Y0 - CY)

traces = json.load(open(traces_path))
# label, colour, line style for each trace name prefix (names not listed are skipped)
STYLE = {
    "square": ("square's own ditch (4 sides)", "k", "-"),
    "nw": ("north-west line", "#d62728", "-"),
    "east": ("east line", "#ff7f0e", "-"),
    "se": ("south-east line, down to the main stream", "#9467bd", "-"),
    "sw": ("south-west line, from across the main stream", "#2ca02c", "-"),
    "unstable": ("fainter branch (traces disagree)", "#17becf", "--"),
}
# where a line meets the main stream, its end is marked by the crossing note instead
NO_END_LABEL = {"se", "sw_back", "nw_back"}

fig = plt.figure(figsize=(21, 11.5), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1])
a0, a1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad("#d9d9d9")
a0.imshow(np.clip(rel, -0.35, 0.35), cmap=cmap, vmin=-0.35, vmax=0.35, extent=ext, alpha=0.55)

done = set()
profiles = {}
for name, t in traces.items():
    key = next((k for k in STYLE if name.startswith(k)), None)
    if key is None or not t["path"]:
        continue
    label, colour, ls = STYLE[key]
    p = np.array(t["path"]) - [CX, CY]
    a0.plot(p[:, 0], p[:, 1], color=colour, lw=3.2, ls=ls, label=None if key in done else label, solid_capstyle="round")
    done.add(key)
    if key != "unstable":
        profiles.setdefault(key, []).append((np.array(t["median_profile_m"]), len(p)))
    if key != "square" and name not in NO_END_LABEL:
        why = t["stopped_because"]
        text = {"ran off the data": "survey ends", "fell into a stream valley": "reaches a stream"}.get(
            why, "trace lost" if why.startswith("came back") else ("fades" if why == "groove faded" else None)
        )
        if text:
            a0.plot(*p[-1], "o", color=colour, ms=9, mec="k")
            a0.annotate(text, p[-1], xytext=(8, -4), textcoords="offset points", fontsize=12,
                        bbox=dict(fc="white", ec=colour, alpha=0.9))
    offs = np.array(t["profile_offsets_m"])

if "se" in traces and "sw_back" in traces:
    e1 = np.array(traces["se"]["end"]) - [CX, CY]
    e2 = np.array(traces["sw_back"]["end"]) - [CX, CY]
    a0.plot([e1[0], e2[0]], [e1[1], e2[1]], ":", color="k", lw=2.5)
    a0.annotate(f"{np.linalg.norm(e1 - e2):.0f} m across the main stream", (e1 + e2) / 2, xytext=(30, -10),
                textcoords="offset points", fontsize=12, bbox=dict(fc="white", ec="k", alpha=0.9))

a0.set_xlabel("metres east of the square's centre", fontsize=13)
a0.set_ylabel("metres north of the square's centre", fontsize=13)
a0.set_title("A. Where the lines go (grey = outside the 4 processed tiles or the survey)", fontsize=15, loc="left")
a0.legend(loc="upper right", fontsize=12, framealpha=0.95)

for key, plist in profiles.items():
    label, colour, _ = STYLE[key]
    w = np.array([n for _, n in plist], float)
    prof = np.average(np.array([pr for pr, _ in plist]), axis=0, weights=w)
    a1.plot(offs, prof, color=colour, lw=3, label=label)
a1.axhline(0, color="0.6", lw=1)
a1.set_xlabel("metres across the line (0 = groove centre)", fontsize=13)
a1.set_ylabel("height relative to ground 6-9 m to either side (m)", fontsize=13)
a1.set_title("B. Median cross-section of each line", fontsize=15, loc="left")
a1.legend(fontsize=12)
a1.annotate("square: a low bank beside the ditch", (5.8, 0.115), xytext=(1.5, 0.25), fontsize=12,
            arrowprops=dict(arrowstyle="->", color="k"))
a1.annotate("lines: a groove with no bank", (0, -0.29), xytext=(-8.8, -0.5), fontsize=12,
            arrowprops=dict(arrowstyle="->", color="#d62728"))
a1.set_ylim(-0.6, 0.35)
fig.savefig(out, dpi=85)
print("wrote", out)
