"""One figure: does each spot on a flat 5-D plane belong to exactly one generated 1?

Reads figures/generated_one_flat_plane/{planes.npz,results.json}. Run after
analyze_generated_one_flat_plane.py.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

from grey_ones import render

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("figures/generated_one_flat_plane")
res = json.loads((OUT / "results.json").read_text())
planes = np.load(OUT / "planes.npz")
INK, MUTED, BLUE, GREY, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#a3a29c", "#eb6834"
MIDDLE = np.array([14.5, 14.5, 19.75, 3.2, 12.5])
NAME = "YOURS at middle of the ranges"
U = planes[NAME]
gap, a, b = res["planes"][NAME]["pairs"][0]
A, B = render(a, dtype=np.float64)[0], render(b, dtype=np.float64)[0]
ra, rb = A.ravel() @ U, B.ravel() @ U
assert np.abs(ra - rb).max() < 1e-4

fig = plt.figure(figsize=(14, 11), facecolor="white")
fig.text(0.04, 0.965, "Does each spot on a flat 5D plane belong to exactly one 1?", fontsize=21,
         fontweight="bold", color=INK)
fig.text(0.04, 0.933, "Your recipe: pick one 1, take the flat plane there, project every 1 onto it. "
         "Generated 1s only, within the default knob ranges.", fontsize=13, color=MUTED)

# ---- why the plane goes wrong
ax = fig.add_axes([0.04, 0.555, 0.24, 0.305])
base = render(MIDDLE, dtype=np.float64)[0]
weight = np.sqrt((U ** 2).sum(1)).reshape(28, 28)
ax.imshow(base, cmap="Greys", vmin=0, vmax=2.4, interpolation="nearest")
watch = np.zeros((28, 28, 4))
watch[..., :3] = matplotlib.colors.to_rgb(BLUE)
watch[..., 3] = np.where(weight > 1e-6, 0.35 + 0.65 * np.sqrt(weight / weight.max()), 0)
ax.imshow(watch, interpolation="nearest")
ax.contour(A, levels=[0.5], colors=ORANGE, linewidths=2.2)
ax.set_xticks([])
ax.set_yticks([])
outside = 1 - A[weight > 1e-6].sum() / A.sum()
fig.text(0.04, 0.875, f"A. The plane watches {int((weight > 1e-6).sum())} of 784 pixels", fontsize=15, color=INK)
ax.text(0, -0.03, f"Grey: the 1 this plane was taken at.\nBlue: the only pixels the plane can see.\n"
        f"Orange: the left-hand 1 from B. {100 * outside:.0f}% of its ink is off\nthe blue, where it can't affect the five numbers.",
        transform=ax.transAxes, va="top", fontsize=11.5, color=MUTED, linespacing=1.4)

# ---- the proof by example
for i, (img, p, r) in enumerate([(A, a, ra), (B, b, rb)]):
    ax = fig.add_axes([0.37 + 0.215 * i, 0.555, 0.165, 0.21])
    ax.imshow(img, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, -0.04, f"lean {p[4]:.1f}°, width {p[3]:.1f}, height {p[2]:.1f}", transform=ax.transAxes,
            ha="center", va="top", fontsize=12, color=INK)
fig.text(0.37, 0.875, "B. Two different 1s, the same five numbers", fontsize=15, color=INK)
nums = "  ".join(f"{v:.3f}" for v in ra)
fig.text(0.37, 0.845, f"Position of both on the plane:  {nums}", fontsize=12, color=INK, family="DejaVu Sans")
fig.text(0.37, 0.828, f"They match to better than 0.0001, yet the pictures are {gap:.2f}× as far apart "
         "as two typical 1s.\nSo this position does not name one image. Both rendered and checked.",
         fontsize=11.5, color=MUTED, linespacing=1.4, va="top")
fig.add_artist(plt.Line2D([0.04, 0.96], [0.44, 0.44], color="#dddcd6", lw=1))

# ---- the count across planes
P = res["planes"]
worst = "searched: random start, images B"
rows = [
    ("group", "The test on your paper example"),
    ("Gently curved sheet", res["paper"]["gently curved"], GREY),
    ("Crumpled sheet", res["paper"]["crumpled"], GREY),
    ("group", "Your recipe: the plane at one 1"),
    ("At root 1", P["YOURS at root 1"], BLUE),
    ("At the middle of the knob ranges", P[NAME], BLUE),
    ("At the best of 200 points", P["YOURS at best of 200 points"], BLUE),
    ("group", "Other flat planes, for comparison"),
    ("Best-fit plane for all 1s", P["best-fit plane for all 1s"], GREY),
    ("Random plane", P["random plane"], GREY),
    ("Five weighted pixel sums (my addition)", P["five weighted sums"], GREY),
    ("Searched from images, best of 6 (my addition)", P["searched: best-fit start, images A"], GREY),
    ("Searched from images, worst of 6 (my addition)", P[worst], GREY),
]
fig.text(0.04, 0.408, "C. How often each plane fails", fontsize=15, color=INK)
left = fig.add_axes([0.36, 0.075, 0.27, 0.29])
right = fig.add_axes([0.69, 0.075, 0.27, 0.29])
ypos, y = [], 0.0
for row in rows:
    if row[0] == "group":
        y += 0.35
        ypos.append(y)
        y += 1.0
    else:
        ypos.append(y)
        y += 1.0
for ax, key, title in [(left, lambda r: 100 * r["share_with_lookalike"],
                        "1s with a clearly different look-alike"),
                       (right, lambda r: 100 * (r["share_direction_stops_registering"] + r["share_plane_flipped"]),
                        "Settings where the plane goes blind or flips")]:
    for row, yy in zip(rows, ypos):
        if row[0] == "group":
            continue
        v = key(row[1])
        ax.barh(yy, max(v, 0.0), height=0.68, color=row[2], zorder=2)
        ax.text(v + 1.5, yy, "none" if v == 0 else f"{v:.0f}%" if v >= 10 else f"{v:.1f}%", va="center",
                fontsize=11.5, color=INK)
    ax.set_ylim(y - 0.3, -0.5)
    ax.set_yticks([])
    ax.set_title(title, fontsize=12.5, color=INK, loc="left", pad=8)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color("#b9b8b1")
    ax.set_xticks([])
left.set_xlim(0, 45)
right.set_xlim(0, 112)
for row, yy in zip(rows, ypos):
    if row[0] == "group":
        left.text(-0.03, yy, row[1], transform=left.get_yaxis_transform(), ha="right", va="center", fontsize=12,
                  fontweight="bold", color=INK)
    else:
        left.text(-0.03, yy, row[0], transform=left.get_yaxis_transform(), ha="right", va="center", fontsize=11.5,
                  color=INK)
fig.text(0.04, 0.012, "Look-alike: same position to one part in a million, pictures at least a quarter of the typical "
         "distance apart. 600 1s checked per plane (200 for paper). None found is evidence, not proof.\n"
         "Blind: some change to the 1 leaves all five numbers unchanged. Flips: the plane sees that part of the "
         "family turned over, as with a fold. 1,500 settings checked per plane (600 for paper).",
         fontsize=10, color=MUTED, linespacing=1.4)
fig.savefig(OUT / "flat_plane_one_to_one.png", dpi=150, facecolor="white")
print("ink off the watched pixels", outside)
print("gap", gap, "a", a, "b", b, "\nra", ra, "\nrb", rb)
