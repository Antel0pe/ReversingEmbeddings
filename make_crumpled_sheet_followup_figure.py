"""Figure for analyze_crumpled_sheet_followup.py: real paper comes back square; stretched
paper cannot, and its unrolling hides a fold that the dot-overlap test missed."""
import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.collections import PolyCollection

from analyze_crumpled_sheet_followup import fit_to_square, folded_share, good_triangles

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("figures/crumpled_sheet_unrolling")
res = json.loads((OUT / "followup_results.json").read_text())
lay = np.load(OUT / "followup_layouts.npz")
UV = np.load(OUT / "sheets.npz")["UV"]
INK, MUTED, RED = "#0b0b0b", "#52514e", "#d03b3b"
BLUE, ORANGE = np.array(matplotlib.colors.to_rgb("#2a78d6")), np.array(matplotlib.colors.to_rgb("#eb6834"))
a = ((UV[:, 0] + 1) / 2)[:, None]
shade = (0.30 + 0.70 * (UV[:, 1] + 1) / 2)[:, None]
COL = ((1 - a) * BLUE + a * ORANGE) * shade + (1 - shade)
ORDER = np.random.default_rng(0).permutation(len(UV))

keep = np.arange(0, len(UV), 4)   # 1,000-dot triangles: big enough that wobble cannot flip them
T = good_triangles(UV[keep])

fig = plt.figure(figsize=(14, 12.2), facecolor="white")
fig.text(0.03, 0.968, "Why the unrolled sheets weren't squares", fontsize=21, fontweight="bold", color=INK)
fig.text(0.03, 0.94, "Every sheet started as the same square of paper. Same colour print as before.",
         fontsize=12.5, color=MUTED)
rows = [("rolled, no stretch", "Real paper: bent, never stretched",
         "Comes back as the original square\n(grey outline). Dots sit {med:.1f}% of\nthe side from their true place\n"
         "(worst {worst:.1f}%). No folds anywhere,\nincluding between the dots."),
        ("crumpled, layers apart", "My crumpled sheet: stretched up to {hi:.1f}× when I made it",
         "Can't come back square: its\nstretching was uneven, and no flat\nshape matches it.\n\nRed: folded back over "
         "itself\n({fold:.1f}% of the paper). My first\noverlap test missed these.")]
for i, (name, title, note) in enumerate(rows):
    r = res[name]
    top = 0.885 - 0.455 * i
    X, Y = lay[f"{name}|sheet"], lay[f"{name}|unroll"]
    fig.text(0.03, top + 0.02, title.format(hi=r["stretch_when_made_p5_p95"][1]), fontsize=16, color=INK)
    ax = fig.add_axes([0.0, top - 0.37, 0.42, 0.37], projection="3d")
    ax.scatter(X[ORDER, 0], X[ORDER, 1], X[ORDER, 2], c=COL[ORDER], s=3, linewidths=0, depthshade=False,
               rasterized=True)
    ax.view_init(elev=22, azim=-62)
    ax.set_box_aspect(np.ptp(X, 0))
    ax.set_axis_off()
    ax = fig.add_axes([0.47, top - 0.33, 0.30, 0.33])
    Yf = fit_to_square(Y)
    _, flipped = folded_share(Yf[keep], UV[keep], T)
    if flipped.any():
        ax.add_collection(PolyCollection(Yf[keep][T[flipped]], facecolor=RED, edgecolor=RED, lw=0.6, alpha=0.75,
                                         zorder=3))
    ax.scatter(Yf[ORDER, 0], Yf[ORDER, 1], c=COL[ORDER], s=3, linewidths=0, rasterized=True, zorder=2)
    if i == 0:
        ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color="#8a8983", lw=1.6, zorder=4)
    ax.set_aspect("equal", "datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#c9c8c2")
    ax.set_title("Unrolled by walking along the sheet", fontsize=13, color=INK, loc="left")
    fold = 100 * r["folded_share_by_dots_used"]["1000"]
    txt = note.format(med=100 * r.get("distance_from_true_square_median", 0) / 2,
                      worst=100 * r.get("distance_from_true_square_worst", 0) / 2, fold=fold)
    fig.text(0.785, top - 0.165, txt, fontsize=11, color=MUTED, va="center", linespacing=1.45)
fig.add_artist(plt.Line2D([0.03, 0.97], [0.475, 0.475], color="#dddcd6", lw=1))
fig.text(0.03, 0.012, "Red = triangles of neighbouring dots (using every 4th dot, well-shaped, away from the edge) that come out "
         "mirror-flipped in the flat layout: a fold.",
         fontsize=10, color=MUTED)
fig.savefig(OUT / "sheet_flattening_why_not_square.png", dpi=130, facecolor="white")
