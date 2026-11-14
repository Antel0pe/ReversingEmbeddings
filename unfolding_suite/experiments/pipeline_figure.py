"""Before / after pictures for the combined pipeline (pipeline.py).

One figure per example sheet, three panels:
  A  the crumpled input in 3-D, printed so every spot of the paper has its own colour
  B  what the pipeline made of it from the dots alone (red = paper lying mirror-image, i.e. folded)
  C  the same steps given perfect information (the control that shows which decision leaked)

Run: .venv/bin/python -m unfolding_suite.experiments.pipeline_figure
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from unfolding_suite.report import FOLD, box_shape, print_colours
from unfolding_suite.scoring import align_to_square, signed_area, to_grid_order, triangles
from unfolding_suite.sheets import SHEETS, make_instance

RUNS = Path(__file__).resolve().parent.parent / "runs" / "pipeline"
OUT = Path(__file__).resolve().parent.parent / "figures" / "pipeline"
GOOD = "#1f7a45"


def flat_panel(ax, inst, flat, title, note, passed):
    F = to_grid_order(np.asarray(flat, float), inst)
    F = align_to_square(F, inst.uv)[0]
    stride = 2 if inst.grid > 65 else 1
    T = triangles(inst.grid, stride)
    a = signed_area(F, T)
    paper = signed_area(inst.uv, T)
    flip = (a < 0) if paper[a > 0].sum() >= paper[a < 0].sum() else (a > 0)
    ax.add_collection(PolyCollection(F[T], facecolors=print_colours(inst.uv[T].mean(1)), edgecolors="none",
                                     rasterized=True, zorder=2))
    if flip.any():
        ax.add_collection(PolyCollection(F[T[flip]], facecolors=FOLD, edgecolors=FOLD, lw=0.3, rasterized=True, zorder=3))
    lim = max(1.25, np.percentile(np.abs(F), 99.5) * 1.08)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GOOD if passed else FOLD); s.set_linewidth(2.2)
    ax.set_title(title, fontsize=13, pad=8)
    ax.text(0.5, -0.03, note, transform=ax.transAxes, ha="center", va="top", fontsize=10.5,
            color=GOOD if passed else FOLD, linespacing=1.35)


PART = {"unfold": ("after_unfold", "flat_unfold"), "final": ("final", "flat_final"), "guarded": ("guarded", "flat_guarded")}


def example(rows, arrays, sheet, grid, label, b, c, b_title, c_title, name=None):
    """b, c = (unfold method, control, stage), stage in unfold / final / guarded."""
    inst = make_instance(SHEETS[sheet], grid, 100)

    def pick(spec):
        u, control, st = spec
        r = next(x for x in rows if x["sheet"] == sheet and x["grid"] == grid and x["seed"] == 100
                 and x["unfold"] == u and x["control"] == control)
        part, key = PART[st]
        return r[part], arrays[f"{sheet}|{grid}|100|{u}|{control}|{key}"]

    fig = plt.figure(figsize=(17, 7.1))
    fig.text(0.5, 0.97, label, ha="center", va="top", fontsize=15.5, fontweight="bold")
    ax = fig.add_axes([0.01, 0.17, 0.31, 0.68], projection="3d")
    ax.scatter(*inst.xyz.T, c=print_colours(inst.uv), s=2.0 if grid == 65 else 0.35, linewidths=0, depthshade=False,
               rasterized=True)
    ax.view_init(elev=26, azim=-58)
    ax.set_box_aspect(box_shape(inst.xyz))
    ax.set_axis_off()
    ax.set_title("A. Before: the crumpled paper in 3-D", fontsize=13)
    fig.text(0.165, 0.145, f"{len(inst.uv):,} dots. Each spot of the paper has its own colour\n"
                           "(blue to orange left to right, pale to deep bottom to top)",
             ha="center", va="top", fontsize=10.5, color="#52514e")
    for x, spec, title in ((0.36, b, b_title), (0.68, c, c_title)):
        st, flat = pick(spec)
        note = "passes: nothing folded" if st["passed"] else (f"{100 * st['fold']:.1f}% of the paper folded over"
                                                              if st["fold"] > 0 else st["words"])
        flat_panel(fig.add_axes([x, 0.17, 0.29, 0.68]), inst, flat, title, note, st["passed"])
    fig.text(0.5, 0.035, "B and C show the flat result printed with the original paper's colours. Red = paper lying "
                         "mirror-image on the table (folded over). Green border = passed, red border = failed.",
             ha="center", fontsize=10.5, color="#52514e")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name or sheet + '_' + str(grid)}.png"
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)
    return path


LADDER = [("own", "from the dots alone"), ("true_links", "+ perfect neighbours"),
          ("true_links_edge", "+ perfect edge dots"), ("true_mesh_circle", "+ connections that never cross,\n   edge in its true order")]
NAMES = {"crumple": "Wide twists", "hard_crumple": "Tight twists", "h_crumple17": "Wide twists, new sheet\n(held back)",
         "h_crumple29": "Tighter twists, new sheet\n(held back)"}


def ladder(rows):
    """How much of each crumple is folded after the initial unfold, as each decision is made perfect."""
    import matplotlib.colors as mc
    fig, ax = plt.subplots(figsize=(14, 6.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.2)
    sheets = list(NAMES)
    colours = ["#b3361f", "#e08a3c", "#e8c24a", "#1f7a45"]
    w = 0.2
    for k, (control, label) in enumerate(LADDER):
        for i, s in enumerate(sheets):
            rs = [r for r in rows if r["sheet"] == s and r["grid"] == 65 and r["unfold"] == "edge" and r["control"] == control]
            vals = [100 * r["after_unfold"]["fold"] for r in rs]
            m = float(np.mean(vals))
            x = i + (k - 1.5) * w
            ax.bar(x, max(m, 0.05), w * 0.92, color=colours[k], label=label if i == 0 else None)
            ax.plot([x] * len(vals), vals, "o", color="#222222", ms=3.5, zorder=3)
            npass = sum(r["after_unfold"]["passed"] for r in rs)
            ax.text(x, max(vals) + 0.6, f"{npass}/3\npass", ha="center", va="bottom", fontsize=9,
                    color="#1f7a45" if npass == 3 else "#52514e")
    ax.set_xticks(range(len(sheets)))
    ax.set_xticklabels([NAMES[s] for s in sheets], fontsize=11.5)
    ax.set_ylabel("% of the paper folded over\nafter the initial unfold", fontsize=12)
    ax.set_ylim(0, 40)
    ax.legend(fontsize=11, loc="upper left", ncol=2, frameon=False)
    fig.text(0.5, 0.965, "Where the crumples leak: fix one decision at a time and watch the fold shrink",
             ha="center", va="top", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.905, "Each bar = average of 3 random dot samplings (black dots = each sampling), 4,225 dots, "
                         "edge-on-an-outline unfold. Bars from left to right add one perfect input each.",
             ha="center", va="top", fontsize=11, color="#52514e")
    fig.text(0.5, 0.06, "Green bars at 0 with 3/3 pass: with every input perfect, every crumple flattens with no fold. "
                        "So every fold left in the red bars traces to one of the three decisions.",
             ha="center", fontsize=11, color="#52514e")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "crumple_ladder.png", dpi=100, facecolor="white")
    plt.close(fig)
    return OUT / "crumple_ladder.png"


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "main"
    rows = json.loads((RUNS / f"{part}.json").read_text())
    arrays = dict(np.load(RUNS / f"{part}_layouts.npz"))
    if part == "main":
        print(ladder(rows))
        print(example(rows, arrays, "h_crumple17", 65, "A crumpled sheet the pipeline never saw while being built (wide random twists)",
                      ("edge", "own", "final"), ("edge", "true_mesh_circle", "unfold"),
                      "B. After: the whole pipeline, from the dots alone", "C. Initial unfold with every input perfect"))
        print(example(rows, arrays, "h_crumple29", 65, "The hardest crumple (tighter random twists, held back)",
                      ("edge", "own", "final"), ("edge", "true_mesh_circle", "unfold"),
                      "B. After: the whole pipeline, from the dots alone", "C. Initial unfold with every input perfect"))
        print(example(rows, arrays, "h_crumple29", 65, "The flip step breaking a correct result (hardest crumple, every input perfect)",
                      ("edge", "true_mesh_circle", "unfold"), ("edge", "true_mesh_circle", "final"),
                      "B. Initial unfold: already correct", "C. After the check wrongly flipped 1% and the strings settled",
                      name="flip_breaks_h_crumple29"))
        print(example(rows, arrays, "s_curl", 65, "A sheet that curls back over itself (S-shape)",
                      ("walk", "own", "final"), ("edge", "own", "final"),
                      "B. Pipeline with walking, from the dots alone", "C. Pipeline with edge-on-an-outline"))
    else:
        print(example(rows, arrays, "tight_roll", 257, "The tight roll, given enough dots (66,049)",
                      ("edge", "own", "final"), ("edge", "true_links", "final"),
                      "B. After: the whole pipeline, from the dots alone", "C. Same, with perfect neighbours"))
