"""Twitter graphic (2026-11-15): three crumpled sheets, the graphics recipe vs the best simple method,
and an overall score for every method on all 30 sheets.

Run from the repo root: .venv/bin/python -m unfolding_suite.experiments.make_twitter_frontier SHEET SHEET SHEET
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.collections import PolyCollection

import unfolding_suite.experiments.mesh_recipe  # noqa: F401 - registers mesh_full
from unfolding_suite.experiments.simple_score import needed_grid
from unfolding_suite.methods import METHODS
from unfolding_suite.report import box_shape, print_colours
from unfolding_suite.scoring import align_to_square, score, signed_area, to_grid_order, triangles
from unfolding_suite.sheets import SHEETS, make_instance

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = Path(__file__).resolve().parents[1] / "runs"
OUT = Path(__file__).resolve().parents[1] / "figures" / "twitter_frontier.png"
CACHE = Path(__file__).resolve().parents[1] / "runs" / "simple" / "twitter_layouts.npz"
INK, MUTED, FAINT, GOOD, BAD = "#14161a", "#6a6963", "#e6e5e0", "#1f8a4c", "#d03b3b"
MINE, FRONT = "#e8743b", "#2f6fd6"
NAMES = {"crumple": "Crumpled", "hard_crumple": "Tightly crumpled", "h_crumple17": "Crumpled", "h_crumple29": "Badly crumpled",
         "crescent_roll": "C-shaped sheet, rolled", "patchy_scurl": "S-curl, patchy dots", "tight_roll": "Tight roll",
         "s_curl": "S-curl", "strip_roll": "Rolled strip", "hairpin_flap": "Folded flat", "h_roll5": "Roll",
         "accordion": "Paper fan", "cone": "Cone", "diagonal_roll": "Diagonal roll", "h_three_lobes": "Three lobes"}
VIEWS = {"tight_roll": (16, -74), "s_curl": (16, -74), "patchy_scurl": (16, -74)}
COLUMNS = [("mesh_full", "Crude recreation of\nresearch papers", FRONT),
           ("best_simple", "Consistent basis vectors,\ncarried dot to dot", MINE)]
CLEAN = True     # the version to post: no grey notes, no scoreboard
BARS = [("best_simple", "My method (carry + vote)", MINE),
        ("convex_edge", "Pin the edge, average inside", "#a39a92"),
        ("mesh_full", "State of the art (graphics recipe)", FRONT),
        ("simple_unfold", "Walk along the paper", "#a39a92"),
        ("isomap_k10", "Plain walking*", "#bdb6af"),
        ("best_fit_plane", "Squash it flat", "#d3cec8")]


def scores():
    rows = []
    for name in ("main", "dense", "best", "frontier", "plane"):
        f = RUNS / "simple" / f"{name}.json"
        if f.exists():
            rows += [r for r in json.loads(f.read_text()) if not r.get("noise")]
    cur = json.loads((RUNS / "current" / "results.json").read_text())["tests"]
    out = {}
    sheets = list(SHEETS)
    for m, *_ in BARS:
        passed = 0
        for s in sheets:
            got = [r for r in rows if r["method"] == m and r["sheet"] == s]
            if got:
                g = max(r["grid"] for r in got)
                passed += all(r["passed"] for r in got if r["grid"] == g)
            elif f"{s} / {m}" in cur:
                passed += bool(cur[f"{s} / {m}"]["passed"])
        out[m] = passed
    return out, len(sheets)


def flat_panel(ax, inst, flat):
    F = align_to_square(to_grid_order(np.asarray(flat, float), inst), inst.uv)[0]
    T = triangles(inst.grid, 1 if inst.grid == 65 else 2)
    a, paper = signed_area(F, T), signed_area(inst.uv, T)
    flip = (a < 0) if paper[a > 0].sum() >= paper[a < 0].sum() else (a > 0)
    ax.add_collection(PolyCollection(F[T], facecolors=print_colours(inst.uv[T].mean(1)), edgecolors="none", rasterized=True))
    if flip.any():
        ax.add_collection(PolyCollection(F[T[flip]], facecolors=BAD, edgecolors=BAD, lw=0.4, rasterized=True))
    lim = max(1.18, np.percentile(np.abs(F), 99.7) * 1.05)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")


def main(picks):
    score_by, total = scores()
    cache = dict(np.load(CACHE)) if CACHE.exists() else {}
    cell, top, left = 3.25, 2.15, 0.35
    if CLEAN:
        top = 1.55
    row_h = cell + (0.45 if CLEAN else 0.55)
    chart_h = 0 if CLEAN else 0.62 * len(BARS) + 1.95
    width = left + cell * 3 + 0.35
    height = top + row_h * len(picks) + chart_h + 0.35
    fig = plt.figure(figsize=(width, height), facecolor="white")

    def box(x, y, w, h):
        return [x / width, 1 - (y + h) / height, w / width, h / height]

    fig.text(left / width, 1 - 0.5 / height, "Uncrumpling paper from 3D dots alone", fontsize=29, fontweight="bold",
             color=INK, va="center")
    if not CLEAN:
        fig.text(left / width, 1 - 1.05 / height, "Each dot only knows where it sits in space. Same colour = same spot on the paper.\n"
             "Red = paper that came out folded over itself.", fontsize=14.5, color=MUTED, va="center", linespacing=1.4)
    for j, head in enumerate(["The crumpled sheet"] + [c[1] for c in COLUMNS]):
        colour = INK if j == 0 else COLUMNS[j - 1][2]
        fig.text((left + cell * (j + 0.5)) / width, 1 - (top - 0.3) / height, head, ha="center", va="center",
                 fontsize=16, fontweight="bold", color=colour, linespacing=1.15)

    for i, sheet in enumerate(picks):
        y = top + row_h * i
        grid = needed_grid(sheet)
        inst = make_instance(SHEETS[sheet], grid, 100)
        ax = fig.add_axes(box(left - 0.15, y - 0.1, cell + 0.3, cell + 0.2), projection="3d")
        order = np.random.default_rng(0).permutation(len(inst.uv))[: 9000]
        ax.scatter(*inst.xyz[order].T, c=print_colours(inst.uv[order]), s=3.2 if grid == 65 else 3.0, linewidths=0,
                   depthshade=False, rasterized=True)
        ax.set_box_aspect(box_shape(inst.xyz))
        ax.view_init(*VIEWS.get(sheet, (24, -60)))
        ax.set_axis_off()
        if not CLEAN:
            fig.text((left + cell * 0.5) / width, 1 - (y + cell + 0.2) / height, NAMES.get(sheet, sheet), ha="center",
                     va="center", fontsize=14.5, color=MUTED)
        for j, (m, _, _) in enumerate(COLUMNS):
            key = f"{sheet}|{grid}|{m}"
            if key in cache:
                flat, secs = cache[key], float(cache[key + "|s"])
            else:
                t0 = time.time()
                a = METHODS[m](inst.points.copy(), 2, np.random.default_rng(1100))
                secs = time.time() - t0
                flat = a[0] if isinstance(a, tuple) else a
                cache[key], cache[key + "|s"] = np.asarray(flat, np.float32), np.float64(secs)
                np.savez_compressed(CACHE, **cache)
            r = score(flat, inst)
            flat_panel(fig.add_axes(box(left + cell * (j + 1) + 0.22, y + 0.12, cell - 0.44, cell - 0.44)), inst, flat)
            fold = 100 * max(r["fold_share"], r["overlap_share"])
            words = "Unfolded" if r["passed"] else (f"Folded {fold:.0f}%" if fold >= 1 else
                                                    f"Folded {fold:.2g}%" if fold > 0 else "Torn")
            fig.text((left + cell * (j + 1.5)) / width, 1 - (y + cell + 0.12) / height, words, ha="center", va="center",
                     fontsize=16, fontweight="bold", color=GOOD if r["passed"] else BAD)
            t = f"{secs:.1f} s" if secs < 60 else f"{secs / 60:.1f} min"
            if not CLEAN:
                fig.text((left + cell * (j + 1.5)) / width, 1 - (y + cell + 0.42) / height, f"took {t}", ha="center",
                     va="center", fontsize=11.5, color=MUTED)

    if CLEAN:
        fig.savefig(OUT, dpi=140, facecolor="white", metadata={"Software": None})
        print(OUT, (fig.get_size_inches() * 140).astype(int))
        return
    y0 = top + row_h * len(picks) + 0.15
    fig.add_artist(plt.Line2D([left / width, (width - 0.35) / width], [1 - y0 / height] * 2, color=FAINT, lw=1.5))
    fig.text(left / width, 1 - (y0 + 0.45) / height, f"Sheets flattened correctly, out of all {total} test sheets",
             fontsize=19, fontweight="bold", color=INK, va="center")
    fig.text(left / width, 1 - (y0 + 0.98) / height, "Rolls, curls, folds, fans, cones, dents, odd shapes and crumples,\n"
             "each given enough dots to tell its layers apart. Pass = nothing folded, on 3 tries out of 3.", fontsize=12.5, color=MUTED, va="center", linespacing=1.4)
    ax = fig.add_axes(box(left + 3.9, y0 + 1.5, width - left - 3.9 - 1.75, 0.62 * len(BARS)))
    vals = [score_by[m] for m, *_ in BARS]
    ys = np.arange(len(BARS))[::-1]
    ax.barh(ys, [total] * len(BARS), color=FAINT, height=0.62)
    ax.barh(ys, vals, color=[c for *_, c in BARS], height=0.62)
    for yy, v, (m, label, c) in zip(ys, vals, BARS):
        bold = m in ("mesh_full", "best_simple")
        ax.text(-0.6, yy, label, ha="right", va="center", fontsize=13, color=INK if bold else MUTED,
                fontweight="bold" if bold else "normal")
        ax.text(total + 0.5, yy, f"{v}/{total}  ({100 * v / total:.0f}%)", ha="left", va="center", fontsize=13,
                color=INK if bold else MUTED, fontweight="bold" if bold else "normal")
    ax.set_xlim(0, total)
    ax.set_ylim(-0.6, len(BARS) - 0.4)
    ax.axis("off")
    fig.text(left / width, 1 - (height - 0.3) / height, "*Plain walking needs too much memory for the 66,000-dot tight roll, "
             "so it is scored there with fewer dots.", fontsize=10.5, color=MUTED, va="center")
    fig.savefig(OUT, dpi=140, facecolor="white", metadata={"Software": None})
    print(OUT, (fig.get_size_inches() * 140).astype(int), score_by)


if __name__ == "__main__":
    main(sys.argv[1:] or ["crescent_roll", "tight_roll", "h_crumple29"])
