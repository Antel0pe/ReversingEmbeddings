"""Before / after pictures for the best simple methods (2026-11-15).

One figure per example sheet: A the input (for sheets in more than 3-D, its three most spread-out
directions), B the walking version of the method, C the carry-and-vote version. Both use the same
neighbour choice (closest dots, minus tilted and mirror-facing connections).

Run: .venv/bin/python -m unfolding_suite.experiments.simple_figure
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import unfolding_suite.experiments.up_loops  # noqa: F401 - registers up_0.0_surface
from unfolding_suite.experiments.pipeline_figure import flat_panel
from unfolding_suite.methods import METHODS
from unfolding_suite.report import box_shape, print_colours
from unfolding_suite.run import failure_words
from unfolding_suite.scoring import score
from unfolding_suite.sheets import SHEETS, make_instance

OUT = Path(__file__).resolve().parent.parent / "figures" / "simple"
PANELS = (("check_k10", "B. Tilt check, then walking along the paper"),
          ("best_simple", "C. Tilt check, then carry and vote (the new method)"))


def note(r):
    if r["passed"]:
        return "passes: nothing folded"
    return f"{100 * max(r['fold_share'], r['overlap_share']):.1f}% of the paper folded over" \
        if max(r["fold_share"], r["overlap_share"]) > 0 else failure_words(r)


def example(sheet, label, grid=65, seed=100, panels=PANELS):
    inst = make_instance(SHEETS[sheet], grid, seed)
    X = inst.xyz
    shown = "3-D"
    if X.shape[1] > 3:
        Xc = X - X.mean(0)
        X = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:3].T
        shown = f"{inst.xyz.shape[1]}-D, drawn by its 3 most spread-out directions"
    fig = plt.figure(figsize=(17, 7.1))
    fig.text(0.5, 0.97, label, ha="center", va="top", fontsize=15.5, fontweight="bold")
    ax = fig.add_axes([0.01, 0.17, 0.31, 0.68], projection="3d")
    ax.scatter(*X[:, :3].T, c=print_colours(inst.uv), s=2.0 if grid == 65 else 0.5, linewidths=0, depthshade=False,
               rasterized=True)
    ax.view_init(elev=26, azim=-58)
    ax.set_box_aspect(box_shape(X[:, :3]))
    ax.set_axis_off()
    ax.set_title(f"A. Before ({shown})", fontsize=13)
    fig.text(0.165, 0.145, f"{len(inst.uv):,} dots. Each spot of the paper has its own colour\n"
                           "(blue to orange left to right, pale to deep bottom to top)",
             ha="center", va="top", fontsize=10.5, color="#52514e")
    for x, (m, title) in zip((0.36, 0.68), panels):
        a = METHODS[m](inst.points.copy(), 2, np.random.default_rng(seed + 1000))
        flat = a[0] if isinstance(a, tuple) else a
        r = score(flat, inst)
        flat_panel(fig.add_axes([x, 0.17, 0.29, 0.68]), inst, flat, title, note(r), r["passed"])
    fig.text(0.5, 0.035, "B and C: the flat result printed with the original paper's colours. Red = paper lying "
                         "mirror-image (folded over). Green border = passed, red border = failed.",
             ha="center", fontsize=10.5, color="#52514e")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{sheet}.png"
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(example("crescent_flat", "C-shaped paper, lying perfectly flat (non-convex outline)"))
    if False:
        print(example("bumps_8d", "Paper with gentle bumps in 8-D: the old 'which side is up' check versus the new one",
                  panels=(("up_0.0_surface", "B. Old up-side check (3-D only), walking"),
                          ("simple_unfold", "C. New facing check (any dimension), walking"))))
    print(example("h_crumple17", "A crumple (held back): where both still fail, and walking fails less"))


AUDIT_NAMES = {"patchy_scurl": "S-curl, uneven dots\n(4x denser in places)", "patchy_roll": "Roll, uneven dots",
               "crescent_flat": "C-shaped paper, flat\n(outline with a dent)", "crescent_roll": "C-shaped paper, rolled",
               "strip_roll": "4:1 strip, rolled\n(dots 4x closer across it)", "hairpin_flap": "Hairpin: one end folded\nflat back over the rest"}


def audit_gallery():
    fig = plt.figure(figsize=(17, 9))
    fig.text(0.5, 0.975, "The six new audit sheets", ha="center", va="top", fontsize=16, fontweight="bold")
    for i, (name, label) in enumerate(AUDIT_NAMES.items()):
        inst = make_instance(SHEETS[name], 65, 100)
        X = inst.xyz
        if X.shape[1] > 3:
            Xc = X - X.mean(0)
            X = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:3].T
        ax = fig.add_axes([0.02 + 0.33 * (i % 3), 0.53 - 0.47 * (i // 3), 0.30, 0.36], projection="3d")
        ax.scatter(*X[:, :3].T, c=print_colours(inst.uv), s=1.6, linewidths=0, depthshade=False, rasterized=True)
        ax.view_init(elev=26, azim=-58)
        ax.set_box_aspect(box_shape(X[:, :3]))
        ax.set_axis_off()
        fig.text(0.17 + 0.33 * (i % 3), 0.93 - 0.47 * (i // 3), label, ha="center", va="top", fontsize=12,
                 linespacing=1.3)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "audit_sheets.png", dpi=95, facecolor="white")
    plt.close(fig)
    return OUT / "audit_sheets.png"
