"""Figures for press_unfold (2026-11-14).

press_unfold.png  : loose roll, filtered strings. A: pressed result from above (folded);
                    B: the detector's area change per dot, drawn on the original paper;
                    C: after flipping across the creases and letting the strings settle.
crease_profile.png: area change along one strip across the paper, for the loose roll and the
                    S-shaped sheet: how gradually does it pass through zero at a crease?

Run: .venv/bin/python -m unfolding_suite.experiments.press_unfold_figure
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from unfolding_suite.report import print_colours
from unfolding_suite.scoring import align_to_square, signed_area, to_grid_order, triangles
from unfolding_suite.sheets import SHEETS, make_instance

RUNS = Path(__file__).resolve().parent.parent / "runs"
OUT = Path(__file__).resolve().parent.parent / "figures" / "press_unfold"


def from_above(ax, inst, flat, title, note):
    F = align_to_square(to_grid_order(flat, inst), inst.uv)[0]
    T = triangles(65, 1)
    a = signed_area(F, T)
    flipped = np.sign(a) != np.sign(np.median(a))
    col = print_colours(inst.uv[T].mean(1))
    col[flipped] = (0.82, 0.23, 0.23)
    ax.add_collection(PolyCollection(F[T], facecolors=col, edgecolors="none", alpha=0.85))
    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color="#666666", lw=1, ls="--")
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(note, fontsize=10.5)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = json.loads((RUNS / "press_unfold" / "results.json").read_text())
    pressed = json.loads((RUNS / "press_flat" / "main_seeds100.json").read_text())
    r = next(x for x in rows if x["sheet"] == "roll" and x["links"] == "checked")
    p = next(x for x in pressed if x["sheet"] == "roll" and x["links"] == "checked" and x["push"] == 0)
    inst = make_instance(SHEETS["roll"], 65, 100)
    uv = inst.uv[inst.order]
    fig, axs = plt.subplots(1, 3, figsize=(17, 6.6), constrained_layout=True)
    from_above(axs[0], inst, np.array(p["flat"]), "A. After pressing, seen from above",
               "print: blue-to-orange across the paper; red = mirror-image.\n"
               "Dashed = where the square should be. 56% lies on other paper")
    det = np.array(r["det"])
    sc = axs[1].scatter(uv[:, 0], uv[:, 1], c=np.clip(det, -1.2, 1.2), cmap="RdBu", vmin=-1.2, vmax=1.2, s=9)
    axs[1].set_aspect("equal"); axs[1].set_xticks([]); axs[1].set_yticks([])
    axs[1].set_xlim(-1.08, 1.08); axs[1].set_ylim(-1.08, 1.08)
    axs[1].set_title("B. Found from the dots alone,\ndrawn on the original paper", fontsize=13)
    axs[1].set_xlabel(f"blue = laid flat the right way up; red = laid flat mirror-image.\n"
                      f"Agrees with the truth on {100 * r['agreement']:.0f}% of dots; the 2 colour borders are the creases",
                      fontsize=10.5)
    cb = fig.colorbar(sc, ax=axs[1], shrink=0.7, pad=0.01)
    cb.set_label("area change of each dot's little patch\n(+1 as-is, -1 mirror-image, 0 squashed)", fontsize=9.5)
    from_above(axs[2], inst, np.array(r["flat_settled"]),
               "C. After flipping across the 2 creases\nand letting the strings settle",
               "Passes: nothing folded; typical dot 0.03% of the side from its true place")
    fig.savefig(OUT / "press_unfold.png", dpi=110)

    fig, ax = plt.subplots(figsize=(11, 4.6), constrained_layout=True)
    for name, colour, label in [("roll", "#2a78d6", "loosely rolled sheet"), ("s_curl", "#eb6834", "S-shaped sheet")]:
        x = next(y for y in rows if y["sheet"] == name and y["links"] == "true")
        inst = make_instance(SHEETS[name], 65, 100)
        uv = inst.uv[inst.order]
        strip = np.abs(uv[:, 1]) < 0.02
        o = np.argsort(uv[strip, 0])
        ax.plot(uv[strip, 0][o], np.array(x["det"])[strip][o], "-o", color=colour, ms=3.5, lw=1.4, label=label)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xlabel("position along one strip across the original paper (one dot spacing = 0.031)", fontsize=11)
    ax.set_ylabel("area change of each dot's patch\nafter pressing", fontsize=11)
    ax.set_title("At a crease the area change jumps from +1 to -1 within 2-4 dots: the crease is sharp, not gradual",
                 fontsize=12.5)
    ax.legend(fontsize=11, loc="lower left")
    ax.set_xlim(-1.02, 1.02)
    fig.savefig(OUT / "crease_profile.png", dpi=110)


if __name__ == "__main__":
    main()
