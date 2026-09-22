"""The five intrinsic axes drawn like graph axes: tick marks, and what each tick IS in 784-d.

Each row is one knob, starting from the plain upright 1 and stepping the knob in its
own units (pixels, or degrees for lean). Under each tick: the knob value and the
straight-line distance back to the first tick. Between ticks: the step image
(what has to be added to get from one tick to the next; red = add ink, blue =
remove ink) and its size.

Run: python make_tick_axes.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import grey_ones as G

OUT = "figures"
INK, GOOD, BAD, REF = "#1a1a1a", "#1a7f37", "#c62828", "#1565c0"
C = np.array([14.5, 14.5, 19.75, 3.2, 0.0])
TICKS = [("cx", 0, [12, 13, 14, 15, 16, 17], "px"),
         ("cy", 1, [12, 13, 14, 15, 16, 17], "px"),
         ("height", 2, [14, 16, 18, 20, 22, 24], "px"),
         ("width", 3, [1, 2, 3, 4, 5, 6], "px"),
         ("lean", 4, [-10, 0, 10, 20, 30, 40], "°")]


def main():
    fig = plt.figure(figsize=(16.2, 11.4))
    fig.text(.5, .972, "The five intrinsic axes with tick marks — what each tick is in 784-d, and how you get to the next",
             ha="center", va="center", fontsize=15, fontweight="bold")
    fig.text(.5, .938, "Each tick is a knob value in its own units. Plugging it into the renderer gives one full 784-pixel image (black). "
                       "Between ticks, the step image is what you must add\nto go from one tick to the next (red = add ink, blue = remove). "
                       "“gap” = size of that step.   “from start” = straight-line distance back to the first tick.",
             ha="center", va="center", fontsize=9.5, color="#555", linespacing=1.55)
    L, T, B = .075, .895, .06
    rh = (T - B) / 5
    tw, sw = .0605, .0605
    pitch = .0785
    for r, (name, k, ticks, unit) in enumerate(TICKS):
        X = []
        for t in ticks:
            p = C.copy(); p[k] = t; X.append(G.render(p).ravel().astype(np.float64))
        X = np.array(X)
        y0 = T - (r + 1) * rh
        fig.text(L - .012, y0 + rh * .56, f"{name}\naxis", ha="right", va="center",
                 fontsize=11.5, fontweight="bold", linespacing=1.3)
        for c, t in enumerate(ticks):
            ax = fig.add_axes([L + 2 * c * pitch, y0 + rh * .30, tw, rh * .60])
            ax.imshow(X[c].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_edgecolor(INK); s.set_linewidth(1.1)
            ax.text(.5, -.06, f"{name} = {t}{unit}", transform=ax.transAxes, ha="center",
                    va="top", fontsize=8.6, fontweight="bold")
            ax.text(.5, -.24, f"from start {np.linalg.norm(X[c]-X[0]):.2f}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=7.8, color="#555")
            if c < len(ticks) - 1:
                d = X[c + 1] - X[c]; m = max(np.abs(d).max(), 1e-9)
                ax2 = fig.add_axes([L + (2 * c + 1) * pitch, y0 + rh * .36, sw * .80, rh * .48])
                ax2.imshow(d.reshape(28, 28), cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
                ax2.set_xticks([]); ax2.set_yticks([])
                for s in ax2.spines.values(): s.set_edgecolor("#ccc"); s.set_linewidth(.6)
                ax2.text(.5, 1.05, "→ step →", transform=ax2.transAxes, ha="center",
                         va="bottom", fontsize=7.4, color="#888")
                ax2.text(.5, -.08, f"gap {np.linalg.norm(d):.2f}", transform=ax2.transAxes,
                         ha="center", va="top", fontsize=8.2, fontweight="bold", color=REF)
    fig.text(.5, .026, "cx, cy, height, width: every gap is identical — an even ruler. lean: the gaps grow (4.85 → 5.91) — an uneven ruler, "
                       "like lines of longitude. Every row: “from start” grows\nmore slowly than the gaps add up, and for cx it stops "
                       "growing at all (10.24) — once the stroke no longer overlaps where it began, walking further along the axis cannot take you further away in a straight line.",
             ha="center", va="center", fontsize=9.1, color="#333", style="italic", linespacing=1.6)
    fig.savefig(f"{OUT}/tick_axes.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
