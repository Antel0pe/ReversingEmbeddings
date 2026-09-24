"""Show a longer fixed-vector extrapolation beside the actual lean path.

Run: python make_long_fixed_vector.py
Outputs: figures/fixed_vector_long_run.png, figures/fixed_vector_long_run.gif
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from lean_axis import Geometry, coverage


GEOMETRY = Geometry(14.5, 14.5, 19.75, 3.0, 0.0)
STEPS = np.arange(19)
ANGLES = -2 + 2 * STEPS
START = coverage(GEOMETRY, -2)
D = coverage(GEOMETRY, 0) - START
FIXED = np.stack([START + n * D for n in STEPS])
TRUE = np.stack([coverage(GEOMETRY, a) for a in ANGLES])
GAPS = np.linalg.norm((FIXED - TRUE).reshape(len(STEPS), -1), axis=1)
INVALID = np.count_nonzero((FIXED < -1e-10) | (FIXED > 1 + 1e-10), axis=(1, 2))


def image_panel(ax, image, change=None):
    ax.imshow(np.clip(image, 0, 1), cmap="gray_r", vmin=0, vmax=1,
              interpolation="nearest")
    if change is not None:
        rgba = np.zeros((*image.shape, 4), dtype=float)
        rgba[change > 1e-8, :3] = (.84, .17, .10)
        rgba[change < -1e-8, :3] = (.11, .35, .84)
        rgba[..., 3] = np.minimum(np.abs(change) / .5, 1) * .55
        ax.imshow(rgba, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaa")
        spine.set_linewidth(.8)


def static_figure():
    shown = [0, 1, 2, 3, 4, 6, 8, 10, 14, 18]
    fig = plt.figure(figsize=(18.5, 16.4), facecolor="white")
    fig.suptitle("Repeat the same vector all the way to a nominal +34°",
                 fontsize=21, fontweight="bold", y=.986)
    fig.text(.5, .955,
             "Each addition uses D = image(0°) − image(−2°). "
             "Red gains ink; blue loses ink. Grayscale is clipped for display only.",
             ha="center", fontsize=11.5, color="#444")
    grid = fig.add_gridspec(5, 5, left=.06, right=.97, bottom=.055, top=.91,
                            height_ratios=[1, 1, 1, 1, .9],
                            hspace=.43, wspace=.25)
    for j, n in enumerate(shown):
        block = 0 if j < 5 else 2
        col = j % 5
        ax = fig.add_subplot(grid[block, col])
        image_panel(ax, FIXED[n], n * D if n else None)
        ax.set_title(f"n = {n}  ·  nominal {ANGLES[n]:+g}°",
                     fontsize=11, fontweight="bold", pad=7)
        ax.set_xlabel(f"raw range {FIXED[n].min():.2f} to {FIXED[n].max():.2f}"
                      f"   ·   {INVALID[n]} invalid", fontsize=9, labelpad=6,
                      color="#ad332d" if INVALID[n] else "#555")
        if col == 0:
            ax.text(-.20, .5, "REPEAT D", transform=ax.transAxes,
                    rotation=90, ha="center", va="center", fontsize=12,
                    fontweight="bold", color="#bb3d32")

        ax = fig.add_subplot(grid[block + 1, col])
        change = TRUE[n] - TRUE[n-1] if n else None
        image_panel(ax, TRUE[n], change)
        ax.set_xlabel(f"true lean {ANGLES[n]:+g}°   ·   L₂ gap {GAPS[n]:.2f}",
                      fontsize=9, labelpad=6)
        if col == 0:
            ax.text(-.20, .5, "TRUE LEAN", transform=ax.transAxes,
                    rotation=90, ha="center", va="center", fontsize=12,
                    fontweight="bold", color="#16825d")

    ax = fig.add_subplot(grid[4, 0:3])
    ax.plot(ANGLES, GAPS, "o-", color="#b53f34", lw=2, markersize=4)
    for n in (2, 10, 18):
        ax.annotate(f"{GAPS[n]:.1f}", (ANGLES[n], GAPS[n]),
                    (ANGLES[n]-2, GAPS[n]+1.7), fontsize=9)
    ax.set(xlabel="nominal lean angle", ylabel="784-pixel L₂ gap",
           title="Distance from the true generated image", xlim=(-2, 34), ylim=(-.8, 29))
    ax.grid(alpha=.2)

    ax = fig.add_subplot(grid[4, 3:5])
    ax.plot(ANGLES, FIXED.min(axis=(1, 2)), "o-", color="#2375af", label="smallest value")
    ax.plot(ANGLES, FIXED.max(axis=(1, 2)), "o-", color="#c73e35", label="largest value")
    ax.axhline(0, color="#555", ls=":")
    ax.axhline(1, color="#555", ls=":")
    ax.set(xlabel="nominal lean angle", ylabel="raw pixel value",
           title="The raw values leave valid coverage [0, 1]", xlim=(-2, 34))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=.2)

    fig.savefig("figures/fixed_vector_long_run.png", dpi=145, facecolor="white")
    plt.close(fig)


def animated_figure():
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.5), facecolor="white")
    fig.subplots_adjust(left=.035, right=.94, bottom=.16, top=.78, wspace=.13)
    left, middle, right = axes
    raw_artist = left.imshow(np.clip(FIXED[0], 0, 1), cmap="gray_r", vmin=0, vmax=1,
                             interpolation="nearest")
    true_artist = middle.imshow(TRUE[0], cmap="gray_r", vmin=0, vmax=1,
                                interpolation="nearest")
    difference_artist = right.imshow(FIXED[0]-TRUE[0], cmap="RdBu_r",
                                     vmin=-2, vmax=2, interpolation="nearest")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    left.set_title("repeated fixed vector", fontsize=11)
    middle.set_title("true generated lean", fontsize=11)
    right.set_title("fixed minus true", fontsize=11)
    fig.colorbar(difference_artist, ax=right, fraction=.047, pad=.025,
                 label="red: excess ink · blue: missing ink")
    header = fig.suptitle("", fontsize=14, fontweight="bold", y=.965)
    footer = fig.text(.5, .055, "", ha="center", fontsize=11, color="#444")

    def update(n):
        raw_artist.set_data(np.clip(FIXED[n], 0, 1))
        true_artist.set_data(TRUE[n])
        difference_artist.set_data(FIXED[n]-TRUE[n])
        header.set_text(f"Add D {n} times from −2°  ·  nominal {ANGLES[n]:+g}°")
        footer.set_text(f"L₂ gap {GAPS[n]:.2f}    ·    {INVALID[n]} invalid pixels    ·    "
                        f"raw range {FIXED[n].min():.2f} to {FIXED[n].max():.2f}")
        return raw_artist, true_artist, difference_artist, header, footer

    animation = FuncAnimation(fig, update, frames=len(STEPS), interval=350,
                              blit=False, repeat=True)
    animation.save("figures/fixed_vector_long_run.gif", writer=PillowWriter(fps=3))
    plt.close(fig)


if __name__ == "__main__":
    static_figure()
    animated_figure()
