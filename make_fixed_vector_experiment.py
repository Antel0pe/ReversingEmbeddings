"""Plot repeated pixel-vector addition against the exact generated lean path.

Run: python make_fixed_vector_experiment.py
Output: figures/fixed_vector_vs_lean.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from lean_axis import Geometry, coverage


G = Geometry(cx=14.5, cy=14.5, height=19.75, width=3.0, lean=0.0)
RED = "#c73e35"
BLUE = "#2375af"
GREEN = "#16825d"


def tinted_image(ax, image, change=None):
    # Raw extrapolations can leave [0, 1]. Clip only their display, not data.
    ax.imshow(np.clip(image, 0, 1), cmap="gray_r", vmin=0, vmax=1,
              interpolation="nearest")
    if change is not None:
        rgba = np.zeros((*image.shape, 4), dtype=float)
        pos = change > 1e-8
        neg = change < -1e-8
        rgba[pos, :3] = (0.78, 0.16, 0.12)
        rgba[neg, :3] = (0.10, 0.38, 0.80)
        rgba[..., 3] = np.minimum(np.abs(change) / .32, 1) * .57
        ax.imshow(rgba, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#888")
        spine.set_linewidth(.8)


def main():
    start = coverage(G, -2)
    corner = coverage(G, 0)
    d = corner - start
    n_values = np.arange(6)
    nominal_angles = -2 + 2 * n_values
    straight = np.stack([start + n * d for n in n_values])
    true = np.stack([coverage(G, a) for a in nominal_angles])
    assert np.allclose(straight[0], true[0]) and np.allclose(straight[1], true[1])

    fig = plt.figure(figsize=(20, 8.7), facecolor="white")
    fig.suptitle("What happens if we keep adding the same 784-pixel vector?",
                 fontsize=21, fontweight="bold", y=.982)
    fig.text(.5, .94,
             "Start at −2°, reach 0°, then repeat D = image(0°) − image(−2°). "
             "Red pixels gain ink (darken); blue pixels lose ink (lighten).",
             ha="center", fontsize=12, color="#444")
    outer = fig.add_gridspec(2, 2, width_ratios=[3.45, 1],
                             left=.045, right=.975, bottom=.12, top=.87,
                             wspace=.09, hspace=.18)
    left = outer[:, 0].subgridspec(2, 6, wspace=.16, hspace=.30)

    for n, angle in zip(n_values, nominal_angles):
        ax = fig.add_subplot(left[0, n])
        tinted_image(ax, straight[n], d if n else None)
        invalid = int(np.count_nonzero((straight[n] < -1e-10) |
                                        (straight[n] > 1 + 1e-10)))
        title = "start: −2°" if n == 0 else "0°: on the axis" if n == 1 \
            else f"add D again ×{n-1}"
        ax.set_title(title, fontsize=10.4, pad=5, fontweight="bold" if n < 2 else None)
        ax.set_xlabel(f"{invalid} invalid pixels" if n > 1 else "valid generated image",
                      fontsize=9, color=RED if invalid else "#555", labelpad=5)
        if n == 0:
            ax.text(-.18, .5, "FIXED\nVECTOR", rotation=90,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, fontweight="bold", color=RED)

        ax = fig.add_subplot(left[1, n])
        prev = true[n - 1] if n else None
        tinted_image(ax, true[n], true[n] - prev if prev is not None else None)
        ax.set_title(f"true lean: {angle:g}°", fontsize=10.4, pad=5)
        error = np.linalg.norm(straight[n] - true[n])
        ax.set_xlabel(f"L₂ gap: {error:.2f}" if n > 1
                      else "same image as above", fontsize=9, labelpad=5)
        if n == 0:
            ax.text(-.18, .5, "EXACT\nLEAN", rotation=90,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, fontweight="bold", color=GREEN)

    right = outer[:, 1].subgridspec(2, 1, hspace=.36)
    ax = fig.add_subplot(right[0])
    im = ax.imshow(d, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-.34, vcenter=0, vmax=.34),
                   interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("The one vector D, reused each time", fontsize=11, fontweight="bold", pad=8)
    fig.colorbar(im, ax=ax, fraction=.05, pad=.025, label="change in ink coverage")
    ax.text(.5, -.09, "Red = more black   ·   blue = less black",
            transform=ax.transAxes, ha="center", fontsize=9, color="#444")

    ax = fig.add_subplot(right[1])
    # A literal projection of all 784 coordinates on the two unit step vectors.
    before = d.ravel() / np.linalg.norm(d)
    after = (coverage(G, 2) - corner).ravel()
    after /= np.linalg.norm(after)
    sampled_angles = np.linspace(-2, 2, 101)
    sampled = np.stack([coverage(G, angle).ravel() - corner.ravel()
                        for angle in sampled_angles])
    coords = sampled @ np.stack([before, after], axis=1)
    extrapolated = (straight[:4].reshape(4, -1) - corner.ravel()) @ np.stack([before, after], axis=1)
    ax.plot(coords[:, 0], coords[:, 1], color=GREEN, lw=3,
            label="actual generated axis")
    ax.plot(extrapolated[:, 0], extrapolated[:, 1], "o--", color=RED, lw=2,
            label="keep adding D")
    ax.scatter([0], [0], s=70, c="black", zorder=5)
    ax.annotate("0° corner", (0, 0), (.08, -.23), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#333"))
    ax.set(xlabel="direction before 0°", ylabel="direction after 0°",
           title="The 90° turn in pixel space", xlim=(-1.4, 2.7), ylim=(-.45, 1.5))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8.8, loc="upper left")
    ax.set_aspect("equal", adjustable="box")

    fig.text(.5, .035,
             "Values outside [0, 1] are clipped only to draw the grayscale images. "
             "The calculations, distances, and invalid-pixel counts use the raw values.",
             ha="center", fontsize=10, color="#555")
    fig.savefig("figures/fixed_vector_vs_lean.png", dpi=155, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
