"""Draw an exact lean path and its geometry in the original 784 coordinates.

Run: python make_exact_lean_figure.py
Output: figures/exact_lean_axis.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import grey_ones
from lean_axis import geometry_from_image, trace_lean


def main():
    # A valid member with pixel-aligned edges at 0 degrees, so its crease is
    # particularly clear. The supplied input image itself is leaned 20 degrees.
    p = np.array([14.5, 14.5, 19.75, 3.0, 20.0])
    x = grey_ones.render(p)[0]
    g = geometry_from_image(x)
    angles, images = trace_lean(x, np.linspace(-10, 35, 181))
    points = images.reshape(len(angles), -1)
    steps = np.diff(points, axis=0)
    step_lengths = np.linalg.norm(steps, axis=1)
    unit = steps / step_lengths[:, None]
    turn = np.degrees(np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1, 1)))
    arc = np.r_[0, np.cumsum(step_lengths)]
    chord = np.linalg.norm(points - points[0], axis=1)

    fig = plt.figure(figsize=(13.5, 9.2), facecolor="white")
    fig.suptitle("One exact lean axis through 784-dimensional pixel space",
                 fontsize=17, fontweight="bold", y=.98)
    fig.text(.5, .938,
             "The input is the boxed 20° image. Every other image follows from its pixels and the angle alone. "
             "The path has exact straight pieces in tan(angle); the graphs use all 784 coordinates.",
             ha="center", fontsize=9.8, color="#555")

    shown = [-10, -5, 0, 5, 10, 15, 20, 27.5, 35]
    for j, angle in enumerate(shown):
        ax = fig.add_axes([.045 + j * .103, .645, .081, .245])
        idx = int(round((angle + 10) * 4))
        ax.imshow(images[idx], cmap="gray_r", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#1565c0" if angle == 20 else "#bbb")
            spine.set_linewidth(2.5 if angle == 20 else .7)
        ax.set_title(f"{angle:g}°" + ("  input" if angle == 20 else ""),
                     fontsize=9.5, color="#1565c0" if angle == 20 else "#222")

    left = fig.add_axes([.075, .115, .385, .435])
    left.plot(angles, arc, color="#222", lw=2.2, label="length travelled along the axis")
    left.plot(angles, chord, color="#1565c0", lw=2, ls="--",
              label="straight distance from −10°")
    left.axvline(g.lean, color="#777", lw=1, ls=":")
    left.set(xlabel="lean angle", ylabel="pixel-space distance",
             title="A curved route is longer than its chord")
    left.legend(fontsize=8.4, loc="upper left")
    left.grid(alpha=.2)

    right = fig.add_axes([.555, .115, .385, .435])
    right.plot(angles[1:-1], turn, color="#c62828", lw=1.6)
    right.axvline(0, color="#333", lw=1, ls=":")
    right.annotate("90° turn when the upright stroke's\nedge crosses a pixel boundary",
                   xy=(0, 90), xytext=(4, 73), fontsize=9,
                   arrowprops=dict(arrowstyle="->", color="#333"))
    right.set(xlabel="lean angle", ylabel="angle between neighbouring steps",
              title="The direction turns inside 784 dimensions", ylim=(0, 100))
    right.grid(alpha=.2)

    fig.text(.5, .035,
             f"Full path length {arc[-1]:.2f}; endpoint distance {chord[-1]:.2f}. "
             "The 90° corner is a coverage event, not a cap on how much the 1 can lean.",
             ha="center", fontsize=9.5, color="#444")
    fig.savefig("figures/exact_lean_axis.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
