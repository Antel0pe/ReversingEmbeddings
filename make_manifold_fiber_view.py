"""Show an exact 3+2 coordinate view of the five-knob generated 1 family.

Run: python make_manifold_fiber_view.py
Output: figures/manifold_fiber_view.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lean_axis import Geometry, coverage
from make_direction_atlas import lean_direction


def unit_lean(g):
    v = lean_direction(g.lean, g)
    return v / np.linalg.norm(v)


def main():
    lean_values = [-10, 0, 10, 20, 35]
    width_values = [1.8, 3.0, 4.6]
    height_values = [19.0, 19.75, 20.5]
    center_values = [14.0, 14.5, 15.0]
    base = Geometry(14.5, 14.5, 19.75, 3.0, 20.0)
    image0 = coverage(base).ravel()
    tangent0 = unit_lean(base)

    center_grid = np.linspace(14, 15, 35)
    distance = np.empty((len(center_grid), len(center_grid)))
    direction_angle = np.empty_like(distance)
    for i, cy in enumerate(center_grid):
        for j, cx in enumerate(center_grid):
            g = Geometry(cx, cy, base.height, base.width, base.lean)
            distance[i, j] = np.linalg.norm(coverage(g).ravel() - image0)
            dot = np.dot(unit_lean(g), tangent0)
            direction_angle[i, j] = np.rad2deg(np.arccos(np.clip(dot, -1, 1)))

    fig = plt.figure(figsize=(17.2, 7.5), facecolor="white")
    fig.suptitle("A five-dimensional manifold as a 3D base with a 2D fiber",
                 fontsize=19, fontweight="bold", y=.98)
    fig.text(.5, .934,
             "Each location in the 3D plot has a whole 2D sheet of different generated images behind it. "
             "The two panels open one of those sheets.",
             ha="center", fontsize=10.8, color="#555")
    gs = fig.add_gridspec(1, 3, left=.04, right=.96, bottom=.15, top=.88,
                          width_ratios=[1.38, 1, 1], wspace=.29)

    ax = fig.add_subplot(gs[0, 0], projection="3d")
    plotted = []
    for cy in center_values:
        for cx in center_values:
            for height in height_values:
                for width in width_values:
                    for lean in lean_values:
                        plotted.append((lean, width, height, cx, cy))
    p = np.asarray(plotted)
    dots = ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=p[:, 3], cmap="viridis",
                      vmin=14, vmax=15, s=7 + 14 * (p[:, 4]-14),
                      alpha=.56, depthshade=False, edgecolors="none")
    ax.scatter([base.lean], [base.width], [base.height], s=190,
               color="#d93e28", edgecolors="white", linewidths=1.1,
               marker="*", depthshade=False, zorder=10)
    ax.set(xlabel="lean θ (degrees)", ylabel="width (px)", zlabel="height (px)",
           title="3D base: lean, width, height")
    ax.set_box_aspect((1.4, 1, .8))
    ax.view_init(elev=22, azim=-59)
    ax.tick_params(labelsize=8)
    cb = fig.colorbar(dots, ax=ax, shrink=.54, pad=.09, location="left")
    cb.set_label("4th knob: cx", fontsize=9)
    ax.text2D(.52, -.13, "9 sampled images share each 3D point",
              transform=ax.transAxes, ha="center", fontsize=9, color="#555")

    extent = (14, 15, 14, 15)
    for index, values, cmap, title, label in [
        (1, distance, "magma", "Image distance within the selected fiber",
         "pixel-space L2 distance from center"),
        (2, direction_angle, "viridis", "Lean direction angle within that fiber",
         "784D tangent angle from center (degrees)"),
    ]:
        ax = fig.add_subplot(gs[0, index])
        im = ax.imshow(values, origin="lower", extent=extent, cmap=cmap,
                       aspect="equal", interpolation="bilinear")
        ax.plot([14.5], [14.5], marker="*", markersize=14,
                markerfacecolor="#e5efef", markeredgecolor="#222")
        ax.set(xlabel="cx", ylabel="cy")
        ax.set_title(title, fontsize=11.2, pad=10)
        ax.set_xticks([14, 14.25, 14.5, 14.75, 15])
        ax.set_yticks([14, 14.25, 14.5, 14.75, 15])
        ax.tick_params(labelsize=8)
        cb = fig.colorbar(im, ax=ax, shrink=.73, pad=.03)
        cb.set_label(label, fontsize=8.5)
        ax.text(.5, -.19, "fixed lean=20°, width=3, height=19.75",
                ha="center", transform=ax.transAxes, fontsize=8.7, color="#555")

    fig.text(.5, .06,
             "The full coordinate is (lean, width, height ; cx, cy). No generated degree of freedom is removed. "
             "Each heatmap color is computed from all 784 pixels or their full lean direction.",
             ha="center", fontsize=9.2, color="#555")
    fig.text(.15, .095, "3D marker size = cy", ha="center", fontsize=9, color="#555")
    fig.savefig("figures/manifold_fiber_view.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
