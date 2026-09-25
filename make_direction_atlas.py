"""Visualize local pixel-space directions of the generated grey-1 family.

Run: python make_direction_atlas.py
Output: figures/direction_atlas.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from lean_axis import Geometry, coverage


def image_at(q):
    """q = (cx, cy, height, width, tan(lean)); output is 784 pixel values."""
    angle = np.rad2deg(np.arctan(q[4]))
    return coverage(Geometry(*q[:4], angle)).ravel()


def jacobian(q, step=1e-5):
    """Five local derivatives, with lean parameterized by its tangent."""
    columns = []
    for j in range(5):
        hi, lo = q.copy(), q.copy()
        hi[j] += step
        lo[j] -= step
        columns.append((image_at(hi) - image_at(lo)) / (2 * step))
    return np.stack(columns, axis=1)


def lean_direction(angle, geometry, step=1e-6):
    u = np.tan(np.deg2rad(angle))
    q = np.array([geometry.cx, geometry.cy, geometry.height, geometry.width, u])
    hi, lo = q.copy(), q.copy()
    hi[4] += step
    lo[4] -= step
    return (image_at(hi) - image_at(lo)) / (2 * step)


def direction_image(ax, v, scale):
    v = v / np.linalg.norm(v)
    ax.imshow(v.reshape(28, 28), cmap="RdBu_r",
              norm=TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale),
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#aaa")
        spine.set_linewidth(.8)


def main():
    # A generic interior point avoids an edge/pixel-boundary coincidence.
    q = np.array([14.43, 14.62, 19.77, 3.13, np.tan(np.deg2rad(17.3))])
    J = jacobian(q)
    singular = np.linalg.svd(J, compute_uv=False)

    # The lower panels follow one fixed geometry across its entire lean range.
    geometry = Geometry(14.5, 14.5, 19.75, 3.0, 0.0)
    shown_angles = [-2, 2, 8, 20, 30]
    selected = [lean_direction(a, geometry) for a in shown_angles]
    # Midpoints avoid measuring a centered derivative exactly at the 0° corner.
    angle_grid = np.linspace(-9.875, 34.875, 180)
    V = np.stack([lean_direction(a, geometry) for a in angle_grid])
    speed = np.linalg.norm(V, axis=1)
    unit = V / speed[:, None]
    cosine = unit @ unit.T
    before = lean_direction(-2, geometry)
    after = lean_direction(2, geometry)
    corner_cosine = np.dot(before, after) / (np.linalg.norm(before) * np.linalg.norm(after))
    corner_angle = np.rad2deg(np.arccos(np.clip(corner_cosine, -1, 1)))

    fig = plt.figure(figsize=(17.5, 13.0), facecolor="white")
    fig.suptitle("The generated manifold, viewed through its directions",
                 fontsize=21, fontweight="bold", y=.987)
    fig.text(.5, .954,
             "Each colored tile is a 784-component vector. Red pixels gain ink; blue pixels lose ink. "
             "Tiles are unit length, so color shows direction rather than speed.",
             ha="center", fontsize=11.2, color="#444")
    gs = fig.add_gridspec(3, 6, left=.055, right=.955, bottom=.075, top=.91,
                          height_ratios=[1, 1, 1.43], hspace=.38, wspace=.28)

    knobs = ["move right (cx)", "move down (cy)", "increase height",
             "increase width", "increase lean (tan θ)"]
    for j, label in enumerate(knobs):
        ax = fig.add_subplot(gs[0, j])
        direction_image(ax, J[:, j], .42)
        ax.set_title(label, fontsize=10.5, pad=6)
        ax.set_xlabel(f"raw speed {np.linalg.norm(J[:, j]):.2f}",
                      fontsize=9, color="#666", labelpad=4)
    ax = fig.add_subplot(gs[0, 5])
    ax.axis("off")
    ax.text(.05, .82, "FIVE DIRECTIONS\nAT ONE IMAGE", fontsize=12,
            fontweight="bold", color="#333", va="top")
    ax.text(.05, .58,
            "They are independent here:\n"
            f"Jacobian rank = {np.count_nonzero(singular > 1e-7)}.\n\n"
            "Any small knob change\ncombines these five vectors.",
            fontsize=10, va="top", linespacing=1.5)

    for j, (angle, v) in enumerate(zip(shown_angles, selected)):
        ax = fig.add_subplot(gs[1, j])
        direction_image(ax, v, .28)
        ax.set_title(f"lean at {angle:+g}°", fontsize=10.5, pad=6)
        ax.set_xlabel(f"raw speed {np.linalg.norm(v):.2f}",
                      fontsize=9, color="#666", labelpad=4)
    ax = fig.add_subplot(gs[1, 5])
    ax.axis("off")
    ax.text(.05, .82, "ONE KNOB,\nCHANGING DIRECTION", fontsize=12,
            fontweight="bold", color="#333", va="top")
    ax.text(.05, .53,
            "The −2° and +2°\ndirections are orthogonal\n"
            f"(angle {corner_angle:.1f}°).\n\n"
            "The changed pixels\nswitch at the 0° corner.",
            fontsize=10, va="top", linespacing=1.5)

    ax = fig.add_subplot(gs[2, :2])
    ax.plot(angle_grid, speed, color="#b8463b", lw=1.8)
    ax.axvline(0, color="#777", lw=1, ls=":")
    ax.set(xlabel="lean angle (degrees)",
           ylabel="||d(image)/d(tan θ)||",
           title="Speed: size of the pixel-change vector",
           xlim=(-10, 35), ylim=(0, 41))
    ax.grid(alpha=.2)
    ax.text(.04, .10,
            "Speed depends on the units used for the knob.\n"
            "Normalizing removes that choice.", transform=ax.transAxes,
            fontsize=9, bbox=dict(facecolor="white", edgecolor="#ddd", pad=5))

    ax = fig.add_subplot(gs[2, 2:6])
    im = ax.imshow(cosine, cmap="PuOr", vmin=-1, vmax=1,
                   origin="lower", extent=(-10, 35, -10, 35),
                   interpolation="nearest", aspect="equal")
    ax.set(xlabel="lean at first image (degrees)",
           ylabel="lean at second image (degrees)",
           title="Cosine similarity of 784-dimensional directions")
    ax.axhline(0, color="white", lw=.8, alpha=.8)
    ax.axvline(0, color="white", lw=.8, alpha=.8)
    cb = fig.colorbar(im, ax=ax, fraction=.038, pad=.02)
    cb.set_label("cosine: +1 same · 0 perpendicular · −1 opposite", fontsize=9)
    ax.text(.02, .02, "Each cell compares full 784-pixel vectors; no projection.",
            transform=ax.transAxes, fontsize=9, color="#333",
            bbox=dict(facecolor="white", alpha=.9, edgecolor="none", pad=4))

    fig.text(.5, .028,
             "The top row uses one generic five-knob image. The other panels hold four knobs fixed "
             "and vary lean; derivatives are with respect to tan(angle). Raw speeds depend on knob units.",
             ha="center", fontsize=9.7, color="#555")
    fig.savefig("figures/direction_atlas.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
