"""Explain shared pixel support versus shared five-dimensional motion axes.

Run: python make_shared_motion_axes.py
Outputs: figures/shared_pixel_motion.png, figures/five_axis_decomposition.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

from grey_ones import N_PIX, SUB
from lean_axis import Geometry, coverage


OUT = Path("figures")
OUT.mkdir(exist_ok=True)
RED = "#b64035"
BLUE = "#346eaa"
INK = "#28323c"


def image(ax, values, *, limit=None, outline=None):
    if limit is None:
        ax.imshow(values.reshape(28, 28), cmap="gray_r", vmin=0, vmax=1,
                  interpolation="nearest")
    else:
        ax.imshow(values.reshape(28, 28), cmap="RdBu_r",
                  norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                  interpolation="nearest")
    if outline is not None:
        ax.contour(outline.reshape(28, 28), levels=[.5], colors=[INK], linewidths=.65)
    ax.set_xticks([])
    ax.set_yticks([])


def exact_local_jacobian(q):
    """Analytic five-column derivative with q=(cx,cy,h,w,tan(lean)).

    At this generic example no stroke edge or vertical boundary lies exactly
    on a pixel/subrow boundary, so the ordinary local derivative is defined.
    """
    cx, cy, h, w, u = q
    ylo = np.arange(N_PIX * SUB, dtype=float) / SUB
    yhi = ylo + 1 / SUB
    ymid = (ylo + yhi) / 2
    top, bottom = cy - h/2, cy + h/2
    vertical = np.clip((np.minimum(yhi, bottom)-np.maximum(ylo, top))*SUB, 0, 1)
    dvert_cy = SUB * ((ylo < bottom) & (bottom < yhi)).astype(float)
    dvert_cy -= SUB * ((ylo < top) & (top < yhi)).astype(float)
    dvert_h = SUB/2 * (((ylo < bottom) & (bottom < yhi))
                       + ((ylo < top) & (top < yhi)))

    left = cx - w/2 - u*(ymid-cy)
    right = left + w
    col = np.arange(N_PIX)[None, :]
    overlap = np.clip(np.minimum(right[:, None], col+1)
                      - np.maximum(left[:, None], col), 0, 1)
    left_active = ((col < left[:, None]) & (left[:, None] < col+1)).astype(float)
    right_active = ((col < right[:, None]) & (right[:, None] < col+1)).astype(float)
    common_shift = right_active-left_active
    horizontal = np.stack([
        common_shift,
        u*common_shift,
        np.zeros_like(common_shift),
        .5*(left_active+right_active),
        -(ymid-cy)[:, None]*common_shift,
    ], axis=-1)
    dimage = vertical[:, None, None]*horizontal
    dimage[:, :, 1] += dvert_cy[:, None]*overlap
    dimage[:, :, 2] += dvert_h[:, None]*overlap
    return dimage.reshape(N_PIX, SUB, N_PIX, 5).mean(axis=1).reshape(784, 5)


def edge_card(ax, first, second, title, *, delta=False):
    vals = np.array([first, second])
    if delta:
        ax.imshow(vals[None, :], cmap="RdBu_r", vmin=-.1, vmax=.1,
                  extent=(0, 2, 0, 1), interpolation="nearest", aspect="auto")
    else:
        ax.imshow(vals[None, :], cmap="gray_r", vmin=0, vmax=1,
                  extent=(0, 2, 0, 1), interpolation="nearest", aspect="auto")
    ax.axvline(1, color="#777", lw=1)
    for i, val in enumerate(vals):
        ax.text(i+.5, .5, f"{val:+.1f}" if delta else f"{val:.1f}",
                ha="center", va="center", fontsize=15, fontweight="bold",
                color="white" if (abs(val)>.065 if delta else val>.6) else INK)
    ax.set(xlim=(0, 2), ylim=(0, 1), title=title)
    ax.set_xticks([.5, 1.5], labels=["left edge pixel", "right edge pixel"])
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=9)
    ax.spines[:].set_visible(False)


def make_shared_pixel_motion():
    # A single generator-style horizontal subrow, width 3 and left edge 12.8.
    # A +0.2 width change moves each edge outward 0.1 pixels.
    # A lean-slope change of 1/60 moves this upper slice right 0.1 pixels.
    baseline = (.2, .8)
    width_after = (.3, .9)
    lean_after = (.1, .9)
    fig = plt.figure(figsize=(14.5, 8.2), facecolor="white")
    fig.suptitle("The same pixels can move in different directions",
                 y=.979, fontsize=19, fontweight="bold")
    fig.text(.5, .934,
             "Zoom into one thin upper slice of a generated 1. Only its two partly covered edge pixels respond to these small moves.",
             ha="center", fontsize=10.4, color="#555")
    gs = fig.add_gridspec(2, 3, left=.055, right=.965, bottom=.19, top=.86,
                          height_ratios=[1, 1.05], width_ratios=[.9, 1, 1],
                          hspace=.45, wspace=.25)
    ax = fig.add_subplot(gs[:, 0])
    example_angle = np.rad2deg(np.arctan((12.8-13)/6))
    full = coverage(Geometry(14.5, 14.5, 19.75, 3, example_angle))
    image(ax, full)
    ax.axhspan(8-.5, 8+.5, facecolor="none", edgecolor="#d5842c", lw=2.2)
    ax.set_title("1 · The whole 28×28 stroke", fontsize=11.5, pad=8)
    ax.text(.5, -.06, "Orange band: the magnified upper slice", ha="center",
            transform=ax.transAxes, fontsize=9.6, color="#555")

    ax = fig.add_subplot(gs[0, 1])
    edge_card(ax, *baseline, "2 · Starting ink fractions")
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    ax.text(.04, .93,
            "Both operations touch the same\nleft and right edge pixels.",
            fontsize=12.3, fontweight="bold", va="top", color=INK,
            linespacing=1.4)
    ax.text(.04, .58,
            "A vector is the change in pixel values.\n"
            "The 784D vectors are sparse, so here\n"
            "we show just these two entries.",
            fontsize=10.5, va="top", color="#555", linespacing=1.5)
    ax = fig.add_subplot(gs[1, 1])
    edge_card(ax, width_after[0]-baseline[0], width_after[1]-baseline[1],
              "3 · Widen: both edges gain ink", delta=True)
    ax.text(.5, -.19, "width move = (+0.1, +0.1)", transform=ax.transAxes,
            ha="center", fontsize=10.4, color=INK)
    ax = fig.add_subplot(gs[1, 2])
    edge_card(ax, lean_after[0]-baseline[0], lean_after[1]-baseline[1],
              "4 · Lean: one edge loses, one gains", delta=True)
    ax.text(.5, -.19, "lean move = (−0.1, +0.1)", transform=ax.transAxes,
            ha="center", fontsize=10.4, color=INK)
    fig.text(.5, .107,
             "The two moves use exactly the same pixels, yet (+0.1,+0.1) · (−0.1,+0.1) = 0: their directions are perpendicular.",
             ha="center", fontsize=10.7, color=INK)
    fig.text(.5, .07,
             "For a complete generated 1 at the example used below, width and lean share all 52 responding pixels; their normalized dot product is −0.028.",
             ha="center", fontsize=9.9, color="#555")
    fig.savefig(OUT / "shared_pixel_motion.png", dpi=155, facecolor="white")
    plt.close(fig)


def make_five_axis_decomposition():
    q = np.array([14.43, 14.62, 19.77, 3.13, np.tan(np.deg2rad(17.3))])
    original = coverage(Geometry(q[0], q[1], q[2], q[3], 17.3)).ravel()
    J = exact_local_jacobian(q)
    H = J / np.linalg.norm(J, axis=0)[None, :]
    gram = H.T @ H
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    inverse_root = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    modes = H @ inverse_root
    coefficients = root.T  # row = knob; column = orthogonal pixel mode
    mode_limit = np.max(np.abs(modes))
    names = ["move right", "move down", "grow taller", "grow wider", "lean more"]

    fig = plt.figure(figsize=(17.5, 11.3), facecolor="white")
    fig.suptitle("One exact answer to 'which five motions does each knob use?'",
                 y=.985, fontsize=18.5, fontweight="bold")
    fig.text(.5, .949,
             "At this one generated 1, choose five mutually perpendicular 784-pixel motion patterns. "
             "Each knob move is an exact weighted sum of all five.",
             ha="center", fontsize=10.6, color="#555")
    top = fig.add_gridspec(1, 6, left=.047, right=.96, bottom=.61, top=.887,
                           wspace=.18)
    ax = fig.add_subplot(top[0, 0])
    image(ax, original)
    ax.set_title("starting 1", fontsize=11, pad=7)
    ax.text(.5, -.10, "same pixel positions\nin every panel",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    for i in range(5):
        ax = fig.add_subplot(top[0, i+1])
        image(ax, modes[:, i], limit=mode_limit, outline=original)
        ax.set_title(f"motion axis {i+1}", fontsize=11, pad=7)
        ax.text(.5, -.10, "red = gains ink\nblue = loses ink",
                transform=ax.transAxes, ha="center", fontsize=9, color="#555")

    lower = fig.add_gridspec(1, 2, left=.10, right=.93, bottom=.17, top=.53,
                             width_ratios=[1, 1], wspace=.32)
    ax = fig.add_subplot(lower[0, 0])
    heat = ax.imshow(coefficients, cmap="RdBu_r", vmin=-1, vmax=1,
                     interpolation="nearest", aspect="equal")
    ax.set_xticks(range(5), labels=[str(i) for i in range(1, 6)])
    ax.set_yticks(range(5), labels=names)
    ax.set(xlabel="orthogonal pixel motion axis above",
           title="How much of each axis a knob uses")
    for row in range(5):
        for col in range(5):
            value = coefficients[row, col]
            ax.text(col, row, f"{value:+.3f}", ha="center", va="center",
                    fontsize=9.5, fontweight="bold" if abs(value)>.2 else "normal",
                    color="white" if abs(value)>.65 else INK)
    ax.add_patch(Rectangle((-.5, 3-.5), 5, 2, fill=False,
                           edgecolor="#d5842c", linewidth=2.1))
    cb = fig.colorbar(heat, ax=ax, shrink=.74, pad=.04)
    cb.set_label("signed coefficient", fontsize=9)

    ax = fig.add_subplot(lower[0, 1])
    gmap = ax.imshow(gram, cmap="RdBu_r", vmin=-1, vmax=1,
                    interpolation="nearest", aspect="equal")
    ax.set_xticks(range(5), labels=names, rotation=38, ha="right")
    ax.set_yticks(range(5), labels=names)
    ax.set_title("How much the knob moves align with each other", fontsize=10.7)
    for row in range(5):
        for col in range(5):
            value = gram[row, col]
            ax.text(col, row, f"{value:+.3f}", ha="center", va="center",
                    fontsize=9.5, color="white" if abs(value)>.65 else INK)
    cb = fig.colorbar(gmap, ax=ax, shrink=.74, pad=.04)
    cb.set_label("dot product: +1 same, 0 perpendicular, −1 opposite", fontsize=8.7)

    fig.text(.5, .095,
             "Read one row of the left table: multiply each numbered pixel pattern above by its coefficient, then add all five patterns.",
             ha="center", fontsize=10.3, color=INK)
    fig.text(.5, .064,
             "All five axes are retained. Numbers are rounded for display; the analytic local derivatives and full coefficients were used in the computation.",
             ha="center", fontsize=9.6, color="#555")
    fig.text(.5, .036,
             "This symmetric orthogonal basis is a choice, not a unique intrinsic labelling. The right table is basis-independent.",
             ha="center", fontsize=9.3, color="#666")
    fig.savefig(OUT / "five_axis_decomposition.png", dpi=155, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make_shared_pixel_motion()
    make_five_axis_decomposition()
