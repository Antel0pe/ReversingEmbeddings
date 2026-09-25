"""Analyze joint knob motion and the geometry of normalized direction frames.

Run: python make_joint_direction_analysis.py
Output: figures/joint_direction_analysis.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.stats import qmc
from sklearn.manifold import Isomap, trustworthiness
from sklearn.metrics import pairwise_distances

from grey_ones import RANGES
from make_direction_atlas import image_at, jacobian


BASE = np.array([14.43, 14.62, 19.77, 3.13,
                 np.tan(np.deg2rad(17.3))])
CHANGE = np.array([0, 0, 0, .4,
                   np.tan(np.deg2rad(22.3)) - BASE[4]])


def direction_frame(q, step=1e-6):
    j = jacobian(q, step=step)
    return (j / np.linalg.norm(j, axis=0)).T.reshape(-1)


def local_spectrum(func, q, step=1e-5):
    columns = []
    for j in range(5):
        hi, lo = q.copy(), q.copy()
        hi[j] += step
        lo[j] -= step
        columns.append((func(hi) - func(lo)) / (2 * step))
    return np.linalg.svd(np.stack(columns, axis=1), compute_uv=False)


def redblue(ax, values, limit=.9):
    ax.imshow(values.reshape(28, 28), cmap="RdBu_r",
              norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
              interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#aaa")
        spine.set_linewidth(.8)


def main():
    x = image_at(BASE)
    J = jacobian(BASE, step=1e-6)
    width_component = CHANGE[3] * J[:, 3]
    lean_component = CHANGE[4] * J[:, 4]
    local_sum = width_component + lean_component
    actual = image_at(BASE + CHANGE) - x
    residual = actual - local_sum

    fractions = np.linspace(0, 1, 41)
    linear_error = []
    endpoint_error = []
    for t in fractions:
        d = t * CHANGE
        target = image_at(BASE + d)
        linear_error.append(np.linalg.norm(x + J @ d - target))
        width_only = image_at(BASE + np.array([0, 0, 0, d[3], 0]))
        lean_only = image_at(BASE + np.array([0, 0, 0, 0, d[4]]))
        endpoint_error.append(np.linalg.norm(width_only + lean_only - x - target))

    n = 250
    params = qmc.LatinHypercube(5, seed=2026).random(n)
    params = params * (RANGES[:, 1] - RANGES[:, 0]) + RANGES[:, 0]
    q_samples = params.copy()
    q_samples[:, 4] = np.tan(np.deg2rad(params[:, 4]))
    images = np.stack([image_at(q) for q in q_samples])
    frames = np.stack([direction_frame(q, step=1e-5) for q in q_samples])
    image_dist = pairwise_distances(images)
    frame_dist = pairwise_distances(frames)
    image_2d = Isomap(n_neighbors=30, n_components=2,
                      metric="precomputed").fit_transform(image_dist)
    frame_2d = Isomap(n_neighbors=30, n_components=2,
                      metric="precomputed").fit_transform(frame_dist)
    image_trust = trustworthiness(images, image_2d, n_neighbors=10)
    frame_trust = trustworthiness(frames, frame_2d, n_neighbors=10)
    k = 10
    image_knn = np.argsort(image_dist, axis=1)[:, 1:k+1]
    frame_knn = np.argsort(frame_dist, axis=1)[:, 1:k+1]
    neighbor_overlap = np.mean([len(set(a) & set(b)) / k
                                for a, b in zip(image_knn, frame_knn)])

    image_s = np.linalg.svd(J, compute_uv=False)
    frame_s = local_spectrum(direction_frame, BASE)
    lean_s = local_spectrum(lambda q: direction_frame(q)[4*784:], BASE)

    fig = plt.figure(figsize=(17.5, 14.0), facecolor="white")
    fig.suptitle("Moving two knobs together, then comparing images with directions",
                 fontsize=19, fontweight="bold", y=.985)
    fig.text(.5, .954,
             "At one image, local changes add. Across a finite move, the vectors change "
             "as pixel boundaries are crossed.", ha="center", fontsize=11, color="#555")
    gs = fig.add_gridspec(3, 5, left=.06, right=.95, bottom=.075, top=.91,
                          height_ratios=[1, 1.28, .87], hspace=.49, wspace=.30)

    changes = [width_component, lean_component, local_sum, actual, residual]
    titles = ["+0.4 width component", "+5° lean component",
              "local sum of vectors", "actual joint change", "actual − local sum"]
    for j, (values, title) in enumerate(zip(changes, titles)):
        ax = fig.add_subplot(gs[0, j])
        redblue(ax, values, .9)
        ax.set_title(title, fontsize=10.3, pad=6)
        ax.set_xlabel(f"L₂ size {np.linalg.norm(values):.2f}",
                      fontsize=9, color="#555")

    for slot, data, title, trust in [
        (gs[1, 0:2], image_2d, "250 generated images", image_trust),
        (gs[1, 2:4], frame_2d, "their five normalized direction fields", frame_trust),
    ]:
        ax = fig.add_subplot(slot)
        scatter = ax.scatter(data[:, 0], data[:, 1], c=params[:, 4],
                             cmap="coolwarm", vmin=-10, vmax=35,
                             s=24, alpha=.85, linewidths=0)
        ax.set(title=f"Isomap of {title}", xlabel="2D map axis 1",
               ylabel="2D map axis 2")
        ax.grid(alpha=.15)
        ax.text(.03, .04, f"10-neighbor trustworthiness {trust:.3f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor="white", alpha=.9, edgecolor="#ddd", pad=4))
    cb = fig.colorbar(scatter, ax=fig.axes[-1], fraction=.05, pad=.02)
    cb.set_label("lean angle", fontsize=9)

    ax = fig.add_subplot(gs[1, 4])
    ax.axis("off")
    ax.text(.03, .90, "WHAT THE MAPS SAY", fontsize=11,
            fontweight="bold", va="top")
    ax.text(.03, .71,
            f"Shared 10-neighbor fraction:\n{neighbor_overlap:.1%}\n\n"
            "The same images are colored\nby their actual lean angle.\n\n"
            "Isomap shows neighborhoods;\nit cannot prove dimension.",
            fontsize=10, va="top", linespacing=1.5)

    ax = fig.add_subplot(gs[2, 0:3])
    ax.plot(fractions, linear_error, color="#b94335", lw=2.1,
            label="one starting tangent sum")
    ax.plot(fractions, endpoint_error, color="#2d6caf", lw=2,
            label="sum of two separate finite effects")
    ax.set(xlabel="fraction of (+0.4 width, +5° lean) move",
           ylabel="L₂ error against actual joint image",
           title="Why local vector addition must be updated")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9, loc="upper left")

    ax = fig.add_subplot(gs[2, 3:5])
    ax.plot(range(1, 6), image_s/image_s[0], "o-", color="#2c6ca0",
            label="image map F(q)")
    ax.plot(range(1, 6), frame_s/frame_s[0], "o-", color="#b94335",
            label="full unit-direction frame H(q)")
    ax.plot(range(1, 6), np.maximum(lean_s/lean_s[0], 1e-6), "o--",
            color="#76539a", label="lean unit vector alone")
    ax.set(xlabel="local singular direction", ylabel="relative response",
           title="Local rank at a generic interior image",
           xticks=np.arange(1, 6), yscale="log", ylim=(1e-6, 2))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.text(.98, .89, "image: rank 5\nfull frame: rank 5\nlean alone: rank 2",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="#ddd", pad=4))

    fig.text(.5, .027,
             "Local ranks use finite differences at one generic point; UMAP is unavailable, "
             "so the 2D views use Isomap. All distance and rank calculations use the full pixel or frame vectors.",
             ha="center", fontsize=9.2, color="#555")
    fig.savefig("figures/joint_direction_analysis.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
