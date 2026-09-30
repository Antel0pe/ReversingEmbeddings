"""Explain and plot nearest-neighbor preservation in the generated-one 3D maps.

Run after the neighbor benchmark: python make_generated_one_neighbor_figure.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from benchmark_generated_one_neighbors import COORDS, FIGURES, exact_neighbors
from grey_ones import render
from make_generated_one_3d import make_samples


OUT = FIGURES / "generated_one_neighbor_comparison.png"
DISPLAY = (
    ("TSNE-p2", "t-SNE, perplexity 2", "#af4c45"),
    ("TSNE-p5", "t-SNE, perplexity 5", "#d18834"),
    ("MultiTSNE-p5-50-w0.5", "multi-scale, 5 + 50", "#267b78"),
    ("Rank-refined Isomap", "refined Isomap", "#4c649a"),
    ("Isomap", "original Isomap", "#89949e"),
)


def recall_curve(points, true_neighbors, max_k=30):
    projected = cKDTree(points).query(points, k=max_k + 1, workers=-1)[1][:, 1:]
    hits = np.zeros(max_k + 1, dtype=np.float64)
    for original, display in zip(true_neighbors, projected):
        display_rank = {int(j): rank for rank, j in enumerate(display, 1)}
        for rank, image in enumerate(original[:max_k], 1):
            low_rank = display_rank.get(int(image), max_k + 1)
            if low_rank <= max_k:
                hits[max(rank, low_rank):] += 1
    return hits[1:] / (len(points) * np.arange(1, max_k + 1))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings, _, center = make_samples()
    images = render(settings)
    pixels = images.reshape(len(images), -1)
    _, true_neighbors = exact_neighbors(pixels)
    with np.load(COORDS) as archive:
        candidates = {name: archive[name] for name in archive.files}
    refined = FIGURES / "generated_one_refined_candidates.npz"
    if refined.exists():
        with np.load(refined) as archive:
            candidates.update({name: archive[name] for name in archive.files})

    fig = plt.figure(figsize=(14, 7.3), facecolor="white")
    left = fig.add_axes([0.055, 0.15, 0.30, 0.70])
    right = fig.add_axes([0.45, 0.17, 0.50, 0.66])
    left.set_axis_off()
    left.text(0, 1.00, "Start with an actual generated ‘1’", fontsize=14,
              fontweight="bold", va="top", color="#22343e")
    left.text(0, 0.92, "One image is one 784-number point: 28 × 28 ink coverages.\n"
              "0 means white; 1 means fully inked. ‘Closest’ means smallest\n"
              "Euclidean distance across all 784 coverages.", fontsize=10.5,
              va="top", color="#405762", linespacing=1.4)

    source = fig.add_axes([0.085, 0.47, 0.17, 0.19])
    source.imshow(images[center], cmap="gray_r", vmin=0, vmax=1,
                  interpolation="nearest")
    source.set(xticks=[], yticks=[], title="Source image")
    source.title.set_fontsize(11)
    for spine in source.spines.values():
        spine.set_edgecolor("#beced3")

    left.text(0, 0.42, "Its 5 closest images in all 784 pixels:", fontsize=10.5,
              va="top", color="#22343e", fontweight="bold")
    for rank, image in enumerate(true_neighbors[center, :5], 1):
        ax = fig.add_axes([0.058 + (rank - 1) * 0.061, 0.255, 0.052, 0.115])
        ax.imshow(images[image], cmap="gray_r", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set(xticks=[], yticks=[], title=f"#{rank}")
        ax.title.set_fontsize(9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#beced3")
    left.text(0, 0.12,
              "Now place every image as a 3D dot. For K = 5, count how many\n"
              "of these five images are still among the source dot’s five\n"
              "closest 3D dots. Average that fraction over all images.",
              fontsize=10, va="top", color="#405762", linespacing=1.4)

    x = np.arange(1, 31)
    for name, label, color in DISPLAY:
        if name not in candidates:
            continue
        y = recall_curve(candidates[name], true_neighbors)
        right.plot(x, 100 * y, lw=2.7 if name.startswith("Multi") else 2.2,
                   label=label, color=color)
        right.scatter([1, 5, 15, 30], 100 * y[[0, 4, 14, 29]],
                      color=color, s=18, zorder=3)
    right.set(xlim=(1, 30), ylim=(0, 100),
              xlabel="K: how many closest images we care about",
              ylabel="True K-nearest images also K-nearest in 3D (%)")
    right.set_xticks([1, 5, 10, 15, 20, 25, 30])
    right.grid(True, color="#e3e9eb", linewidth=0.8)
    right.spines[["top", "right"]].set_visible(False)
    right.legend(loc="upper right", frameon=True, fontsize=9.5)
    right.set_title("Which 3D layout keeps close images close?", fontsize=14,
                    fontweight="bold", color="#22343e", loc="left", pad=12)
    fig.suptitle("Generated ‘1’ images: nearest-neighbor accuracy after compression to 3D",
                 fontsize=17, fontweight="bold", x=0.055, ha="left", y=0.972,
                 color="#22343e")
    fig.text(0.055, 0.058,
             "Measured on the same 4,329 sampled images. Layouts use pixel values, not generator knobs. "
             "Perplexity sets t-SNE’s neighborhood scale; smaller emphasizes closer images.",
             fontsize=9.5, color="#526670")
    fig.text(0.055, 0.038,
             "This is a finite-sample projection of a continuous 5D family, not an exact 3D manifold.",
             fontsize=9.5, color="#526670")
    fig.savefig(OUT, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT.name}")


if __name__ == "__main__":
    main()
