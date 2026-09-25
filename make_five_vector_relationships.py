"""Study only the mutual angles among the five generated-image knob vectors.

Run: python make_five_vector_relationships.py
Output: figures/five_vector_relationships.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc
from sklearn.manifold import Isomap, trustworthiness
from sklearn.metrics import pairwise_distances

from grey_ones import RANGES
from make_direction_atlas import image_at, jacobian


Q0 = np.array([14.43, 14.62, 19.77, 3.13, np.tan(np.deg2rad(17.3))])
NAMES = ["cx", "cy", "height", "width", "lean"]
UPPER = np.triu_indices(5, 1)
PAIRS = [f"{NAMES[i]} / {NAMES[j]}" for i, j in zip(*UPPER)]


def gram(q, step=1e-6):
    j = jacobian(q, step=step)
    unit = j / np.linalg.norm(j, axis=0)
    return unit.T @ unit


def angle_signature(q):
    return gram(q)[UPPER]


def local_singulars(q, step=1e-5):
    cols = []
    for i in range(5):
        hi, lo = q.copy(), q.copy()
        hi[i] += step
        lo[i] -= step
        cols.append((angle_signature(hi) - angle_signature(lo)) / (2 * step))
    return np.linalg.svd(np.stack(cols, axis=1), compute_uv=False)


def main():
    g0 = gram(Q0)
    singulars = local_singulars(Q0)

    angles = np.linspace(-9.875, 34.875, 180)
    signatures = []
    for angle in angles:
        q = Q0.copy()
        q[4] = np.tan(np.deg2rad(angle))
        signatures.append(angle_signature(q))
    signatures = np.stack(signatures)

    n = 250
    params = qmc.LatinHypercube(5, seed=2026).random(n)
    params = params * (RANGES[:, 1] - RANGES[:, 0]) + RANGES[:, 0]
    q_samples = params.copy()
    q_samples[:, 4] = np.tan(np.deg2rad(params[:, 4]))
    X = np.stack([image_at(q) for q in q_samples])
    A = np.stack([angle_signature(q) for q in q_samples])
    image_d = pairwise_distances(X)
    angle_d = pairwise_distances(A)
    positions = Isomap(n_neighbors=30, n_components=2,
                       metric="precomputed").fit_transform(angle_d)
    trust = trustworthiness(A, positions, n_neighbors=10)
    image_knn = np.argsort(image_d, axis=1)[:, 1:11]
    angle_knn = np.argsort(angle_d, axis=1)[:, 1:11]
    overlap = np.mean([len(set(a) & set(b)) / 10
                       for a, b in zip(image_knn, angle_knn)])

    fig = plt.figure(figsize=(17, 9.8), facecolor="white")
    fig.suptitle("How the five unit vectors relate to one another",
                 fontsize=20, fontweight="bold", y=.984)
    fig.text(.5, .945,
             "The 5×5 Gram table contains every pairwise cosine. It forgets where the vectors "
             "point in pixel space, but retains their mutual angles.",
             ha="center", fontsize=11, color="#555")
    gs = fig.add_gridspec(2, 3, left=.06, right=.96, bottom=.09, top=.88,
                          width_ratios=[1, 1.34, 1], height_ratios=[1.15, .76],
                          hspace=.39, wspace=.31)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(g0, cmap="PuOr", vmin=-1, vmax=1,
                   interpolation="nearest")
    ax.set_xticks(range(5), NAMES, rotation=35, ha="right")
    ax.set_yticks(range(5), NAMES)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{g0[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if abs(g0[i,j])>.68 else "#222")
    ax.set_title("At one generated image", fontsize=12, pad=9)
    fig.colorbar(im, ax=ax, fraction=.048, pad=.04, label="cosine similarity")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(signatures.T, cmap="PuOr", vmin=-1, vmax=1,
              interpolation="nearest", aspect="auto",
              extent=(-10, 35, 9.5, -.5))
    ax.set_yticks(range(10), PAIRS, fontsize=8.5)
    ax.set(xlabel="lean angle, with other knobs held fixed",
           title="All ten mutual angles as lean changes")
    ax.axvline(0, color="#444", ls=":", lw=1)

    ax = fig.add_subplot(gs[0, 2])
    scatter = ax.scatter(positions[:, 0], positions[:, 1], c=params[:, 4],
                         cmap="coolwarm", vmin=-10, vmax=35,
                         s=24, alpha=.85, linewidths=0)
    ax.set(title="Isomap of the ten angle values", xlabel="2D map axis 1",
           ylabel="2D map axis 2")
    ax.grid(alpha=.18)
    cb = fig.colorbar(scatter, ax=ax, fraction=.048, pad=.04)
    cb.set_label("lean angle", fontsize=9)

    ax = fig.add_subplot(gs[1, 0:2])
    ax.plot(range(1, 6), singulars, "o-", color="#326c9c", lw=2,
            markersize=6)
    ax.set(xticks=range(1, 6), xlabel="independent parameter direction",
           ylabel="local change in the ten angle values",
           title="All five local responses are nonzero")
    ax.grid(alpha=.2)
    ax.text(.98, .87,
            "The five arrows span 5D pixel space.\n"
            "Their mutual-angle table also varies\n"
            "in 5 independent ways nearby.",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
            bbox=dict(facecolor="white", edgecolor="#ddd", pad=6))

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.text(.03, .95, "SAME LOCAL DIMENSION,\nDIFFERENT NEIGHBORS",
            fontsize=11.5, fontweight="bold", va="top")
    ax.text(.03, .72,
            f"250 sampled images\n"
            f"10-neighbor overlap with images: {overlap:.1%}\n"
            f"2D map trustworthiness: {trust:.3f}\n\n"
            "Mutual angles can preserve five\n"
            "local degrees of freedom while\n"
            "changing global geometry.",
            fontsize=10.2, va="top", linespacing=1.45)

    fig.text(.5, .028,
             "Local rank comes from finite differences at one generic point; "
             "the map and neighbor comparison use all ten angle values. "
             "The 2D Isomap does not establish intrinsic dimension.",
             ha="center", fontsize=9.2, color="#555")
    fig.savefig("figures/five_vector_relationships.png", dpi=150,
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
