"""Build an offline 3D atlas learned from generated 1 pixel vectors.

Run: python make_generated_one_3d.py

The reducers see only the 784 coverage values. Generator settings are retained
solely for diagnostic coloring and five reference paths in the viewer. UMAP is
optional; when installed, it is included as a comparison to Isomap and PCA.
"""

from __future__ import annotations

import base64
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances

from grey_ones import KNOBS, RANGES, render


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures" / "generated_one_3d.html"
TEMPLATE = ROOT / "generated_one_3d_template.html"
SEED = 41
SOBOL_POWER = 12
PATH_STEPS = 41
NEIGHBORS = 15


def make_samples():
    unit = qmc.Sobol(d=5, scramble=True, seed=SEED).random_base2(SOBOL_POWER)
    samples = list(RANGES[:, 0] + unit * (RANGES[:, 1] - RANGES[:, 0]))
    center = RANGES.mean(axis=1)
    paths = []
    center_index = None
    for knob in range(5):
        ids = []
        for step, value in enumerate(np.linspace(*RANGES[knob], PATH_STEPS)):
            if knob > 0 and step == PATH_STEPS // 2:
                ids.append(center_index)
                continue
            q = center.copy()
            q[knob] = value
            ids.append(len(samples))
            samples.append(q)
            if knob == 0 and step == PATH_STEPS // 2:
                center_index = ids[-1]
        paths.append(ids)
    for corner in itertools.product(*RANGES):
        samples.append(np.asarray(corner, dtype=float))
    return np.asarray(samples, dtype=np.float64), paths, center_index


def sorted_neighbors(distances, count):
    candidates = np.argpartition(distances, count, axis=1)[:, : count + 1]
    candidate_distances = np.take_along_axis(distances, candidates, axis=1)
    order = np.argsort(candidate_distances, axis=1)
    ranked = np.take_along_axis(candidates, order, axis=1)
    return ranked[:, 1 : count + 1]


def make_model_data(name, model, pixels, pixel_distances, pixel_neighbors, pairs):
    print(f"Fitting {name}...", flush=True)
    embedded = np.asarray(model.fit_transform(pixels), dtype=np.float64)
    if not np.isfinite(embedded).all():
        raise ValueError(f"{name} returned nonfinite coordinates")
    display = embedded - embedded.mean(axis=0)
    radius = np.percentile(np.linalg.norm(display, axis=1), 98)
    display *= 1.45 / radius

    reduced_distances = pairwise_distances(embedded)
    display_neighbors = sorted_neighbors(reduced_distances, 60)
    recall = np.asarray(
        [len(set(a[:NEIGHBORS]) & set(b[:NEIGHBORS])) / NEIGHBORS
         for a, b in zip(pixel_neighbors, display_neighbors)],
        dtype=float,
    )
    false_neighbor = []
    for source, candidates in enumerate(display_neighbors):
        true_set = set(pixel_neighbors[source, :30])
        false_neighbor.append(next((int(i) for i in candidates if i not in true_set), -1))

    a, b = pairs
    original = pixel_distances[a, b]
    reduced = reduced_distances[a, b]
    scale = float(np.dot(original, reduced) / np.dot(reduced, reduced))
    stress = float(np.linalg.norm(original - scale * reduced) / np.linalg.norm(original))
    rank_agreement = float(spearmanr(original, reduced).statistic)
    stats = {
        "neighbor_recall_15": round(float(recall.mean()), 4),
        "global_pair_rank_correlation": round(rank_agreement, 4),
        "scaled_pair_distance_error": round(stress, 4),
    }
    print(name, stats, flush=True)
    return {
        "name": name,
        "positions": np.round(display, 5).tolist(),
        "local_recall": np.round(recall, 3).tolist(),
        "nearest_3d": display_neighbors[:, 0].tolist(),
        "nearest_3d_pixel_distance": np.round(
            pixel_distances[np.arange(len(pixels)), display_neighbors[:, 0]], 5
        ).tolist(),
        "false_3d": false_neighbor,
        "false_3d_pixel_distance": [
            round(float(pixel_distances[i, j]), 5) if j >= 0 else None
            for i, j in enumerate(false_neighbor)
        ],
        "stats": stats,
    }


def encoded_ink(pixels):
    quantized = np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8)
    row, col = np.nonzero(quantized)
    offsets = np.searchsorted(row, np.arange(len(pixels) + 1)).astype("<u4")
    indices = col.astype("<u2")
    values = quantized[row, col]

    def b64(array):
        return base64.b64encode(array.tobytes()).decode("ascii")

    return {"offsets": b64(offsets), "indices": b64(indices), "values": b64(values)}


def make_overview(settings, embeddings):
    """Small static comparison for readers who cannot rotate the HTML view."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [name for name in ("Isomap", "UMAP", "PCA") if name in embeddings]
    figure = plt.figure(figsize=(5.4 * len(order), 6), facecolor="white")
    lean = settings[:, 4]
    for panel, name in enumerate(order, 1):
        points = np.asarray(embeddings[name]["positions"])
        ax = figure.add_subplot(1, len(order), panel, projection="3d")
        marks = ax.scatter(
            points[:, 0], points[:, 1], points[:, 2], c=lean, s=1.6,
            cmap="coolwarm", vmin=RANGES[4, 0], vmax=RANGES[4, 1],
            alpha=0.6, rasterized=True,
        )
        ax.view_init(elev=18, azim=-65)
        recall = embeddings[name]["stats"]["neighbor_recall_15"]
        ax.set(
            xlabel="learned X", ylabel="learned Y", zlabel="learned Z",
            title=f"{name} · {100 * recall:.1f}% of 15 neighbors kept",
        )
        ax.tick_params(labelsize=7)
    figure.suptitle(
        "Generated 1s · one dot per 784-pixel image · three learned layouts",
        fontsize=15,
    )
    figure.text(
        0.5, 0.035,
        "Color shows the known lean for diagnosis only. Each view compresses a sampled 5D family; its axes have arbitrary orientation and scale.",
        ha="center", fontsize=9.4, color="#526670",
    )
    figure.subplots_adjust(top=0.87, bottom=0.28, left=0.02, right=0.98, wspace=0.06)
    colorbar_ax = figure.add_axes([0.32, 0.15, 0.36, 0.024])
    figure.colorbar(
        marks, cax=colorbar_ax, orientation="horizontal",
        label="generated lean angle (degrees)",
    )
    figure.savefig(OUT.with_name("generated_one_3d_overview.png"), dpi=170, facecolor="white")
    plt.close(figure)


def main():
    settings, paths, center_index = make_samples()
    pixels = render(settings).reshape(len(settings), -1)
    print(f"Rendered {len(pixels)} images, each with {pixels.shape[1]} pixels", flush=True)
    pixel_distances = pairwise_distances(pixels)
    pixel_neighbors = sorted_neighbors(pixel_distances, 30)
    rng = np.random.default_rng(SEED + 1)
    a = rng.integers(0, len(pixels), size=40000)
    b = rng.integers(0, len(pixels), size=40000)
    keep = a != b
    pairs = a[keep], b[keep]

    models = [
        ("Isomap", Isomap(n_neighbors=70, n_components=3, eigen_solver="arpack", n_jobs=1)),
        ("PCA", PCA(n_components=3, random_state=SEED)),
    ]
    try:
        import umap
    except ImportError:
        print("UMAP not installed; building Isomap and PCA views", flush=True)
    else:
        models.append(("UMAP", umap.UMAP(
            n_neighbors=40, n_components=3, min_dist=0.12,
            metric="euclidean", random_state=SEED, n_epochs=400,
        )))

    embeddings = {
        name: make_model_data(name, model, pixels, pixel_distances, pixel_neighbors, pairs)
        for name, model in models
    }
    data = {
        "count": len(pixels),
        "image_size": 28,
        "sobol_count": 2 ** SOBOL_POWER,
        "path_steps": PATH_STEPS,
        "knobs": KNOBS,
        "ranges": RANGES.tolist(),
        "settings": np.round(settings, 5).tolist(),
        "paths": paths,
        "center_index": center_index,
        "nearest_pixel": pixel_neighbors[:, 0].tolist(),
        "nearest_pixel_distance": np.round(
            pixel_distances[np.arange(len(pixels)), pixel_neighbors[:, 0]], 5
        ).tolist(),
        "embeddings": embeddings,
        "pair_count": len(pairs[0]),
        "ink": encoded_ink(pixels),
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "__DATA__", json.dumps(data, separators=(",", ":"))
    )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.2f} MB)", flush=True)
    make_overview(settings, embeddings)


if __name__ == "__main__":
    main()
