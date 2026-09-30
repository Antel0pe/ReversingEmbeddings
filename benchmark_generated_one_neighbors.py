"""Compare 3D layouts by exact 784-pixel nearest-neighbor preservation.

Run: python benchmark_generated_one_neighbors.py

The generator settings only select the sample; fitting and evaluation use pixels.
The existing viewer supplies baseline 3D coordinates. Results and candidate
coordinates are saved under figures/ for rebuilding the viewer.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

from grey_ones import render
from make_generated_one_3d import ROOT, SEED, make_samples


FIGURES = ROOT / "figures"
VIEWER = FIGURES / "generated_one_3d.html"
OUT = FIGURES / "generated_one_neighbor_benchmark.json"
COORDS = FIGURES / "generated_one_neighbor_candidates.npz"
K_VALUES = (1, 5, 15, 30)


def load_viewer_data():
    text = VIEWER.read_text(encoding="utf-8")
    start = '<script id="embedding-data" type="application/json">'
    return json.loads(text.split(start, 1)[1].split("</script>", 1)[0])


def exact_neighbors(pixels, count=60):
    distances = pairwise_distances(pixels, metric="euclidean", n_jobs=1)
    np.fill_diagonal(distances, np.inf)
    candidate = np.argpartition(distances, count, axis=1)[:, :count]
    ordered = np.take_along_axis(
        candidate,
        np.argsort(np.take_along_axis(distances, candidate, axis=1), axis=1),
        axis=1,
    )
    return distances, ordered


def evaluate(positions, true_neighbors, pixel_distances):
    positions = np.asarray(positions, dtype=np.float64)
    if not np.isfinite(positions).all():
        raise ValueError("Coordinates contain nonfinite values")
    n = len(positions)
    candidate = cKDTree(positions).query(positions, k=61, workers=-1)[1]
    # Duplicate display coordinates are not expected, but remove self by index.
    near = np.empty((n, 60), dtype=np.int32)
    for i, row in enumerate(candidate):
        near[i] = row[row != i][:60]
    metrics = {}
    for k in K_VALUES:
        matches = np.asarray([
            len(set(a[:k]) & set(b[:k])) / k
            for a, b in zip(true_neighbors, near)
        ])
        metrics[f"recall_{k}"] = round(float(matches.mean()), 5)
        metrics[f"p10_recall_{k}"] = round(float(np.percentile(matches, 10)), 5)
    # Among the same 30 original neighbors, does the layout retain their
    # near-to-far order? This detects rank scrambling that set overlap misses.
    local_distances = np.linalg.norm(
        positions[:, None, :] - positions[true_neighbors[:, :30]], axis=2
    )
    local_ranks = np.argsort(np.argsort(local_distances, axis=1), axis=1) + 1
    original_ranks = np.arange(1, 31)
    local_spearman = 1 - 6 * np.sum((local_ranks - original_ranks) ** 2, axis=1) / (30 * (30**2 - 1))
    metrics["local_rank_spearman_30"] = round(float(local_spearman.mean()), 5)
    true_closest_rank = np.asarray([
        int(np.flatnonzero(row == true_neighbors[i, 0])[0]) + 1
        if true_neighbors[i, 0] in row else 61
        for i, row in enumerate(near)
    ])
    metrics["true_closest_median_3d_rank"] = float(np.median(true_closest_rank))
    metrics["true_closest_within_15"] = round(float(np.mean(true_closest_rank <= 15)), 5)
    nearest_3d_pixel = pixel_distances[np.arange(n), near[:, 0]]
    nearest_pixel = pixel_distances[np.arange(n), true_neighbors[:, 0]]
    ratio = nearest_3d_pixel / np.maximum(nearest_pixel, 1e-10)
    metrics["nearest_3d_pixel_distance_ratio_median"] = round(float(np.median(ratio)), 4)
    metrics["nearest_3d_pixel_distance_ratio_p90"] = round(float(np.percentile(ratio, 90)), 4)
    return metrics


def main():
    settings, _, _ = make_samples()
    pixels = render(settings).reshape(len(settings), -1)
    old = load_viewer_data()
    if len(pixels) != old["count"]:
        raise ValueError("Sample does not match the existing viewer")
    pixel_distances, true_neighbors = exact_neighbors(pixels)
    saved = json.loads(OUT.read_text()) if OUT.exists() else None
    if saved is not None and saved.get("sample_count") == len(pixels) and COORDS.exists():
        results = saved["metrics"]
        with np.load(COORDS) as old_coordinates:
            coordinates = {name: old_coordinates[name] for name in old_coordinates.files}
    else:
        results, coordinates = {}, {}

    def record(name, points, elapsed=None):
        metrics = evaluate(points, true_neighbors, pixel_distances)
        if elapsed is not None:
            metrics["fit_seconds"] = round(elapsed, 2)
        results[name] = metrics
        coordinates[name] = np.asarray(points, dtype=np.float32)
        print(name, json.dumps(metrics), flush=True)
        OUT.write_text(json.dumps({"sample_count": len(pixels), "metrics": results}, indent=2) + "\n")
        np.savez_compressed(COORDS, **coordinates)

    for name, data in old["embeddings"].items():
        if name not in results:
            record(name, data["positions"])

    for perplexity in (2, 3, 5, 10, 15, 25, 50, 100):
        name = f"TSNE-p{perplexity}"
        if name in results:
            continue
        print(f"Fitting {name}", flush=True)
        start = time.monotonic()
        model = TSNE(
            n_components=3,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=1200,
            method="barnes_hut",
            angle=0.35,
            random_state=SEED,
        )
        record(name, model.fit_transform(pixels), time.monotonic() - start)


if __name__ == "__main__":
    main()
