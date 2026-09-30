"""Try mixtures of small and larger pixel-neighborhood affinities in 3D t-SNE.

Run after benchmark_generated_one_neighbors.py. This uses scikit-learn's
private t-SNE optimizer, so it is pinned to the installed scikit-learn API.
"""

from __future__ import annotations

import json
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold._t_sne import _joint_probabilities_nn
from sklearn.neighbors import NearestNeighbors

from benchmark_generated_one_neighbors import COORDS, OUT, exact_neighbors, evaluate
from grey_ones import render
from make_generated_one_3d import SEED, make_samples


MIXES = ((5, 25, 0.5), (5, 50, 0.5), (10, 50, 0.5), (5, 25, 0.75))


def main():
    settings, _, _ = make_samples()
    pixels = render(settings).reshape(len(settings), -1)
    d, neighbors = exact_neighbors(pixels)
    n = len(pixels)
    graph = NearestNeighbors(n_neighbors=151, metric="euclidean", n_jobs=1).fit(pixels).kneighbors_graph(mode="distance")
    graph.data **= 2
    affinities = {p: _joint_probabilities_nn(graph, p, 0) for p in (5, 10, 25, 50)}
    init = PCA(n_components=3, svd_solver="randomized", random_state=SEED).fit_transform(pixels).astype(np.float32)
    init = init / np.std(init[:, 0]) * 1e-4
    results = json.loads(OUT.read_text())
    with np.load(COORDS) as old:
        coords = {name: old[name] for name in old.files}

    for small, large, small_weight in MIXES:
        name = f"MultiTSNE-p{small}-{large}-w{small_weight:g}"
        if name in results["metrics"]:
            continue
        print(f"Fitting {name}", flush=True)
        start = time.monotonic()
        probabilities = (small_weight * affinities[small] +
                         (1 - small_weight) * affinities[large]).tocsr()
        model = TSNE(n_components=3, learning_rate="auto", max_iter=1200,
                     method="barnes_hut", angle=0.35, random_state=SEED)
        model.learning_rate_ = max(n / model.early_exaggeration / 4, 50)
        points = model._tsne(probabilities, 2, n, init.copy())
        metrics = evaluate(points, neighbors, d)
        metrics["fit_seconds"] = round(time.monotonic() - start, 2)
        print(name, metrics, flush=True)
        results["metrics"][name] = metrics
        coords[name] = np.asarray(points, np.float32)
        OUT.write_text(json.dumps(results, indent=2) + "\n")
        np.savez_compressed(COORDS, **coords)


if __name__ == "__main__":
    main()
