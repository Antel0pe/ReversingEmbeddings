"""Re-fit leading 3D methods on an independent generated-one sample.

Run: python replicate_generated_one_neighbors.py

The evaluation sample uses a different Sobol seed but the same knob ranges,
number of images, reference paths, and exact pixel-distance definition.
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.stats import qmc
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
from sklearn.manifold._t_sne import _joint_probabilities_nn
from sklearn.neighbors import NearestNeighbors

from benchmark_generated_one_neighbors import FIGURES, exact_neighbors, evaluate
from grey_ones import RANGES, render
from make_generated_one_3d import SEED, SOBOL_POWER, make_samples
from refine_generated_one_neighbors import refine


OUT = FIGURES / "generated_one_neighbor_replicate.json"


def main():
    settings, _, _ = make_samples()
    independent = qmc.Sobol(d=5, scramble=True, seed=SEED + 101).random_base2(SOBOL_POWER)
    settings[:2**SOBOL_POWER] = RANGES[:, 0] + independent * (RANGES[:, 1] - RANGES[:, 0])
    pixels = render(settings).reshape(len(settings), -1)
    distances, true_neighbors = exact_neighbors(pixels)
    results = json.loads(OUT.read_text()) if OUT.exists() else {
        "sample_count": len(pixels), "seed": SEED + 101, "metrics": {}
    }
    if results["sample_count"] != len(pixels) or results["seed"] != SEED + 101:
        raise ValueError("Existing replicate report is from a different sample")

    def measure(name, fit):
        if name in results["metrics"]:
            return
        print("Fitting", name, flush=True)
        start = time.monotonic()
        points = fit()
        metrics = evaluate(points, true_neighbors, distances)
        metrics["fit_seconds"] = round(time.monotonic() - start, 2)
        results["metrics"][name] = metrics
        OUT.write_text(json.dumps(results, indent=2) + "\n")
        print(name, metrics, flush=True)

    measure("Isomap", lambda: Isomap(n_neighbors=70, n_components=3,
                                      eigen_solver="arpack", n_jobs=1).fit_transform(pixels))
    for perplexity in (2, 5, 15):
        measure(f"TSNE-p{perplexity}", lambda p=perplexity: TSNE(
            n_components=3, perplexity=p, init="pca", learning_rate="auto",
            max_iter=1200, method="barnes_hut", angle=0.35,
            random_state=SEED,
        ).fit_transform(pixels))

    def fit_multi():
        n = len(pixels)
        graph = NearestNeighbors(n_neighbors=151, metric="euclidean", n_jobs=1).fit(pixels).kneighbors_graph(mode="distance")
        graph.data **= 2
        small = _joint_probabilities_nn(graph, 5, 0)
        large = _joint_probabilities_nn(graph, 50, 0)
        probabilities = (0.5 * small + 0.5 * large).tocsr()
        init = PCA(n_components=3, svd_solver="randomized", random_state=SEED).fit_transform(pixels).astype(np.float32)
        init = init / np.std(init[:, 0]) * 1e-4
        model = TSNE(n_components=3, learning_rate="auto", max_iter=1200,
                     method="barnes_hut", angle=0.35, random_state=SEED)
        model.learning_rate_ = max(n / model.early_exaggeration / 4, 50)
        return model._tsne(probabilities, 2, n, init)

    measure("MultiTSNE-p5-50-w0.5", fit_multi)
    measure("Rank-refined Isomap", lambda: refine(
        Isomap(n_neighbors=70, n_components=3, eigen_solver="arpack", n_jobs=1).fit_transform(pixels),
        true_neighbors, distances, "Rank-refined Isomap replicate",
    ))


if __name__ == "__main__":
    main()
