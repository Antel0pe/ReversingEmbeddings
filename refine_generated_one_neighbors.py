"""Refine sampled 3D layouts for pixel-neighbor ranking.

Run after benchmark_generated_one_neighbors.py. This is a transductive layout:
it optimizes the displayed sample coordinates, not a map for unseen images.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from benchmark_generated_one_neighbors import (
    COORDS, FIGURES, exact_neighbors, evaluate,
)
from grey_ones import render
from make_generated_one_3d import make_samples


OUT = FIGURES / "generated_one_refined_candidates.npz"
SCORES = FIGURES / "generated_one_refinement_results.json"
LEVELS = ((1, (0,), 8, 0.35), (5, (0, 1, 2, 3, 4), 5, 0.30),
          (15, (0, 3, 7, 11, 14), 5, 0.35))


def mine_triplets(points, true_neighbors):
    low = cKDTree(points).query(points, k=81, workers=-1)[1][:, 1:]
    anchors, positives, negatives, weights = [], [], [], []
    for k, ranks, negative_count, level_weight in LEVELS:
        for i, row in enumerate(low):
            real = set(true_neighbors[i, :k])
            false = [int(j) for j in row if j not in real][:negative_count]
            for rank in ranks:
                for neg in false:
                    anchors.append(i)
                    positives.append(int(true_neighbors[i, rank]))
                    negatives.append(neg)
                    weights.append(level_weight / (len(ranks) * len(false)))
    return (
        np.asarray(anchors, dtype=np.int32),
        np.asarray(positives, dtype=np.int32),
        np.asarray(negatives, dtype=np.int32),
        np.asarray(weights, dtype=np.float64),
    )


@njit(cache=True)
def rank_loss(flat, origin, anchors, positives, negatives, weights,
              margin, temperature, anchor_strength):
    points = flat.reshape((-1, 3))
    initial = origin.reshape((-1, 3))
    gradient = np.zeros_like(points)
    loss = 0.0
    total_weight = 0.0
    for t in range(len(anchors)):
        i, j, k = anchors[t], positives[t], negatives[t]
        weight = weights[t]
        pos_sq, neg_sq = 0.0, 0.0
        for dim in range(3):
            a = points[i, dim] - points[j, dim]
            b = points[i, dim] - points[k, dim]
            pos_sq += a * a
            neg_sq += b * b
        u = (pos_sq - neg_sq + margin) / temperature
        if u > 30.0:
            softplus, sigmoid = u, 1.0
        elif u < -30.0:
            softplus, sigmoid = np.exp(u), np.exp(u)
        else:
            softplus = np.log1p(np.exp(u))
            sigmoid = 1.0 / (1.0 + np.exp(-u))
        loss += weight * softplus
        total_weight += weight
        g = weight * sigmoid / temperature
        for dim in range(3):
            a = points[i, dim] - points[j, dim]
            b = points[i, dim] - points[k, dim]
            gradient[i, dim] += 2.0 * g * (a - b)
            gradient[j, dim] -= 2.0 * g * a
            gradient[k, dim] += 2.0 * g * b
    loss /= total_weight
    gradient /= total_weight
    for i in range(len(points)):
        for dim in range(3):
            delta = points[i, dim] - initial[i, dim]
            loss += anchor_strength * delta * delta / len(points)
            gradient[i, dim] += 2.0 * anchor_strength * delta / len(points)
    return loss, gradient.ravel()


def refine(points, true_neighbors, pixel_distances, label, anchor_strength=3.0):
    original = np.asarray(points, dtype=np.float64).copy()
    scale = np.median(np.linalg.norm(
        original - original[true_neighbors[:, 4]], axis=1
    ))
    original /= scale
    current = original.copy()
    best = current.copy()
    best_scores = evaluate(best, true_neighbors, pixel_distances)
    best_quality = sum(best_scores[f"recall_{k}"] for k in (1, 5, 15))
    for round_index in range(1, 9):
        triplets = mine_triplets(current, true_neighbors)
        start = time.monotonic()
        def objective(flat):
            return rank_loss(flat, original.ravel(), *triplets,
                             0.20, 0.20, anchor_strength)
        result = minimize(objective, current.ravel(), jac=True, method="L-BFGS-B",
                          options={"maxiter": 20, "ftol": 1e-7})
        current = result.x.reshape((-1, 3))
        scores = evaluate(current, true_neighbors, pixel_distances)
        quality = sum(scores[f"recall_{k}"] for k in (1, 5, 15))
        if quality > best_quality:
            best, best_scores, best_quality = current.copy(), scores, quality
        print(f"{label} round {round_index}: {scores} "
              f"({time.monotonic() - start:.1f}s)", flush=True)
    print(f"{label} best: {best_scores}", flush=True)
    return best


def main():
    settings, _, _ = make_samples()
    pixels = render(settings).reshape(len(settings), -1)
    pixel_distances, true_neighbors = exact_neighbors(pixels)
    baseline = np.load(COORDS)
    results, coordinates = {}, {}
    for name in ("Isomap", "TSNE-p10", "TSNE-p25"):
        if name not in baseline:
            continue
        label = f"Rank-refined {name}"
        points = refine(baseline[name], true_neighbors, pixel_distances, label)
        coordinates[label] = points.astype(np.float32)
        results[label] = evaluate(points, true_neighbors, pixel_distances)
        SCORES.write_text(json.dumps(results, indent=2) + "\n")
        np.savez_compressed(OUT, **coordinates)


if __name__ == "__main__":
    main()
