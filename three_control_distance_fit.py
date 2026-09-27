"""Fit a topology-aware 3D display to the exact three-knob image renderer.

The result is a visual approximation, not an isometry. It balances pixel-space
distances for all sampled pairs, small mixed-knob moves, and a penalty against
reversed or nearly collapsed cells in a finer continuous parameter grid.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, squareform

from make_three_dimensional_ones import BOW, LEAN, WIDTH, render


FINE_W = np.linspace(float(WIDTH[0]), float(WIDTH[-1]), 9)
FINE_B = np.linspace(float(BOW[0]), float(BOW[-1]), 21)
FINE_A = np.linspace(float(LEAN[0]), float(LEAN[-1]), 21)
LOW = np.array([float(LEAN[0]), float(BOW[0]), float(WIDTH[0])])
HIGH = np.array([float(LEAN[-1]), float(BOW[-1]), float(WIDTH[-1])])


def fine_settings():
    w, b, a = np.meshgrid(FINE_W, FINE_B, FINE_A, indexing="ij")
    return np.stack((a, b, w), axis=-1).reshape(-1, 3)


def interpolation_matrix(settings):
    """Sparse trilinear map from the 405 coarse anchors to given settings."""
    rows, cols, weights = [], [], []
    for row, (lean, bow, width) in enumerate(settings):
        brackets = []
        for value, grid in ((width, WIDTH), (bow, BOW), (lean, LEAN)):
            t = np.interp(value, grid, np.arange(len(grid)))
            index = min(len(grid) - 2, int(t))
            brackets.append((index, t - index))
        (wi, wt), (bi, bt), (ai, at) = brackets
        for dw in (0, 1):
            for db in (0, 1):
                for da in (0, 1):
                    rows.append(row)
                    cols.append(((wi + dw) * len(BOW) + bi + db)
                                * len(LEAN) + ai + da)
                    weights.append((wt if dw else 1 - wt)
                                   * (bt if db else 1 - bt)
                                   * (at if da else 1 - at))
    return coo_matrix(
        (weights, (rows, cols)),
        shape=(len(settings), len(WIDTH) * len(BOW) * len(LEAN)),
    ).tocsr()


def fine_topology():
    ids = np.arange(len(FINE_W) * len(FINE_B) * len(FINE_A)).reshape(
        len(FINE_W), len(FINE_B), len(FINE_A)
    )
    edge_pairs = np.concatenate([
        np.stack((ids[1:].ravel(), ids[:-1].ravel()), axis=1),
        np.stack((ids[:, 1:].ravel(), ids[:, :-1].ravel()), axis=1),
        np.stack((ids[:, :, 1:].ravel(), ids[:, :, :-1].ravel()), axis=1),
    ])
    corners = (
        ids[:-1, :-1, :-1].ravel(), ids[1:, :-1, :-1].ravel(),
        ids[:-1, 1:, :-1].ravel(), ids[:-1, :-1, 1:].ravel(),
    )
    return edge_pairs, corners


def oriented_cell_volumes(points, corners):
    v0, v1, v2, v3 = corners
    a = points[v1] - points[v0]
    b = points[v2] - points[v0]
    c = points[v3] - points[v0]
    return -np.einsum("ij,ij->i", a, np.cross(b, c))


def sampled_cell_jacobians(points):
    """Check the displayed trilinear cells at corners and their midpoints."""
    grid = points.reshape(len(FINE_W), len(FINE_B), len(FINE_A), 3)
    patches = np.stack([
        grid[i:len(FINE_W)-1+i, j:len(FINE_B)-1+j,
             k:len(FINE_A)-1+k]
        for i in (0, 1) for j in (0, 1) for k in (0, 1)
    ]).reshape(2, 2, 2, len(FINE_W)-1, len(FINE_B)-1,
               len(FINE_A)-1, 3)
    determinants = []
    for u in (0, .5, 1):
        for v in (0, .5, 1):
            for t in (0, .5, 1):
                wu, wv, wt = ((1-u, u), (1-v, v), (1-t, t))
                du = sum(wv[j] * wt[k] * (patches[1, j, k]
                                               - patches[0, j, k])
                         for j in (0, 1) for k in (0, 1))
                dv = sum(wu[i] * wt[k] * (patches[i, 1, k]
                                               - patches[i, 0, k])
                         for i in (0, 1) for k in (0, 1))
                dt = sum(wu[i] * wv[j] * (patches[i, j, 1]
                                               - patches[i, j, 0])
                         for i in (0, 1) for j in (0, 1))
                determinants.append(-np.einsum(
                    "...i,...i->...", du, np.cross(dv, dt)
                ))
    return np.stack(determinants)


def fit_view(start, images):
    """Return a distance-shaped chart and measured errors on unseen settings."""
    n = len(start)
    pixel_images = images.reshape(n, -1)
    pixel_distances = squareform(pdist(pixel_images))
    global_i, global_j = np.triu_indices(n, 1)
    global_d = pixel_distances[global_i, global_j]

    fine_q = fine_settings()
    fine_map = interpolation_matrix(fine_q)
    fine_images = np.array([render(*q).ravel() for q in fine_q])
    fine_edges, corners = fine_topology()
    fine_i, fine_j = fine_edges.T
    fine_d = np.linalg.norm(
        fine_images[fine_i] - fine_images[fine_j], axis=1
    )
    initial_volumes = oriented_cell_volumes(fine_map @ start, corners)
    if np.any(initial_volumes <= 0):
        raise ValueError("Initial chart has reversed fine-grid cells")
    minimum_volume = .25 * initial_volumes

    random = np.random.default_rng(923)
    near_q = LOW + random.random((3000, 3)) * (HIGH - LOW)
    near_r = np.clip(
        near_q + random.uniform(-1, 1, (3000, 3)) * [.13, .09, .06],
        LOW, HIGH,
    )
    near_map = interpolation_matrix(np.concatenate((near_q, near_r)))
    near_i = np.arange(len(near_q))
    near_j = near_i + len(near_q)
    near_d = np.linalg.norm(
        np.array([render(*q).ravel() for q in near_q])
        - np.array([render(*q).ravel() for q in near_r]), axis=1,
    )

    def pair_loss(points, i, j, target, weight, gradient):
        diff = points[i] - points[j]
        lengths = np.maximum(np.linalg.norm(diff, axis=1), 1e-8)
        relative = (lengths - target) / target
        cost = weight * np.mean(relative**2)
        factor = weight * 2 * relative / (target * lengths * len(target))
        updates = diff * factor[:, None]
        for k in range(3):
            gradient[:, k] += (
                np.bincount(i, weights=updates[:, k], minlength=len(points))
                - np.bincount(j, weights=updates[:, k], minlength=len(points))
            )
        return cost

    def objective(flat, repulsion):
        anchors = flat.reshape(n, 3)
        fine = fine_map @ anchors
        random_near = near_map @ anchors
        grad = np.zeros_like(anchors)
        fine_grad = np.zeros_like(fine)
        near_grad = np.zeros_like(random_near)
        cost = pair_loss(
            anchors, global_i, global_j, global_d, 1.0, grad
        )
        cost += pair_loss(
            fine, fine_i, fine_j, fine_d, .2, fine_grad
        )
        cost += pair_loss(
            random_near, near_i, near_j, near_d, 1.0, near_grad
        )
        grad += near_map.T @ near_grad

        # A few strongly compressed pairs can create a false self-overlap even
        # when the median error is low. Penalize those pairs separately.
        diff = anchors[global_i] - anchors[global_j]
        lengths = np.maximum(np.linalg.norm(diff, axis=1), 1e-8)
        ratio = lengths / global_d
        shortfall = np.maximum(0, .7 - ratio)
        cost += repulsion * np.sum(shortfall**2) / n
        factor = -repulsion * 2 * shortfall / (n * global_d * lengths)
        updates = diff * factor[:, None]
        for k in range(3):
            grad[:, k] += (
                np.bincount(global_i, weights=updates[:, k], minlength=n)
                - np.bincount(global_j, weights=updates[:, k], minlength=n)
            )

        v0, v1, v2, v3 = corners
        a = fine[v1] - fine[v0]
        b = fine[v2] - fine[v0]
        c = fine[v3] - fine[v0]
        volumes = -np.einsum("ij,ij->i", a, np.cross(b, c))
        shortfall = np.maximum(0, (minimum_volume - volumes) / minimum_volume)
        cost += np.mean(shortfall**2)
        factor = -2 * shortfall / (minimum_volume * len(shortfall))
        da = factor[:, None] * (-np.cross(b, c))
        db = factor[:, None] * (-np.cross(c, a))
        dc = factor[:, None] * (-np.cross(a, b))
        for ids, updates in (
            (v0, -(da + db + dc)), (v1, da), (v2, db), (v3, dc),
        ):
            for k in range(3):
                fine_grad[:, k] += np.bincount(
                    ids, weights=updates[:, k], minlength=len(fine)
                )
        grad += fine_map.T @ fine_grad
        return cost, grad.ravel()

    stage_one = minimize(
        objective, start.ravel(), args=(1.0,), method="L-BFGS-B",
        jac=True, options={"maxiter": 250, "ftol": 1e-10},
    )
    final = minimize(
        objective, stage_one.x, args=(5.0,), method="L-BFGS-B",
        jac=True, options={"maxiter": 600, "ftol": 1e-10},
    )
    fitted = final.x.reshape(n, 3)
    diagnostics = evaluate_view(fitted, fine_map, corners)
    return fitted, diagnostics


def evaluate_view(anchors, fine_map, corners):
    """Use independent random image pairs and moves, never optimization pairs."""
    random = np.random.default_rng(284)
    q = LOW + random.random((500, 3)) * (HIGH - LOW)
    far = LOW + random.random((500, 3)) * (HIGH - LOW)
    near = np.clip(
        q + random.uniform(-1, 1, (500, 3)) * [.12, .08, .05],
        LOW, HIGH,
    )
    placed = interpolation_matrix(np.concatenate((q, far, near))) @ anchors
    first, far_placed, near_placed = np.split(placed, 3)
    original = np.array([render(*state).ravel() for state in q])
    far_images = np.array([render(*state).ravel() for state in far])
    near_images = np.array([render(*state).ravel() for state in near])
    global_true = np.linalg.norm(original - far_images, axis=1)
    local_true = np.linalg.norm(original - near_images, axis=1)
    global_error = np.abs(np.linalg.norm(first - far_placed, axis=1)
                          - global_true) / global_true
    local_error = np.abs(np.linalg.norm(first - near_placed, axis=1)
                         - local_true) / local_true
    volumes = oriented_cell_volumes(fine_map @ anchors, corners)
    jacobians = sampled_cell_jacobians(fine_map @ anchors)
    return {
        "global_median": float(np.median(global_error)),
        "global_p90": float(np.percentile(global_error, 90)),
        "local_median": float(np.median(local_error)),
        "local_p90": float(np.percentile(local_error, 90)),
        "reversed_fine_cells": int(np.sum(np.any(jacobians <= 0, axis=0))),
        "fine_cell_count": len(volumes),
        "jacobian_samples_per_cell": len(jacobians),
        "minimum_sampled_jacobian": float(np.min(jacobians)),
    }


def five_image_witness():
    """Five legal images with fixed width that require four pixel directions."""
    settings = [(0, 0, 3), (3, 0, 3), (-3, 0, 3),
                (0, 2, 3), (0, -2, 3)]
    images = np.array([render(*q).ravel() for q in settings])
    singular = np.linalg.svd(images - images.mean(axis=0),
                             compute_uv=False)
    return [float(x) for x in singular[:4]]
