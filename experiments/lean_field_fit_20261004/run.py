"""Small non-neural lean-field experiment using the repository renderer.

Run from the repository root:
    .venv/bin/python experiments/lean_field_fit_20261004/run.py

Writes numerical results beside this file. No figures are generated here.
The model inputs are known width and lean settings, not inferred image coordinates.
"""

from pathlib import Path
import csv
import json
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from grey_ones import KNOBS, RANGES, render  # noqa: E402


OUT = Path(__file__).resolve().parent / "results"
NAMES = list(KNOBS)
BOUNDS = np.asarray(
    [RANGES[name] for name in NAMES] if isinstance(RANGES, dict) else RANGES,
    dtype=float,
)
assert NAMES == ["cx", "cy", "height", "width", "lean"], NAMES
assert BOUNDS.shape == (5, 2), BOUNDS.shape
SEED = BOUNDS.mean(axis=1)
WIDTH, LEAN = NAMES.index("width"), NAMES.index("lean")
DELTA = 0.025  # degrees; all derivative measurements remain inside the range box


def images(settings):
    settings = np.atleast_2d(settings)
    return np.concatenate(
        [np.asarray(render(part)).reshape(len(part), -1)
         for part in np.array_split(settings, max(1, (len(settings) + 63) // 64))],
        axis=0,
    ).astype(float)


def settings_from_slice(points):
    settings = np.tile(SEED, (len(points), 1))
    settings[:, WIDTH] = points[:, 0]
    settings[:, LEAN] = points[:, 1]
    return settings


def measured_field(points, delta=DELTA):
    return measured_field_settings(settings_from_slice(points), delta)


def measured_field_settings(settings, delta=DELTA):
    lower = np.asarray(settings, dtype=float).copy()
    upper = lower.copy()
    lower[:, LEAN] = np.maximum(lower[:, LEAN] - delta, BOUNDS[LEAN, 0])
    upper[:, LEAN] = np.minimum(upper[:, LEAN] + delta, BOUNDS[LEAN, 1])
    return (images(upper) - images(lower)) / (upper[:, LEAN] - lower[:, LEAN])[:, None]



def edge_basis(settings):
    """Supplied geometry defines switching features; one scale is fitted."""
    from grey_ones import N_PIX, SUB
    settings = np.atleast_2d(settings)
    ys_lo = np.arange(N_PIX * SUB) / SUB
    ys_mid = ys_lo + 0.5 / SUB
    columns = np.arange(N_PIX)[None, None, :]
    parts = []
    for params in np.array_split(settings, max(1, (len(settings) + 63) // 64)):
        cx, cy, height, width, lean = [params[:, k, None] for k in range(5)]
        radians = np.radians(lean)
        vertical_weight = np.clip(np.minimum(ys_lo + 1 / SUB, cy + height / 2)
            - np.maximum(ys_lo, cy - height / 2), 0, 1 / SUB) * SUB
        offset = cy - ys_mid
        center = cx + np.tan(radians) * offset
        left, right = center - width / 2, center + width / 2
        left_inside = (left[:, :, None] > columns) & (left[:, :, None] < columns + 1)
        right_inside = (right[:, :, None] > columns) & (right[:, :, None] < columns + 1)
        pattern = (right_inside.astype(float) - left_inside.astype(float)) * (
            vertical_weight * offset / np.cos(radians) ** 2)[:, :, None]
        parts.append(pattern.reshape(len(params), N_PIX, SUB, N_PIX).mean(2).reshape(len(params), -1))
    return np.concatenate(parts)


def grid(n_width, n_lean, offset):
    axes = [
        low + (high - low) * (np.arange(n) + offset) / n
        for n, (low, high) in zip((n_width, n_lean), BOUNDS[[WIDTH, LEAN]])
    ]
    a, b = np.meshgrid(*axes, indexing="ij")
    return np.column_stack((a.ravel(), b.ravel()))


def basis(points, degree):
    normalized = 2 * (points - BOUNDS[[WIDTH, LEAN], 0]) / np.diff(
        BOUNDS[[WIDTH, LEAN]], axis=1
    ).ravel() - 1
    width_terms = np.polynomial.chebyshev.chebvander(normalized[:, 0], degree)
    lean_terms = np.polynomial.chebyshev.chebvander(normalized[:, 1], degree)
    terms = [(i, j) for total in range(degree + 1)
             for i in range(total + 1) for j in [total - i]]
    matrix = np.column_stack([width_terms[:, i] * lean_terms[:, j] for i, j in terms])
    return matrix, terms


def summary(values):
    values = np.asarray(values)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)), "max": float(values.max())}


def field_scores(predicted, observed):
    norms = np.linalg.norm(observed, axis=1)
    predicted_norms = np.linalg.norm(predicted, axis=1)
    error = np.linalg.norm(predicted - observed, axis=1)
    valid = (norms > 1e-10) & (predicted_norms > 1e-10)
    cosine = np.full(len(norms), np.nan)
    cosine[valid] = np.sum(predicted[valid] * observed[valid], axis=1) / (
        norms[valid] * predicted_norms[valid]
    )
    relative = error / np.maximum(norms, 1e-10)
    scores = {
        "relative_arrow_error": summary(relative),
        "absolute_arrow_error_per_degree": summary(error),
        "direction_angle_degrees": summary(np.degrees(np.arccos(
            np.clip(cosine[valid], -1, 1)))) if valid.any() else None,
        "zero_observed_arrow_count": int((norms <= 1e-10).sum()),
        "undefined_angle_count": int((~valid).sum()),
    }
    return scores, relative, cosine


def integrate(arrows, angles, starting_image):
    increments = 0.5 * (arrows[:-1] + arrows[1:]) * np.diff(angles)[:, None]
    return np.vstack((starting_image, starting_image + np.cumsum(increments, axis=0)))


def path_scores(predicted, actual):
    errors = np.linalg.norm(predicted - actual, axis=1)
    outside = actual == 0
    return {
        "image_l2_error": summary(errors),
        "endpoint_image_l2_error": float(errors[-1]),
        "largest_absolute_pixel_error": float(np.abs(predicted - actual).max()),
        "largest_absolute_value_outside_target_ink": float(np.abs(predicted[outside]).max()),
        "largest_violation_of_zero_one_bounds": float(max(
            0, -predicted.min(), predicted.max() - 1)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = grid(17, 25, 0.5)
    heldout = grid(19, 37, 0.371)
    corners = np.asarray([(w, l) for w in BOUNDS[WIDTH] for l in BOUNDS[LEAN]])
    test = np.vstack((heldout, corners))
    assert not any(np.all(np.isclose(train, p, atol=1e-10, rtol=0), axis=1).any()
                   for p in test), "Training and evaluation settings overlap"
    train_arrows = measured_field(train)
    test_arrows = measured_field(test)
    half_step_arrows = measured_field(test, DELTA / 2)
    double_step_arrows = measured_field(test, DELTA * 2)
    results = {
        "status": "measured",
        "scope": "Width and lean vary; cx, cy, height fixed at range midpoints",
        "knob_order": NAMES,
        "fixed_settings": dict(zip(NAMES, map(float, SEED))),
        "ranges": dict(zip(NAMES, BOUNDS.tolist())),
        "training_points": len(train),
        "heldout_points": len(heldout),
        "boundary_stress_points": len(corners),
        "finite_difference_step_degrees": DELTA,
        "finite_difference_sensitivity_half_step": field_scores(
            half_step_arrows, test_arrows)[0],
        "finite_difference_sensitivity_double_step": field_scores(
            double_step_arrows, test_arrows)[0],
        "models": {},
        "limits": [
            "Known knob settings are model inputs; no replacement knobs are discovered.",
            "Measured arrows are finite differences, not guaranteed exact derivatives.",
            "Each polynomial term carries a 784-component coefficient pattern.",
            "Image errors compare with the target render, not distance to the entire family.",
            "No clipping, denoising, or projection is applied to integrated images.",
            "A finite validation set cannot establish exact coverage or validity.",
        ],
    }
    models = {}
    error_rows = []
    for degree in (1, 3, 5):
        design, terms = basis(train, degree)
        coefficients, _, rank, _ = np.linalg.lstsq(design, train_arrows, rcond=None)
        assert rank == len(terms), "Polynomial fit is rank deficient"
        predictions = basis(test, degree)[0] @ coefficients
        scores, relative, cosine = field_scores(predictions, test_arrows)
        name = f"polynomial_degree_{degree}"
        models[name] = (degree, coefficients)
        results["models"][name] = {
            "term_count": len(terms),
            "stored_scalar_coefficients": int(coefficients.size),
            "chebyshev_term_orders_width_lean": terms,
            "evaluation": scores,
            "paths": [],
        }
        np.savez_compressed(OUT / f"{name}.npz", coefficients=coefficients,
                            terms=np.asarray(terms), fixed_settings=SEED, ranges=BOUNDS)
        for index, (point, err, cos) in enumerate(zip(test, relative, cosine)):
            error_rows.append((name, *point, "heldout" if index < len(heldout)
                               else "boundary", err, cos))

    edge_train = edge_basis(settings_from_slice(train))
    edge_scale = float(np.sum(edge_train * train_arrows) / np.sum(edge_train ** 2))
    edge_predictions = edge_scale * edge_basis(settings_from_slice(test))
    edge_scores, relative, cosine = field_scores(edge_predictions, test_arrows)
    edge_name = "piecewise_edge_rule"
    results["models"][edge_name] = {
        "term_count": 1, "stored_scalar_coefficients": 1,
        "fitted_scale": edge_scale,
        "analytic_degrees_to_radians_scale": float(np.pi / 180),
        "construction": "Known stroke geometry, vertical coverage weights, and subrow sampling; one global coefficient fitted on training arrows",
        "equation": "V_lean[row,col] = c / SUB * sum(vertical_weight * (cy-y_subrow) / cos(lean_in_radians)^2 * (right_edge_inside_pixel - left_edge_inside_pixel))",
        "nondifferentiable_cases": "Exact pixel-boundary coincidences require one-sided or averaged derivatives",
        "evaluation": edge_scores, "paths": [],
    }
    for index, (point, err, cos) in enumerate(zip(test, relative, cosine)):
        error_rows.append((edge_name, *point, "heldout" if index < len(heldout) else "boundary", err, cos))
    np.savez_compressed(OUT / f"{edge_name}.npz", fitted_scale=edge_scale, fixed_settings=SEED, ranges=BOUNDS)
    rng = np.random.default_rng(20261004)
    control_settings = BOUNDS[:, 0] + rng.random((256, 5)) * np.diff(BOUNDS, axis=1).ravel()
    control_points = control_settings[:, [WIDTH, LEAN]]
    control_arrows = measured_field_settings(control_settings)
    results["full_space_control"] = {
        "sample_count": len(control_settings),
        "meaning": "All five settings vary; polynomial slice models cannot see cx, cy, or height. The edge rule explicitly uses all five known settings.",
        "models": {},
    }
    control_rows = []
    for name in list(models) + [edge_name]:
        if name == edge_name:
            prediction = edge_scale * edge_basis(control_settings)
        else:
            degree, coefficients = models[name]
            prediction = basis(control_points, degree)[0] @ coefficients
        scores, relative, cosine = field_scores(prediction, control_arrows)
        results["full_space_control"]["models"][name] = scores
        for params, err, cos in zip(control_settings, relative, cosine):
            control_rows.append((name, *params, err, cos))

    angles = np.linspace(*BOUNDS[LEAN], 901)
    widths = BOUNDS[WIDTH, 0] + np.asarray([0, 0.117, 0.333, 0.617, 0.883, 1]) * (
        BOUNDS[WIDTH, 1] - BOUNDS[WIDTH, 0])
    results["oracle_integration_paths"] = []
    display_indices = np.arange(0, len(angles), 10)
    for path_index, width in enumerate(widths):
        points = np.column_stack((np.full(len(angles), width), angles))
        actual = images(settings_from_slice(points))
        actual_arrows = measured_field(points)
        oracle = integrate(actual_arrows, angles, actual[0])
        results["oracle_integration_paths"].append(
            {"width": float(width), **path_scores(oracle, actual)})
        arrays = {"width": np.asarray(width), "lean_degrees": angles[display_indices],
                  "actual_images": actual[display_indices].reshape(-1, 28, 28).astype(np.float32),
                  "oracle_integrated_images": oracle[display_indices].reshape(-1, 28, 28).astype(np.float32),
                  "measured_arrows": actual_arrows[display_indices].reshape(-1, 28, 28).astype(np.float32)}
        for name, (degree, coefficients) in models.items():
            arrows = basis(points, degree)[0] @ coefficients
            predicted = integrate(arrows, angles, actual[0])
            results["models"][name]["paths"].append(
                {"width": float(width), **path_scores(predicted, actual)})
            arrays[f"{name}_images"] = predicted[display_indices].reshape(-1, 28, 28).astype(np.float32)
            arrays[f"{name}_arrows"] = arrows[display_indices].reshape(-1, 28, 28).astype(np.float32)
        arrows = edge_scale * edge_basis(settings_from_slice(points))
        predicted = integrate(arrows, angles, actual[0])
        results["models"][edge_name]["paths"].append({"width": float(width), **path_scores(predicted, actual)})
        arrays[f"{edge_name}_images"] = predicted[display_indices].reshape(-1, 28, 28).astype(np.float32)
        arrays[f"{edge_name}_arrows"] = arrows[display_indices].reshape(-1, 28, 28).astype(np.float32)
        np.savez_compressed(OUT / f"path_{path_index}.npz", **arrays)

    with (OUT / "arrow_errors.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "width", "lean_degrees", "evaluation_group",
                         "relative_arrow_error", "direction_cosine"])
        writer.writerows(error_rows)
    with (OUT / "full_space_control.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", *NAMES, "relative_arrow_error", "direction_cosine"])
        writer.writerows(control_rows)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print("Results saved to experiments/lean_field_fit_20261004/results")
    for name, model in results["models"].items():
        print(name, "terms:", model["term_count"],
              "heldout-and-boundary relative arrow error:",
              model["evaluation"]["relative_arrow_error"])


if __name__ == "__main__":
    main()
