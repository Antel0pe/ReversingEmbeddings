"""Predict a calibrated width update from pixel arrays and explicit polynomials.

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/image_only_width_field.py

The fitted rule receives image arrays and observed update arrays only. The
renderer and control settings belong to the independent data adapter below.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/image-only-width-field-mpl")

from pathlib import Path
import itertools
import json
import sys

import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STEP = 0.01
SEED = 20261016
OUT = ROOT / "figures/image_only_width_field"


def polynomial_features(z, degree):
    """Only scalar products of measured pixel coordinates; no physical inputs."""
    z = np.atleast_2d(z)
    columns = [np.ones(len(z))]
    columns.extend(z[:, i] for i in range(z.shape[1]))
    if degree == 2:
        columns.extend(z[:, i] * z[:, j]
                       for i in range(z.shape[1]) for j in range(i, z.shape[1]))
    return np.column_stack(columns)


def fit_field(images, changes, coordinates=5, degree=2):
    """All fitting inputs are vectors. Return an inspectable fixed equation."""
    origin = images.mean(axis=0)
    centered = images - origin
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    resolved = int(np.sum(singular > max(singular[0] * 1e-12, 1e-12)))
    count = resolved if coordinates is None else min(coordinates, resolved)
    reader = right[:count].T / (singular[:count] / np.sqrt(len(images)))[None, :]
    measured = centered @ reader
    design = polynomial_features(measured, degree)
    patterns = np.linalg.lstsq(design, changes, rcond=1e-12)[0]
    return dict(origin=origin, reader=reader, patterns=patterns,
                degree=degree, resolved_image_rank=resolved,
                singular_values=singular)


def predict_field(images, model):
    """No settings, renderer, edges, grid adjacency, or knob inversion."""
    measured = (np.atleast_2d(images) - model["origin"]) @ model["reader"]
    return polynomial_features(measured, model["degree"]) @ model["patterns"]


def compact_linear_field(model):
    """Factor the affine image rule into only its resolved change patterns."""
    left, singular, right = np.linalg.svd(model["patterns"][1:], full_matrices=False)
    count = int(np.sum(singular > max(singular[0] * 1e-10, 1e-12)))
    reader = (model["reader"] @ left[:, :count]) * singular[:count]
    return dict(origin=model["origin"], anchor=model["patterns"][0],
                reader=reader, patterns=right[:count])


def predict_compact(images, model):
    return model["anchor"] + ((images - model["origin"]) @ model["reader"]) @ model["patterns"]


def metrics(truth, prediction):
    error = prediction - truth
    norms = np.linalg.norm(truth, axis=1)
    relative = np.linalg.norm(error, axis=1) / np.maximum(norms, 1e-30)
    return dict(pooled_relative_L2=float(np.linalg.norm(error) / np.linalg.norm(truth)),
                median_relative_L2=float(np.median(relative)),
                worst_relative_L2=float(relative.max()),
                worst_pixel_absolute_error=float(np.abs(error).max()),
                worst_image_L2_error=float(np.linalg.norm(error, axis=1).max()),
                states_under_1e_12_pixel_error=int(np.sum(np.abs(error).max(axis=1) < 1e-12)),
                test_states=len(truth))


# The adapter below is the only part allowed to use the original generator or
# numeric controls. Fitted functions above never call this adapter.
def observations(settings):
    from grey_ones import render
    images = render(settings, dtype=np.float64).reshape(-1, 784)
    targets = settings.copy()
    targets[:, 3] += STEP
    target_images = render(targets, dtype=np.float64).reshape(-1, 784)
    return images, target_images - images


def domains():
    root = np.array([14., 14., 19.75, 3.2, 0.])
    result = []

    def add(name, label, fixed, axes, bounds):
        result.append(dict(name=name, label=label, fixed=np.array(fixed),
                           axes=axes, bounds=np.array(bounds)))

    add("upright_patch", "Upright patch", root, [2, 3],
        [[19.1, 19.9], [2.2, 3.8]])
    add("tiny_tilted_patch", "Tiny tilted patch", [14.43, 14.62, 19.73, 3.17, 17.3],
        [2, 3, 4], [[19.7299, 19.7301], [3.1699, 3.1701], [17.299, 17.301]])
    add("narrow_lean", "Add lean ±0.5°", root, [2, 3, 4],
        [[19.1, 19.9], [2.2, 3.8], [-0.5, 0.5]])
    add("full_slice", "Full height/width/lean", root, [2, 3, 4],
        [[19., 20.5], [1.8, 4.6 - STEP], [-10., 35.]])
    add("all_five", "All five controls", root, list(range(5)),
        [[14., 15.], [14., 15.], [19., 20.5], [1.8, 4.6 - STEP], [-10., 35.]])
    return result


def settings_from_unit(unit, domain):
    result = np.tile(domain["fixed"], (len(unit), 1))
    bounds = domain["bounds"]
    result[:, domain["axes"]] = bounds[:, 0] + unit * np.diff(bounds, axis=1).ravel()
    return result


def independent_test_states(domain, seed):
    """Random, grid, corner, and screened transition cases, all excluded from fit."""
    d = len(domain["axes"])
    random = qmc.Sobol(d, scramble=True, seed=seed).random_base2(9)
    grid_count = 7 if d < 5 else 3
    lattice = np.array(list(itertools.product(np.linspace(0, 1, grid_count), repeat=d)))
    # Narrow pairs near all box faces exercise sampled edges, not only interiors.
    edge = qmc.Sobol(d, scramble=True, seed=seed + 1).random_base2(7)
    for j in range(d):
        edge[j::d, j] = np.where(np.arange(len(edge[j::d])) % 2, 1e-8, 1 - 1e-8)
    candidates = settings_from_unit(
        qmc.Sobol(d, scramble=True, seed=seed + 2).random_base2(10), domain)
    before, forward = observations(candidates)
    backward_settings = candidates.copy()
    backward_settings[:, 3] -= STEP
    from grey_ones import render
    backward_image = render(backward_settings, dtype=np.float64).reshape(-1, 784)
    second_difference = forward - (before - backward_image)
    score = np.linalg.norm(second_difference, axis=1) / np.maximum(np.linalg.norm(forward, axis=1), 1e-30)
    screened = candidates[np.argsort(score)[-64:]]
    ordinary = settings_from_unit(np.vstack([random, lattice, edge]), domain)
    settings = np.vstack([ordinary, screened])
    tags = (["sobol"] * len(random) + ["lattice"] * len(lattice)
            + ["near_box_face"] * len(edge) + ["large_width_second_difference"] * len(screened))
    return settings, np.array(tags), dict(candidate_count=len(candidates),
                                        retained=64, largest_score=float(score.max()))


def upright_four_image_control():
    """New independent renderer validation of the preceding chat's pixel reader."""
    train = np.tile([14., 14., 19.1, 2.2, 0.], (4, 1))
    train[:, 2] += .4 * np.array([0, 1, 0, 1])
    train[:, 3] += np.array([0, 0, 1, 1])
    x, _ = observations(train)
    A = x[0]
    B, C = x[1] - A, x[2] - A
    D = x[3] - x[1] - x[2] + A
    basis = np.column_stack([B, C, D])
    readers = np.linalg.pinv(basis, rcond=1e-12)
    uv = qmc.Sobol(2, scramble=True, seed=SEED).random_base2(10)
    p = np.tile([14., 14., 19.1, 2.2, 0.], (len(uv), 1))
    p[:, 2] = 19.1 + .8 * uv[:, 0]
    p[:, 3] = 2.2 + 1.6 * uv[:, 1]
    images, truth = observations(p)
    measured = (images - A) @ readers.T
    prediction = STEP * (C + measured[:, 0, None] * D)
    u, v = measured[:, 0], measured[:, 1]
    reconstructed = A + u[:, None] * B + v[:, None] * C + (u * v)[:, None] * D
    result = metrics(truth, prediction)
    result.update(training_images=4,
                  worst_source_reconstruction_pixel_error=float(np.abs(reconstructed - images).max()),
                  worst_source_reconstruction_image_L2=float(np.linalg.norm(reconstructed - images, axis=1).max()))
    assert result["worst_pixel_absolute_error"] < 1e-12
    assert result["worst_source_reconstruction_pixel_error"] < 1e-12
    np.savez_compressed(OUT / "four_image_control.npz", A=A, B=B, C=C, D=D,
                        readers=readers, test_settings=p, test_images=images,
                        true_changes=truth, predicted_changes=prediction)
    return result


def run_domain(domain, index):
    train_unit = qmc.Sobol(len(domain["axes"]), scramble=True,
                           seed=SEED + 20 * index + 1).random_base2(10)
    train_settings = settings_from_unit(train_unit, domain)
    train_images, train_changes = observations(train_settings)
    test_settings, tags, screening = independent_test_states(domain, SEED + 20 * index + 4)
    images, truth = observations(test_settings)
    assert not np.any(np.all(test_settings[:, None, :] == train_settings[None, :, :], axis=2))
    models = {}
    predictions = {}
    models["linear_5"] = fit_field(train_images, train_changes, 5, 1)
    # Reuse the pixel readers: changing the degree must not change observations.
    models["quadratic_5"] = dict(models["linear_5"])
    z = (train_images - models["quadratic_5"]["origin"]) @ models["quadratic_5"]["reader"]
    models["quadratic_5"]["degree"] = 2
    models["quadratic_5"]["patterns"] = np.linalg.lstsq(
        polynomial_features(z, 2), train_changes, rcond=1e-12)[0]
    models["linear_all_resolved"] = fit_field(train_images, train_changes, None, 1)
    for name, model in models.items():
        predictions[name] = predict_field(images, model)
        np.savez_compressed(OUT / f"{domain['name']}_{name}.npz", **model)
    compact = compact_linear_field(models["linear_5"])
    predictions["compact_linear"] = predict_compact(images, compact)
    np.savez_compressed(OUT / f"{domain['name']}_compact_linear.npz", **compact)
    predictions["frozen"] = np.broadcast_to(train_changes.mean(axis=0), truth.shape)
    anchor = train_changes.mean(axis=0)
    _, singular, patterns = np.linalg.svd(train_changes - anchor, full_matrices=False)
    resolved = int(np.sum(singular > max(singular[0] * 1e-10, 1e-12))) if singular[0] > 1e-15 else 0
    pattern_floor = {}
    for k in sorted(set([1, 2, 3, 5, 9, 16, 32, 64, 128, resolved])):
        if k <= 0 or k > len(patterns):
            continue
        # This diagnostic uses true held-out updates to choose their best weights.
        # It measures representability in stored patterns, not an available predictor.
        best = anchor + ((truth - anchor) @ patterns[:k].T) @ patterns[:k]
        pattern_floor[str(k)] = metrics(truth, best)
        if k == 5:
            predictions["best_fit_5_patterns"] = best
    all_metrics = {name: metrics(truth, pred) for name, pred in predictions.items()}
    groups = {tag: {name: metrics(truth[tags == tag], pred[tags == tag])
                    for name, pred in predictions.items()} for tag in np.unique(tags)}
    quadratic = models["quadratic_5"]
    result = dict(label=domain["label"], fixed_settings=domain["fixed"].tolist(),
                  varied_axes=domain["axes"], source_bounds=domain["bounds"].tolist(),
                  training_pairs=len(train_images), training_renderer_images=2 * len(train_images),
                  test_states=len(images), screening=screening,
                  resolved_source_affine_rank=quadratic["resolved_image_rank"],
                  measured_pixel_coordinates=quadratic["reader"].shape[1],
                  quadratic_pixel_patterns=len(quadratic["patterns"]),
                  compact_linear_patterns=len(compact["patterns"]),
                  resolved_width_change_affine_rank=resolved,
                  metrics=all_metrics, test_groups=groups,
                  best_fit_pattern_errors=pattern_floor)
    np.savez_compressed(OUT / f"{domain['name']}_evaluation.npz",
                        train_settings=train_settings, train_images=train_images,
                        train_changes=train_changes, test_settings=test_settings,
                        test_groups=tags, test_images=images, true_changes=truth,
                        pattern_singular_values=singular, **predictions)
    print(json.dumps(dict(domain=domain["name"], results=all_metrics,
                          source_rank=result["resolved_source_affine_rank"],
                          change_rank=resolved)), flush=True)
    return result


def permutation_control():
    """Confirm that neither image adjacency nor index meaning supplies the rule."""
    saved = np.load(OUT / "narrow_lean_evaluation.npz")
    order = np.random.default_rng(SEED).permutation(784)
    x, y = saved["train_images"][:, order], saved["train_changes"][:, order]
    model = fit_field(x, y, 5, 2)
    shuffled = predict_field(saved["test_images"][:, order], model)
    restored = shuffled[:, np.argsort(order)]
    discrepancy = float(np.max(np.abs(restored - saved["quadratic_5"])))
    assert discrepancy < 1e-10
    return dict(maximum_prediction_difference=discrepancy,
                interpretation="Invariance to shuffled pixel order, not proof of generality.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(width_step_pixels=STEP, storage="float64",
                  prediction_inputs="Current 784-pixel vector; fixed trained arrays. No control settings.",
                  training_information="Paired before/after vectors for a labeled width +0.01 px action.",
                  measured_target="Finite +0.01 px width change, not an instantaneous derivative.",
                  coordinate_equation="z=(image-origin)@reader",
                  quadratic_equation="change=P0+sum_i(z_i*P_i)+sum_i<=j(z_i*z_j*P_ij)",
                  action_scope="One fixed positive width step. No other action or arbitrary step size is learned.",
                  four_image_control=upright_four_image_control(), cases={})
    for index, domain in enumerate(domains()):
        result["cases"][domain["name"]] = run_domain(domain, index)
    result["lean_extent_sweep"] = {}
    for extent in [1., 2., 5.]:
        domain = dict(domains()[2])
        domain["name"] = f"lean_extent_{extent:g}"
        domain["label"] = f"Lean ±{extent:g}°"
        domain["bounds"] = domain["bounds"].copy()
        domain["bounds"][-1] = [-extent, extent]
        # Identical normalized train/test coordinates to the narrow-lean case.
        result["lean_extent_sweep"][str(extent)] = run_domain(domain, 2)
    result["pixel_permutation_control"] = permutation_control()
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Image-only width field experiment complete.", flush=True)


if __name__ == "__main__":
    main()
