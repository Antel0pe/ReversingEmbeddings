"""Measure support and affine span of only the generated-one coverage family.

Uses the same overlap law as grey_ones.render, preserving float64 values before
the generator's final float32 cast. No external dataset is loaded.
"""

import json
from itertools import product
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import svd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grey_ones import RANGES, SUB


def coverage64(parameters):
    parameters = np.atleast_2d(np.asarray(parameters, np.float64))
    ylo = np.arange(28 * SUB) / SUB
    ymid = ylo + 0.5 / SUB
    cols = np.arange(28)
    output = np.empty((len(parameters), 784), dtype=np.float64)
    for start in range(0, len(parameters), 256):
        p = parameters[start:start + 256]
        cx, cy, height, width, lean = [p[:, j, None] for j in range(5)]
        top, bottom = cy - height / 2, cy + height / 2
        vertical = np.clip(np.minimum(ylo + 1 / SUB, bottom) - np.maximum(ylo, top), 0, 1 / SUB) * SUB
        center = cx - np.tan(np.deg2rad(lean)) * (ymid - cy)
        left, right = center - width / 2, center + width / 2
        horizontal = np.clip(np.minimum(right[:, :, None], cols + 1) - np.maximum(left[:, :, None], cols), 0, 1)
        output[start:start + len(p)] = (vertical[:, :, None] * horizontal).reshape(len(p), 28, SUB, 28).mean(2).reshape(len(p), 784)
    return output


def possible_ink_mask():
    """Exact possible nonzero pixels under the allowed continuous knob box.

For a given sub-strip, restrict vertical center to settings for which a
maximal-height stroke reaches it. Extrema of horizontal stroke center over
that restricted interval occur at box corners in center and lean slope.
"""
    ylo = np.arange(28 * SUB) / SUB
    ymid = ylo + 0.5 / SUB
    slopes = np.tan(np.deg2rad(RANGES[4]))
    cy_lower = np.maximum(RANGES[1, 0], ylo - RANGES[2, 1] / 2)
    cy_upper = np.minimum(RANGES[1, 1], ylo + 1 / SUB + RANGES[2, 1] / 2)
    vertical_possible = cy_lower < cy_upper
    candidates = np.stack([
        cx - slope * (ymid - cy)
        for cx in RANGES[0] for cy in (cy_lower, cy_upper) for slope in slopes
    ])
    left_min = candidates.min(0) - RANGES[3, 1] / 2
    right_max = candidates.max(0) + RANGES[3, 1] / 2
    cols = np.arange(28)
    overlap_possible = (left_min[:, None] < cols[None, :] + 1) & (right_max[:, None] > cols[None, :])
    return (vertical_possible[:, None] & overlap_possible).reshape(28, SUB, 28).any(1).reshape(784)


def main():
    rng = np.random.default_rng(20260930)
    random_unit = rng.random((4000, 5))
    lattice_unit = np.array(list(product(np.linspace(0, 1, 5), repeat=5)))
    near_edge_unit = np.array(list(product([0.001, 0.5, 0.999], repeat=5)))
    unit = np.concatenate([random_unit, lattice_unit, near_edge_unit])
    settings = RANGES[:, 0] + unit * (RANGES[:, 1] - RANGES[:, 0])
    images = coverage64(settings)
    certified_mask = possible_ink_mask()
    assert np.max(np.abs(images[:, ~certified_mask])) == 0
    active_mask = np.ptp(images, axis=0) > 1e-12
    centered = images[:, certified_mask] - images[0, certified_mask]
    _, singular, vt = svd(centered, full_matrices=False, check_finite=False)
    numerical_rank = int(np.count_nonzero(singular > 1e-10 * singular[0]))
    full_row_mass = images.reshape(-1, 28, 28).sum(axis=2)[:, 6:23]
    max_row_mass_error = float(np.max(np.abs(full_row_mass - settings[:, 3, None])))
    assert max_row_mass_error < 1e-12
    assert numerical_rank == certified_mask.sum() - 16
    test_unit = rng.random((1000, 5))
    test_settings = RANGES[:, 0] + test_unit * (RANGES[:, 1] - RANGES[:, 0])
    heldout = coverage64(test_settings)[:, certified_mask] - images[0, certified_mask]
    out_of_span = heldout - (heldout @ vt[:numerical_rank].T) @ vt[:numerical_rank]
    heldout_max_residual = float(np.linalg.norm(out_of_span, axis=1).max())
    thresholds = [1e-6, 1e-8, 1e-10, 1e-12]
    summary = {
        "settings": len(settings),
        "settings_by_design": {"random": 4000, "lattice": len(lattice_unit), "near_edges": len(near_edge_unit)},
        "certified_possible_ink_pixels": int(certified_mask.sum()),
        "certified_constant_zero_pixels": int((~certified_mask).sum()),
        "sampled_variable_pixels": int(active_mask.sum()),
        "fully_covered_rows": list(range(6, 23)),
        "equal_full_row_mass_constraints": 16,
        "max_full_row_mass_error": max_row_mass_error,
        "analytic_affine_upper_bound": int(certified_mask.sum() - 16),
        "numerical_affine_rank": numerical_rank,
        "sampled_affine_rank_by_relative_singular_threshold": {
            str(t): int(np.count_nonzero(singular > t * singular[0])) for t in thresholds
        },
        "last_nonzero_singular_value": float(singular[numerical_rank - 1]),
        "first_numerical_zero_singular_value": float(singular[numerical_rank]),
        "heldout_count": len(test_settings),
        "heldout_max_pixel_l2_residual": heldout_max_residual,
        "singular_values_first_ten": singular[:10].tolist(),
        "singular_values_last_ten": singular[-10:].tolist(),
    }
    destination = Path(__file__).with_name("global_span_results.json")
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
