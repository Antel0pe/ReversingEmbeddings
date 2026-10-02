"""Identify linear laws among generated-one pixel coordinates.

This inspects the float64 coverage formula only. No learned projection is used.
"""

import json
from itertools import product
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_global_span import coverage64, possible_ink_mask
from grey_ones import RANGES


def main():
    rng = np.random.default_rng(20260930)
    unit = np.concatenate([
        rng.random((4000, 5)),
        np.array(list(product(np.linspace(0, 1, 5), repeat=5))),
        np.array(list(product([.001, .5, .999], repeat=5))),
    ])
    q = RANGES[:, 0] + unit * (RANGES[:, 1] - RANGES[:, 0])
    images = coverage64(q)
    variable = np.ptp(images, axis=0) > 1e-12
    coordinates = np.flatnonzero(variable)
    x = images[:, variable] - images[0, variable]
    _, singular, vt = np.linalg.svd(x, full_matrices=False)
    numerical_null = vt[singular < 1e-10 * singular[0]]
    full_rows = np.arange(6, 23)
    row_sums = []
    for row in full_rows:
        mask = ((coordinates // 28) == row).astype(np.float64)
        row_sums.append(mask)
    expected_constraints = np.diff(np.stack(row_sums), axis=0)
    expected_basis = np.linalg.qr(expected_constraints.T)[0][:, :len(expected_constraints)]
    expected_residual = np.max(np.linalg.norm(x @ expected_basis, axis=0))
    null_projected = numerical_null - numerical_null @ expected_basis @ expected_basis.T
    remainder_singular = np.linalg.svd(null_projected, compute_uv=False)
    summary = {
        "variable_pixels": int(variable.sum()),
        "numerical_affine_rank": int(np.count_nonzero(singular >= 1e-10 * singular[0])),
        "numerical_linear_constraint_count": len(numerical_null),
        "equal_row_mass_constraints": len(expected_constraints),
        "max_equal_row_mass_residual": float(expected_residual),
        "null_space_after_known_row_constraints_max_singular": float(remainder_singular.max()),
        "additional_numerical_constraints": int(np.count_nonzero(remainder_singular > 1e-8)),
    }
    destination = Path(__file__).with_name("linear_constraint_results.json")
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
