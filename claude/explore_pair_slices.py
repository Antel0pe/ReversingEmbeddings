"""Compare image-space affine spans of the ten two-knob slices."""

import json
from itertools import combinations, product
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import svdvals

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_global_span import coverage64
from grey_ones import RANGES, KNOBS


def main():
    rng = np.random.default_rng(20260930)
    base = np.array([14.43, 14.62, 19.77, 3.2, 17.3])
    grid = np.array(list(product(np.linspace(0, 1, 61), repeat=2)))
    extras = rng.random((2000, 2))
    near = np.array(list(product([0.001, .5, .999], repeat=2)))
    unit = np.vstack([grid, extras, near])
    records = []
    for i, j in combinations(range(5), 2):
        q = np.tile(base, (len(unit), 1))
        q[:, [i, j]] = RANGES[[i, j], 0] + unit * (RANGES[[i, j], 1] - RANGES[[i, j], 0])
        images = coverage64(q)
        variable = np.ptp(images, axis=0) > 1e-12
        centered = images[:, variable] - images[0, variable]
        singular = svdvals(centered, check_finite=False)
        rank = int(np.count_nonzero(singular > 1e-10 * singular[0]))
        records.append({
            "knobs": [KNOBS[i], KNOBS[j]], "samples": len(unit),
            "sampled_variable_pixels": int(variable.sum()),
            "numerical_affine_rank": rank,
            "last_nonzero_singular": float(singular[rank-1]),
            "next_singular": float(singular[rank]) if rank<len(singular) else None,
        })
    result = {"fixed_settings": base.tolist(),
              "sampling": "61-by-61 parameter grid, 2000 random points, 9 near-edge points per pair",
              "records": records}
    output = Path(__file__).with_name("pair_slice_results.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
