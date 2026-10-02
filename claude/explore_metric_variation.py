"""Measure variation of image-space lengths across the five-knob generator."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_tangent_events import jacobian64
from grey_ones import RANGES, KNOBS


def describe(values):
    return dict(zip(["min", "p10", "median", "p90", "max"],
                    np.quantile(values, [0, .1, .5, .9, 1]).tolist()))


def main():
    rng = np.random.default_rng(20260930)
    settings = RANGES[:, 0] + (.01 + .98 * rng.random((3000, 5))) * (RANGES[:, 1] - RANGES[:, 0])
    singular = np.empty((len(settings), 5))
    column_lengths = np.empty((len(settings), 5))
    for i, q in enumerate(settings):
        j = jacobian64(q)
        singular[i] = np.linalg.svd(j, compute_uv=False)
        column_lengths[i] = np.linalg.norm(j, axis=0)
    volume = singular.prod(axis=1)
    condition = singular[:, 0] / singular[:, -1]
    summary = {
        "sample_count": len(settings),
        "parameterization": "Each knob scaled to its full allowed range; interior uniform random settings",
        "column_pixel_l2_per_full_knob_range": {name: describe(column_lengths[:, j]) for j, name in enumerate(KNOBS)},
        "largest_local_stretch": describe(singular[:, 0]),
        "smallest_local_stretch": describe(singular[:, -1]),
        "condition_number": describe(condition),
        "five_dimensional_local_image_volume_factor": describe(volume),
        "volume_factor_max_over_min": float(volume.max() / volume.min()),
        "volume_factor_p90_over_p10": float(np.quantile(volume, .9) / np.quantile(volume, .1)),
    }
    output = Path(__file__).with_name("metric_variation_results.json")
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
