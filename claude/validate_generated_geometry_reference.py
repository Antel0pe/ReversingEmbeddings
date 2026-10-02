"""Compare saved JavaScript float32 fixtures against the repository's actual renderer.

Run from the repository root:
    python claude/validate_generated_geometry_reference.py

This validator was prepared but could not be executed in the original session.
It imports only grey_ones.py and never reads an MNIST dataset.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grey_ones import render

fixture_path = Path(__file__).with_name("generated_geometry_reference_fixtures.json")
fixtures = json.loads(fixture_path.read_text())["fixtures"]
parameters = np.array([item["q"] for item in fixtures], dtype=np.float64)
expected = np.array([item["pixels"] for item in fixtures], dtype=np.float32)
actual = render(parameters).reshape(-1, 784)
difference = actual.astype(np.float64) - expected.astype(np.float64)
print(json.dumps({
    "images": len(fixtures),
    "max_pixel_difference": float(np.abs(difference).max()),
    "max_image_distance": float(np.linalg.norm(difference, axis=1).max()),
    "unequal_pixels": int(np.count_nonzero(actual != expected)),
}, indent=2))
# Allow one float32-scale discrepancy across platform math implementations.
assert np.max(np.abs(difference)) < 1e-6

