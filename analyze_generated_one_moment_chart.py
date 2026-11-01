"""Five knobs read out of six linear measurements of the pixels, in closed form.

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python analyze_generated_one_moment_chart.py

`figures/linear_5d_to_784d` showed that no linear map from five numbers to 784
pixels can describe this family: the best five-dimensional affine subspace holds
80% of the image variance and the rest is unreachable. That rules out linearity
on the DECODER side.

This script puts the linearity on the ENCODER side instead, where it survives.
Six fixed linear projections of the 784 coverage values -- total ink, and the
ink-weighted averages of x, y, x*x, y*y and x*y -- determine the five knobs
through an algebraic formula with nothing fitted:

    cx = mean x
    cy = mean y
    height = sqrt(12 * var y)
    tan(lean) = -cov(x, y) / var y
    width = sqrt(12 * (var x - tan(lean)^2 * var y))

The parallelogram is symmetric about its centre, so its centroid is exactly
(cx, cy); its vertical spread fixes the height; the tilt of its ink cloud fixes
the lean; and whatever horizontal spread the lean does not account for is the
width. Variances are corrected by pitch^2/12, the variance of a pixel box
itself, which is the leading grid artefact.

Four measurements:

1. RECOVERY. Error in the five knobs over the whole default box, and the ink
   identity (total ink should equal width*height exactly).
2. IS THE RESIDUAL THE METHOD OR THE RULER? Re-measure on finer grids. An error
   that falls as the square of the pixel pitch is the ruler.
3. THE LINEAR CEILING, for contrast: best rank-5/10/20/50 reconstruction.
4. A DECODER. Feed the recovered knobs into an affine warp of ONE image (root 1)
   and score held-out reconstruction, against the rank-5 and rank-10 ceilings.
   Control: the same warp driven by the TRUE knobs, which separates resampling
   error from recovery error.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy.ndimage import affine_transform

from coverage_grids import render_coverage
from grey_ones import KNOBS, N_PIX, RANGES, render

OUT = Path(__file__).resolve().parent / "figures/generated_one_moment_chart"
ROOT_SETTINGS = np.array([14.0, 14.0, 19.75, 3.2, 0.0])
ROOT_CX, ROOT_CY, ROOT_HEIGHT, ROOT_WIDTH = 14.0, 14.0, 19.75, 3.2


def settings(n, seed):
    rng = np.random.default_rng(seed)
    return RANGES[:, 0] + rng.random((n, 5)) * np.diff(RANGES).ravel()


def moments(images, pitch=1.0):
    """The six linear projections. Returns (total_ink, mean/variance summary)."""
    n_pix = images.shape[-1]
    centres = (np.arange(n_pix) + 0.5) * pitch
    x = np.broadcast_to(centres, (n_pix, n_pix))
    y = np.broadcast_to(centres[:, None], (n_pix, n_pix))
    ink = images.sum((1, 2))
    mean_x = (images * x).sum((1, 2)) / ink
    mean_y = (images * y).sum((1, 2)) / ink
    var_x = (images * x * x).sum((1, 2)) / ink - mean_x ** 2 - pitch ** 2 / 12
    var_y = (images * y * y).sum((1, 2)) / ink - mean_y ** 2 - pitch ** 2 / 12
    cov_xy = (images * x * y).sum((1, 2)) / ink - mean_x * mean_y
    return ink * pitch ** 2, mean_x, mean_y, var_x, var_y, cov_xy


def knobs_from_images(images, pitch=1.0):
    """The closed form. No fitting, no knowledge of the generator."""
    _, mean_x, mean_y, var_x, var_y, cov_xy = moments(images, pitch)
    height = np.sqrt(12 * var_y)
    shear = -cov_xy / var_y
    width = np.sqrt(np.clip(12 * (var_x - shear ** 2 * var_y), 0, None))
    return np.stack([mean_x, mean_y, height, width, np.degrees(np.arctan(shear))], 1)


results = {}

# ---- 1. recovery over the default box ----
P = settings(4000, seed=0)
images = render(P, dtype=np.float64)
recovered = knobs_from_images(images)
error = np.abs(recovered - P)
ink = images.sum((1, 2))
results["recovery_28"] = {
    "max_abs_error": dict(zip(KNOBS, error.max(0).tolist())),
    "median_abs_error": dict(zip(KNOBS, np.median(error, 0).tolist())),
    "max_ink_identity_error": float(np.abs(ink - P[:, 2] * P[:, 3]).max()),
}
print("1. Recovering the knobs from six linear projections, 28x28, 4000 settings.")
print(f"   {'':>10} " + " ".join(f"{k:>10}" for k in KNOBS))
print(f"   {'max error':>10} " + " ".join(f"{v:>10.5f}" for v in error.max(0)))
print(f"   {'median':>10} " + " ".join(f"{v:>10.5f}" for v in np.median(error, 0)))
print(f"   total ink vs width*height, worst case: "
      f"{results['recovery_28']['max_ink_identity_error']:.2e}")

# ---- 2. is the leftover the method or the grid? ----
Q = settings(200, seed=2)
print("\n2. The same closed form on finer grids.")
print(f"   {'grid':>9} {'pitch':>7} " + " ".join(f"{k:>9}" for k in KNOBS))
results["grid_refinement"] = []
for n_pix in (28, 56, 112, 224):
    pitch = 28.0 / n_pix
    err = np.abs(knobs_from_images(render_coverage(Q, n_pix), pitch) - Q).max(0)
    results["grid_refinement"].append(
        {"n_pix": n_pix, "pitch": pitch, "max_abs_error": err.tolist()})
    print(f"   {n_pix:>4}x{n_pix:<4} {pitch:>7.3f} "
          + " ".join(f"{v:>9.5f}" for v in err))
first = np.array(results["grid_refinement"][0]["max_abs_error"])
last = np.array(results["grid_refinement"][-1]["max_abs_error"])
results["error_shrink_over_8x_finer"] = (first / last).tolist()
print(f"   error shrink over an 8x finer grid: "
      + ", ".join(f"{k}={v:.0f}x" for k, v in zip(KNOBS, first / last))
      + "   (64x if it scales as pitch squared)")

# ---- 3. the linear ceiling ----
train, test = settings(4000, seed=11), settings(1500, seed=12)
A = render(train, dtype=np.float64).reshape(len(train), -1)
B = render(test, dtype=np.float64).reshape(len(test), -1)
mean = A.mean(0)
_, _, basis = np.linalg.svd(A - mean, full_matrices=False)
print("\n3. The linear ceiling, scored on held-out images.")
results["linear_ceiling"] = {}
for rank in (5, 10, 20, 50):
    centred = B - mean
    approx = (centred @ basis[:rank].T) @ basis[:rank]
    rel = float(np.linalg.norm(centred - approx) / np.linalg.norm(centred))
    results["linear_ceiling"][f"rank_{rank}"] = rel
    print(f"   rank {rank:>2}: {rel * 100:>6.2f}% of the variation left over")

# ---- 4. a decoder: warp one image ----
root = render(ROOT_SETTINGS[None], dtype=np.float64)[0]


def warp_root(knobs, order=1):
    cx, cy, height, width, lean = knobs
    shear = np.tan(np.radians(lean))
    y_scale, x_scale = ROOT_HEIGHT / height, ROOT_WIDTH / width
    matrix = np.array([[y_scale, 0.0], [x_scale * shear, x_scale]])
    shift = np.array([ROOT_CY - y_scale * cy,
                      ROOT_CX - x_scale * cx - x_scale * shear * cy])
    offset = matrix @ np.array([0.5, 0.5]) + shift - 0.5
    return affine_transform(root, matrix, offset=offset, order=order,
                            mode="constant", cval=0.0)


est = knobs_from_images(render(test, dtype=np.float64))
truth_images = render(test, dtype=np.float64)
print("\n4. Decoding by warping the single root image.")
results["warp_decoder"] = {}
for label, source, order in (("recovered knobs, linear resampling", est, 1),
                             ("recovered knobs, cubic resampling", est, 3),
                             ("TRUE knobs, cubic (control)", test, 3)):
    num = den = 0.0
    for knobs, target in zip(source, truth_images):
        num += np.linalg.norm(warp_root(knobs, order) - target) ** 2
        den += np.linalg.norm(target) ** 2
    rel = float(np.sqrt(num / den))
    results["warp_decoder"][label] = rel
    print(f"   {label:<36} {rel * 100:>6.2f}% relative RMS")
print(f"   {'rank-5 linear, for comparison':<36} "
      f"{results['linear_ceiling']['rank_5'] * 100:>6.2f}%")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "results.json").write_text(json.dumps(results, indent=2))
print(f"\nwrote {OUT / 'results.json'}")
