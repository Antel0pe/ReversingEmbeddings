# Local linear bases at 17 generated-one starts

This repeats the root-neighborhood SVD at 17 starting images, using 512 fitting
neighbors and 256 independent checking neighbors at each of three image-space
radii: 0.1, 0.01, and 0.001. Every neighborhood varies all five generator
parameters. Only the 784 pixel differences from the corresponding start enter
SVD; parameter labels are used for generating samples and later diagnostics.

The root is `(cx, cy, height, width, lean) = (14, 14, 19.75, 3.2, 0)`.
Eight controlled starts change one root setting. Eight additional starts are
scrambled Sobol samples inside the middle 80% of the default parameter ranges.
All settings are saved in `summary.csv`, `results.json`, and the archive.

## Precision and counts

Numerical rank retains singular values above `1e-10 * largest_singular_value`.
Crucially, we separately reconstruct the held-out images: the largest absolute
784-pixel L2 error across all 51 fits is **1.50e-14**. Thus the reported fitted
spaces reconstruct independent samples at floating-point precision; the rank
threshold alone is not our evidence of precision.

| Starting image | Radius 0.1 rank | Radius 0.01 rank | Radius 0.001 rank |
| --- | ---: | ---: | ---: |
| Root | 9 | 9 | 9 |
| Right by 0.3 px | 9 | 9 | 9 |
| Down by 0.3 px | 9 | 9 | 9 |
| Height 20.1 px | 9 | 9 | 9 |
| Width 2.2 px | 9 | 9 | 9 |
| Width 4.2 px | 9 | 9 | 9 |
| Lean −7° | 15 | 11 | 9 |
| Lean 12.5° | 17 | 11 | 9 |
| Lean 30° | 33 | 13 | 9 |
| Spread 01 | 20 | 14 | 9 |
| Spread 02 | 32 | 13 | 9 |
| Spread 03 | 25 | 13 | 10 |
| Spread 04 | 13 | 12 | 10 |
| Spread 05 | 15 | 12 | 9 |
| Spread 06 | 20 | 14 | 9 |
| Spread 07 | 26 | 12 | 9 |
| Spread 08 | 12 | 12 | 9 |

These count fixed linear vectors for finite displacements. They do not count
physical generator parameters. Changing raster clipping regimes can increase
the linear rank even in a small neighborhood. Results describe sampled shells,
not a proof about every image on the continuous shell.

The canonical renderer is used with float64 output storage. Its coverage
calculation is unchanged, and every generated image is checked to cast exactly
to the default float32 result. This distinguishes geometry from float32 rounding;
the counts are not claims about the exact rank of quantized float32 arrays.

## Which vectors match?

All basis vectors are unit length. Coefficients may be positive or negative, so
one vector accounts for both directions of motion. The plotted sign is arbitrary.
In the root figure, positive coefficients predominantly give lean, increasing
width, rightward motion, downward motion, and increasing height, in that order.
Red gains ink and blue loses ink. These are approximate interpretations of the
leading vectors, not constraints imposed during fitting.

For individual-vector comparison, Hungarian assignment maximizes the sum of
absolute dot products between the five leading vectors at each start and the
root's five leading vectors. Signs are then aligned for display. No images are
translated, warped, or cropped. Low similarities remain low; the forced
assignment is not a claim that low-match columns have the same physical meaning.
Numbers in the figures are rounded: a displayed 1.000 need not be exact equality.

At radius 0.001, the first three matched vectors for the right-shifted start have
cosines 0.99999988, 0.99999970, and 0.99998666 with the root. For width 2.2 they are
0.99999950, 0.99998972, and 0.99999686. The other two leading vectors differ more.
At width 4.2, the first three matches are nearly orthogonal because the stroke's
side edges occupy different pixel columns. At lean 30°, every root match is below
0.20. Thus there are strong near-matches between some starts, but no universal
set of these five fixed pixel vectors over all sampled starts.

We also compare the spaces rather than just the chosen basis vectors. Principal
angles between five-dimensional spaces allow any orthogonal change of basis;
the full angle lists and all 136 pairs of starts are in `results.json`. These
confirm that the differences are not merely column order or sign changes.

A direct transfer check projects another start's displacement vectors into the
root's full nine-vector space. To condition the small directions better, a
separate reported check uses the root basis learned at radius 0.1 and transfers
it to radius-0.001 neighbors. It reconstructs the right-shifted and width-2.2
neighbors with absolute errors 5.50e-15 and 5.66e-15, respectively. In contrast,
its worst relative errors are 67.45% after moving down, 68.81% at height 20.1,
approximately 100% at width 4.2, and 99.998% at lean 30°.

Matching the individual SVD vectors and transferring an entire learned space
are different tests. The latter can succeed when particular basis vectors differ.
Shared full-space direction counts use an angle/rank tolerance of `1e-6` because
normalizing the smallest singular directions amplifies rounding error; these
counts are similarity diagnostics, not floating-point reconstruction guarantees.

## A renderer boundary changes the small-radius story

The straight root is a special reference. Its ends are at `y = 4.125` and
`23.875`, exactly on boundaries of the renderer's 1/16-pixel subrows. With lean
zero, adjacent subrow cross sections coincide. At nonzero lean, they differ.
The horizontal overlap is evaluated at each subrow's midpoint while vertical
coverage changes at subrow boundaries. This yields continuous images with
unequal one-sided derivatives at some parameter settings.

At the 30°-leaning root, decreasing the radius does not make the five-vector
relative error vanish in this test. Moving the center down by 0.023 px, away
from those boundaries and nearby horizontal clipping events, changes the result:

| Radius | Five-vector worst error, original tilted start | Five-vector worst error, center shifted down 0.023 px |
| --- | ---: | ---: |
| 0.001 | 1.01483% | 0.0127574% |
| 0.0003 | 1.01700% | 0.0038273% |
| 0.0001 | 1.01764% | 0.0012758% |

All six diagnostic fits retain nine vectors and reconstruct their held-out
changes with absolute errors below 1.5e-14. One-sided finite differences directly
check the kink: at the original tilted start, forward and backward height
derivatives differ by approximately 2.997% at steps of both 1e-5 and 1e-6 px.
At the shifted start that difference is below 1e-9 relative. This supports the
renderer-boundary explanation rather than attributing the persistent residual
solely to smooth curvature.

Therefore the earlier claim that five always emerges as the radius shrinks
needs a smoothness qualification. The finite-subrow renderer is piecewise
smooth; at its kinks the linear span of small changes can exceed the dimension
of the underlying five-parameter family even in the small-radius limit.

## Reproduction and saved data

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python analyze_generated_one_local_bases_across_starts.py
```

The script reuses the root experiment's shell search with an explicitly supplied
start and bounds. Position sampling extends to `[13, 16]` for both coordinates
to allow two-sided neighborhoods; height, width, and lean keep default bounds.
Fitting and checking use independent scrambled Sobol seeds 20261010 and 20261011,
transformed to Gaussian directions in normalized parameter space. The same rays
are reused across starts for consistent comparisons. This sampling is not uniform
in image-space shell area. Spread starts use seed 20261012.

- `matched_vectors.png`: seven controlled start images and their five matched vectors.
- `ranks_and_matches.png`: counts and root similarities for all 17 starts.
- `results.json`: rank tolerance sweeps, reconstruction errors, generator derivative
  diagnostics, root comparisons, and all pairwise comparisons at radius 0.001.
- `summary.csv`: compact settings, counts, errors, and matching scores.
- `bases_and_samples.npz`: source images, sampled settings/images, local bases,
  fitting/checking coefficients, matched vectors, and unit knob derivatives.
- `subrow_boundary_diagnostic.json`: the six smaller-radius boundary checks and
  one-sided derivative comparisons.

For example, reconstruct a held-out neighbor of the 30° start:

```python
import numpy as np

d = np.load("figures/generated_one_local_bases_across_starts/bases_and_samples.npz")
key = "lean_30_r_0p001"
basis = d[key + "_basis"]
coefficients = d[key + "_test_coefficients"][0]
image = (d["lean_30_start_image"] + coefficients @ basis).reshape(28, 28)
target = d[key + "_images"][512].reshape(28, 28)
print(np.linalg.norm(image - target))
```

The run checks shell distances, sampling bounds, float32 casting, orthonormality,
and fitting reconstruction. Saved arrays were independently loaded to verify
held-out reconstruction and reported counts. Both figures were rendered and
inspected using the repository's experiment-figure understanding check.
