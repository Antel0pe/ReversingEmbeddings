# Linear bases from small neighborhoods of the root 1

The root is `(cx, cy, height, width, lean) = (14, 14, 19.75, 3.2, 0)`.
For each radius, 512 generated neighbors fit a basis and 256 independent
neighbors check it. All neighbors have that Euclidean distance from the root
in 784-dimensional pixel space. Seven radii range from 1 to 0.001.

## What the method returns

Each input is `neighbor_image - root_image`. Uncentered SVD produces orthonormal
pixel-change vectors, singular values measuring their importance, and a
coefficient for each basis vector for each image. The basis vectors need not be
observed sample vectors. This run checks that none of its first five basis
vectors equals a normalized fitting vector, even allowing sign reversal.

For a basis `B` with vectors in its rows:

```python
coefficients = (image.ravel() - root.ravel()) @ B.T
reconstruction = root.ravel() + coefficients @ B
```

There is no mean subtraction beyond subtracting the specified root, no pixel
standardization, and no parameter label input to SVD. An SVD prefix minimizes
aggregate squared fitting error; a count meeting a worst-error threshold below
is the count within this ordered basis, not a proof that no other subspace could
meet that worst-error threshold with fewer vectors.

## Results

Errors are the worst among 256 held-out images, divided by the image's distance
from the root. They measure errors in the changes, rather than dividing by the
much larger norm of the full image.

| Shell radius | Five-vector worst error | SVD vectors for at most 1% worst error | Numerical linear rank |
| --- | ---: | ---: | ---: |
| 1 | 51.7492% | 15 | 15 |
| 0.3 | 3.5087% | 9 | 9 |
| 0.1 | 1.1814% | 6 | 9 |
| 0.03 | 0.3559% | 5 | 9 |
| 0.01 | 0.1188% | 5 | 9 |
| 0.003 | 0.0356% | 5 | 9 |
| 0.001 | 0.0119% | 5 | 9 |

At radius 0.001, four SVD vectors leave 60.4364% worst error, whereas five leave
0.0119%. The five-vector error shrinks with radius. Four additional small
directions remain measurable at finite radii; using all nine reconstructs the
small-shell fitting and held-out changes to floating-point accuracy. These are
finite-sample numerical ranks at a singular-value threshold of `1e-10 * s[0]`,
not exact continuum dimension proofs. Tolerance sweeps are in `results.json`.

After fitting, the first five basis vectors at radius 0.001 were compared with
central finite differences of each known generator knob. They align most closely
with lean, width, horizontal position, vertical position, and height, respectively.
Their absolute cosines with those derivatives are approximately 1.0000, 1.0000,
1.0000, 0.9987, and 0.9884. These labels are a diagnostic using the known generator;
they were not supplied to fitting. The vectors are not necessarily pure knobs.

Pivoted QR also selects actual observed displacement vectors as a nonorthogonal
basis. The selected full-rank bases reconstruct held-out changes to the same
floating-point accuracy. Their sample indices, vectors, and coefficients are saved.

## Sampling and precision

Scrambled Sobol samples, transformed to Gaussian directions, define rays in
range-normalized parameter space. Bisection locates a shell crossing on each
ray. These are diverse samples, not a uniform image-metric shell distribution
or an exhaustive shell. Fitting and checking use independent seeds, 20261010
and 20261011. The largest observed shell-distance error was below `2e-10`.

Position bounds explicitly extend to `[13, 15]` for both center coordinates,
so the root has neighbors on both sides. Height, width, and lean retain their
default bounds. Parameters are used to generate data and later diagnose the
basis; only pixel displacements enter SVD.

The canonical renderer now accepts an optional output-storage `dtype`, retaining
float32 by default. Rank diagnostics use float64 storage of the same 16-subrow
coverage calculation; every sampled float64 image is checked to cast exactly
to the default float32 image. Independently fitting the default float32 images
also gives an excellent five-vector approximation: 0.0187% worst error at radius
0.001. Float32 rounding affects the smallest residual directions, so numerical
rank should not be inferred from that rounding floor.

Ten pure-knob probes (positive and negative motion for each knob) are generated
only after fitting. Their radius is `min(shell_radius, 0.3)` because pure height
cannot reach distance 1 within its bounds. At radius 0.001, their five-vector
worst relative error is 0.000960%.

This experiment discovers a local linear description. Arbitrary coefficients
are not guaranteed to produce a valid generated 1, and these fixed vectors do
not define controls over the entire family.

## Files and reproduction

Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python analyze_root_one_local_svd.py
```

- `basis_vectors.png`: root followed by the first five learned vectors at radius 0.001.
- `reconstruction_by_radius.png`: held-out reconstruction errors for all radii.
- `results.json`, `summary.csv`: counts, errors, singular values, and diagnostics.
- `bases_and_samples.npz`: all sampled settings/images, learned bases, coefficients,
  selected observed-vector bases, and known-knob derivatives used for diagnostics.

Load the smallest-neighborhood basis and reconstruct one held-out image:

```python
import numpy as np

d = np.load("figures/root_one_local_svd/bases_and_samples.npz")
basis = d["r_0p001_basis"]         # (9, 784); first five are dominant
c = d["r_0p001_test_coefficients"][0]
approximation = (d["root_image"] + c[:5] @ basis[:5]).reshape(28, 28)
full_reconstruction = (d["root_image"] + c @ basis).reshape(28, 28)
```

The run verifies shell distances, bounds, float32 casting, basis orthonormality,
fitting reconstruction, and agreement of float32 and float64 five-vector errors.
Both figures were rendered and inspected against the experiment-figure
understanding check: object, baseline, encoding, units, visible evidence,
sampling limits, and numerical precision are stated on each figure.
