# PCA versus SVD for the generated-one neighborhood method

**Centered PCA changes the finite-neighborhood fit and sometimes improves
reconstruction. It did not improve the physical-family matching in this
comparison, or remove the mixing of physical knobs within individual axes.**

This is a new comparison on the saved generated-one neighborhoods, executed
2026-10-17. It concerns the controlled `grey_ones.py` renderer, not MNIST.

## What was compared

We reused the exact images and training/checking split from:

- [17-start local bases](../generated_one_local_bases_across_starts/README.md):
  17 starts at image-L2 radii 0.1, 0.01, and 0.001.
- [Smaller-radius separation](../generated_one_scale_separation/README.md):
  five of those starts at radii 0.0001 and 0.00001.

There are **61 neighborhoods**, each with 512 training neighbors and 256
independently sampled checking neighbors. These checking samples were excluded
from each new basis fit, but have been inspected in earlier project work.
There is no new sampling or global nonlinear decoder in this comparison.
Coverage is float64, retaining the same 16-subrow rendering calculation.
Position sampling extends on both sides of the root, as in the source work.

For training rows `D = neighbor image - starting image`:

```text
Anchored SVD: SVD(D)
Centered PCA: SVD(D - mean_training_change)

SVD reconstruction = sum(coefficient_i * basis_i)
PCA reconstruction = mean_training_change + sum(coefficient_i * basis_i)
```

Both methods use direct full SVD, with no pixel standardization or whitening.
PCA stores a mean in addition to its counted basis vectors. Coefficients come
from projection of the observed target, so these are reconstruction tests,
not prediction of an unseen vector from its starting image.

The centered covariance equals the uncentered second moment minus the outer
product of the mean change with itself. Thus zero mean gives identical axes;
otherwise centering can change the fit. PCA is not a different, inherently
more accurate matrix decomposition. See the official
[PCA definition and solver notes](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

## Reconstruction of neighbor changes

Worst checking error = largest `L2(reconstruction - true change) / L2(true change)`.
These percentages describe reconstruction of full 784-pixel changes, not
variance retention, physical-knob recovery, or intrinsic dimension.

| Starting image | Radius | SVD, five vectors | PCA, five vectors plus mean |
| --- | ---: | ---: | ---: |
| Root | 0.1 | 1.181426% | 1.178713% |
| Root | 0.001 | 0.0118844% | 0.0118573% |
| Root | 0.00001 | 0.000118854% | 0.000118583% |
| Tilt 30 degrees, cap boundary | 0.001 | 1.014834% | 0.840840% |
| Near-boundary spread 03 | 0.001 | 1.379278% | 0.815830% |
| Near-boundary spread 03 | 0.00001 | 0.000129386% | 0.000130510% |

PCA has lower worst checking error in 54 of the 61 neighborhoods. At radius
0.1 it improves all 17 cases; the median PCA/SVD error ratio is 0.7573.
At radius 0.001 it improves 13 of 17 cases, with median ratio 0.997708.
There is no uniform improvement; spread 03 at the smallest radius is one
counterexample. The boundary case also retains a nonzero error floor.

The large near-boundary gain is mostly the affine offset. At spread 03 and
radius 0.001, using the PCA axes through the original start, without the mean,
gives **1.370247%**, close to SVD's 1.379278%. Adding the mean gives 0.815830%.
At the tilted boundary, the corresponding numbers are 1.010246% through the
start and 0.840840% with the mean. An affine fit is useful, but this does not
make its axes purer or give a direction basis through the original image.

With every retained nonzero vector, both methods reconstruct the checking
changes to below **1.5e-14 absolute image L2**. Numerical ranks agree in 60/61
cases; spread 04 at radius 0.00001 changes from 10 to 9 at relative singular
cutoff 1e-10. Counts near numerical floors do not establish new or removed
physical controls.

## Consistent matching across starting images

We reused the earlier
[observable matching rule](../generated_one_basis_followup/README.md) without
modification: six image measurements, their gradients, normalized direction
effect profiles, and one-to-one absolute-cosine assignment. Candidate orders
and signs are scrambled before matching. Physical labels enter only to evaluate
the assignments, using the same tiny central-difference targets as before.
At kinks those targets may average different one-sided derivatives.

| Radius | Direct pixel matching, SVD | Direct pixel matching, PCA | Observable matching, SVD | Observable matching, PCA |
| --- | ---: | ---: | ---: | ---: |
| 0.1 | 57/80 | 57/80 | 80/80 | 80/80 |
| 0.01 | 59/80 | 59/80 | 80/80 | 80/80 |
| 0.001 | 59/80 | 59/80 | 80/80 | 80/80 |

The scores are unchanged. A correctly matched category still need not be a
pure physical derivative. At the root and radius 0.001, the best scaled single
axis approximates the physical height vector with error **15.2113% for SVD**
and **15.2085% for PCA**. At radius 0.00001, the corresponding errors are
15.2109% and 15.2081%. Shrinking the neighborhood improves the local span
without removing this individual-axis mixture.

There is a geometric obstruction: both methods produce perpendicular axes,
while true height and width derivatives at the root have cosine 0.144059.
They cannot both be members of the same orthonormal basis. See
[the analytic overlap diagnostic](../generated_one_manifold_transport/README.md).
The manifold also admits alternative coordinate systems with the same root
directions and exact decoding, so identifying original physical controls
requires a selection principle beyond local geometry and consistency.

## Which other changes have evidence behind them?

- **For more accurate local-space estimates:** the previous
  [zero-radius extrapolation](../local_svd_limit_extrapolation/README.md)
  reduces root axis error against the analytically evaluated SVD limit from
  about 3.21e-6 at radius 0.0001 to 4.38e-11, using three finite radii.
  It estimates the SVD limit more accurately; that limit remains mixed.
  Boundary crossings spoil some windows, and excessively fine scales amplify
  roundoff. These are saved earlier results, not rerun here.
- **For actual physical derivatives:** use combinations of basis vectors,
  allowing nonorthogonal families. The earlier
  [true-knob reconstruction](../true_knobs_in_svd/README.md) reduces root-height
  error from about 0.153 with one vector to 3.14e-6 with five and 5.09e-12
  with nine at radius 0.0001. Its 256-ray sampling differs from this comparison's
  512-ray sampling. The coefficient solve receives the true target derivative;
  learning those weights without the original controls remains additional work.
- **For reconstructing joint changes:** the earlier
  [second-level experiment](../generated_one_second_level_svd/README.md)
  fits a quadratic decoder in a certified renderer cell and achieves worst
  held-out image L2 error 2.08e-14. This captures changing directions and
  interactions rather than freezing five vectors. It uses supplied coordinates
  and applies inside that cell; a global image-only chart was not established.
- **For stitching local spaces:** overlapping coordinate charts can allow full
  mixing and rescaling rather than matching only signs and permutations.
  [Local Tangent Space Alignment](https://arxiv.org/abs/cs/0212008) is a relevant
  method. Its use here remains a proposed experiment, with decoding and original
  knob identification requiring separate checks.

Kernel PCA is not automatically more accurate either. The existing
[physical-direction family comparison](../generated_one_knob_direction_families/README.md)
already tested it with a nonlinear inverse. For the global width-direction
family, five-coordinate median/worst errors were 68.5%/88.3% for SVD,
67.2%/88.8% for PCA, 37.9%/80.1% for a collection of 12 local PCA patches,
and 42.8%/91.8% for kernel PCA. Those tests compress observed, already labeled
physical directions across the full box; they do not discover knobs or use
the same input as the local-neighborhood comparison. Local patches store
additional means, bases, and a patch label. These are inspected earlier results,
not new runs.

## Reproduce and inspect

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/compare_local_pca_svd.py
```

- [`results.json`](results.json): all 61 comparisons, per-knob errors, matching
  assignments and scores, singular spectra, and method caveats.
- [`comparison.csv`](comparison.csv): 122 method/neighborhood rows.
- [`experiments/compare_local_pca_svd.py`](../../experiments/compare_local_pca_svd.py):
  reproducible calculation, with assertions verifying earlier SVD/matching
  results, the centering identity, and full-basis held-out errors.

At the root and radius 0.001, diagonalizing the centered covariance gives the
same five PCA axes as direct centered SVD to **5.33e-15** in unit-vector L2
after sign/permutation matching. This independently checks the mathematical
equivalence. Direct SVD is preferable for the tiny singular components because
forming covariance squares the spectral condition number.
