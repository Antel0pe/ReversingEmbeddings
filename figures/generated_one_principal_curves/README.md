# Principal curves of the generated-one family

This experiment fits curves directly to the 784 coverage values from `grey_ones.render`; it does not fit in a PCA projection. All five knobs vary independently and uniformly over `grey_ones.RANGES`.

## Results on unseen images

| Coordinates | Sequential curves | Centered PCA |
| --- | ---: | ---: |
| 1 | 89.92% | 46.11% |
| 2 | 94.09% | 63.77% |
| 3 | 95.74% | 72.04% |
| 4 | 96.68% | 76.64% |
| 5 | 97.28% | 80.20% |

The first curve is strongly associated with lean (held-out Spearman rank correlation 0.971), but it also snakes back and forth through width and horizontal position. Its length is 145.59 pixel-distance units, compared with 28.77 for an exact lean-only sweep at the other knobs’ midpoints. That flexibility is a major source of its high score. Its approximate closest allowed strokes range from thin to thick repeatedly; their heights remain near the midrange.

Later curves fit the residual coverage after earlier projections. They mix width, position, and remaining lean-dependent patterns: the largest absolute single-knob correlations are only 0.30, 0.37, 0.21, and 0.07 for curves 2–5. They should not be interpreted as five clean physical knobs.

## How this relates to PCA

PC1 is a straight line through the mean image. A principal curve replaces that straight line with a bendable path through image space. Each location on the path is a complete 28×28 image, and a sample is encoded by its nearest location on the path. The fit alternates nearest-point projection with a smooth fit of the pixel values against path position.

The numerical implementation uses 65 polyline nodes and a second-difference smoothing penalty. Each projection searches every segment using all 784 pixels. Smoothing is a penalized linear spline, followed by arc-length reparameterization. These are finite-resolution, regularized principal-curve approximations, not a proof of exact self-consistency or a globally optimal curve.

Curve 1 is a principal-curve fit; curves 2–5 are a sequential residual extension, not a canonical ordered list analogous to PC1–PC5. The encoder projects an image onto curve 1, subtracts that predicted image difference, projects the residual onto curve 2, and so on. The decoder adds the training mean and the five selected curve contributions. Independent choices of these coordinates need not give valid strokes. The curves are not required to be orthogonal and the later fit is order-dependent.

## What “variance captured” means

`100 × (1 − sum of squared reconstruction errors / sum of squared distances to the training-mean image)`

Every square is a coverage-value error across all 784 pixels. There are 4,096 training images, 2,048 validation images for smoothing/iteration selection, and 4,096 untouched test images, from independently scrambled Sobol sequences with seeds 731–733. Rendering is float32; fitting and scoring are float64. The PCA basis and mean are learned on the same training set. This experiment concerns the full default box, not the earlier radius-4.5 shell.

A nonlinear curve has many more fitted shape parameters than a PCA line. Equal numbers of output coordinates do not mean equal model complexity. These nonlinear reconstruction scores do not have PCA’s eigenvalue decomposition or orthogonality guarantee.

## Smoothing controls the answer

| Smoothing penalty | Curve length | Test variation captured |
| ---: | ---: | ---: |
| 1 | 145.59 | 89.92% |
| 100 | 66.17 | 84.38% |
| 1000 | 38.95 | 78.43% |
| 10000 | 27.15 | 74.06% |
| 100000 | 16.67 | 66.92% |

All sequential fits selected penalty 1 from the tested values 1, 10, 100, 1000, using validation reconstruction error. Fits are capped at 35 iterations and retain the best validation iterate, so this is an approximate finite computation. The extended sensitivity check shows that a smoother, lean-like path of length 27.15 captures 74.06%, while a highly smoothed 16.67-long path captures 66.92%. There is no unique variance percentage independent of smoothing, node count, initialization, and the knob-sampling distribution.

A separate PC2-initialized first curve captures 88.23%, despite a very different coordinate association with lean (rank correlation 0.265 rather than 0.971). This is further evidence against treating the curve as a uniquely identified physical coordinate.

## Boundaries, joint changes, and validity

| Probe set | Images | One curve | Five curves | Worst five-curve image L2 error |
| --- | ---: | ---: | ---: | ---: |
| lattice_5_per_knob | 3125 | 86.82% | 95.85% | 2.283 |
| all_32_corners | 32 | 81.92% | 92.02% | 2.283 |
| near_faces | 1024 | 88.32% | 96.69% | 1.755 |
| joint_parameter_midpoints | 1024 | 89.15% | 97.08% | 1.209 |

The 5×5×5×5×5 lattice and the corners include boundary combinations that uniform sampling gives little weight. The joint-midpoint probe renders the parameter midpoint of two independent settings and encodes that image; it is not a test of linearly interpolating the curve coordinates. These probes test sampled reconstruction, not continuous coverage or collision-free coordinates.

The first curve has pixels from −0.1065 to 1.0786; the five-curve test reconstruction has pixels from -0.2942 to 1.3335. The fit is unconstrained, so averaged or spline-interpolated images can leave the coverage cube or violate the parallelogram rule. Each plotted curve-point validity error is distance to an approximately fitted allowed renderer image; it is not a certificate of the globally nearest point. Across the 65 first-curve nodes those errors have median 0.373 and maximum 0.570.

The five physical parameters plus the renderer remain the exact construction. The fitted curves are useful nonlinear summaries, not a replacement for that complete coordinate system.

## Reproduce and inspect

```sh
.venv/bin/python analyze_generated_one_principal_curves.py
.venv/bin/python supplement_generated_one_principal_curves.py
.venv/bin/python make_generated_one_principal_curve_figures.py
```

- `index.html`: offline explorer for curve shape, smoothing, and held-out reconstructions.
- `first_curve.png`: actual curve images, approximately fitted valid strokes, and projected curve shape.
- `variance_capture.png`: unseen-image scores, boundary scores, and smoothing tradeoff.
- `residual_curves.png`: later curve changes, with a common signed coverage scale.
- `results.json`: metrics and probe results.
- `fit_history.json`: training/validation fitting history.
- `model_and_samples.npz`: learned mean, PCA basis, curves, sample settings, and reconstructions.
- `smoothing_sensitivity.npz`: first-curve smoothing controls and exact lean-only comparison path.

Independent checks confirm that the vectorized nearest-segment search matches a direct segment loop, and that the sum decoder equals stored sequential reconstructions. Figure interpretation checks cover source object, scales, starting state, measurements, clipping, sampled scope, and hidden projection dimensions.

Method reference: [Hastie and Stuetzle, Principal Curves (1989)](https://hastie.su.domains/Papers/Principal_Curves.pdf).
