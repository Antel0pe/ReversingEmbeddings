# Can local SVD bases reproduce the true knob vectors?

**Yes, as weighted combinations of the full retained basis, to numerical
accuracy in this test.** Five leading vectors alone can retain geometric error,
and one rescaled vector retains its mixture. The coefficient solve here receives
the known true direction as its target: this is a test of what the SVD basis
contains, not blind discovery of physical knobs.

## Scope and definitions

- 62 starting images: the canonical root, 16 other reference starts, and 45
  distinct non-root states from the five successful nine-hop walks.
- All five knobs, checking both one-sided derivatives. Their average is also
  saved. Three tilted reference starts have genuinely different one-sided
  derivatives; the other 59 starts have a common two-sided derivative.
- Ordinary uncentered SVD of 256 seed-relative displacements on equal pixel-L2
  radius shells: 0.001, 0.0001, and 0.00001. Fixed Sobol seed 20261012 and
  default-range-scaled parameter rays reproduce the earlier sampling design.
- Pixel coverage uses the same 16-subrow renderer, with float64 storage.
- The full retained basis includes singular values exceeding 1e-9 times the
  largest. It has 9 or 10 vectors here. No arbitrary completion to all 784
  ambient directions is used.
- Each target is normalized to length 1. Error is the Euclidean length of the
  difference between target and reconstruction. To reconstruct a derivative
  per original pixel/degree knob unit, multiply its weights by the stored
  original derivative norm. This does not infer the physical knob scale.
- Two-sided neighborhoods can extend beyond default parameter bounds, as in
  the previous root experiments. All strokes remain positive and unclipped.

## Reconstruction rule

For an orthonormal SVD basis, the least-squares weights are simply:

```text
weight for basis vector i = true direction dot basis vector i

reconstructed direction = sum(weight i × basis vector i)
```

This is the best possible weighted reconstruction in the selected span. Any
remaining residual is perpendicular to every selected basis vector, so changing
weights cannot remove it.

For comparison, the single-vector case chooses the best scalar multiple of
**any** of the five leading vectors. It is not restricted to an earlier family
assignment. A scalar changes magnitude and sign, but cannot rotate a mixture
into the true direction.

## Results at the canonical root

The radius is 0.0001. The true directions in this table are unit vectors.

| Knob | Best scaled single vector | Best mixture of five | Best mixture of nine |
| --- | ---: | ---: | ---: |
| Horizontal position | 2.94e-2 | 1.20e-7 | 1.51e-12 |
| Vertical position | 3.71e-2 | 6.68e-7 | 3.03e-12 |
| Height | 1.53e-1 | 3.14e-6 | 5.09e-12 |
| Width | 3.13e-2 | 2.45e-7 | 8.45e-13 |
| Lean | 1.35e-2 | 5.72e-8 | 6.69e-13 |

For example, the unit height direction is approximately:

```text
0.000685679 × V1
+ 0.152880991 × V2
+ 0.005929626 × V3
+ 0.011954833 × V4
+ 0.988154266 × V5
```

The five-term expression still has error 3.14e-6. Including V6–V9 with their
small fitted weights reduces the residual to 5.09e-12. The full-precision
weights, including the original per-knob scale, are in `root_weights.csv`.

This is why the near-height SVD vector cannot simply be shrunk: recovering
height needs a substantial contribution from V2 as well as V5. True knob
vectors can overlap; forcing them to be perpendicular would change them.

## Across all starts

Worst errors over all five knobs and both sides:

| Starts | Radius | Five-vector error | Full-basis error |
| --- | ---: | ---: | ---: |
| 59 common-derivative starts | 0.001 | 8.67e-3 | 3.19e-12 |
| 59 common-derivative starts | 0.0001 | 3.00e-3 | 2.31e-11 |
| 59 common-derivative starts | 0.00001 | 4.48e-7 | 2.78e-10 |
| 3 tilted cap-boundary starts | 0.001 | 1.47e-2 | 1.93e-11 |
| 3 tilted cap-boundary starts | 0.0001 | 1.48e-2 | 2.07e-10 |
| 3 tilted cap-boundary starts | 0.00001 | 1.48e-2 | 2.23e-9 |

The worst five-vector smooth-start error at radius 0.0001 is horizontal position
at `spread_03`. A start can have a unique derivative while its finite
neighborhood crosses a nearby formula boundary. Smaller neighborhoods resolve
that issue in these samples; they also amplify numerical subtraction noise.

At the canonical root and in ordinary local formula cells, the extra sampled
linear directions include curvature/interaction effects. They also let the
coefficient solve correct the slight tilt of the leading finite-neighborhood
span relative to the actual tangent space.

## Check against the infinitesimal geometry

As a **known-renderer control**, construct an SVD from exact analytic directional
derivatives of the same random parameter rays. This removes finite-neighborhood
curvature and subtraction error. It is not a new data-only estimator.

- At all 59 common-derivative starts, this limiting span has rank 5. Combining
  its five vectors reconstructs all true knob directions with worst error
  **2.16e-15**.
- At the three tilted cap-boundary starts, the full limiting span has rank 7.
  Seven vectors reconstruct both sides of the five knobs with worst error
  **2.13e-15**. Keeping only five leaves worst error **0.0147568**, or 1.48%.

There are still five independent input knobs. A nonsmooth join can require
more than five vectors to linearly span the limiting changes from all sides.
An exact two-sided knob vector does not exist for the affected controls there;
the one-sided directions are the appropriate targets.

## Numerical checks

The true derivatives are obtained by differentiating the actual renderer's
min/max/clip operations, using directional tie rules at boundaries. Independently
checking with one-sided, second-order finite differences at several step sizes
across all 62 starts gives a worst best-step relative discrepancy of **4.70e-10**.
The average of the two analytic sides matches the existing analytic Jacobian
implementation within **8.91e-16** relative norm.

A second sampling seed at the root, `spread_02`, and `lean_30` produces different
bases but reconstructs the same true targets to the observed arithmetic floor.

Higher-precision coverage differences, followed by ordinary float64 SVD, check
that the tiny full-basis residuals are numerical. At radius 0.0001:

| Start | Ordinary coverage differences | Higher-precision differences |
| --- | ---: | ---: |
| Root | 5.09e-12 | 2.90e-15 |
| Smooth `spread_02` | 1.40e-11 | 5.95e-15 |
| Tilted `lean_30` boundary | 2.07e-10 | 2.54e-14 |

The higher-precision calculation uses the same coverage formula with extended
arithmetic before subtraction. It was checked at these three selected starts,
not all 62. The geometric parameters and sample locations are unchanged.

Float32 controls at the same three starts show substantially larger errors when
keeping the same nine vectors: approximately 6.06e-4, 5.93e-4, and 9.10e-4.
Retaining every float32 singular direction above the float64 cutoff creates
42–67 directions, many due to quantization noise. Their near-zero projection
residual is **not** evidence of a recovered low-dimensional signal basis.

## What this settles

The mostly-correct axes are not the limit of what the SVD representation can
express. A weighted change of basis can recover the exact physical direction
when it is contained in the span. The finite sampled full bases contain those
directions to numerical accuracy in this experiment. At regular points, the
five-dimensional infinitesimal span contains them exactly in exact arithmetic.

This does not identify the weights without ground truth. SVD supplies a span;
many invertible mixtures describe the same span. The current calculation uses
the known true vectors on the right-hand side. Inferring physical weights from
unlabeled neighborhoods or the cross-path pattern rule remains separate work.
No claim is made that these are unique physical coordinates of the manifold.

## Figures and reproduction

- `01_root_reconstruction.png`: true directions, best single-vector fits,
  five-vector mixtures, full nine-vector mixtures, and remaining errors.
- `02_across_states.png`: worst errors over starts, radii, both sides, and the
  exact-derivative control, with higher-precision checks annotated.

Run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/verify_true_knobs_in_svd.py
```

`results.json` and `reconstruction_errors.csv` record every check.
`vectors_and_weights.npz` retains targets, bases, and fitted weights at radius
0.0001 for every state. `root_weights.csv` gives the root weights directly.
