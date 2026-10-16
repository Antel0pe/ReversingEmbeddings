# The zero-radius limit of local SVD

**A limiting SVD space exists at regular differentiable points, and individual
axes converge when their limiting eigenvalues are separated.** The resulting
axes are generally mixtures of the physical knobs. Extrapolating the limit
more accurately does not select the original physical coordinate system.

A direct finite-radius extrapolation check accompanies the mathematical
argument below. No zero-radius or true-knob data enters the extrapolation fit.

## The local measurement and its limit

Let `F` be the image renderer, `p` the starting five-control state, and `J` its
784-by-5 Jacobian. Each column of `J` is a true local knob vector. Assume `F`
is differentiable at `p` and `J` has rank five.

Choose fixed parameter-ray directions `a_i`. For an image radius `r`, move
an amount `t_i(r)` along each ray until the image displacement has length `r`.
The normalized image displacement is:

```text
direction_i(r) = [F(p + t_i(r) × a_i) - F(p)] / r
```

Differentiability means:

```text
F(p + t × a_i) - F(p) = t × J a_i + a remainder smaller than t
```

Because the image displacement length is `r`, its limiting parameter-step
ratio is:

```text
t_i(r) / r  →  1 / length(J a_i)
```

Consequently:

```text
direction_i(r)  →  J a_i / length(J a_i)
```

This is the derivative-like limit relevant to the experiment. It is a limit
of **normalized pixel changes**, not a claim that each SVD axis is a knob.

SVD of the displacement rows has the same pixel axes as the eigenvectors of
the uncentered second-moment matrix:

```text
C(r) = average of outer(direction_i(r), direction_i(r))

C(0) = J M J-transpose

M = average of [outer(a_i, a_i) / length(J a_i)^2]
```

If the sampled `a_i` span all five parameter directions, `M` is positive
definite. Thus `C(0)` has rank five and its range is exactly the range of `J`.
The five-dimensional limiting SVD space therefore contains every true knob
vector. This conclusion does not require knowing their names to compute SVD.

The gap between the fifth positive eigenvalue and the zero eigenvalues makes
the leading five-dimensional space stable as `C(r)` approaches `C(0)`.
Individual axes additionally require gaps between their own eigenvalues.
Theorem 2 in [Yu, Wang, and Samworth, A useful variant of the Davis–Kahan theorem
for statisticians](https://www.statslab.cam.ac.uk/~rjs57/Biometrika-2015-Yu-315-23.pdf)
bounds eigenspace error by matrix error divided by an eigenvalue gap.

If limiting eigenvalues coincide, axes inside that block can rotate without
changing the space. For example, uniform sampling over the tangent unit sphere
has population second moment equal to one fifth of its tangent projector:
all five positive eigenvalues are equal. There is then no distinguished
individual PCA/SVD axis, even with infinitely many noiseless samples.

## Why this does not recover the original knobs automatically

The limiting axes diagonalize `J M J-transpose`: they describe the local image
geometry together with the sampling rule. They are not generally the columns
of `J`.

A direct obstruction in this renderer is that true knob vectors need not be
perpendicular. At the canonical root, the unit height and width vectors have
inner product approximately **0.144059**. Distinct SVD axes are perpendicular.
All five SVD axes therefore cannot equal the five physical knob directions,
even in exact arithmetic at radius zero.

Weighted combinations of the limiting five axes do recover the true vectors
at regular points, given the correct weights. The limit itself does not
identify those weights: changing coordinates mixes the columns of `J` while
describing the same manifold.

The exact limiting root axes, with this fixed 256-ray sampling design, have:

| True knob | Cosine to its assigned limiting SVD axis | Best scaled-single-axis relative vector error |
| --- | ---: | ---: |
| Horizontal position | 0.999568506 | 2.93735% |
| Vertical position | 0.999312109 | 3.70852% |
| Height | 0.988154264 | 15.34638% |
| Width | 0.999509477 | 3.13178% |
| Lean | 0.999908389 | 1.35356% |

These are different measures: cosine is angular similarity; vector error is
the remaining pixel-vector length after the best scalar fit. The height error
remains in the limit. It is not finite-radius curvature that further shrinking
would remove.

Changing only the finite sampling seed changes the root's limiting axes by up
to 0.0553 in unit-vector distance, while their five-dimensional span differs
by less than 9.2e-15 in projection error. Sampling details can rotate a basis
without changing the underlying tangent space.

## Extrapolating without a measurement at zero

Normalize every sampled displacement by its measured image length and compute
`C(r)` at three finite radii. Fit the model:

```text
C(r) approximately = limiting matrix + r × first correction
                                     + r² × second correction
```

Each matrix entry is fitted independently by quadratic interpolation, then
evaluated at zero. Diagonalizing that estimated limiting matrix gives the
predicted limiting axes. Working with the matrix avoids arbitrary SVD signs
and permutations during the fit.

This model requires an appropriate small-radius expansion over the chosen
window. Differentiability alone establishes convergence, but not necessarily
a quadratic expansion or a numerical error certificate. In a smooth region,
more regularity supplies stronger expansions. This piecewise analytic renderer
also has such expansions along fixed rays once the active formulas stabilize.

The fit never receives the true zero-radius matrix. Known directional
derivatives construct the exact limiting matrix afterward, solely to evaluate
the extrapolation. A smaller finite radius excluded from each fit provides a
separate prediction check.

### Direct check

Five starts: the root, two previously sampled spread starts, a tilted cap
boundary, and a generic smooth start. Each shell has 256 samples, using the
same fixed Sobol parameter-ray directions and default-range scaling.

| Fitting window | Three finite radii | Excluded finite prediction radius |
| --- | --- | ---: |
| Larger | 0.001, 0.0003, 0.0001 | 0.00003 |
| Smaller | 0.0001, 0.00003, 0.00001 | 0.000003 |
| Finest | 0.00001, 0.000003, 0.000001 | 0.0000003 |

Maximum unit-vector error versus the analytic limiting **five leading axes**,
allowing signs and permutation:

| Start | Larger-window estimate | Smaller-window estimate | Finest-window estimate |
| --- | ---: | ---: | ---: |
| Root | 4.38e-11 | 5.22e-10 | 6.07e-9 |
| Smooth spread 02 | 4.29e-11 | 2.54e-9 | 5.81e-9 |
| Near-boundary spread 03 | 2.44e-3 | 1.54e-4 | 4.84e-9 |
| Tilted 30-degree cap boundary | 2.61e-10 | 7.57e-10 | 7.26e-9 |
| Smooth generic | 6.37e-11 | 3.53e-10 | 4.63e-9 |

At the root, the measured radius-0.0001 axes differ from the analytic limit by
about 3.21e-6. Extrapolation from the larger window reduces that to 4.38e-11.
It predicts the excluded radius-0.00003 second moment with relative matrix
error **2.73e-11**.

Spread 03 demonstrates an important failure. Its larger windows cross a
renderer formula boundary, so one polynomial model is unsuitable. The finest
window reaches the stable regime and gives an accurate estimate. The excluded
finite-radius prediction errors also flag the larger windows: approximately
5.52e-4 and 8.88e-5 relative matrix error.

At the other starts, finer windows generally worsen the zero-radius estimates:
subtraction roundoff gets amplified when the image changes become tiny.
Shrinking indefinitely is not a numerical-accuracy strategy.

All sampling uses four fixed radial refinement iterations. Row normalization
uses the actual measured displacement length. This prevents a radius-dependent
solver stopping threshold from introducing artificial changes in the
second-moment expansion. Maximum relative shell deviation is 3.65e-8 at the
smallest checked scales. Float64 arithmetic is used throughout this check.

Quadratic extrapolation need not preserve positive semidefiniteness. In the
stable windows, negative eigenvalues are about 1e-16 from roundoff. The bad
larger Spread 03 window instead produces a negative eigenvalue of -3.52e-5,
an additional warning that its extrapolated matrix is unsuitable. All values
are saved; they are not negative variances in a measured second moment.

## Tilted boundary exception

At the tilted cap-boundary start, there is no common two-sided linear Jacobian
for all controls. The limiting changes from both sides span seven linear
directions. The five leading axes still have well-separated eigenvalues here
and converge, but their span cannot contain all one-sided true knob vectors.

The analytic limiting seven-vector space recovers the one-sided targets to
numerical accuracy. Axes six and seven have much smaller eigenvalue gaps than
the leading five, making their individual extrapolations less stable; errors
for both the first five and all seven are recorded separately.

This is a nonsmooth join of a five-control family, not seven independent
controls. The regular-point proof applies within appropriate smooth regions,
not unchanged across the join.

## What finite data can and cannot establish

A limit can be estimated without observing a sample at radius zero. That is
precisely what the extrapolation check does. The mathematical argument also
establishes the limit under its stated regularity and sampling conditions.

However, finitely many measurements do not prove an exact limit by themselves.
Different smooth functions can agree at finitely many observed radii while
having different intercepts at zero. Model assumptions and validation matter.
With only a finite point cloud, sufficiently small neighborhoods eventually
contain no usable neighbors. Extrapolation does not create new observations
or uniquely determine physical knobs from missing data.

## Files and reproduction

- `limit_extrapolation.png`: measured and extrapolated axis errors, starting
  images, and the root's persistent physical-knob mismatch.
- `results.json`: every finite and extrapolated error, excluded-radius checks,
  limiting eigenvalues/gaps, and sampling-seed sensitivity.
- `extrapolated_bases.npz`: estimated axes, exact limiting axes, true unit knob
  targets, and reference settings.

Run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/extrapolate_local_svd_limit.py
```
