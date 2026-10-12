# How small a neighborhood separates first-order changes from interactions?

This continues the controlled generated-1 experiment using ordinary,
equally weighted SVD of neighbor-minus-start images. No knob labels enter
SVD. The radius is Euclidean distance over 784 coverage values, not a knob
increment. Each fit has 512 training and 256 independently sampled test
neighbors on an equal-distance shell. The same directional sampling design
is reused at each scale.

New radii are 0.0001 and 0.00001, at the canonical root, three previously
sampled random starts, and the 30-degree tilted root. These are selected
diagnostics, not a new sweep of all possible starts.

## Measured five-vector reconstruction errors

Worst held-out error divided by distance from the start, in percent:

| Start | Radius 0.001 (prior) | Radius 0.0001 | Radius 0.00001 |
| --- | ---: | ---: | ---: |
| Root | 0.0118844% | 0.00118850% | 0.000118854% |
| Spread 02 | 0.0140427% | 0.00140420% | 0.000140413% |
| Spread 03 | 1.37928% | 0.874631% | 0.000129386% |
| Spread 04 | 0.357472% | 0.00100759% | 0.000100761% |
| Tilted root, lean 30 degrees | 1.01483% | 1.01764% | 1.01793% |

The random starts have different distances to renderer formula boundaries.
Spread 03 illustrates that the clean small-patch regime can require a radius
as small as 0.00001 for these settings. The exact tilted cap kink retains
first-order variation beyond five at both new radii. Its underlying generator
still has five controls.

## Separation is about scaling, not waiting for exact zero

For a smooth regular patch with well-spread sampling, tangent singular values
scale as `sigma ~ r`, while second-order normal effects typically have
`sigma ~ r**2`. Therefore inspect `sigma/r` over scales:

- A nonzero plateau indicates first-order variation.
- Decay approximately proportional to radius indicates second-order variation.
- A small fixed absolute rounding floor eventually grows as `1/r` after this normalization.

For the root, the first five normalized singular-value powers between the two
new radii are approximately zero, while powers for directions 6-9 are
approximately one. Spread 02 and Spread 04 reproduce this pattern. The
tilted kink instead has seven approximately zero powers. Spread 03 crosses
formula boundaries over this interval and does not follow one clean power
law across the two scales.

At the root, `sigma_5/sigma_1` remains about 0.138284. Meanwhile
`sigma_6/sigma_1` falls from `2.94e-5` at radius 0.001 to `2.94e-6` at 0.0001
and `2.94e-7` at 0.00001. This already gives clear dimension-separation
evidence at 0.001; numerical-precision five-vector reconstruction is a much
stronger and different requirement.

This is a diagnostic under regularity and sampling assumptions, not a proof
from finitely many radii. It identifies a limiting tangent subspace, not the
generator's uniquely labeled physical controls. See
[Multiscale Estimation of Intrinsic Dimensionality of Data Sets](https://cdn.aaai.org/ocs/950/950-4178-1-PB.pdf).

## Floating-point storage limits

The main results retain the canonical renderer's float64 coverage calculation.
The same settings were also rendered to the default float32 storage and
independently fitted with ordinary five-vector SVD. At radius 0.00001,
the root's float32 worst relative error is **1.74287%**, compared with
**0.000118854%** for float64 coverage. The three random starts have float32
errors between 1.17% and 1.71% at that radius.

Some numerical ranks at relative cutoff `1e-10` rise from nine to ten at
radius 0.00001 because a roughly `1e-14` absolute rounding component passes
the now-smaller threshold. These counts are not evidence of another
generator control. Full fitted bases still reconstruct all new held-out
changes below `1e-12` absolute L2. Scale slopes should only be interpreted
above their numerical floor.

## What association across neighborhoods still requires

The spectral test distinguishes first-order variation from higher-order
effects. It does not label individual SVD axes as physical knobs.

A mathematical association can be built by fitting five-coordinate charts
with decoders, using shared images in overlapping neighborhoods to estimate
chart-transition maps, and transporting changes by derivatives of those
maps. Transition maps generally allow mixing and rescaling, not merely
permutation or sign changes. This is related to
[Local Tangent Space Alignment](https://arxiv.org/abs/cs/0212008), though exact
decoding and identification of physical controls are additional requirements.

A knob is then a coordinate function and its associated vector field:
`V_i = dF/dq_i`. Its curves change `q_i` while holding the other coordinates
fixed. Independent calibrated coordinate fields commute on smooth patches;
compatible chart transitions and path checks test that they define the same
family across neighborhoods. See the [MIT integrability notes](https://math.mit.edu/classes/18.966/2014SP/965/class4.pdf).

The root height/width example gives a useful stronger local signature:
`dV_height/dwidth = dV_width/dheight = D`, with a constant pixel array `D`
inside that formula region. Such joint-change structure can constrain a
choice of families. Consistency alone does not uniquely select the original
physical chart, as the prior explicit alternate-coordinate example shows.

Pixel-only chart association and family selection are proposed next steps;
they were not implemented by this scale-separation experiment. The
known-coordinate transport checks remain a separate diagnostic in
`../generated_one_manifold_transport/`.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python analyze_generated_one_scale_separation.py
```

- Source: [`analyze_generated_one_scale_separation.py`](../../analyze_generated_one_scale_separation.py).
- `results.json`: new measurements, two-scale normalized spectrum slopes, and prior larger-radius spectra.
- `smaller_radius_samples.npz`: settings, float64 images, spectra, and fitted bases for all ten new fits.
- `formula_cell_checks.json`: separate analytic diagnostic counting neighbors outside their start's formula cell.

No new explanatory figure was generated; the tables report measured results.
