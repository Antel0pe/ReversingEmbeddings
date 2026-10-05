# Lean-field fitting experiment

Executed against `grey_ones.render`. The viewer, numerical outputs, and figure
are in `results/`. Browser checks passed for 36 model/width/lean combinations,
pixel selection, animation, and a 390-pixel viewport, with no page errors.

## Scope and result

Width and lean vary. Center is fixed at (14.5, 14.5) px and height at 19.75 px.
425 states train the models; 703 distinct interior states and four boundary
corners evaluate them. All inputs are known knob settings. The experiment does
not infer replacement knobs or recover geometry from an unlabeled point cloud.

| Model | Scalar coefficients | Mean relative arrow error |
| --- | ---: | ---: |
| Linear polynomial | 2,352 | 85.56% |
| Cubic polynomial | 7,840 | 67.83% |
| Degree-5 polynomial | 16,464 | 55.62% |
| Boundary-aware edge rule | 1 | 0.65% |

Relative arrow error means the length of the arrow difference divided by the
measured-arrow length, over all 784 coverage values. These are errors against
finite differences with a 0.025-degree step, not percentages of correct images.
The edge rule's 95th-percentile error is 1.66%; its worst error is 4.75%.

The polynomial fits are smooth functions of normalized width and lean. Their
3, 10, or 21 terms each carry a 784-pixel coefficient pattern. The edge rule
uses known stroke geometry and subrow sampling to supply switching features;
only a global scalar is fitted. Its assumptions are stronger than those of the
polynomials, and its coefficient count does not capture that supplied structure.

## The short rule

For each pixel:

```text
lean arrow = c * average over its 16 subrows of
    vertical coverage weight
  * (vertical center - subrow midpoint)
  * 1 / cos(lean in radians)^2
  * (right edge inside pixel - left edge inside pixel)
```

Each inside test is 1 or 0. Width changes which pixels contain the edges. The
fitted `c = 0.01745084920395062` is close to the analytic degree-to-radian factor
`pi/180 = 0.017453292519943295`. Exact edge/pixel coincidences can have different
one-sided derivatives. The implementation uses strict inside tests there.

On 256 new random states across the entire five-knob box, the edge rule's mean
relative arrow error is 0.76%. This rule receives all five known settings;
polynomial slice models only receive width and lean. Their full-space control
errors therefore measure a deliberate extrapolation beyond the fitted slice.

## Following the field

Six lean paths run from -10 to 35 degrees with fixed width, starting from an
exact rendered image. Trapezoidal integration uses 901 samples per path; 91
frames are stored for display. No prediction is clipped, denoised, or projected
before scoring. The edge rule's worst absolute pixel error across those paths
is 0.00283 on the 0-1 coverage scale. Small negative or above-one values remain;
closeness does not establish membership in the renderer's family.

A measured-arrow integration baseline separates fitting effects from the
chosen numerical integration procedure. Figure panels mark values outside
[0,1] in purple; the viewer can toggle those marks and shows magnified residuals.

The follow-up numerical checks keep the fitted coefficient fixed. Reducing the
finite-difference step from 0.025 to 0.00625 degrees reduces mean arrow mismatch
from 0.65% to 0.20%. On one path at width 2.7324 px, reducing the integration
step from 0.05 to 0.0125 degrees reduces mean image L2 error from 0.00737 to
0.00123. These are checks on numerical resolution, not additional fitting.

## Reproduce

From the repository root:

```sh
.venv/bin/python experiments/lean_field_fit_20261004/run.py
.venv/bin/python experiments/lean_field_fit_20261004/check_numerics.py
.venv/bin/python experiments/lean_field_fit_20261004/make_visuals.py
```

Open `results/viewer.html`. It is self-contained and uses measured Python data,
with no JavaScript reimplementation of the renderer.

Outputs: `metrics.json`, per-state error CSVs, model coefficients, sampled paths,
`numerical_checks.json`, `comparison.png`, and `viewer.html`. The figure followed
`skills/experiment-figures/SKILL.md` and was inspected and corrected for overlap.
