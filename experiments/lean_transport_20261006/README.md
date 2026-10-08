# Compressing how the lean direction changes with state

**Vector-only follow-up:** This is a known-geometry reconstruction, not a learned
vector-variation solution. See `../lean_vector_variation_20261007/README.md` for
the corrected experiment based on observed pixel vectors.

The answer for the controlled generated-one family is **transport two continuous
edge patterns, then rasterize them**. A single fixed 784-pixel vector, a scalar
multiple of it, and simple addition/averaging of independent axis vectors do
not cover the changing states. The continuous edge rule does.

This experiment uses the actual `grey_ones.render`, with all 784 coverage values.
The reference settings are `(cx, cy, height, width, lean) =
(14.5, 14.5, 19.75, 3.2, 0)` in pixels and degrees. It does not use MNIST,
neighborhood embeddings, PCA, or a dimensionality reduction.

Open `results/viewer.html` for signed pixel comparisons and finite lean moves.
The page contains 504 precomputed states: nine widths, fourteen starting lean
angles, and four center/height presets. These controls sample the continuous
family; the equation below is the statement about the continuous domain.

## What a lean vector means

Let `F(p)` be the 784-pixel image at state `p`. The unnormalized local lean
vector is `V(p) = partial F(p) / partial lean`, in coverage change per degree.
Its signs specify which pixels gain or lose ink, and its magnitudes specify
how fast. Calling it a group of pixels that change together is useful, provided
we retain these magnitudes and remember that the group changes during a move.

A finite move from angle `theta0` to `theta1` follows the changing field:

```text
F(other settings, theta1) - F(other settings, theta0)
    = integral from theta0 to theta1 of V(other settings, theta) dtheta
```

It generally does not equal `(theta1-theta0) * V(start)`.
Changing the vector with width is a first derivative of the vector-valued
function `V(width)`. Relative to the original image function, that is a mixed
partial derivative, `partial² F / (partial width partial lean)`; it is not a
second derivative along the lean path. The renderer has switching events, so
this classical derivative need not exist at every state.

## Width first: the pattern that actually changes

At fixed center, height, and lean, increasing width by `dw` moves the left
continuous edge left by `dw/2` and the right continuous edge right by `dw/2`.
The horizontal speed of either edge under lean does not depend on width.
There is no universal width-dependent amplification such as multiplying the
whole vector by 1.5. Which pixels contain those moving edges changes.

For the upright reference:

- For `1 < width < 3`, the edges lie in columns 13 and 15 (zero-based).
- For `3 < width < 5`, the edges lie in columns 12 and 16.
- Throughout either open interval the infinitesimal lean vector is identical.
  Widths 1.8 and 2.6 share one vector; 3.2 and 4.6 share another.
- At width 3, edges coincide with integer pixel boundaries. Increasing and
  decreasing lean have different one-sided derivatives; their turn is about
  90.003 degrees. A centered finite difference averages these two limits.

The fine width sweep contains 1,401 regularly spaced values in the default
box plus near-event probes, and extra illustrative widths 1, 2, and 3. Width 1
is outside the default box and is explicitly flagged in `width_sweep.csv`.
These event plateaus are a rasterization property, not a claim that the
continuous boundary stops moving when width changes. At nonzero lean,
different subrows switch columns at different widths.

Two widths, 3.1 and 3.3, have exactly equal upright raster tangents. At 2 degrees
their tangent difference has L2 length 0.167907 per degree. The raster tangent
alone therefore loses the subpixel phase that determines future changes.
Keeping the continuous edge locations supplies that information.

## One template, transported to every state

Use `u = tan(theta)` with `theta` in radians. At each of the renderer's
16 subrow midpoints `y` per pixel row, define:

```text
vertical_offset = cy - y
left_edge       = cx + u * vertical_offset - width/2
right_edge      = cx + u * vertical_offset + width/2
vertical_weight = fraction of that subrow lying between cy-height/2 and cy+height/2
```

For a pixel column `c`, the universal continuous edge template is

```text
H_c(b) = clip(b - c, 0, 1)
```

It measures the fraction of the pixel's horizontal interval to the left of
boundary `b`. The pixel coverage equation is

```text
image[row, c] = average over the row's 16 subrows of
    vertical_weight * [H_c(right_edge) - H_c(left_edge)]
```

This is exactly the renderer's horizontal interval overlap, expressed as the
difference of two translated copies of one ramp template. Differentiating:

```text
lean_vector[row, c] = average over the row's 16 subrows of
    vertical_weight
  * (cy - y)
  * (pi/180) / cos(theta)^2
  * [edge_inside_c(right_edge) - edge_inside_c(left_edge)]
```

Here `edge_inside_c(b)` is 1 when `c < b < c+1`, otherwise 0.
At a boundary `b=c` or `b=c+1`, use the appropriate one-sided limit.
The figure/viewer's direction comparison uses the average of these limits
where they disagree; this average is not a classical two-sided derivative.
The factor `pi/180` converts an angle change in degrees to radians.

Every term has a role:

- **Width** separates the two edge patterns, without changing edge velocity.
- **Horizontal center** translates both edge patterns together.
- **Vertical center** changes the active rows, the lever arm `cy-y`, and the
  shear position. With nonzero lean, the horizontal shift includes `u * dcy`.
- **Height** changes vertical coverage, adding/removing weighted subrows.
- **Current lean** changes which pixels contain the edges and multiplies
  degree-based speed by `1/cos(theta)^2`.

That is the required blending: compose positions, lever arms, and vertical
weights, then apply the common pixel template. The formula is nonlinear in
the state but has no per-state table. It is derived from supplied geometry;
this experiment does not claim to have learned that geometry from scratch.

## Do independent axis vectors combine?

The independent-axis model receives perfect arrows from the formula above,
at each target's current lean angle. For a target with four other settings
changed, it predicts

```text
additive = V(reference with current lean)
         + sum over other knobs k of
             [V(only k changed, with current lean) - V(reference with current lean)]

average = mean over the changed knobs k of V(only k changed, with current lean)
```

The two-knob tests average only their two relevant arrows. The full-box tests
average all four other-knob arrows. Each evaluation uses 512 new random joint
states. Relative vector error is `norm(prediction-reference)/norm(reference)`.

| Joint changes | Mean additive error | Mean averaging error |
| --- | ---: | ---: |
| Width + height, upright | 7.56% | 36.12% |
| Width + horizontal center, upright | 73.94% | 51.02% |
| All four other knobs, upright | 81.48% | 61.84% |
| Full five-knob box | 69.45% | 64.14% |

The largest additive error in the full box is 172.60%, and its mean direction
angle error is 36.27 degrees. These are arrow errors, not image accuracy rates.
Failure with perfect single-axis inputs establishes that this particular
addition/averaging rule is insufficient. It does not rule out a different
nonlinear model trained on single-axis observations plus structural assumptions.

An explicit width–height interaction explains one failure. Let `A(h)` denote
the subrow vertical weights and `B(w)` the signed edge pattern, with lean and
center fixed. Then the joint arrow has the bilinear form `A(h) * B(w)` before
subrow averaging. Independent additive changes omit the cross term:

```text
joint - additive = average [(A(h)-A(h0)) * (B(w)-B(w0))]
```

The same structural issue occurs when multiple changes relocate an edge
across a pixel boundary. Adding changes in the old raster positions cannot
reliably place them in the newly active pixels.

## Calibration using only independent changes

A second model receives 95 single-axis rendered images (19 per knob), plus
one reference image. The existing `lean_axis.geometry_from_image` reads
continuous geometry from each image. A linear calibration fits how each
input changes the descriptor `(cx, cy, height, width, tan(lean))` relative to
the reference: a 5×5 transformation matrix plus five baseline descriptors,
30 scalar coefficients. Joint examples are never used in this calibration.

Composition happens in those continuous descriptors; the common edge formula
then generates the raster arrows and finite moves. The fitted matrix is
essentially the identity, with tiny float32 discrepancies. This is calibration
in known physical coordinates, not discovery of new coordinates. Its supplied
image-to-geometry recovery and decoder are the important structural prior;
the count of fitted coefficients alone does not measure the algorithm's
entire information content.

One lattice state coincided with a single-axis training state and was excluded
from this model's held-out score. On **1,281 genuinely held-out states**, its
maximum finite-move pixel error was **1.05e-7**, and maximum image L2 error was
**3.94e-7**. On 1,024 random states, its mean relative arrow error was 2.82e-9.
That arrow score uses the analytic reference field, not finite differences;
exact switching points can be extremely sensitive to rounding of state inputs.

## Can the rule be used from an image and actually followed?

Yes, on this generated family and its declared parameter box. Geometry can be
read from row ink totals and cumulative horizontal coverage using the existing
`lean_axis.geometry_from_image`. Width alone is sufficient only for the fixed
reference slice. An arbitrary image also needs its center, height, and current
lean; all can be recovered here.

Tests against `grey_ones.render` included 1,024 random states, a 243-state
three-level full-box lattice, and 15 deliberate near-event states. The new
transport code never calls the renderer to make its prediction; rendering is
used to supply inputs and evaluate reference targets.

| Validation | Count | Worst absolute pixel error |
| --- | ---: | ---: |
| Forward universal edge-template coverage | 1,282 | 2.98e-8 |
| Finite lean moves, known settings and source image | 1,282 | 5.95e-8 |
| Finite lean moves, geometry recovered from source image only | 1,282 | 1.02e-7 |
| Independent geometry calibration, no joint training | 1,281 | 1.05e-7 |
| Integrate changing tangents along full −10° to 35° paths | 6 × 181 frames | 5.87e-8 |

For the full paths, `u=tan(theta)` makes the field piecewise constant between
edge/pixel crossings. The integrator solves those event locations and uses the
midpoint tangent on every interval, including all requested frame angles.
There are 2,819–3,046 intervals per path. Starting images are source float32
renders; the remaining differences are at float32 rounding scale. Predictions
are scored without clipping, projection, or denoising.

Ordinary finite-difference derivative checks on 256 random states have mean
relative errors of 1.56%, 0.84%, and 0.31% for half-steps 0.1°, 0.025°, and
0.00625°. Finite differences smear switching events; making the step smaller
also eventually exposes float32 subtraction noise. This is why near-equality
of an integrated finite move is a clearer correctness test than one global
finite-difference tolerance at corners.

The coverage algebra proves the transport formula for every state to which
this renderer equation applies (positive width); finite tests check its
implementation. Image-only geometry recovery and the empirical validation are
restricted to the declared generated-one box. A lean-only field does not,
by itself, characterize all possible images, nor does this experiment
establish a learned model for real handwriting.

## Reproduce and inspect

From the repository root:

```sh
.venv/bin/python experiments/lean_transport_20261006/run.py
.venv/bin/python experiments/lean_transport_20261006/make_visuals.py
# Optional: requires an existing local Chromium browser.
.venv/bin/python experiments/lean_transport_20261006/check_viewer.py
```

Outputs are in `results/`: `metrics.json`, `width_sweep.csv`, joint-state and
path arrays, two explanatory PNGs, and `viewer.html`. The viewer uses Python
arrays; there is no JavaScript copy of the generator. Color scales are shared
across comparable panels, and invalid image predictions are marked in purple.
The figure understanding check in `skills/experiment-figures/SKILL.md` was
applied: baseline first, axes and signs defined, all pixels visible, joint-space
control shown, numerical claims beside the evidence, no clipped signed values.

Local headless Chromium verification passed for 1,008 state/mode combinations
at both 1,200-pixel and 390-pixel viewport widths, including mode switches,
width input, reset, and play/pause wiring. No horizontal overflow was detected.
Desktop/mobile screenshots and `browser_checks.json` preserve the checks;
this is separate from the unavailable in-app browser automation runtime.
