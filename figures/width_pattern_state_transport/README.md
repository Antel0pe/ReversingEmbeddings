# A width-change pattern that depends on the current state

The idea under test is: start with a reference pixel-change pattern for a knob,
then update that pattern according to the current state, including combinations
of other knobs and the knob's own current value.

This experiment supplies the true five state coordinates. It learns only how
the finite width-change pattern varies. It does not recover coordinates from
images or use SVD. This isolates the representation/composition question from
SVD mixtures and family-tracking errors.

## Plain statement of the idea

A state is a location in the five-coordinate system. It does not identify the
particular route used to reach it. A knob rule takes that location and returns
a signed pixel-change pattern. A large knob move either needs a finite-change
rule or requires repeatedly updating the direction along the move.

The experiment's target is:

```text
width pattern at current state
    = image after width +0.28 px − image before widening
```

The pattern depends on all five current values, including current width.
For the centered height convention, height +1 px extends the top and bottom
by 0.5 px each, not 1 px each.

## Learned rule

State values are expressed relative to the canonical root
`(cx, cy, height, width, lean) = (14,14,19.75,3.2,0)`.
One normalized state unit equals 10% of each knob's default range width:

| State coordinate | One normalized unit |
| --- | ---: |
| Horizontal position | 0.1 px |
| Vertical position | 0.1 px |
| Height | 0.15 px |
| Width | 0.28 px |
| Lean | 4.5 degrees |

Negative horizontal/vertical changes explicitly extend the default sampling
bounds around the canonical root. All tested strokes remain positive and
unclipped. Rendering uses float64 storage with the same 16-subrow coverage
calculation as the standard renderer.

Fit one quadratic rule for the entire width-change pattern:

```text
predicted pattern = root pattern
                  + sum(current coordinate × linear response pattern)
                  + sum(current coordinate squared × self-interaction pattern)
                  + sum(coordinate i × coordinate j × pair-interaction pattern)
```

This contains 21 fixed arrays of 784 pixel values: one root array, five linear
arrays, five square arrays, and ten mixed arrays. Arrays can be added; the
important point is that the correction for one knob depends on the other
current coordinates through the mixed terms.

For example, the height update contains a term proportional to current
horizontal offset. That represents the change in the height-driven correction
to the width pattern when horizontal offset changes. Such a correction is a
change *to a pixel-change pattern*, not a direct ink change itself.

The coefficients come from a symmetric finite-difference stencil: the root,
plus/minus endpoints on five axes, and four corners for each of ten axis pairs.
There are 51 training states, with at most two coordinates altered at once.
No held-out five-coordinate joint combination contributes to the coefficients.
This is an interpolation stencil, not a best-fit regression or a search over
model types.

Given an update of size `d` in coordinate `i`, the model's exact finite update is:

```text
new pattern = old pattern
            + d × (linear response i
                   + 2 × current coordinate i × square response i
                   + sum(other coordinate j × mixed response i,j))
            + d squared × square response i
```

This is the concrete extra layer in the proposal: the response used for the
next update is itself conditioned on the current state.

## Joint-state results

Use 128 fixed Sobol joint settings at each scale, with all five coordinates
varied together. The same directions at different scales permit paired
comparisons. The first three scales interpolate inside the training coordinate
box; the final scale extrapolates to twice its extent. Actual true patterns and
model predictions are compared over all 784 pixels without clipping.

The root-pattern and additive rows below are ablations of the same model: keep
only the constant array, or only constant and linear arrays. There is one
primary model and no method/parameter sweep.

Errors divide pooled prediction-error L2 by pooled **true width-change L2**.
They are not fractions of the norm of the entire starting image.

| State variation, fraction of each default knob span | Fixed root pattern | Additive responses | Quadratic with interactions |
| --- | ---: | ---: | ---: |
| 2.5% | 0.58% | 7.23% | 7.47% |
| 5% | 12.01% | 16.55% | 16.03% |
| 10% | 47.49% | 41.16% | 26.35% |
| 20%, extrapolation | 84.53% | 84.11% | 92.89% |

At 10%, the worst individual joint-state quadratic error is 43.58% of its
true width effect. All 128 quadratic predictions at each scale contain some
negative width-change values, whereas actual widening only adds ink. Their
magnitude varies; these signs are retained in the plots and measurements.

Interactions help substantially at the training scale, but this broad quadratic
rule is not an accurate full solution. In particular, it is worse very near the
root and extrapolates poorly.

The sampled geometric boundaries explain a concrete source of error: close to
the root, widening still affects the same columns; farther samples cross pixel
boundaries and activate different columns. A single polynomial averages over
those different formula regions. The experiment does not establish that all
remaining errors have this cause or that any particular alternative will solve
them.

## Composition consistency

For 16 joint target states, apply the model's five coordinate updates in all
120 orders. Their final patterns agree with direct model evaluation to maximum
pixel L2 difference **1.44e-16**.

This demonstrates internal order consistency of this quadratic representation.
It is expected from the formula and is not a separate claim of reconstruction
accuracy: an order-consistent model can still predict the wrong pixels.
It does not calibrate the accumulated counts of previously tracked SVD fields.

## Straight physical motion versus a straight pixel-vector path

In the upright height-only example, the side bands for width +0.28 extend as
height grows. At height 20, their cap changes move into new pixel rows.
Using equal height steps of 0.125 px, the changes to the width patterns turn
by **90 degrees in pixel space** at this event. Within either neighboring row
region, the same curve is straight. The physical boundaries themselves simply
continue upward/downward along straight lines.

Thus a simple physical motion need not have one constant pixel-space direction.
Within a fixed pixel formula region, a constant change-of-pattern can exist;
across regions, the active pixel coordinates change. A vector is a single
arrow; it is the sequence of vectors over states that traces a curve.

## Figures

- `01_height_pattern.png` shows reference and varying-height images, their true
  widening patterns, and the turn in the height-driven change of those patterns.
  Heights above the default upper bound 20.5 are explicitly used for the next
  pixel-row region.
- `02_joint_prediction.png` shows true/predicted patterns at the root, a training
  axis endpoint, and median/worst-error held-out joint states. The graph reports
  all joint tests and both ablations. True/predicted maps share one scale;
  error maps use an explicitly separate scale. Blue in a predicted width
  pattern means erroneous ink loss. Blue/red in an error map means
  under/overprediction, not a direct ink change.

## What the result supports

The proposed object is well-defined: each knob has a state-conditioned change
rule, and those rules can be represented with interaction terms and composed
consistently. The particular low-order pixel-array model tested here is too
simple across the chosen pixel events. This supports working with local
pieces or a representation that moves a change template through space; those
alternatives were not tested here.

A rule using five known coordinates is a representation feasibility test, not
unsupervised discovery of physical knobs. Likewise, reconstructing a width
effect is only one component of a full manifold decoder. The experiment makes
no claim about the whole valid family or uniqueness of a coordinate system.

## Reproduce

Run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/width_pattern_state_transport.py
```

`results.json` records every metric and assumption. `joint_errors.csv` contains
per-state errors. `patterns_and_model.npz` retains the learned coefficient
arrays, training states/patterns, all joint test states, actual patterns, and
predictions for each ablation.
