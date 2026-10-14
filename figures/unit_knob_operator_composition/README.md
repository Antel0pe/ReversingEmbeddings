# Single-knob operations that carry a widening pattern forward

Question: can one reusable update operation per knob act on the current pattern,
so height, lean, and position interactions arise through composition rather than
an explicit table for every pair/triple of controls?

**For this supplied geometric model, yes.** Five state-aware operations on two
continuous widening bands match the original renderer across 64 joint targets,
all 120 block orders per target, and a nine-step mixed trace. This is an
original-control feasibility experiment, not learned geometry or discovery from
individual pixel vectors. The representation and operations were supplied by
hand from the known parallelogram stroke class.

## What is carried between steps

Start at the canonical root `(cx,cy,height,width,lean)=(14,14,19.75,3.2,0)`.
The target pattern at every state is the finite difference

```text
image(state with width +0.28 px) − image(state)
```

Widening adds two parallelogram bands. Each is 0.14 px horizontally thick;
its height and lean match the stroke. Carry their four vertices each, along
with the five current control values. The pixel pattern is an observation of
these bands, rather than the complete internal state.

One unit means 5% of the respective default parameter-range span:

| Knob | One unit |
| --- | ---: |
| Horizontal position | 0.05 px |
| Vertical position | 0.05 px |
| Height | 0.075 px |
| Current width | 0.14 px |
| Lean | 2.25 degrees |

These units use the original controls, not equal image-distance ticks.
The separate requested widening remains +0.28 px throughout. Updating current
width moves where the widening bands begin; it does not change that requested
increment.

## Five reusable operations

Every step transforms the current vertices, then updates the one changed
control. No call to the original renderer and no reconstruction of polygons
from target settings occurs inside an update.

1. Horizontal position: translate both bands horizontally.
2. Vertical position: translate both bands vertically.
3. Height: stretch vertically about the current center. Include a horizontal
   shear correction using the current lean, so the slanted side locations stay
   consistent with the original knob definition.
4. Current width: move the left band left and right band right, by half the
   current-width increment each.
5. Lean: shear about the current vertical center. The slope increment is
   `tan(new lean) − tan(current lean)`, with degree-to-radian conversion.

These are two-channel 3×3 homogeneous affine maps. They act on spatial
coordinates, not by averaging or multiplying pixel intensities. Current state
matters: height ratios, shear increments, and pivots are recomputed. Reusing a
matrix frozen at the root would not express the same physical knob increments.

When lean follows height, it acts on the entire lengthened bands, including
new endpoints. When height follows lean, its state-aware correction preserves
the existing lean. This is how the combined effect is accounted for without
storing a separate three-knob interaction table.

An independent band-coverage routine observes the transported polygons using
the original 16-subrow convention. It receives vertices only, not the five
knob values. `grey_ones.render` supplies the evaluation targets. Matching the
same sampling convention matters; this does not assert exact continuous-area
integration by the original renderer.

## Results

Use 64 deterministic Sobol combinations. Every control changes by 1–4 signed
units, so target shifts are 5–20% of the default parameter-span widths.
For each target, try all 120 orders of the five control blocks, applying each
block as repeated one-unit steps. All sources and targets remain positive and
unclipped. Position bounds are explicitly extended around the root.

| Check | Result |
| --- | ---: |
| Compared target/order endpoints | 7,680 |
| Largest absolute error in any coverage pixel | 1.01e-14 |
| Largest relative width-pattern L2 error | 3.65e-14 |
| Largest order-to-order pixel L2 disagreement | 3.08e-14 |
| Largest stored state discrepancy | 3.55e-15 |

Errors are numerical roundoff. These are checks of a supplied geometric model,
not statistical generalization rates. The sampled states are not the whole
allowed family, although the supplied operations have an algebraic geometric
definition beyond this sample.

The illustrated mixed trace applies height, height, lean, horizontal position,
lean, vertical position, current width, height, height. Every intermediate
pattern is compared with the renderer. Its largest relative geometric-pattern
error is 6.60e-15. Thus the output of each operation really is the input to the
next; there is no reset from the target state between steps.

## Why retaining only resampled pixels differs

Give a comparison the same state, band identities, and affine maps, but carry
only two 28×28 coverage maps. Bilinearly resample those maps at every unit step
using SciPy `ndimage.affine_transform`. The pixel centers and inverse-map
convention are handled explicitly. The maps are not freshly rendered from
geometry at later steps.

| Schedule | Pooled prediction error / true widening effect |
| --- | ---: |
| Forward control blocks | 62.24% |
| Reverse control blocks | 62.35% |
| Interleaved unit steps | 62.28% |

Forward/reverse predictions differ from one another by pooled 0.42% of the true
widening effect. On the first eight targets, subdividing each unit into 1, 4,
and 16 resampling steps gives pooled errors 62.85%, 64.19%, and 64.48%.

This particular interpolation baseline loses the subpixel arrangement and
blurs thin bands. It is not a proof that every pixel-based model must fail;
a different model or richer internal state could behave differently. Both
representations are supplied the geometric operations, so this comparison
isolates representation/resampling rather than operator learning.

## Does merely splitting the previous learned rule help?

Load the previous 21-array quadratic model from
`figures/width_pattern_state_transport/patterns_and_model.npz`. On the same 128
joint states at 10% variation, split each coordinate update into 1, 2, 4, 8,
or 16 parts. Apply its exact finite-update rule to the current predicted
pattern and state each time.

All five subdivisions give pooled error **26.3506557%**. Their endpoints differ
from direct quadratic evaluation by at most 1.48e-15 in pixel L2. Subdivision
changes the route through the same fitted function, not that function's
approximation error. This does not test retraining a one-step model or a model
that updates from observed true intermediate patterns.

## What can be reduced to one rule per knob?

A chain can reuse `n` update-rule definitions, with cost proportional to the
number of requested steps times each operation's cost. In this example each
operation touches only eight geometric points. This avoids explicitly listing
pair/triple formulas because the geometric representation and operator class
supply strong structure.

It is not a universal learning-complexity reduction. An update rule may still
need the full current state. Isolated vectors do not identify arbitrary joint
interactions: a term proportional to `height × lean × horizontal offset` can
vanish on every single-axis and pairwise test while changing a three-way joint
state. Additional data or structural assumptions are required to distinguish
such possibilities.

The previous-knob dependency count `0+1+...+(n−1)` is quadratic, not cubic.
Modeling, for each of `n` fields, every pair among its other `n−1` inputs gives
`n × choose(n−1,2)`, which is cubic. Explicitly allowing every interaction order
can grow exponentially. Counts of named inputs alone do not measure the
complexity of learning their functions.

## Related mathematical tools

- **Flows and state transitions:** a rule repeatedly evolves the current state.
  See MIT's [notes on vector fields and their flows](https://math.mit.edu/classes/18.966/2014SP/965/class4.pdf).
- **Transformation composition and Lie groups/algebras:** affine transformations
  use matrix multiplication; their generators describe small continuous moves.
  [Learning Visual Flows: A Lie Algebraic Approach](https://people.csail.mit.edu/fisher/publications/papers/dhlin09cvpr.pdf)
  is a relevant example of learning structured spatial flow models. Its
  observation model and assumptions differ from unlabeled pixel-change vectors.
- **Raster implementation:** [SciPy affine_transform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html)
  applies a supplied warp using inverse coordinates. It does not learn the warp
  or restore information lost by repeated resampling.

These tools supply ways to represent and compose updates. None provides a
universal formula that averages isolated knob vectors into the correct joint
vector without assumptions.

## Artifacts and reproduction

- `01_unit_chain.png`: reference images, actual widening patterns, transported
  boundary predictions, repeatedly resampled pixel predictions, and pixel errors.
- `02_composition_checks.png`: all joint-target results, subdivision of the prior
  quadratic model, and smaller repeated pixel-resampling steps.
- `results.json`: assumptions, all aggregate metrics, and numerical checks.
- `states_and_patterns.npz`: target settings, count schedules, all 7,680 geometric
  endpoints, raster comparisons, and the illustrated trace.

First generate the previous model if absent:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/width_pattern_state_transport.py
```

Then run:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/unit_knob_operator_composition.py
```
