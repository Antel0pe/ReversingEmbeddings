# Explicit rules discovered from numeric controls and shuffled outputs

The question was whether an understandable pixel-change rule can be found
without supplying knowledge that the images are straight strokes. The preceding
interval-overlap experiment supplied that structure. This experiment does not
supply it to the rule-building procedure.

**Result:** a generic numeric procedure recovered a six-region rule for the
height/width slice, matching all tested changes to float64 roundoff. It did not
find a small accurate rule for the harder slices. Width/lean remained approximate
with a much larger table, and a coarse five-control table failed substantially.
This is an inspectable interpolation experiment, not a solution to discovering a
compact representation of every unknown five-dimensional manifold.

## What the procedure was allowed to see

`experiments/blind_coordinate_rules/coordinate_rule.py` receives only:

- Coordinates in a supplied numeric box, normalized to `[0,1]` on each axis.
- Vectors of output values returned for requested coordinates.
- Numeric error tolerances, a sample budget, and numeric probe contexts.

It has no generator import, image reshape, grid adjacency, stroke model,
coverage equation, edge detector, physical control names, or semantic feature
extraction. It can work with any number of output channels. All 784 output
channels were shuffled before rule discovery. The adapter and evaluation code
know the original controls and layout; they restore the layout after fitting.

This is algorithmic isolation, not evidence that the experiment designer had no
prior knowledge. The two-control slices and the interpolation assumption were
chosen by the designer. The supplied controls are already useful coordinates;
recovering coordinates from an image or an unlabeled point cloud is not tested.

## The one procedure

1. Probe each numeric control with the other controls held at nine numeric
   contexts: the box center plus eight scrambled Sobol contexts.
2. Compare observed quarter-, half-, and three-quarter points with straight
   interpolation. Bisect the worst failing interval. The criterion is maximum
   absolute error over all output channels and contexts.
3. Remove redundant knots. Where measured affine traces on either side meet,
   intersect them numerically to refine a detected breakpoint. No interpretation
   of the output channels is involved. Sorted reductions make this step
   reproducible under output permutation.
4. Observe the Cartesian product of the retained coordinate knots.
5. Subtract an anchor vector and use SVD to factor the table into explicit fixed
   output patterns and coordinate-dependent weights. Retain singular values
   above `1e-12` times the largest singular value.
6. Between table states, interpolate weights using the surrounding corners:
   four corners for two controls, 32 for five controls. The rule is linear in
   each coordinate separately, including their product interactions.
7. For an update, evaluate the rule at the numeric source and target, subtract,
   and return indices with absolute change above `1e-7`, their estimated new
   values, and their signed changes.

No neural network, optimizer, symbolic shape model, or supplied boundary rule
is used. This is still a data-derived approximation method. Its transparency
comes from explicit knots, coefficients, arrays, and arithmetic, rather than
from calling it something other than a learned function.

### The equation

Inside one two-control region, let `u` and `v` be the fractions across its two
coordinate intervals. Four observed corner vectors give:

```text
base        = corner(0,0)
first       = corner(1,0) - corner(0,0)
second      = corner(0,1) - corner(0,0)
interaction = corner(1,1) - corner(1,0) - corner(0,1) + corner(0,0)

output(u,v) = base + u*first + v*second + u*v*interaction
```

Each coefficient is an explicit array of output values. SVD expresses the same
table using shared patterns; it does not change the assumed interpolation law.
The SVD patterns are not the original physical controls. Their count is not an
estimate of intrinsic dimension.

## Reference data and budgets

The adapter uses `grey_ones.render(..., dtype=np.float64)`. All domains use the
existing default range box; no bounds are extended.

| Case | Controls varied | Fixed controls | Maximum retained knots per axis |
| --- | --- | --- | ---: |
| Height/width | C, D | centers `(14,14)`, lean `0` | 129 |
| Width/lean | D, E | centers `(14,14)`, height `19.75` | 65 |
| All five | A, B, C, D, E | none | 5 |

After fitting, names are revealed: A = horizontal center, B = vertical center,
C = height, D = width, E = lean. The ranges supplied numerically were
`[14,15]`, `[14,15]`, `[19,20.5]`, `[1.8,4.6]`, and `[-10,35]` respectively.

The trace tolerance is `1e-9` per output value. Width/lean and all-five hit their
knot budgets before satisfying this tolerance. The all-five grid is intentionally
coarse: a 65-knot grid in five dimensions would require over a billion states.
Thus the cases do not have equal coordinate resolution; the full-space failure
is a budgeted result, not an impossibility claim.

For each case, 256 fresh scrambled Sobol source coordinates are sampled from
`[0,0.9]^d`. Each control is increased separately by `0.1` of its supplied range.
This means center `+0.1` pixel, height `+0.15` pixel, width `+0.28` pixel, or lean
`+4.5` degrees. Every target remains inside the range box. None of these source
or target coordinates was used during discovery. The cases use two actions per
source or five actions per source, yielding 512, 512, and 1,280 updates.

## Results

A changed index has absolute pixel change above `1e-7`. Values are not clipped
or projected before evaluation.

| Case | Test updates | Extra indices/update | Missed indices/update | Updates with exactly correct index sets | Combined change error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Height/width | 512 | 0 | 0 | 100% | `7.56e-15` relative |
| Width/lean | 512 | 4.00 | 0 | 15.23% | 1.29% |
| All five | 1,280 | 54.01 | 0.22 | 0% | 55.74% |

Combined change error is the Euclidean norm of all change errors divided by the
Euclidean norm of all true changes. It is not a percentage of correctly changed
pixels or a proof that the generated images are valid.

Largest individual change-value errors were `3.22e-15`, `0.03635`, and `0.74531`.
A further 256 height/width updates close to the discovered breakpoints had no
extra or missed indices and maximum change-value error `3.11e-15`.

A fixed update pattern taken from the canonical root had combined errors of
62.93%, 123.73%, and 125.81% respectively. This baseline uses exact renderer
changes at the root; its failure measures why state dependence matters.

### Was the success compact?

| Case | Retained knots per axis | Retained table states | Oracle states used in discovery | Fixed SVD patterns | Stored scalar entries |
| --- | --- | ---: | ---: | ---: | ---: |
| Height/width | 3, 4 | 12 | 5,375 | 9 | 7,955 |
| Width/lean | 65, 65 | 4,225 | 8,721 | 94 | 471,760 |
| All five | 5, 5, 5, 5, 5 | 3,125 | 3,865 | 267 | 1,044,512 |

Scalar counts include the dense pattern arrays, weights, anchor, and knots,
without serialization overhead or singular values. Zero entries are included.
Discovery samples are more numerous than retained references; the height/width
algorithm is not sample-efficient at the strict `1e-9` tolerance.

### A concrete discovered rule

Without semantic inputs, the procedure found internal breaks at C = 20,
D = 2, and D = 4 (to floating-point precision). These produce six rectangles.
After revealing the control names, one rectangle is height `[19,20]`,
width `[2,4]`. Inside it:

```text
u = height - 19
v = (width - 2) / 2
image = base + u*height_effect + v*width_effect + u*v*interaction
```

Consequently, when both source and target widths remain in `[2,4]`:

```text
pixel_change_for_width_step = (step / 2) *
                             (width_effect + (height - 19)*interaction)
```

`width_effect` has 40 nonzero output indices. `interaction` has only four:
flat indices `124, 127, 656, 659`, each with coefficient `0.5` to roundoff.
Those four indices quantify how increasing C changes the D-update pattern.
They were extracted from numeric corner differences, not from a rule about
which edge pixels to append. The other five rectangles have their own explicit
coefficients. Cross-boundary updates evaluate the appropriate piecewise rules.

Every region's nonzero coefficients are listed in
`height_width_explicit_cells.json`. Matching these finite tests to roundoff is
strong numerical evidence for this slice, not a proof of the formula everywhere.

## What failed and why

The harder tables average output patterns from reference states even when the
true active pixel indices change inside the coordinate cell. This can spread a
change onto an output channel whose reference value should remain fixed.

A numeric inspection chose the largest false width/lean change, independently
of pixel location. For original output channel 155, the reference stays
`1 -> 1`, but the inferred change is `+0.018931`. A 129-by-129 numeric sweep of
one learned cell shows its full-ink transition crosses the cell interior. A
corner-weighted average smooths across that transition. No pixel-grid adjacency
or shape detector was used for this diagnostic.

This suggests a more specific missing ingredient: discover boundaries between
output behaviors in **joint control coordinates**, rather than only adding
axis-aligned knots. It does not establish that such boundaries can always be
represented compactly or discovered from a small sample.

Raising the changed-index threshold does not eliminate the issue. At `0.01`,
width/lean still has 534 extra indices across 512 updates; the all-five case has
33,237 extra and 3,892 missed indices across 1,280 updates. The full threshold
sensitivity and per-action results are in `results.json`.

## Checks and limitations

- The entire procedure was repeated with a different output order. All three
  cases chose exactly the same observation coordinates and knots, retained the
  same pattern count, and produced restored predictions differing by less than
  `3e-14`. This checks independence from pixel ordering, not generality across
  unrelated image families.
- SVD table error was below `2e-13` per output value. Direct interpolation and
  SVD-factored interpolation agree below `1e-10`. The substantial failures are
  not caused by throwing away significant SVD components.
- Sparse output indices and new values are checked against the corresponding
  dense numeric rule. These are estimated new values; in the unsuccessful cases
  they are not guaranteed correct.
- Five independent control coordinates are supplied. No current-image inverse,
  unknown-knob discovery, manifold-membership test, off-manifold guarantee, or
  general low-dimensional chart discovery is claimed.
- Probing finitely many contexts can miss features between observations. The
  generic assumption is that piecewise multilinear rules with axis-aligned
  subdivisions provide a useful representation. It succeeds here on one slice,
  but it is not a universal consequence of having five controls.

## Figures and reproduction

- `01_pixel_updates.png`: source, true change, inferred change, true target,
  source plus inferred change, and index errors. Examples are selected by median
  relative error among width updates, not by visual attractiveness. All change
  panels share a color scale. Gold marks inferred values outside `[0,1]` that a
  normal image display would clip.
- `02_discovered_structure.png`: learned width/height regions, an explicit
  control-effect array and interaction array, and full held-out results.
- `03_joint_coordinate_boundary.png`: a numeric view of the false-change channel.
  The first two color scales explicitly zoom near full ink; the last shows signed
  error. The gold contour is sampled, not an analytic boundary equation.

Run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/blind_coordinate_rules/run.py
.venv/bin/python experiments/blind_coordinate_rules/make_figures.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/blind_coordinate_rules/inspect_failure.py
```

`*_rule.npz` contains the explicit knots, pattern arrays, and coefficient tables.
`blinding.npz` records how output indices were shuffled. `*_evaluation.npz`,
`metrics.csv`, and `results.json` retain the held-out evidence. The generic
`CoordinateRule.load()` and `sparse_edits()` functions can evaluate a saved rule
without loading the renderer. Coordinates must be normalized using the stated
range box, and sparse output indices use the stored shuffled order until restored
with the saved permutation.

The figure understanding check was applied after rendering: each figure defines
its object, reading order, operation, colors, measurements, scope, and hidden or
clipped values. Final PNGs were visually inspected.
