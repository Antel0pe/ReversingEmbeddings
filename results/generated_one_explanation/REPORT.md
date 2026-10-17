# Can observations explain the generated-1 family?

**Result: conditional exact mechanism recovery succeeded; assumption-free
recovery of the physical knobs is unidentifiable.** A short image encoder,
five small symbolic expressions, and an explicit pixel decoder reconstruct
11,908 held-out states to a maximum pixel error of **9.231e-8** for float32
observations and **1.222e-14** for float64 observations. An independently
implemented checker passed, including fresh oracle states.

The qualification matters: the experiment supplies a straight-strip geometry
and a restricted raster-program grammar. Anonymous single-control intervention
links select unmixed control families. Neither the geometry grammar nor the
exact allowed range box was learned without prior knowledge. No claim of
unrestricted scientific discovery or proof by finite sampling is made.

## 1. Three different questions

| Question | What five generator inputs establish | What needs more evidence |
|---|---|---|
| Intrinsic dimension | For this locally Lipschitz mathematical renderer, dimension is at most five. | Redundant inputs, collisions, or collapsed regions could reduce it. A left inverse or a rank-five regular patch establishes five dimensions. |
| Exact reconstructive coordinates | The known five settings and known renderer already provide an exact construction of their image set. Injectivity is unnecessary for a decoder. | Discovery must produce its own executable decoder, explain its domain, and test joint changes and all pixels. |
| Recovery of the original controls | The input count says nothing about the meaning or uniqueness of coordinates. | Assumptions or extra observations must distinguish original controls from invertible mixtures. |

For the ideal real-arithmetic strip family, the read-off equations below give
a left inverse throughout the stated domain. This establishes intrinsic
dimension five, rather than just estimating it from a singular-value cutoff.
The family has piecewise formula joins; it need not be a globally smooth
embedded manifold. A union of tangent cones can span more than five linear
directions without adding independent controls.

A collection of stored float32 arrays is a finite set, so its literal
topological dimension is zero. Here “five-dimensional” refers to the continuous
pre-storage construction, not to that finite set. Float64 computations also
have finite precision; they are diagnostics of the mathematical construction.

## 2. Identifiability before fitting

Write a decoder as `image = F(p)`. For any invertible coordinate change `T`,

```
z = T(p)
D(z) = F(T_inverse(z))
```

produces exactly the same images. More samples, including the entire image
set, cannot distinguish the two explanations. Their image distributions also
agree when the unknown latent distribution is pushed through `T`.

A concrete alternative is

```
z_cx = cx
z_width = width + 2*(cx - 14)^2
other coordinates unchanged
```

The inverse subtracts the same quadratic term. At the canonical root,
`cx=14`, its differential is the identity: all root directions agree. Away
from the root, changing `z_cx` while holding `z_width` fixed changes both
physical position and width. Both coordinate systems have commuting coordinate
updates. Root directions, reconstruction, and commutation do not identify knobs.
The transformed domain must be transformed too; taking its bounding rectangle
would incorrectly introduce invalid states.

Even requiring the *same rectangular support* does not fix mixing. Normalize
physical settings into a unit box, and change just its width coordinate:

```
k = 4*u_cx^2*(1-u_cx)^2
v_width = u_width + k*u_width*(1-u_width)
v_other = u_other

u_width = 2*v_width /
          (1+k + sqrt((1+k)^2 - 4*k*v_width))
```

Here `0 <= k <= 1/4`, and the derivative with respect to `u_width` is at least
`3/4`. Thus this is a box-to-itself invertible mixing, fixing the boundary
values of width and agreeing with the physical differential at the root.
The driver checks its parameter and image round trips, but the equations and
monotonicity establish invertibility over the continuous box.

Even a known independent uniform latent distribution is not sufficient in
general: rotate a small disk strictly inside a two-coordinate square by an
angle depending smoothly on radius, with the angle vanishing near the disk
boundary. In polar coordinates this preserves `r dr dtheta`, is identity
outside the disk, preserves the uniform box distribution, and mixes controls.
This is a mathematical counterexample, not an additional fitted model.

**Acceptable ambiguity:** permutation, sign, units, additive origin, and
independent invertible scalar reparameterization, provided each scalar rule
is explained. Lean angle versus its tangent is acceptable. Mixing different
controls is acceptable as an intermediate geometric chart, but fails a claim
that its individual axes recover the original knobs.

This obstruction is consistent with the primary identifiability literature;
see [Locatello et al., 2019](https://proceedings.mlr.press/v97/locatello19a.html).
The concrete argument here does not rely on a model survey.

## 3. Exactly what discovery observes

| Observation | Information available | Information absent |
|---|---|---|
| Unlabeled images | Raw 28x28 coverage arrays; pixel locations and coverage units | Settings, action names, dynamics, exact support bounds |
| Nearby images | The same arrays with proximity or neighborhood information | Which hidden factor changed, purity of a change, correspondence across states |
| Added condition used here | Before/after pairs and a stable anonymous action ID; exactly one factor changes and each ID has the same meaning across base states | Control name, values, step sizes, calibration, renderer source, number of subrows, original range box |
| Known settings | Used by the separate oracle driver for data generation and independent evaluation only | Never passed to `discover.py` |

The discovery API is `fit(images, before, after, ids)`. Training data contain
128 unlabeled float32 images and 5,120 linked pairs, 1,024 per anonymous action.
No parameter values or physical names are present in `train_public.npz`.

The evaluator later asks a **copy intervention**: given source image, donor
image, and action ID, predict the image obtained by replacing that one source
factor with the donor's factor. Its target image is hidden during prediction.
This test explicitly supplies copy semantics. It does not pretend anonymous
pair IDs supply a calibrated additive physical increment. Copying is invariant
to independent scalar reparameterization; equal numerical increments are not.

### Structural knowledge supplied, not discovered

- Noise-free raw coverage in fixed pixel coordinates; no crop, blur, gamma,
  unknown intensity transform, or unknown background.
- One unclipped straight strip, constant horizontal width, horizontal caps.
- Enough complete rows, with a common integer column boundary inside the strip
  across each selected row.
- A candidate sensor grammar: exact vertical interval overlap, horizontal
  interval overlap evaluated on uniform midpoint subrows, then summation.
  Subrow candidates are **1, 2, 4, 8, 16, 32, 64**.
- A symbolic simplicity prior: invariants are searched at degree one, then two,
  in five image-derived geometric features. Coefficients must snap to small
  rational numbers with denominator at most 16 and error at most 2e-5 after
  normalization. A unique invariant at the first successful degree is required.
- Globally pure, consistently labeled single-factor changes, with varied base
  states and nontrivial changes of every factor.

The researcher read the renderer before constructing this grammar. Software
separation prevents hidden numerical inputs from entering fitting; it does
not erase that human source of prior knowledge. This is **conditional program
identification**, not blind rediscovery of an unknown renderer.

### Why the extra links should help

Suppose an invertible chart `z(p)` assigns one coordinate to each action ID.
If dense globally pure interventions show that every other action leaves
`z_i` unchanged, then on a regular connected product domain

```
partial z_i / partial p_j = 0    for j different from i
therefore z_i = h_i(p_i)
```

Invertibility makes each `h_i` invertible. Stable IDs fix the same permutation
across the domain. This excludes mixing while leaving independent nonlinear
scales. Continuity extends the statement from dense links to their closure;
at renderer joins the finite invariance statements remain meaningful.

Finite links do not establish these coverage assumptions. This experiment
adds a low-degree program restriction and withheld tests. It does not prove
that anonymous links are the minimum information necessary. In particular,
the strong geometric prior already makes a meaningful chart possible without
links, and can itself motivate the usual physical coordinates. The addition
provides direct evidence about which coordinate families match interventions.

## 4. Read geometry from the image

The intermediate chart is `(t,b,a,w,s)`:

- `t`: vertical top edge, in pixels measured downward from the canvas top.
- `b`: vertical bottom edge, in the same units.
- `w`: horizontal width in pixels.
- `a`: horizontal centerline position extrapolated to vertical coordinate zero.
- `s`: horizontal displacement per vertical pixel, with positive values
  putting the top farther right: centerline at height `y` is `a - s*y`.

These are meaningful geometric coordinates, but changing `t` alone mixes the
original vertical-position and height controls. Likewise `a` is coupled to the
original horizontal center through lean and vertical position.

Let `m_r = sum over columns of image[r,column]` be total ink in row `r`.
An interior complete row has `m_r = w`; take the median of the interior row
masses for numerical robustness. If `r0,r1` are first and last inked rows,

```
t = r0 + 1 - m_r0 / w
b = r1     + m_r1 / w
```

For a complete row, choose an integer boundary `k` whose cumulative ink
`C_r(k) = sum_{column < k} image[r,column]` is closest to half the width.
When this boundary is inside every strip cross-section in that row,

```
center_at_row_midpoint = k - C_r(k) + w/2
```

`k-C_r(k)` is the average left edge, and adding `w/2` gives the center.
This uses coverage sums; an ink-weighted average of *pixel centers* would
introduce a discretization bias. Centers from the first and last chosen full
rows determine the line `a-s*y` directly. No optimizer or nearest-image lookup
is needed.

For the entire default box, `w >= 1.8` and `abs(s) <= tan(35 degrees)`.
Subrow midpoints extend at most `15/32` of a pixel from their row midpoint.
The nearest integer to the row center is inside every cross-section because

```
1.8/2 - 1/2 - (15/32)*tan(35 degrees) > 0.0717 pixels.
```

The closest-half-ink boundary has this property too: cumulative ink is linear
with slope one throughout the common interior interval, monotone outside it,
and the nearest integer is already inside that interval. Both choices at a
halfway tie are valid. This proves why the read-off works here; it is not a
safe assumption for arbitrarily thin, clipped, or steep strokes.

## 5. What the linked pairs recovered

For each anonymous action, the method finds a polynomial unchanged by all
*other* action groups. The recovered expressions, in anonymous ID order, are

```
z0 = t + b
z1 = w
z2 = s
z3 = a - (t*s + b*s)/2
z4 = t - b
```

Their minimum degrees were `[1,1,1,2,1]`. The quadratic expression for `z3`
removes the mixing of centerline intercept, vertical position, and lean.
The executable model stores these small rational coefficients, not network
weights. The inverse is explicit:

```
t = (z0 + z4)/2
b = (z0 - z4)/2
w = z1
s = z2
a = z3 + z0*z2/2
```

Only after freezing these equations did the evaluator compare them with
physical settings. A permutation and five independent affine fits, with the
predeclared scalar angle-to-slope conversion, give

```
vertical center   = z0/2
horizontal width  = z1
lean angle        = atan(z2), converted to degrees
horizontal center = z3
vertical height   = -z4
```

There is no multivariate alignment fitted to hide entanglement. In float32
held-out data, maximum errors are 2.662e-8 pixels for vertical position,
5.364e-8 for width, 3.878e-9 for lean *slope*, 4.230e-8 for horizontal
position, and 5.588e-8 for height. The slope error is not an angle error.

Unlinked images admit both `(t,b,a,w,s)` and this chart with identical
reconstruction. In held-out linked pairs, the mixed chart changes several
coordinates for vertical-position and lean interventions. The recovered
chart's largest off-axis change is **1.832e-7 training-standard-deviation
units**, consistent with storage error. Shuffling the anonymous IDs causes
the invariant search to reject the data.

The first training pilot, with 128 pairs per action, failed the fixed rational
coefficient criterion for the quadratic expression. Increasing training to
1,024 pairs per action resolved that conditioning problem without changing
acceptance thresholds or inspecting test values to choose coefficients.
The code standardizes features before solving the invariant constraints.

## 6. The entire explicit decoder

Use `S=16` subrows. For subrow index `j`, define

```
y_low = j/S
y_high = (j+1)/S
y_mid = (j+1/2)/S
vertical_length = max(0, min(y_high,b) - max(y_low,t))
center = a - s*y_mid
left = center - w/2
right = center + w/2
horizontal_length(column) = max(0, min(right,column+1) - max(left,column))

image[row,column] = sum over j in that row of
                    vertical_length * horizontal_length(column)
```

Every term is an interval length or a line position. A pixel value is total
ink coverage according to this particular subrow sensor. To reproduce default
storage, cast the resulting array to float32. To diagnose the construction
before that cast, retain float64.

Vertical movement shifts both caps together; height moves caps apart;
horizontal position translates the centerline; width separates its two sides;
lean changes the line slope around the stroke's vertical center, since
`a = cx + s*cy`. Each coordinate's effect is therefore explicit throughout
the family, including joint changes.

At fixed clipping branches, the decoder is quadratic in `(t,b,a,w,s)`.
After substitution of the separated `z` coordinates, it is at most cubic in
`z` within each formula cell. Side/pixel and cap/subrow transitions change
which branch applies. The interval program joins them continuously and
selects the branch automatically. No separately fitted local charts or
nearest-direction transport are required. Coordinate conversion is global
on the valid strip domain, including the joins.

The training raster comparison was decisive within the supplied grammar:

| Subrows | Largest training pixel residual |
|---:|---:|
| 1 | 8.423e-2 |
| 2 | 2.112e-2 |
| 4 | 5.435e-3 |
| 8 | 1.319e-3 |
| **16** | **6.118e-8** |
| 32 | 3.262e-4 |
| 64 | 3.268e-4 |

This selects 16 from these candidates; it does not establish uniqueness
against every conceivable alternative sensor program.

## 7. Success thresholds, independent tests, and worst failures

Thresholds were written to `PROTOCOL.md` before fitting:

| Input storage | Maximum absolute pixel error | Maximum Euclidean image error |
|---|---:|---:|
| float64 | 5e-11 | 5e-10 |
| float32 | 2e-6 | 1e-5 |

All 784 raw pixels enter the error calculations. The Euclidean image error is
`sqrt(sum of squared pixel errors)`, without image normalization or a variance
percentage. Test arrays were created after the symbolic model was frozen.

The 11,908 held-out states comprise 4,096 Sobol states, 1,024 full lattice
states including all 32 corners, 2,560 near-face states, 1,515 exact/near cap
subrow crossings, 2,328 exact/near side-column crossings, 384 encoder-boundary
ties, and the canonical root. They are held out, not all statistically
independent random samples. Boundary construction uses oracle knowledge in
the evaluator, not in discovery.

| Test | Largest pixel error | Largest image L2 error |
|---|---:|---:|
| Reconstruction, float64, 11,908 states | 1.222e-14 | 4.990e-14 |
| Reconstruction, float32, 11,908 states | 9.231e-8 | 3.217e-7 |
| Unseen one-factor copy, float64, 2,560 targets | 1.089e-14 | 4.643e-14 |
| Unseen one-factor copy, float32, 2,560 targets | 7.631e-8 | 2.606e-7 |
| Three five-factor action orders, float64, 1,536 endpoints | 1.699e-14 | 9.972e-14 |
| Three five-factor action orders, float32, 1,536 endpoints | 6.954e-8 | 2.283e-7 |
| Joint coordinate interpolation, float32, 1,536 states | 6.608e-8 | 2.121e-7 |
| Fresh independently checked oracle states, float64, 1,056 states | 8.882e-15 | 3.760e-14 |

Five-factor copying changes all controls together to independently selected
values. Joint interpolation is linear in the recovered independent controls,
so lean interpolates in slope rather than angle. Re-encoding intermediate
images checks path consistency. Individual pixel directions need not remain
constant along these paths, and joint changes need not equal a sum of isolated
pixel changes.

No held-out valid state exceeded the predefined thresholds. The worst float32
reconstruction occurs at a side-column transition, with physical settings
approximately `(14.3307978337, 14.4990208074, 19.4232746828, 4.5915701926,
0.4488059366 degrees)`. Exact settings and worst-case indices are saved.

Counterexamples sought and found:

- Using ordinary row-mass vertical centroids introduces up to **0.006412 px**
  vertical-position error.
- Using 8 subrows gives up to **0.001368** pixel error and **0.006984** image
  error on test states.
- Exact integration of the *ideal continuous parallelogram* still gives up to
  **0.000341902** pixel error and **0.001965** image error. More geometrically
  accurate area integration fails to match this sensor. The subrow rule is
  part of the observed mechanism.
- Averaging opposite-lean strokes produces an invalid image despite its
  plausible row geometry. Its decoder residual is **0.586943** per pixel and
  **2.992066** in image L2. Read-off coordinates alone are not membership.
- Shuffled action IDs fail the invariant search.
- A collision below disproves unique exact physical inversion from storage.

All fitted coefficients, split seeds, source hashes, software versions, and
per-group metrics are saved. No threshold was relaxed to pass these tests.

## 8. Extra directions, rendering transitions, and numerical precision

At three selected starts, the driver samples 96 joint directions at each
normalized scale from 1e-2 through 1e-9, separately in float64 and float32.
Normalized scale means a fraction of each original parameter's allowed span;
these controls are used only by this evaluator. The diagnostic is the sixth
singular value divided by the first for seed-to-neighbor pixel differences.
It is not a fitted discovery coordinate.

| Start and storage | Radius 1e-5 | Radius 1e-7 | Interpretation |
|---|---:|---:|---|
| Smooth, float64 | 2.284e-6 | 2.285e-8 | Finite-scale curvature terms shrink with scale. |
| Tilted cap kink, float64 | 1.078e-2 | 5.574e-4 | Crossing branches supplies different first-order directions. At still smaller scales the ratio approaches 5.574e-4. |
| Upright cap join, float64 | 2.224e-6 | 2.226e-8 | This special join has agreeing first derivatives; extra terms shrink. |
| Smooth, float32 | 5.695e-4 | 5.102e-2 | Quantization overwhelms the small change; extra singular directions increase. |

At the tilted cap kink, the union of positive and negative one-sided
finite-difference derivatives has sixth/first singular ratio **6.048e-4**.
At the smooth point the corresponding ratio is **2.169e-9**. These are
finite-step measurements, corroborating the branch formulas rather than
proving tangent ranks from a numerical cutoff. A sixth linear direction
at the kink is a union-of-branches effect, not a sixth freely chosen factor.
At radius 1e-9 even float64 subtraction shows a roundoff floor.

This separates four causes often confused as reconstruction speckle:

1. Curvature: products of the same five coordinates create additional finite
   secant directions, which shrink relative to first-order changes in a cell.
2. Sampling transitions: cap/subrow and side/column branch changes can produce
   one-sided tangent differences; the explicit interval decoder includes them.
3. Quantization: float64-to-float32 casting alone measured at most
   **2.981e-8** per pixel and **1.633e-7** per image. Read-off from rounded data
   adds small coordinate estimation error, yielding the observed 9.231e-8
   maximum reconstruction residual.
4. Model error: the wrong-subrow and ideal-continuous-area residuals persist
   at about 1e-3 to 1e-4 per pixel, orders of magnitude above storage residuals.

No pixel thresholding, denoising, smoothing, or intensity rescaling is applied
before fitting or checking.

## 9. What is exact, what is bounded, and what was tested

### Mathematical identities, conditional on the declared mechanism

In real arithmetic, the row-mass and cumulative-ink equations recover
`(t,b,a,w,s)`, and the recovered polynomial equations give the five separated
controls above. The interval decoder is identical to the stipulated raster
construction. These are algebraic and interval identities, not conclusions
from test sample counts. Their assumptions are checked analytically over
the default domain: the stroke remains unclipped, there are enough full rows,
and the common integer-boundary clearance is positive.

For clarity, the default box puts `t` in `[3.75,5.5]` and `b` in
`[23.5,25.25]`. Even an active subrow midpoint extends at most `1/32`
pixel beyond a cap. With maximum half-height `10.25`, half-width `2.3`,
and slope `tan(35 degrees)`, its horizontal support stays between about
`4.50` and `24.50` on the 0-through-28 canvas. Thus there is no hidden
canvas clipping in these identities. Height at least 19 leaves the selected
complete-row midpoints at least 16 pixels apart.

The resulting continuous family has an inverse and five independent real
coordinates. Matching directions across neighboring images is unnecessary:
the scalar equations and their inverses are global and explicit.

### Certified input-rounding bounds and a conditional propagation bound

IEEE round-to-nearest float32 storage of a value in `[0,1]` changes it by at
most `epsilon = 2^-25 = 2.981e-8`. For a 784-pixel image this gives a certified
casting bound `L2 <= 28*epsilon = 8.345e-7`, relative to the float64 values
being cast. This does **not** bound the preceding float64 coverage arithmetic
against an ideal real-number calculation.

A conservative mathematical propagation bound is also possible, treating
read-off and decoder operations as exact and perturbing each input pixel by
at most `epsilon`:

```
m = 28*epsilon                         # row-mass and width error bound
cap_error <= 2*m/(1.8-m)                # at most 9.272e-7 pixels per cap
row_center_error <= 42*epsilon          # cumulative ink plus half-width
centerline_error <= 42*epsilon*(1+56/16) # at most 5.633e-6 pixels
pixel_roundtrip_error <= 2*cap_error + 2*centerline_error + m + epsilon
                      <= 1.399e-5
image_roundtrip_L2 <= 28*pixel_roundtrip_error <= 3.916e-4
```

The two selected complete-row midpoints are at least 16 pixels apart, which
bounds line extrapolation over `0 <= y <= 28`. The width and cumulative sums
are Lipschitz with constants at most 28 and 42. Horizontal interval overlap
changes by at most the sum of its two endpoint changes. The sum of vertical
overlap changes per row is bounded by the two cap displacements. The proven
0.0717-pixel common-boundary margin dwarfs cumulative rounding errors, so the
integer boundary remains usable, including ties. Exact-zero background remains
zero under casting; a cap row erased by float32 underflow contains at most
28 times half the smallest positive float32 value, far below these bounds.

These bounds are deliberately loose. They justify bounded sensitivity to
storage while explicitly **not certifying the stricter empirical thresholds
over the entire continuous box**. Nor is this an interval-arithmetic
certification of every binary64 operation in the actual implementation.

### Empirical results

All reported maxima, singular-value ratios, finite-step interventions,
coefficient recovery from finite training, and checker results are empirical.
The precise float64 errors depend on floating-point evaluation order. No
finite test establishes a universal numerical maximum or rules out all
alternative symbolic programs. A rigorous tighter numerical claim would need
outward-rounded interval verification across formula cells or an equivalent
error analysis of the executable implementation; that was not performed.

## 10. Completeness, validity, and support boundaries

The discovered program by itself describes a wider straight-strip family;
finite training images do not reveal the exact original range box. The
**evaluator-supplied** permitted domain for the recovered chart is

```
28 <= z0 <= 30                  # twice vertical center
1.8 <= z1 <= 4.6                # width
 tan(-10 degrees) <= z2 <= tan(35 degrees)
14 <= z3 <= 15                  # horizontal center
-20.5 <= z4 <= -19              # negative height
```

For every original allowed physical setting, the explicit scalar conversion
produces a point in this box and the decoder produces its image. Conversely,
every point in this recovered-coordinate box converts to an allowed original
setting and produces a family member. These two implications establish
coverage and soundness for the mathematical construction **with that supplied
support**, rather than only for a sampled cloud. No extra charts are needed.

This is a bijection of real control domains before storage; after storage,
different permitted points can decode to the same stored image. Every stored
valid image still has a representative, but a unique representative is neither
required nor generally available. Domain checks should allow the stated
measurement uncertainty near a boundary; silently clipping measured controls
would mask a potentially invalid input. For arbitrary input images, encode,
check range uncertainty, decode, and inspect the full residual. Read-off alone
does not assert membership. A tolerance-based membership test defines a tube
around the exact family, not an exact zero set for arbitrary noisy images.

The exact support bounds are a second missing piece of information if the
objective is discovery of *this restricted* family from finite observations
alone. A slightly larger unobserved domain could generate every observed
training image too. Declaring these bounds learned would be unjustified.

## 11. Stored collisions and an impossible exact prediction

The following two physical settings produce bitwise identical float32 arrays:

```
(14, 14,             19.75, 3.2, 0 degrees)
(14, 14.000000001,   19.75, 3.2, 0 degrees)
```

Apply the same physical vertical increment `+0.125 px` to both. Their float32
outputs now differ, with maximum pixel difference **1.000001e-9** and image
L2 difference **1.650e-9** (rounded upward here). The saved raw arrays let the
checker reproduce the identical inputs and different outputs.

Thus no deterministic rule given only the stored input image and that exact
increment can predict both outputs bitwise correctly. The increment fails to
descend to a well-defined action on equivalence classes of indistinguishable
stored states. This is information loss, not a failure of model capacity.
The effect is far below the declared reconstruction tolerance, so it does not
invalidate bounded-error predictions. It does invalidate an exact unique
physical inverse or universal bitwise future prediction from stored images.
For exact action semantics one must preserve more state, supply the original
settings, or propagate the entire compatible-state uncertainty set.

The present encoder chooses a numerically consistent estimate and reports
bounded residuals. It does not compute the full parameter feasibility region
for every stored image. Collision handling is explicit in the contract and
in the independently checked counterexample rather than hidden by demanding
an impossible inverse.

## 12. What this says about messier datasets

Potentially transferable ideas:

- Separate reconstructive coordinates from causal or physical identifiability.
  A useful small program may exist without uniquely identifiable original knobs.
- Stable intervention groups, repeated controlled changes, or instrument metadata
  can constrain factor families more directly than geometric proximity alone.
  Their purity and cross-state consistency must be justified, not assumed from
  correlation or time adjacency.
- Search for simple invariants and executable transformation laws; expose the
  grammar and symbolic complexity preference rather than calling it neutral.
- Validate new joint combinations, boundary cases, full residual arrays,
  sensor precision, ambiguous states, and uncertainty in domain membership.
- Treat the sensing process as part of the mechanism. A beautiful geometric
  model can systematically miss quantized or sampled measurements.

Synthetic dependencies that do not automatically transfer:

- A single straight strip, horizontal caps, known coverage units, no occlusion,
  no background, and exact row-ink conservation.
- A uniform subrow sensor in a small candidate grammar, constant width, and
  rational low-degree invariants.
- An exact five-factor product domain and guaranteed pure interventions.
- A supplied exact allowed range box and an oracle for independent evaluation.
- Essentially noiseless coverage, enabling tiny numeric tolerances.

For real observations, a plausible continuation of this ambition is a small
mechanism plus a declared observation/noise model and interventions with
measured purity. Success would be an interpretable decoder and calibrated
residual/uncertainty bounds, not simply replacing the analytic equations with
opaque fitted weights. No additional datasets are needed to establish the
present result or its obstruction.

## Reproduce and inspect

Start with the compact [mechanism explanation](MECHANISM.md) for the equations
and main limits on one page.

From the repository root, using its existing environment:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/generated_one_explanation/run.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/generated_one_explanation/check.py --oracle
```

The second command independently implements image read-off, polynomial
coordinates, inverse coordinates, and pixel coverage. It imports no discovery
module. Without `--oracle` it uses NumPy and the saved raw arrays; with that
flag it additionally checks 1,024 fresh random settings and 32 corners against
the actual repository renderer. The setting arrays used for alignment are
explicitly evaluator-only. The checker verifies the model hash, reconstruction,
coordinate agreement under the permitted scalar calibrations, held-out copy
interventions, reversed encode/decode consistency, the invalid mixture, and
the collision witness.

Files:

- [Frozen protocol](../../experiments/generated_one_explanation/PROTOCOL.md)
- [Discovery program](../../experiments/generated_one_explanation/discover.py)
- [Oracle-isolated evaluator](../../experiments/generated_one_explanation/run.py)
- [Independent checker](../../experiments/generated_one_explanation/check.py)
- [Five symbolic equations and fitting diagnostics](model.json)
- [All measurements, worst cases, splits, and source hashes](results.json)
- [Independent checker results](checker_results.json)

The exact-recovery obstruction has two parts: observations alone do not fix
physical control meaning, and stored images can lose information required for
exact future actions. The conditional positive result is stronger than a
projection or approximate latent model: five explicit scalar equations and a
short piecewise coverage program reconstruct the family, with global domain
identities and independently tested numerical accuracy. The unlearned pieces
are the supplied mechanism grammar, intervention purity, and exact support.
