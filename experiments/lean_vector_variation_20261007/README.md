# A vector representation of how the lean vector changes

This replaces the previous question's geometry-based reconstruction with a
**measured vector-field model**. The predictors receive pixel vectors. They do
not use stroke edges, subrows, centers, width estimates from row sums, inverse
geometry, or calls to the renderer during prediction. The renderer is used
only to generate controlled observations and score predictions.

The successful result is an approximate width-slice model in the form

```text
V_hat(x) = v0 + alpha(q(x)) b
q(x)     = dot(a, x - x0)
```

Here every `x`, `x0`, `v0`, `a`, and `b` is a vector in 784-dimensional pixel
space. `x` is the input image; `x0` is the reference image; `v0` is its lean
vector. `b` is a vector describing a change in the lean vector, not a width
movement of the image. `alpha` is a learned scalar function, and `a` supplies
an image-based scalar coordinate along the observed width curve.

This is not an exact decoder of the five-knob family. Joint prediction from
independent curves alone remains unresolved, and this report exposes its errors.

## The differential object being sought

Let `X(s)` be the image path made by varying width while fixing the other
controls. Let `V(x)` be the lean-vector field at image `x`. Then

```text
C_width(s) = d/ds [ V(X(s)) ]
          = limit as ds -> 0 of [V(X(s+ds)) - V(X(s))] / ds
```

`C_width(s)` is the desired vector: it specifies how each component of the
lean vector changes as we move along the width path. If the width tangent is
`T_width(x) = dX/ds`, the same object is `D V(x)[T_width(x)]`.
This is a first derivative of a vector field. Expressed through an image
generator with width and lean coordinates, it is a mixed derivative of that
image generator. No renderer formula is needed to measure finite versions.

Locally, where these derivatives exist:

```text
V(X(s+ds)) ≈ V(X(s)) + ds C_width(s)
```

A constant `C_width` would give one affine line of lean vectors. If that
change-vector varies, a larger representation is needed. The general learned
vector decoder used here is

```text
V_hat(x) = v0 + sum_j alpha_j(q(x)) b_j
```

The `b_j` are fixed observed variation vectors; only their scalar coefficients
change with state. This is a statement about the space of lean vectors, not
an image reconstruction or a projection of the image manifold.

## What was actually observed and learned

The controlled reference image was observed at
`(14.5, 14.5, 19.75, 3.2, 0)` in pixels/degrees. Those settings specify the
experimental setup; they are not inputs to the width predictor.

At each observed width, the lean response was measured as

```text
V_epsilon(x) = [image after +epsilon lean - image after -epsilon lean]
              / (2 epsilon)
```

with `epsilon = 0.02 degrees`. Thus scores are against observed finite-difference
vectors. They are not assertions of exact infinitesimal derivatives at every
pixel switching event. Separate-axis lean endpoint observations use the available
one-sided window to stay inside the control range.

First, 65 coarse width observations locate the largest change in the measured
lean vectors. The algorithm then adds 257 samples around that interval, for
322 distinct training widths. The refinement interval is discovered from the
vectors, not hard-coded from the renderer.

For the single-change-vector model:

```text
b = V(narrow reference) - v0
span = image(wide endpoint) - image(narrow endpoint)
a = span / dot(span, span)
q_i = dot(a, observed_image_i - x0)
alpha_i = dot(b, observed_lean_vector_i - v0) / dot(b, b)
```

`alpha(q)` interpolates these observed scalar coefficients. Between two learned
knots `q_i` and `q_(i+1)`, the explicit scalar equation is

```text
alpha(q) = alpha_i
         + (q-q_i)/(q_(i+1)-q_i) * (alpha_(i+1)-alpha_i)
```

This supplies an explicit pixel-input encoder and a vector decoder. It does
not secretly receive the image's physical width. The coordinate is monotone
on the sampled width path, not asserted to identify arbitrary states of the
whole five-dimensional family.

To increase capacity, singular vectors of the observed **lean-vector changes**
supply additional fixed `b_j`. Coefficients are measured by vector projection
and interpolated in the same learned image coordinate. No variance-retention
claim or manifold-dimension claim is made; actual prediction errors decide
whether the representation is adequate.

## Width prediction results

Held-out model comparisons used 1,000 random widths and 301 deliberately dense
transition probes. None coincided with a training width.

| Model | Mean random-width vector error | Worst transition vector error |
| --- | ---: | ---: |
| One measured change-vector `b` | 0.0393% | 14.70% |
| Four learned variation vectors | 0.0149% | 1.612% |
| Eight learned variation vectors | 0.00353% | 0.383% |
| Sixteen learned variation vectors | 0.000771% | 0.236% |

After fixing the dictionary size at 16, an independent confirmation used
1,000 fresh random widths and 1,001 fresh transition probes. Mean random-width
error was **0.000612%** and worst transition error **0.375%**. The denser probes
exposed a larger worst case; the earlier 0.236% is not the final worst case.

Vector error means `norm(predicted-observed)/norm(observed)` across all 784
components. The transition matters: one vector gets most of the family right
but misses changes in the relative per-pixel responses there. The larger
dictionary captures those changes approximately. The results concern a fixed
upright width slice and one finite-difference resolution.

Model storage includes reference image/vector, coordinate vector, variation
vectors, and scalar coefficient knots. This is empirical compression with a
learned scalar function; it is not a short closed-form universal law discovered
from all states, and not a per-state table of 784-component lean vectors.

## Do the independent vector changes predict joint states?

Five single-axis observation curves provide changes relative to the same
reference lean vector. The additive model is

```text
V_add(s1,...,s5) = v0 + sum_i [V_i(si) - v0]
```

The ideal-label test supplies the known normalized progress along each control
curve. A second test learns a five-component linear image encoder solely from
the single-axis image differences, then supplies its approximate coordinates
instead. This encoder is empirical; it is not the analytic knob inverse.

On 512 fresh joint states per case:

| Joint states | Mean error, known applied-change labels | Mean error, learned image coordinates |
| --- | ---: | ---: |
| Width + height | 5.98% | 5.71% |
| Width + horizontal shift | 72.05% | 72.05% |
| All five controls | 154.46% | 132.47% |

These differ from the previous experiment's errors: that geometry-based test
queried each other-knob response at the target's current lean. This experiment
has strictly single-axis observation curves through the reference state.
It does not have all of those two-knob slices during training.

A width-only model also fails the full-space control: the same 16-vector decoder
has more than 100% mean error on 512 random full-box states. Adding more vectors
to the independent additive dictionary does not fix the missing joint dependence.

The mathematical reason an arbitrary joint interaction cannot be recovered
from the two single-axis curves alone is visible in this example:

```text
V(s,t) = v0 + s b_width + t b_height + s*t b_width_height
```

On the width-only path `t=0`, and on the height-only path `s=0`.
The interaction vector `b_width_height` vanishes from both observations. Any
choice of it gives the same two single-axis curves. To identify it requires
joint observations or additional structural assumptions. This is an
identifiability limit for unrestricted vector fields, not a claim that every
particular interaction is necessarily large or impossible to infer under priors.

A measured finite interaction vector could be obtained from four field samples:

```text
b_width_height ≈ [V(s,t)-V(s,0)-V(0,t)+V(0,0)] / (s*t)
```

That is still a vector-based construction; it does not recover or reverse
stroke geometry. It is the next term required when independent changes fail.

## Limits and reproduction

The width model does not encode current lean changes. Holding its initial
predicted vector fixed for a 0.1-degree move has substantial error near switching
states. Following the field along arbitrary lean paths, reconstructing the
whole family, and handling all joint states have **not** been solved by this
experiment. The earlier geometry implementation is not evidence for those
claims under the present vector-only method.

From the repository root:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/lean_vector_variation_20261007/run.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/lean_vector_variation_20261007/check_confirmation.py
.venv/bin/python experiments/lean_vector_variation_20261007/make_visuals.py
```

`results/width_model.npz` contains the actual pixel vectors and scalar functions;
`metrics.json` and `confirmation.json` contain measured errors; evaluation arrays
preserve the inputs and predictions. `vector_variation.png` shows the reference
image/vector, the vector of change, its learned coefficient, and an unseen
transition case. `viewer.html` is a self-contained typeset explanation.

The figure follows `skills/experiment-figures/SKILL.md`: image before vectors,
signed magnitudes defined, all pixels retained, an unseen difficult case shown,
a full-space control stated, common scales without clipping, and limits visible.
