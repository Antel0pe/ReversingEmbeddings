# How knob changes move a generated 1 in pixel space

This experiment uses only the five-knob `grey_ones.py` generator. It does not
measure handwritten MNIST 1s or a 2D/3D projection. A rendered image has 784
ink-coverage values between 0 and 1. All image distances below are Euclidean
distances across those 784 values.

## The question in local geometry

Let `q` contain the five knob settings, each normalized to run from 0 to 1 over
its allowed range, and let `F(q)` be its rendered image. For a small knob move
`change`, the image displacement is approximately `J(q) change`, where `J(q)`
has five columns of signed pixel-change rates. The local image distance obeys

`distance^2 approximately equals change-transpose × G(q) × change`,

where `G(q) = J(q)-transpose × J(q)`. The diagonal entries determine each
knob's local image-space speed. The off-diagonal entries measure how two
knob directions align. If the geometry were uniform and Euclidean in these
coordinates, the speeds and alignments would be constant across `q`, and
finite joint moves would be well predicted by the local linear map.

The ideal real-valued coverage rule is **continuous** on the allowed knob box:
its operations are bounded `tan`, `min`, `max`, clipping, multiplication, and
averaging. It is only piecewise differentiable where an ink edge crosses a
pixel or a subrow boundary. The actual implementation returns `float32`, so
rounding creates extremely small numerical jumps; strict continuity applies
to the underlying real-valued coverage formula, not to the quantized return
values. The measured pixel distance shrinks with the tested step sizes from
1.25% to 20% of a knob's range; see [step_scale.png](step_scale.png).

## What a 0.1 move means

Two interpretations of 0.1 must be kept separate. A literal +0.1 means
0.1 pixel for center/height/width and 0.1 degree for lean. A normalized +0.1
means **10% of each knob's full range**: 0.1 pixel for each center, 0.15 pixel
for height, 0.28 pixel for width, and 4.5 degrees for lean. These are medians
over 2,048 deterministic low-discrepancy starting states in the first 90% of
each normalized range, so every positive move stays inside the box:

| Knob | Image distance for raw +0.1 | Image distance for +10% range | 10th–90th percentile for +10% range |
| --- | ---: | ---: | ---: |
| Horizontal center | 0.598 | 0.598 | 0.567–0.615 |
| Vertical center | 0.269 | 0.269 | 0.205–0.361 |
| Height | 0.116 | 0.172 | 0.137–0.204 |
| Width | 0.300 | 0.833 | 0.792–0.856 |
| Lean | 0.062 | 2.588 | 2.376–2.960 |

The output distance is not numerically tied to 0.1. Even after normalizing
the knob ranges, equal coordinate steps give very unequal image steps. The
largest median response, lean, is about 15 times the smallest, height.

An independent 1,024-state scrambled sample gave corresponding normalized
medians `0.598, 0.270, 0.172, 0.833, 2.587`, confirming the scale ordering.

## Uniform speed is different from a fixed direction

For each knob, 128 contexts held the other four knobs fixed while that knob
traversed its full range in ten equal 10% steps. The figure
[along_knob_uniformity.png](along_knob_uniformity.png) shows each segment's
median image distance and its 10th–90th percentile across contexts. The rows
use separate vertical scales to expose changes along a knob; compare absolute
sizes in [step_scale.png](step_scale.png).

| Knob | Median variation in ten step lengths along one path (CV) | Median angle between consecutive signed change vectors | Ten-step path / direct endpoint distance |
| --- | ---: | ---: | ---: |
| Horizontal center | 0.4% | 15.5° | 1.18 |
| Vertical center | 2.4% | 4.5° | 1.12 |
| Height | 3.6% | 1.4° | 1.13 |
| Width | 0.4% | 20.4° | 1.30 |
| Lean | 8.6% | 59.4° | 2.85 |

Thus horizontal center and width can move at nearly constant *speed* along a
single path while their signed pixel direction still turns. Lean has both a
changing speed and strong turning. The path/chord ratio measures one knob path,
not a shortest geodesic.

Finite differences with centered ±0.1%-range perturbations gave five nonzero local singular
values at each of 256 sampled interior states. The smallest singular value
was between 1.09 and 2.16 pixel-distance units per normalized knob unit;
the median ratio of largest to smallest was 16.2. This supports five
independent local directions at those sampled states. It does not prove that
rank is five at every possible setting.

## Two knobs together

For all ten knob pairs at 1,024 starting states, the experiment compared the
true joint move with the sum of two separate moves, each +10% of its knob's
range. [joint_metric.png](joint_metric.png) shows both the alignment of those
separate signed pixel changes and the size of their interaction. The
interaction is

`F(both moved) - F(first only) - F(second only) + F(start)`.

Width plus lean had the largest median interaction under the displayed
normalization: **0.252** times the root-sum-square length of their separate
changes (10th–90th percentile 0.223–0.293). The separate width and lean
changes had median cosine only **0.057**, yet moving lean changed the width
response by a median **50.5°**. Near-median example images and signed pixel
maps are in [joint_response_example.png](joint_response_example.png). In that
example the interaction has L2 size **0.658**, and simply adding the separate
pixel changes would place **11 pixels** outside the valid `[0, 1]` coverage
range. An independent sample reproduced the width–lean interaction median at
**0.252**.

Moving **all five knobs +10% at once** had median true image distance **2.765**.
The sum of their five separate pixel changes had median length **2.902**, but
the median L2 difference between that sum and the true joint change was
**0.967**, or **34.1%** of the true joint-change length. The summed image had
at least one pixel outside `[0, 1]` for **99.9%** of the 2,048 starts (median
10 such pixels). The all-five measurement is summarized at the bottom of
[joint_metric.png](joint_metric.png).

## Practical reading

One normalized knob coordinate is an exact address for this controlled
generator, but it is not a unit of image-space distance. For an approximately
equal small image effect, a knob's step should be chosen using its *current*
local speed. For multiple knobs, the full local metric `G(q)` also includes
directional alignment. At finite step sizes, re-render and solve for the
desired pixel distance: the response directions themselves move and the joint
image need not equal a sum of separate pixel changes.

Reproduce with `python make_generated_knob_metric_experiment.py`. Detailed
measurements and sampling metadata are in [results.json](results.json).
