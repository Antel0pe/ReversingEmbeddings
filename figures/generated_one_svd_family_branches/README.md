# One-root SVD family experiment

One fixed matcher followed five approximately physical SVD directions from the
canonical root. Lean, width, and horizontal position followed their intended
control through all nine executed steps. Vertical position switched with height;
height repeatedly reversed. This is a small diagnostic, not global knob discovery.

## Figures

- [Found paths](01_found_paths.png): four images per family, root-distance curves,
  and angles between consecutive pixel displacements. Hops 0, 1, 2, and 6 reveal
  the height path's backtracking.
- [Pure-control comparison](02_true_control_paths.png): true single-control image
  progressions, with found and reference curves overlaid.
- [Further continuation](03_unseen_continuation.png): three further steps using
  the unchanged rule. New local neighborhoods remain available at each step.

White means zero ink coverage; black means full coverage. All distances and
angles use all 784 pixels. Images show complete arrays, without a projection.
The number below each thumbnail is the named control's current value; other
controls may also drift on found paths.

## Fixed design

- Root: `(14, 14, 19.75, 3.2, 0)` in `(cx, cy, height, width, lean)` order.
- Neighborhood: 256 images at Euclidean pixel distance **0.0001**. Ordinary
  uncentered SVD of displacements from the current image; equal weights and
  float64 coverage storage. Keep five leading vectors for matching. The same
  seeded Sobol parameter-ray directions are reused at every state and scaled by
  default knob ranges. Sampling is therefore controlled, not distribution free.
- Structure: all five first branches and all 25 second branches, then extend
  the five repeated-family paths to six steps. This gives 51 discovery states
  and 50 edges. Three more steps per path add 15 new states: **66 states total**.
- Motion/probe scale: **0.3** image-distance units to first order. This was fixed
  before seeing the outcomes, to make a few steps visible and permit crossings
  of renderer formula regions. It was not fitted to a desired consistency score.
  Neighborhood estimation and travel are deliberately different scales.
- At each state, the simulator maps a requested SVD direction to its nearest
  valid parameter ray using the known Jacobian's least-squares inverse, then
  calibrates initial pixel speed to one. The next state is the ray at 0.3; its
  actual finite chord distance is measured, not forced to 0.3.
- Probe each candidate at ray amounts `-0.6, -0.3, 0, 0.3, 0.6`. Record ten
  pairwise pixel distances and three successive turning angles. Probes follow
  fixed parameter rays rather than integrating an updated SVD field.
- Resolve each candidate's sign by its dot product with the parent direction.
  Compare parent and candidate with the equal average of three costs:
  direction angle divided by 90 degrees; RMS pairwise-distance change divided
  by the corresponding straight-path tick separation; and RMS turning-angle
  change divided by 90 degrees. A five-by-five assignment selects all five
  candidates jointly. No fitting of alternative weights or matchers was done.
- The matcher uses pixel frames, distances, and angles, without physical
  control labels. The generator and Jacobian provide sampling and valid motion.
  This is not an end-to-end method operating on an unordered image cloud alone.
- Physical labels are attached only after graph construction and continuation.
  Reference paths move only the named control, with the same initial-pixel-speed
  calibration at each step. Calibration is not a claim of independent knob counts.
- Default parameter bounds are not enforced: the root lies on position bounds,
  and arbitrary root signs can move left/up. All states and probes have positive
  width/height and remain geometrically inside the canvas. The pure-height
  reference also extends beyond the default height range.

## Results and failure definitions

At the root, the leading axes map one-to-one to lean, width, horizontal position,
vertical position, and height, with cosines **0.999908, 0.999509, 0.999569,
0.999312, and 0.988154**. Vectors 6–9 are present but much smaller: the sixth
singular value is about 0.00000274 times the first. Tail energy after five is
8.92e-12 at the root and no more than 1.10e-11 over observed states.

For evaluation, associate each fitted frame one-to-one with the true normalized
derivatives using maximum absolute cosine. A label error means the selected
axis's assigned physical label differs from that family's root label. This is
an approximate category check, not a proof that an SVD axis is a pure knob.
Sign errors are counted separately relative to each root's chosen orientation.

| Measurement | Discovery | Further continuation |
| --- | ---: | ---: |
| Correct field labels, ignoring sign | 232/250 (92.8%) | 66/75 (88.0%) |
| Label errors | 18/250 (7.2%) | 9/75 (12.0%) |
| Executed steps using expected family and sign | 43/50 (86.0%) | 13/15 (86.7%) |

These comparisons are dependent observations in one deterministic tree, not
independent trials estimating a population-wide success probability.

Of 27 label errors, **22 are height/vertical-position swaps**. The vertical
path alternates between translating down and shortening. The height path
remains height-like in absolute cosine but reverses four of its nine executed
steps: height oscillates around 19.75–20.02, while the pure reference reaches
22.07. The graph of nearly 180-degree turns exposes that reversal.

For the displayed paths, correct executed family-and-sign counts over nine
steps are: lean 9/9, width 9/9, horizontal position 9/9, vertical position 5/9,
and height 5/9. At the final width and horizontal-position states, some next
field labels become wrong; the experiment stops before executing another step.

The weak cases occur as pixel-response support moves across renderer regions.
For example, actual height directions can turn by 90 degrees when the active
cap pixels change. A previous-vector dot-product sign rule can then select the
opposite physical orientation. This sign policy is a concrete limitation of
the chosen rule, not an impossibility claim about distance/angle methods.

The maximum least-squares tangent-projection residual is 3.43e-6. Five leading
directions still capture the local change spaces well. Accurate local spaces
do not automatically provide consistent labels or signs across finite steps.

## What the distance evidence says

Over the first six steps, the true width and horizontal-position paths have
**identical all-pairs distance matrices in the executed arithmetic**. Both are
straight in their current pixel-formula regions and use the same speed scale.
Thus different actual knobs can share the same distance pattern in this data,
without deliberately constructing an adversarial path.

Found-versus-true all-pairs distance errors over six steps are approximately
0.025% for lean, 0.013% for width, 0.007% for horizontal position, 25.16% for
vertical position, and 82.20% for height. These are curve-shape errors, not image
reconstruction errors. Exact endpoint pixel errors are also saved separately.

Equal numerical weights do not make the three components equally informative:
mean candidate spreads in discovery are 0.857 for direction angle, 0.112 for
pairwise-distance mismatch, and 0.454 for turning-angle mismatch. This run
does not isolate the benefit of the distance component; no second method was
run. The rule assumes neighboring probe shapes change modestly, which can fail
at renderer transitions. Large probe spans can also cross multiple regions.

## Why mixtures remain tractable

Any direction inside the estimated five-dimensional space has five coefficients.
For orthonormal SVD axes, those coefficients are just dot products. Five
potentially nonperpendicular families can be represented by a general five-by-five
coefficient matrix. Discovering a useful matrix still needs a selection rule.

As an evaluation-only example at this root, the normalized true height direction
has coefficients approximately `(0.000686, 0.152881, 0.005930, 0.011955, 0.988154)`
on root axes F1 through F5. Its reconstruction residual is 3.14e-6. This shows
that a nearly pure returned axis can be improved with small contributions from
others. Those known-control coefficients did not enter tracking in this run.

## Reproduce and inspect

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/svd_family_branch_experiment.py
```

- [Source](../../experiments/svd_family_branch_experiment.py)
- [Measurements and design](results.json)
- [Every association](associations.csv)
- `states_and_paths.npz`: settings, images, bases, probes, match-cost components,
  edges, five paths, and pure-control references.

The three final figures were inspected against the repository understanding
check: baseline, image meaning, scales, sign, root-distance and turning-angle
definitions, comparison, and sampling limits are explicit. Source settings and
measured arrays are retained for replotting without rerunning discovery.
