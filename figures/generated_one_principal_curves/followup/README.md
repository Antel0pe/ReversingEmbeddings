# What the five curves actually are

The target is the known renderer family, not a prediction problem. The new fits use every point in the covering cloud: 10,240 previous uniform points, a 3,125-point five-level-per-knob lattice, and 1,024 near-face points. No train/validation/test split is used. The continuous parameter box has infinitely many points; an exact construction uses the renderer equations rather than enumerating points.

## Exact definition of the original fitted curves

Each curve stores 65 vectors `C[k, j]`, each containing all 784 pixels. For curve k, form node positions by cumulative segment length, normalized to [0, 1]:

```text
length[j] = Euclidean distance between node[j+1] and node[j]
position[j] = sum(lengths before j) / sum(all segment lengths)
alpha = (t - position[j]) / (position[j+1] - position[j])
curve_k(t) = (1 - alpha) * node[j] + alpha * node[j+1]
```

Here j is the interval containing t. This is an exact piecewise-linear mathematical equation for the stored curve, not merely a list of example source images. `original_curve_nodes.csv` contains every node and its path position. The original mean is in `../model_and_samples.npz`. Curve 1 panels show `mean + curve_1(t)`; curves 2–5 are residual contributions and can have positive or negative coverage.

The five-coordinate decoder is:

```text
predicted image = mean + curve_1(t1) + curve_2(t2) + curve_3(t3) + curve_4(t4) + curve_5(t5)
```

The original encoder chooses t1 by projecting image-minus-mean, subtracts that curve contribution, chooses t2 from the residual, and continues. There are five curves, not five points: the model has 325 stored pixel vectors.

## Where the starting line comes from

No pair of images is selected to define PC1. All sample images determine the mean and covariance. The unit eigenvector of the largest covariance eigenvalue is the PC1 direction. The line is `mean + score * PC1_direction`; its initial endpoint scores are the minimum and maximum scores of the source samples. Those endpoint vectors are generally synthesized pixel vectors, not actual generated images. The sign of the direction can flip without changing the line.

For every current curve segment from node a to node b, a sample x projects to:

```text
fraction = clip(dot(x-a, b-a) / dot(b-a, b-a), 0, 1)
candidate = a + fraction * (b-a)
choose the candidate with the smallest squared pixel distance over all segments
```

All images are projected in the same iteration; none is selected as an image that the curve must pass through. A least-squares fit then uses all assigned images, with a second-difference penalty on neighboring nodes. It is a simultaneous matrix solve, not a left-to-right sequence of bends through chosen images. The assigned curve coordinate supplies the ordering. The curve is then reparameterized by arc length and the projection/smoothing process repeats.

The smoothing fit minimizes:

```text
sum over images: squared distance(image, interpolated curve at its assigned t)
  + smoothing strength * sum over interior nodes:
      squared norm(node[j+1] - 2*node[j] + node[j-1])
```

Fixed samples, initial line, smoothing strength, node count, and stopping rule give deterministic results, apart from floating-point differences and irrelevant direction-sign conventions. A new sampling seed changes the finite cloud; a different starting curve can lead to a different local solution. These are separate effects.

## The original missing 2.7%

The original residual is 2.7235% of total squared image variation. Its RMS pixel error is 0.02708 and its RMS image L2 error is 0.7583.

| Location | Share of the residual squared error |
| --- | ---: |
| top_bottom_bands | 43.36% |
| side_edge_bands | 53.04% |
| remaining_pixels | 3.60% |

Top/bottom bands include the endpoint rows and one neighboring row on each side. Side-edge bands are within 1.5 pixels of the geometric edges, excluding the endpoint bands. The residual contains blurred/misplaced edge coverage and endpoint corrections, rather than one distinct missing knob.

A univariate spline of any single knob predicts only 1.50–2.88% of this residual energy. These individual prediction fractions overlap and are not a causal partition. A local tangent projection attributes about 32.88% of the error to the first-order span of the five physical knob derivatives and 67.12% to its orthogonal complement; two finite-difference steps agree within 0.001 percentage points. This is a local linear diagnostic, not proof that all normal error is globally off the manifold.

Keeping the original curve nodes fixed and improving all five coordinate choices cyclically raises the score to 97.5253%, removing 9.14% of the original residual energy. Thus the one-pass coordinate search explains a small part of the original loss. The rest reflects the fitted curve shapes and additive decoder approximation; it is not evidence for a sixth intrinsic knob.

## All-point fit and initialization comparison

| Curves retained | Pooled-cloud fitted score |
| --- | ---: |
| 1 | 90.006% |
| 2 | 94.193% |
| 3 | 95.888% |
| 4 | 96.796% |
| 5 | 97.430% |

Refining the five coordinate choices on the all-point curves, without moving any curve node, raises the pooled-cloud score to 97.6622%.

These percentages use the mean and equal weights of the pooled covering cloud. Its inclusion of lattice/boundary points changes the measure, so it must not be treated as the same score distribution as the original uniform sample. The new fits use 65 nodes, smoothing penalty 1, and up to 45 iterations. They retain the best whole-cloud iterate, not a validation-selected iterate.

| First-curve initialization | Pooled-cloud score | Correlation with lean | Correlation with width |
| --- | ---: | ---: | ---: |
| Exact lean path | 90.201% | 0.988 | -0.003 |
| Exact width path | 90.003% | -0.290 | 0.560 |

Starting with an exact knob path does not keep the unconstrained curve aligned to that knob. Projection and smoothing can move it toward a different summary of the cloud.

## Aligning curves to the known knobs

Let F be the renderer and let `(cx, cy, height, width, lean)` be the current state. Exact physical coordinate curves are:

```text
horizontal-center curve(t) = F(t, cy, height, width, lean)
vertical-center curve(t)   = F(cx, t, height, width, lean)
height curve(t)            = F(cx, cy, t, width, lean)
width curve(t)             = F(cx, cy, height, t, lean)
lean curve(t)              = F(cx, cy, height, width, t)
```

Each is a family indexed by the other four fixed settings. They are known coordinate curves of the renderer, not unconstrained principal curves inferred from pixels.

| Known-knob construction | Pooled-cloud variation captured |
| --- | ---: | ---: |
| Add five exact midpoint-anchor axis displacements | 47.433% |
| Best joint least-squares sum of five knob-only curves, 33 nodes each | 80.335% |
| Best joint least-squares sum of five knob-only curves, 65 nodes each | 80.464% |
| Best joint least-squares sum of five knob-only curves, 129 nodes each | 80.735% |
| Follow each physical knob curve at the updated current state | 100.000% |

The optimal additive models use the actual known knob values as coordinates, not nearest-curve inference. Their 33/65/129-node comparison isolates the limitation of independent knob-only contributions from crude sampling of the curve. The fixed-midpoint model uses exact renderer values at the requested knob settings, so its error is not polyline-interpolation error.

For a width increase of 1 px and a lean increase of 15° from the midpoint state, the mixed interaction has pixel L2 norm 3.4601. The joint image differs from the sum of the two separate image changes. Therefore no sum of five functions depending only on individual physical knob values can represent this renderer exactly: such a sum would have zero mixed finite differences.

The exact construction updates one knob, uses that updated state to define the next curve, and repeats. Differences telescope to the endpoint renderer image. On all 14,389 covering points, forward and reverse knob orders agree with the source image bit for bit. The endpoint is order-independent even though intermediate curves depend on state. This exactness comes from the known renderer; it is not a claim that a learned principal-curve algorithm discovered the full manifold.

## Artifacts and reproduction

```sh
.venv/bin/python analyze_generated_one_curve_followup.py
.venv/bin/python refine_generated_one_curve_coordinates.py
.venv/bin/python make_generated_one_curve_followup.py
```

- `literal_five_curves.png`: the original five literal curve contributions.
- `missing_variation.png`: exact source, original reconstruction, signed missing coverage, and whole-sample controls.
- `knob_curve_interaction.png`: one direct counterexample to adding independent knob changes.
- `original_mean_pixels.csv`: all 784 values of the original mean image.
- `original_curve_nodes.csv`: every original curve node with its exact stored pixel values and normalized arc coordinate.
- `results.json`, `coordinate_refinement.json`, and numerical arrays preserve all scores and definitions.
- The parent `index.html` includes sliders for literal fitted curve nodes and exact physical knob curves at selectable states.
