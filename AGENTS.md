# Repository instructions

- Never include the user's full local filesystem path in a public or shareable artifact. Use repo-relative paths or sanitized placeholders.
- Never use npm. Use pnpm or bun when a JavaScript package command is needed.
- When making an explanatory experiment figure, read `skills/experiment-figures/SKILL.md` and apply its understanding check before delivery.
- When the user gives you a research paper, read `skills/read-paper/SKILL.md`. Notes live one-file-per-paper in `literature/`, indexed by `literature/README.md`; never mirror a paper's text into the repo.

## Context shortcut: generated 1s / five-knob manifold

When the user says **generated 1s**, **generated-one manifold**, or **five knobs**,
start with `grey_ones.py`: the controlled synthetic family of straight "1"
strokes. Real MNIST images (`OnesManifold.ipynb`) and sentence-embedding inversion
(the root `README.md`) are separate contexts.

An image is a 28×28 array, a point in 784-dimensional pixel space. The stroke is
a parallelogram with horizontal top/bottom and constant horizontal width.
Values represent ink coverage: 0 = no ink, 1 = full ink. Rendering uses 16
subrows per pixel row and exact horizontal overlap, returning `float32`.

Parameter order is always `(cx, cy, height, width, lean)`:

| Knob | Meaning | Default allowed range |
| --- | --- | --- |
| `cx` | Horizontal center; increases rightward | 14–15 px |
| `cy` | Vertical center; increases downward | 14–15 px |
| `height` | Vertical stroke extent | 19–20.5 px |
| `width` | Horizontal stroke width | 1.8–4.6 px |
| `lean` | Angle; positive tilts the top rightward | −10–35 degrees |

`grey_ones.KNOBS` and `grey_ones.RANGES` are the source of truth. These are
MNIST-informed defaults. `render` does not enforce the range box.

### Root 1

Unless an experiment explicitly states another reference, **root 1** means
`(cx, cy, height, width, lean) = (14.0, 14.0, 19.75, 3.2, 0.0)`:
geometrically centered on the 28-by-28 canvas, upright, with height and width
at the midpoints of their default ranges. Its image is
`render([14.0, 14.0, 19.75, 3.2, 0.0])[0]`.

The canvas spans coordinates 0 through 28, so its geometric center is `(14, 14)`.
Older experiments using `(14.5, 14.5)` keep their explicitly stated references.
The root is on the lower default bounds for `cx` and `cy`; a neighborhood that
moves left/up as well as right/down must explicitly extend those sampling bounds.

### Rendering an image

Generate an image from five parameters (Python from the repo root; NumPy and
SciPy are available in the existing `.venv`):

```python
from grey_ones import render

def generated_one(cx, cy, height, width, lean):
    return render([cx, cy, height, width, lean])[0]

img = generated_one(14.5, 14.5, 19.75, 3.2, 12.5)  # (28, 28), values in [0, 1]
# Batch: render([[cx, cy, height, width, lean], ...])  -> (n, 28, 28)
# Display with matplotlib: plt.imshow(img, cmap="gray_r", vmin=0, vmax=1)
```

### Where to look next (only as needed)

- **Generator / inverse:** `grey_ones.py`; `invert_grey.py` recovers settings;
  `GENERATED_ONES_IMPLICIT_EQUATION.md` explains image validity.
- **Interactive 3D atlas:** `figures/generated_one_3d.html`, built by
  `make_generated_one_3d.py`; `GENERATED_ONE_NEIGHBOR_EMBEDDINGS.md` explains
  the layouts. These are projections of the five-parameter family.
- **Equal-distance neighbors:** `make_generated_one_distance_shells.py` and
  `figures/generated_one_distance_shells/README.md`; raw settings are in
  that directory's `sampled_radius_4p5_shells.csv`.
- **Knob speeds / interactions:** `make_generated_knob_metric_experiment.py`
  and `figures/generated_knob_metric/README.md`. `KNOB_PATH_EXPERIMENT.md`
  discusses composition; its numerical results used a JavaScript mirror,
  not a run of the Python renderer.
- **Linear / learned coordinates:** `analyze_generated_one_direction_pca.py`
  and `figures/generated_one_distance_shells/direction_pca/README.md`;
  `compare_generated_one_autoencoder_knobs.py` and
  `figures/generated_one_autoencoder_knobs/README.md` probe learned latents.
- **Root-local linear bases:** `analyze_root_one_local_svd.py` and
  `figures/root_one_local_svd/README.md`; uncentered SVD of equal-distance
  shells around the canonical root, with held-out reconstruction checks.
  `render(..., dtype=np.float64)` retains the same coverage calculation before
  default float32 storage, for numerical-rank diagnostics.
- **Local bases across starts:** `analyze_generated_one_local_bases_across_starts.py`
  and `figures/generated_one_local_bases_across_starts/README.md` compare ranks,
  vector matches, space transfer, and held-out precision across 17 starting 1s.
  The 16-subrow renderer is piecewise smooth: at subrow/clipping boundaries,
  five leading vectors need not capture all first-order changes as radius shrinks.
- **Full bases, scale extrapolation, and family matching:**
  `analyze_generated_one_basis_followup.py` and
  `figures/generated_one_basis_followup/README.md` show all nine/ten vectors,
  sampling-induced mixtures, observable-based associations, and radius power laws.
- **Mathematical knob families and extra directions:**
  `analyze_generated_one_manifold_transport.py` and
  `figures/generated_one_manifold_transport/README.md` use analytic derivatives
  to check physical-direction overlap, the root's four mixed normal directions,
  local invariants, metric transport, coordinate ambiguity, and an exact
  piecewise-quadratic five-coordinate decoder. Known controls are supplied.
- **Smaller-radius separation:** `analyze_generated_one_scale_separation.py`
  and `figures/generated_one_scale_separation/README.md` check radii 0.0001
  and 0.00001 at five selected starts, normalized spectral slopes, and
  float32 storage limits. The spectral test estimates first-order dimension,
  not uniquely labeled physical knobs.
- **Nearest-direction continuation:** `analyze_generated_one_nearest_direction.py`
  and `figures/generated_one_nearest_direction/README.md` test matching a
  direction to the nearest direction at a neighboring image, including moves
  along estimated SVD vectors and changes of other knobs. Small sampled steps
  succeed; larger steps and renderer boundaries can switch the association.
  Known controls generate valid moves and evaluate categories; the matcher
  itself uses only absolute pixel cosine. This does not identify unique knobs.
- **Why the creases exist, and what grid fineness does:**
  `analyze_generated_one_crease_origin.py` and
  `figures/generated_one_crease_origin/README.md` show every crease is one of
  the renderer's `clip/min/max` decisions flipping, predicted in closed form and
  matched to within 0.37 of a sweep step. The worst crease sits at a pixel that
  stays between 0.9926 and 0.9990, so creases are not pixels switching on.
  Refining the grid adds creases without shrinking them and makes path speed
  diverge as one over the square root of the pixel pitch. `smooth_ones.py`
  renders the same five knobs with no decisions at all (zero creases, at a cost
  of about 15% relative RMS against coverage); `coverage_grids.py` renders
  `grey_ones`' coverage maths at any grid fineness.
- **The five knobs as reweightings of two gradients:**
  `analyze_generated_one_gradient_fields.py` and
  `figures/generated_one_gradient_fields/README.md` express every knob direction
  in the six affine fields built from an image's own `d/dx` and `d/dy`. Residual
  tracks the measuring nudge squared to within 1%, so it is measurement error,
  not a missing ingredient. The unused sixth field is vertical shear, which is
  why there are five knobs. Borrowed-field and dropped-field controls included.
- **The vocabulary of motions, discovered from images alone:**
  `analyze_generated_one_motion_discovery.py` and
  `figures/generated_one_motion_discovery/README.md` withhold the affine
  hypothesis: forward selection over twelve generic polynomial-warp motions picks
  the five affine ones first and then plateaus (0.667% leftover at five, 0.629%
  at all twelve), with leftover falling in direct proportion to the step, so the
  five-motion model is exact in the limit. Knob values build the dataset and are
  never fitted or scored. The method does require a dense dataset -- a scattered
  sample's nearest neighbours leave 18.192%. On coverage output the same five are
  still selected first, but a 23.0% floor appears; running the identical
  blur-then-`np.gradient` pipeline on crease-free `smooth_ones` leaves 16.7%, so
  the floor is mostly gradient tooling rather than the creases. Borrowed-field,
  dropped-motion, random-picture and no-density controls included.
- **Real MNIST 1s: the search for their knobs, and what is actually there:**
  `analyze_mnist_one_knob_search.py` on `warp_charts.py` (eleven parts, runnable
  separately), with `figures/mnist_one_knob_search/README.md`. Validated on the
  generated family first: five coordinates decode held-out images to **1.06%**
  (224^2 template, 40 rounds), saturation confirms five, and a closed-form
  inversion recovers the generator's OWN knobs to ~0.001-0.009 units -- about
  0.1% of each range -- once **five** labelled examples fix a five-number gauge
  (five is exactly enough; the gauge has five degrees of freedom, and a
  shuffled-chart control fails completely). On MNIST there is no small knob set:
  27.65% at five coordinates and resolution/rounds move it 0.03 points, so there
  the error is the family, not the fit. Geometry never flattens (to 14.98% at
  ninety warp numbers) and the biggest single gain is affine->quadratic, i.e.
  bending, not serifs. TwoNN and Levina-Bickel say twelve or more but the
  estimate CLIMBS with sample size (9.17 to 12.16 from n=500 to 4000) while the
  generated family's converges on 4.57, so twelve is an unconverged lower bound.
  Do NOT ask where MNIST geometry saturates: PCA catches the warp model at 72
  parameters and beats it at 90. Use parsimony instead -- five warp coordinates
  match ~70 PCA components on the generated family (~14x) and ~17 on MNIST
  (~3.4x). The generation test cuts the other way and is the sharpest
  discriminator: NEW warp coordinates decode to images indistinguishable from
  real MNIST 1s, matching real precision AND recall at twenty coordinates
  (0.302/0.304 vs 0.299/0.291), while PCA samples are not 1s (overlap 0.74, 9%
  negative ink) and get worse with more coordinates. So MNIST's warp
  coordinates are VALID but INCOMPLETE knobs. Interpreted through the same
  closed form, they ARE slant (57x better than chance), width (5.4x) and
  horizontal position (4.9x), but barely determine height (1.4x) or vertical
  position (1.7x) -- a 1 is a long uniform stroke, so vertical placement and
  length barely move the pixels. That independently reproduces
  `figures/generated_one_warp_space`'s finding that a pixel-fitted warp is blind
  to `cy`. Best structured MNIST model: quadratic warp plus cubic gain field, 22
  coordinates, 18.3%. FIVE traps the experiment records: MNIST's nearest
  neighbour is 24.8% of an image away (9.1% for the generated family), so the
  linearised motion-discovery method cannot be used; the decoder is capped by
  TEMPLATE resolution, not the model, which is why it is super-resolved by
  splatting; `fit_chart` at 7 rounds understates a chart twofold (2.93% vs 1.50%
  at 40), so the default is now 25; whole-canvas fold counts are misleading --
  MNIST folds up to a third of the time somewhere but only 1-2.7% ON THE INK, so
  use `jacobian_determinants(..., ink=image)`; and a gain field must be compared
  at MATCHED template rounds or the extra refinement is credited to the gain
  (which made a worthless one-term darkness knob look valuable). The ten-term
  gain is worth +4.28 points over a five-coordinate warp but only +1.34 over a
  twelve-coordinate one, and is slightly NEGATIVE on the generated family, so
  the genuinely appearance-only part -- ink density along the stroke, not overall
  darkness -- is about a point.
- **Knobs from linear measurements, in closed form:**
  `analyze_generated_one_moment_chart.py` and
  `figures/generated_one_moment_chart/README.md` recover all five knobs from six
  fixed linear projections of the pixels with nothing fitted; total ink equals
  `width * height` to 4e-14, and the remaining error falls as the square of the
  pixel pitch. A warped single template decodes held-out images to 13.65%
  against a rank-5 linear ceiling of 44.49%, and the true-knob control shows
  almost all of that is resampling.
- **Warp space without assuming affine:**
  `figures/generated_one_warp_space/README.md` (salvaged from an earlier
  session) fits a free degree-2 polynomial warp and finds the family is a
  5-dimensional flat subspace there, machine-zero past the fifth component,
  while pixel space still leaves 8% at 50 directions. It also locates the
  failure: the vertical warp coefficients are not equally visible, so a
  pixel-fitted warp is blind to `cy` (median error 0.467).
- **One pixel at a time: humps and handoff:**
  `analyze_generated_one_pixel_humps.py` and
  `figures/generated_one_pixel_humps/README.md` describe a lean sweep in the
  pixel basis. Every touched pixel has exactly one active episode with one
  rise; the active span in `tan(lean)` is `(width + 1) / |row - cy|` with
  r^2 0.9919, against a `cx` control whose spans are identical across rows to
  4e-16. At a typical lean only 56 of 784 pixels move, 56% of the motion sits
  in ten of them, and the move direction at leans 10 degrees apart is
  orthogonal. Hump structure survives `smooth_ones`, so it is geometry, not
  the grid. Faithful bookkeeping, but not a route to the knobs.

- **Direction-change history and count consistency:**
  `analyze_generated_one_direction_history.py` and
  `figures/generated_one_direction_history/README.md` test eight-hop tracking,
  a two-step direction predictor, a four-image bilinear decoder, and the
  order dependence of equal-image-distance ticks. Ordinary SVD fields also
  have measured nonzero Lie brackets outside a direction pair at smooth
  starts (`check_generated_one_direction_commutators.py`); matching labels
  alone does not turn accumulated counts into independent coordinates.

### Geometry conventions

- Image distance is Euclidean over all 784 coverage values. A seed-to-neighbor
  direction is `(render(p) - render(p0)).ravel()`. Raw steps use pixels/degrees;
  normalized steps use fractions of each knob's allowed range.
- Five parameters plus `render` give an exact nonlinear construction. Joint
  parameter moves reach the same endpoint in either order, but intermediate
  pixel directions depend on state. The user's goal is coordinates that
  reconstruct the whole family and joint changes: require a decoder and
  held-out joint/interpolation tests, not just PCA variance or a pleasing plot.
