# Repository instructions

- Never include the user's full local filesystem path in a public or shareable artifact. Use repo-relative paths or sanitized placeholders.
- Never use npm. Use pnpm or bun when a JavaScript package command is needed.
- When making an explanatory experiment figure, read `skills/experiment-figures/SKILL.md` and apply its understanding check before delivery.
- When reporting experiment results back to the user (more than one experiment, a what-if chain, or they say they did not understand), read `skills/experiment-recap-doc/SKILL.md` and deliver a recap doc.
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
- **Adding one-knob lines, and how many lines it takes:**
  `analyze_generated_one_line_interpolation.py` and
  `figures/generated_one_line_interpolation/README.md` take the five lines
  `render(root + e_k) - render(root)` and add them. At +1 on each dial two lines
  miss the joint change by 11-61% of the move (all five at once, 63%), but the
  miss is exactly second order (log-log slope 2.001), so the lines are an exact
  LOCAL answer: 1.5% of the move at a twentieth of a unit. Globally they are
  useless -- a typical setting is 80.7% of its own distance-from-root away from
  the 5-line flat, the best five directions still leave 33.1%, and products of
  the dials converge badly (6 lines 40.0%, 126 lines 15.8% held out) against a
  PCA ceiling of 118 pictures for 1%. Patches of five fresh lines hold to 1%
  only within +-1.2% of each range, so tiling the box costs ~1.3e8 patches.
  `lean` is the whole problem (one line keeps 26% of its own sweep; 13 lines for
  1%, against 2-3 for the others). The payoff: in WARP space, with
  `height * tan(lean)` as the fifth dial, the same five lines add EXACTLY
  (residual 5e-13%), and the 15.7% cost of getting back to pixels is the 28x28
  template, falling to 0.11% at pitch 1/16 px. The root is an unusually flat
  spot (40x better than other centres), so controls use four other centres.
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

- **Making nearby pixels genuinely nearby:**
  `analyze_pixel_metric_locality.py` and
  `figures/pixel_metric_locality/README.md` test whether any representation can
  separate perpendicular-because-adjacent from perpendicular-because-unrelated.
  Every linear candidate is one frequency reweighting w(mu) in the DCT basis.
  The pixel metric's sense of space is a STEP (cos 1 at d=0, 0 at every d>=1)
  and its separation score is NEGATIVE (-0.149): it ranks a wrong answer above
  the gesture continued. Blur 1-2 px scores +0.56, best when the connection
  range matches the stroke width. The -1/2 speed law reproduces at -0.500, and
  the 5-knob plane's twist is FLAT in pixel space (55.9, 56.5, 53.4 degrees)
  but halves per halving under every re-encoding -- a tangent plane exists in
  the re-encoded space. The 28x28 coverage render is already locally linear
  below a normalised step of ~0.06, so the repo has been on F_s, not F.
  Optimal transport is EXACT (a cx step of h costs exactly h, nothing blurred)
  but only down to the pixel pitch; below it the 1/sqrt(h) law returns at every
  fineness, so the sub-pixel blow-up is the SAMPLING GRID, not the edges.
  Sharpen, white-noise, scramble, wrong-knob and wrong-place controls included.
  Traps: an unbalanced sign scramble fakes +0.896 through its mean offset;
  Sinkhorn under ~4000 iterations fails in a way that looks like a result; a
  binary grid returns 90 degrees when a knob step flips no pixel. This measures
  RULERS, not coordinates -- no decoder was run.

- **The turning law: knob sweeps are polygons:**
  `analyze_turning_pattern.py` (eight parts) and
  `figures/turning_pattern/README.md`. Sweeping cx, cy, height or width is a
  POLYGON in pixel space: 99.8% of tiny steps turn by under 1 degree (measured
  0.0000015 deg between corners) and all the turning happens at a few corners.
  Corner POSITIONS are predicted exactly -- a corner is a stroke edge crossing a
  whole-number pixel boundary: width every 2.000, cx at .4 and .6 of each pixel,
  matched to within one sweep step (0.0018). Corner SIZE is closed form:
  if a step spreads over N columns and M hand over, turn = arccos((N-M)/N).
  Width N=2 M=2 -> 90.0 deg exactly; cx N=2 M=1 -> 60.0 deg exactly; predicted
  equals measured on every corner. Lean is the SAME mechanism with the corners
  packed too densely to resolve (each row crosses at its own moment): it looks
  smooth over its range but zoomed 900x it is spiky, with exactly 0.0000 deg
  between corners, against a soft-edge control whose ratio is 1.0x. Lean's
  total turning goes as height^1.42 (fitted to 3.2%) -- a height^2 crossing
  count was predicted and is WRONG. tan(lean), not degrees, is lean's natural
  clock: chunk spread falls 77% -> 24%. `smooth_ones` control removes every
  corner (biggest turn 0.06-0.55 deg vs 36-75 deg). Turning is steady PER
  CORNER, not per unit knob (chunk spread 250-500%), so the clock is a corner
  count, not an angle.

- **Knob frames, stored and priced:** `knob_frames.py` is the shared API
  (`jacobian`, `metric`, `frame`, `speeds`, `knob_cosines`), with
  `build_generated_one_knob_atlas.py` and
  `figures/generated_one_knob_atlas/README.md`. A dense locally-linear 5D
  lattice would need 86 billion points (5.1 TB metric, 1.3 PB frames), so the
  stored `atlas.npz` is slices: a 9^5 metric lattice, 81-point single-knob
  sweeps with all five directions under both renderers, and ten 21x21 pair
  slices, 6.98 MB. Interpolating it recovers the metric to 1.7% but a lean
  frame only to cos 0.92, because one lattice step turns the lean direction to
  cos 0.254. Worst degeneracy is cx-cy at |cos| 0.955 (thin and leaned);
  condition number 66-135. Sweeping cx by one pixel makes the cx direction
  EXACTLY orthogonal to itself (columns 12,15 -> 13,16), which is 0.38 on
  `smooth_ones`.

- **Direction-change history and count consistency:**
  `analyze_generated_one_direction_history.py` and
  `figures/generated_one_direction_history/README.md` test eight-hop tracking,
  a two-step direction predictor, a four-image bilinear decoder, and the
  order dependence of equal-image-distance ticks. Ordinary SVD fields also
  have measured nonzero Lie brackets outside a direction pair at smooth
  starts (`check_generated_one_direction_commutators.py`); matching labels
  alone does not turn accumulated counts into independent coordinates.

- **Flat 5-D plane as an address:** `analyze_generated_one_flat_plane.py` and
  `figures/generated_one_flat_plane/README.md` test whether projecting onto a flat
  plane is one-to-one, with no decoder. The plane at one 1 fails: it watches only
  44-58 pixels, and 10-18% of 1s have a clearly different look-alike with the same
  five numbers. Five weighted pixel sums (ink, x, y, y*y, x*y) pass and have a closed
  form back to the knobs. A search from images alone passed in one of six tries, so
  a good plane exists but is not yet found reliably.

- **Toy crumpled sheets, projection vs. walking along the sheet:**
  `analyze_crumpled_sheet_unrolling.py` and
  `figures/crumpled_sheet_unrolling/README.md`. Projecting onto the plane at a
  random point overlaps 26-87% of any sheet that folds over (best of 3,000 tilts
  still 11-70%). Walking-distance flattening (Isomap) unrolls bent-not-stretched
  paper to the original square (median 1.4% of the side off) with no folds, and
  fails (85%) when short steps jump between layers. More points help partly. The
  twist-made crumples were stretched unevenly: their layouts fold over 1-2.4% of
  the paper, which a dot-neighbour overlap test misses. Count folds with flipped
  well-shaped interior triangles (`analyze_crumpled_sheet_followup.py`); edge
  slivers fake folds.

- **Holes and folds in MNIST 1s (persistent homology):**
  `analyze_mnist_one_holes_and_folds.py`, `check_shell_test_touch.py` and
  `figures/mnist_one_holes_and_folds/README.md`. Loops (ripser, H1) find a toy curl
  that rests on itself at 8.5x vs chance <= 2.8x, but MNIST's strongest loop (1.33x) is
  weaker than the hole-free generated 1s' chance loops (1.70x), so nothing stands out.
  A local shell test (split of the 8th-48th neighbours) is blind to a real touch (0%):
  it needs dot spacing < gap < shell radius. Against a Gaussian twin it flags 95.9% of
  fold-free generated 1s, so use generated 1s, not a smooth cloud, as the null. Cost:
  H1 on all 6,742 MNIST 1s ~10 min; 3-D toys need a `thresh` cap.

- **Unfolding test suite (toy sheets):** self-contained in `unfolding_suite/`; start
  with `unfolding_suite/README.md` (contract, scoring, outputs, workflow, results). Run
  `.venv/bin/python -m unfolding_suite --list`. A method gets shuffled, rotated 3-D dots and
  returns one flat position per dot; the scorer certifies one-to-one (no flipped triangle at
  three mesh sizes, outline never crosses) and names failure causes. 14 design + 10 held-out
  sheets; runs live in `unfolding_suite/runs/` (`current` is the reference). Best simple
  methods (`link_check.py`) pass 11 of 14 design and 7 of 10 held-out sheets. `tight_roll`
  needs more dots; twisted crumples stay unsolved. **Best simple method (2026-11-15): `simple.py`'s
  `best_simple`** (closest dots -> tilt check -> carry two directions, every link votes): 25 of 30 sheets
  with enough dots, beating the graphics recipe `mesh_full` (23) and 18x faster; both fail the stretched
  crumples, where `mesh_full` gets much closer. Scope is 2-D paper in 3-D; the `audit` group (6 sheets in
  `audit.py`) adds uneven dots, C-shaped outlines, a 4:1 strip and a flat hairpin. All 30 sheets are
  solvable with perfect information (crumples only by a no-fold Tutte layout). Walking folds a flat C-shape.
  **Always judge a flattening with `unfolding_suite/verify.py`** (the double-layer check: dots
  plus flat result only, no truth; suite runs print it as `layers:`). It reads `stacked` when
  part of the sheet ended up on top of another part; `unfolding_suite/VERIFY.md` explains the
  readings, typical values and blind spots (it cannot see layers much thinner than the dot spacing).

- **MNIST 1s flattened with locally linear pieces:** `analyze_mnist_one_lle_flattening.py`
  and `figures/mnist_one_lle_flattening/README.md` run the suite's `lle_k10` (k 10/20/40,
  2-15 numbers) on the 6,742 MNIST training 1s, with a generated-1s control. It finds only
  slant (|r| 0.98); every further number is a bent copy of it, and it loses to the best
  flat plane (PCA) at 5+ numbers. On the generated 1s it misses cy and height.

- **Directions to reach every neighbour (flat-patch count):**
  `analyze_mnist_one_flat_patch_directions.py` and
  `figures/mnist_one_flat_patch_directions/README.md`. Count the fewest SVD directions
  that bring every arrow to the nearest k neighbours within 10% of its length. Generated
  1s with neighbours made within ~2% of the image size stay at exactly 5 for any k (12-knob
  drawn 1s need ~0.6%); MNIST's nearest real neighbour is 23.6% away and the count tracks
  k (one new direction per neighbour) to k~25, so no flat patch is reachable at MNIST's density.
  Restricting to the most crowded 1% of starting 1s (`dense` part) only brings the
  nearest neighbour to 16.6% and leaves the count unchanged (87 at k = 200).

- **Scrambled MNIST 1s (picture hidden, geometry kept):** `scrambled_ones.py`;
  `load()` gives the 6,742 train 1s as (n, 784) float32 after a fixed pixel shuffle
  plus a fixed random rotation (`level="shuffle"` for re-ordering only). Distances,
  angles and neighbours are unchanged (to 5e-8). Cached in `data/scrambled_ones/`,
  loads in under 0.1 s. `key()`/`unscramble()`/`original_index()` are for grading
  only. Shuffle alone is reversible: a pixel's most-correlated partner is a touching
  neighbour 98.7% of the time.

- **Warp method on drawn 1s with KNOWN knobs (the calibration for MNIST's numbers):**
  `analyze_drawn_one_warp_scorecard.py`, `analyze_drawn_one_ink_pieces.py` and
  `figures/drawn_one_warp_scorecard/README.md`. Same 3,000/1,500 split and settings as MNIST.
  A clean 12-knob drawn family leaves 30.9% at 5 warp numbers (35.9% with MNIST-sharp edges)
  against MNIST's 27.2%, so MNIST's five-number leftover says nothing about whether it has few
  knobs. The warp reads height and vertical position at 0.93-0.96 on drawn 1s; the earlier
  "blind to cy" on MNIST was the moment yardstick (0.64 there). At 20 numbers MNIST keeps 15.6%
  vs 9.9% (sharp 12 knobs); "ink pieces" (learned extra ink carried by the warp) remove 48-61%
  of the drawn leftover but 25% of MNIST's. Edge-heavy leftover is just sharp ink; smooth
  wobbles and row roughness were tested and do not match. MNIST's excess is a worst tenth that
  fits twice as badly (31.4% vs 16.6%; curved centre lines, a few wide rows) plus ~3 points
  over the rest. Sharp drawn families: `load("sharp12")` (softness 0.18 matches MNIST's grey-ink share).

- **Simplest learned decoder (autoencoder) on MNIST 1s:** `analyze_mnist_one_autoencoder.py`,
  `make_mnist_one_autoencoder_figure.py`, `figures/mnist_one_autoencoder/README.md`. Plain
  784-256-k-256-784 autoencoder, k = 1..30, generated-1s control. Control stops at 5-6 numbers
  (6.0% at 5, flat 4.4% after); MNIST never stops (24.2% at 5, 15.0% at 30), matching the warp
  model. MNIST code direction #1 is slant (0.92) and #2 is slant SQUARED (0.84): a bent copy, as
  LLE found. Height/up-down barely enter at k=5. Code midpoints beat pixel averages but smudge.

### Geometry conventions

- Image distance is Euclidean over all 784 coverage values. A seed-to-neighbor
  direction is `(render(p) - render(p0)).ravel()`. Raw steps use pixels/degrees;
  normalized steps use fractions of each knob's allowed range.
- Five parameters plus `render` give an exact nonlinear construction. Joint
  parameter moves reach the same endpoint in either order, but intermediate
  pixel directions depend on state. The user's goal is coordinates that
  reconstruct the whole family and joint changes: require a decoder and
  held-out joint/interpolation tests, not just PCA variance or a pleasing plot.
