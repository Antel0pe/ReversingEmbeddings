# Repository instructions

- Never include the user's full local filesystem path in a public or shareable artifact. Use repo-relative paths or sanitized placeholders.
- Never use npm. Use pnpm or bun when a JavaScript package command is needed.
- When making an explanatory experiment figure, read `skills/experiment-figures/SKILL.md` and apply its understanding check before delivery.

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

### Geometry conventions

- Image distance is Euclidean over all 784 coverage values. A seed-to-neighbor
  direction is `(render(p) - render(p0)).ravel()`. Raw steps use pixels/degrees;
  normalized steps use fractions of each knob's allowed range.
- Five parameters plus `render` give an exact nonlinear construction. Joint
  parameter moves reach the same endpoint in either order, but intermediate
  pixel directions depend on state. The user's goal is coordinates that
  reconstruct the whole family and joint changes: require a decoder and
  held-out joint/interpolation tests, not just PCA variance or a pleasing plot.
