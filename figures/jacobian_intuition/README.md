# The Jacobian of the generated-1 renderer, as pictures

Three explanatory figures. These are analytic/finite-difference diagnostics of
the **known** five-knob renderer: no fitting, no shell sampling, no learned
basis. They exist to make the object concrete before any data-only recovery
method is judged against it.

Root settings are `(cx, cy, height, width, lean) = (14, 14, 19.75, 3.2, 0)`.
All derivatives use float64 coverage and central differences with steps
`cx/cy/height/width = 1e-4` (pixels) and `lean = 1e-3` (degrees). Halving every
step changes the largest Jacobian entry by a relative `3.55e-11`, so the panels
show converged derivatives rather than step-size artefacts.

## What a Jacobian column is here

`render` maps 5 knobs to 784 coverage values, so its Jacobian at a point is a
784x5 matrix: a linear map from knob space into pixel space. Column *k* is
`d(image)/d(knob k)`, which is 784 numbers, which is a 28x28 picture. Each
column is therefore literally viewable — a map of which pixels gain ink and
which lose it when one knob moves and the other four are held fixed.

A column is **not** an image of a 1. It is a rate of change of an image.

## 01 — `01_jacobian_columns_as_pictures.png`

The root 1 followed by all five columns, twice.

- **Row 1, raw knob units.** cx, cy, height and width move in pixels; lean moves
  in degrees. Lengths in different units are not comparable, so each panel has
  its own colour scale.
- **Row 2, range-normalised.** Each column multiplied by its own allowed knob
  range, putting all five in one unit so a single colour scale and the bar chart
  apply to all. This is a linear extrapolation of the local rate across the
  range, not an actual full-range move.

| Knob | Raw column length | Per full knob range |
| --- | ---: | ---: |
| cx | 6.2500 per pixel | 6.250 |
| cy | 2.3324 per pixel | 2.332 |
| height | 1.1662 per pixel | 1.749 |
| width | 3.1250 per pixel | 8.750 |
| lean | 0.6150 per degree | 27.676 |

The five columns have **rank 5 of 5**, so no combination of knob nudges leaves
the image unchanged: the knobs are independent. Per full range, lean moves the
image 15.8x further than height. Row 2's shared scale is +/-1.40, set by the
second-largest panel; lean's 32 pixels that run past it are outlined in green,
because using lean's own +/-6.68 would wash the other four panels out.

## 02 — `02_columns_change_as_you_move.png`

The cx and cy columns measured the same way at two points that differ only in
lean, plus an exact 2-D view of the plane each pair spans (true lengths, true
angle, nothing projected away), plus a sweep over the whole lean range.

| Start | cosine(d/d cx, d/d cy) | angle |
| --- | ---: | ---: |
| root, lean 0 | 0.000000 | 90.0 deg |
| lean 30 | 0.788879 | 37.9 deg |

Over 91 sampled leans the cosine runs from **-0.4503 to 0.8501**, reaching 0 at
lean 0 only. The Jacobian keeps rank 5 across the whole sweep, so the knobs stay
independent everywhere while ceasing to be perpendicular. A single linear map
has one Jacobian everywhere and could not do this; the small steps in the curve
are the renderer's 16-subrow vertical sampling.

This is the reason an SVD basis cannot in general coincide with the knob
directions: SVD returns mutually perpendicular vectors, and the true knob
directions are not perpendicular away from special points.

## 03 — `03_exact_column_drift.png`

The height column at width 3.2 and at width 3.25, their difference, and a
closed-form prediction of that difference. Panels 3 and 4 are a **change in a
direction vector**: their colours describe how the pixel-change pattern itself
shifted, not ink gained by an image, and they use a 40x finer colour scale than
panels 1 and 2.

At the root's centre and zero lean, with height in (18, 20) and width in (2, 4),
the coverage image is an outer product of two affine profiles, so
`F(h, w) = C + h*U + w*V + h*w*D` with C, U, V, D fixed 28x28 arrays. Hence
`d/dh = U + w*D`, and a width step `dw` changes the height column by exactly
`dw*D`.

| Quantity | L2 |
| --- | ---: |
| measured change, dw = 0.05 | 0.025000 |
| predicted dw x D | 0.025000 |
| residual between the two | 2.483e-12 |
| worst gap from the straight line, dw <= 0.4 | 1.998e-11 |

This is a fixed-knob slice: cx, cy and lean stay at the root and only height and
width move. The exactness belongs to this patch. Crossing into another
pixel-formula region changes D, so the straight line must not be extrapolated
past the patch boundary.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python make_jacobian_intuition.py
```

Writes the three PNGs and `results.json` (column norms, clipping counts, the
full lean sweep, the width sweep, and the finite-difference stability check).
