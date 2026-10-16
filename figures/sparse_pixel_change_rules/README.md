# Exact sparse pixel-change rules for width/height and width/lean

This experiment answers the pixel-index question directly. Given a current
state and one original-knob increment, return only the changed pixel indices
and their new coverage values. The update functions do not call the original
renderer or a full-image predictor. The renderer supplies evaluation targets.

There is no machine learning or neural network. The rule is a hand-derived
interval-overlap equation for the known parallelogram stroke family. Its
simplicity comes from that supplied structure, not from discovering geometry
from arbitrary images.

## Result

| Slice | State input | Tested updates | False indices | Missed indices | Largest pixel-value error |
| --- | --- | ---: | ---: | ---: | ---: |
| Width/height | Five original coordinates, float64 | 354 | 0 | 0 | 1.11e-16 |
| Width/height | Current 784-value image, float64 | 354 | 0 | 0 | 1.78e-15 |
| Width/lean | Five original coordinates, float64 | 396 | 0 | 0 | 2.66e-15 |
| Width/lean | Current 784-value image, float64 | 396 | 0 | 0 | 7.11e-15 |

Primary index membership uses absolute coverage-change threshold **1e-10** to
exclude floating-point noise. Values are compared over the entire resulting
image, including unreturned indices. These are numerical agreement results,
not a variance-retention percentage.

Separate default-float32 image tests also returned zero false/missed indices,
at threshold **1e-6**. Maximum value errors were 7.97e-8 for width/height and
1.01e-7 for width/lean. The larger threshold is explicitly a storage-precision
sensitivity check; it is not the primary support threshold.

Known-control and image-input cases use the same source states. These are
controlled checks on one stroke family, not independent statistical trials.

## Inputs and outputs

The canonical root is `(cx,cy,height,width,lean)=(14,14,19.75,3.2,0)`.
If root coordinates are called zero, actual width is `3.2 + width coordinate`
and actual height is `19.75 + height coordinate`; the update equations are
unchanged by this relabeling.

A unit is **+1 horizontal pixel of total width**, **+1 pixel of total height**,
or **+1 degree of lean**. Width and height increments split equally between
the two sides/ends. Source/default ranges and longer illustrated sequences
are explicitly extended where needed; all strokes remain positive and
unclipped on the canvas.

Two state-input routes are tested:

- Original five coordinates, with other controls fixed for each declared slice.
- The current 784 coverage values only. No source or target knob values are
  passed to this update route. The known family and the fixed upright center
  are structural assumptions where specified.

Each result is a list of:

```text
(flat pixel index, old value, value change, new value)
```

Indices are zero-based: `flat index = 28 × row + column`.
Only returned indices are overwritten; every other input pixel remains intact.

For example, after raising root height by 1, the next width +1 operation includes:

| Flat index | Row | Column | Old | Change | New |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 95 | 3 | 11 | 0 | 0.0375 | 0.0375 |
| 96 | 3 | 12 | 0.225 | 0.15 | 0.375 |
| 124 | 4 | 12 | 0.6 | 0.4 | 1 |

A changed pixel need not previously have been white. A single pixel cell can
contain parts affected by both height and width, so those pixel groups can
overlap even when the added continuous regions are adjacent.

## The width/height equation

For a one-pixel cell starting at position `cell`, define:

```text
covered length(cell, left edge, right edge)
    = max(0, min(cell+1, right edge) − max(cell, left edge))
```

This is just the length of the overlap between two intervals.

Make two short lists:

```text
column coverage[column]
    = covered length(column, 14 − width/2, 14 + width/2)

row coverage[row]
    = covered length(row, 14 − height/2, 14 + height/2)
```

An upright pixel's ink coverage is the product of its two list entries.
The current state is represented by those two 28-element profiles rather
than a large collection of independent per-pixel rules.

For width +1:

```text
pixel change[row,column]
    = current row coverage[row]
      × (column coverage at width+1[column] − current column coverage[column])
```

For height +1:

```text
pixel change[row,column]
    = (row coverage at height+1[row] − current row coverage[row])
      × current column coverage[column]
```

The indices are exactly the row/column pairs where that product is nonzero,
subject to the stated numerical threshold. The new value is old value plus
that change. Only two short profiles and their sparse Cartesian product are
needed; no full next image is constructed inside the rule.

These identities are exact for an upright, centered, unclipped rectangular
stroke. They cross pixel boundaries without polynomial extrapolation or a
fixed active-pixel mask. The original 16-subrow vertical coverage reduces to
the same row-overlap equation at zero lean.

## Updating the hidden groups after another knob moves

A second check carries the prospective width and height masks through the
sequence height, width, height, height, width, height.

When height changes:

```text
new width mask = old width mask
               + change in row coverage × current widening-column profile
```

The multiplication here is an outer product: pair each changed row entry with
each current side-column entry. It extends the widening mask at the ends;
it does not add the entire ordinary height-change mask to it.

When width changes, the prospective height mask receives the corresponding
outer-product correction using the new column coverage and its current
height-increment row profile. The changed knob's own prospective mask is
also refreshed: after widening, the *next* widening can affect different
columns; after heightening, the *next* height increase can affect new rows.

Both carried hidden masks agree exactly with the renderer in the illustrated
six-step trace (maximum computed difference zero). This is a second explicit
form of the same separable geometry, not evidence for an unrelated model.

## Image-only input for the upright slice

The fixed center is part of the declared slice. Read the current dimensions
with two operations:

```text
width  = largest row sum
height = total image ink / width
```

Every full row contains exactly `width` units of ink; the total contains
`width × height`. Apply the sparse row/column rule using these values.
This route requires a valid centered upright stroke with at least one full
row and no clipping; it is not an arbitrary-image inverse.

## Extension to width and lean

In a tilted stroke, different heights need different horizontal intervals.
At vertical position `y`:

```text
row center = horizontal center − tan(lean angle) × (y − vertical center)
left edge  = row center − width/2
right edge = row center + width/2
```

Angle-to-radian conversion is included when evaluating the tangent. The
same covered-length function is used in every row.

- Width +1 adds a horizontal strip of thickness 0.5 at each side. Enumerate only
  pixel columns intersected by those strips and add their overlap amounts.
- Lean +1 shifts the interval by
  `−(y−vertical center) × (tan(new lean) − tan(current lean))`.
  Enumerate the interval portions gained and lost. Add gains and subtract
  losses in their corresponding pixel indices.

To match the actual renderer rather than a different continuous-area
approximation, do this at its 16 subrow midpoints, with its vertical clipping
weights. Combine signed contributions to the same pixel before deciding
whether that pixel changed. The update observes old values only at emitted
indices for the coordinate-input route, or reads them directly from the
current image for the image-input route.

No global new image is generated inside the rule. Unlike repeatedly warping a
coarse bitmap, it keeps the fractional interval locations that determine the
next pixel-boundary crossing.

## Image-only input for the lean slice

Row sums recover width and the first/last partial rows recover top and bottom.
Choose two well-separated full rows. In each, select an integer column boundary
inside the stroke and read the cumulative ink to its left:

```text
mean left edge = selected column boundary − ink to its left
row center     = mean left edge + width/2
```

The difference between the two row centers, divided by their vertical
separation, gives the lean slope. This is sums and arithmetic, not an optimizer
or a trained inverse. The chosen boundary must remain inside all subrow
intervals of the selected row; the sampled width/lean domain satisfies this.
The method assumes valid input strokes and at least two full interior rows.
It makes no promise for unrelated or clipped images.

## Validation and repeated use

Width/height is tested first. Only after its two float64 routes pass does the
experiment attempt width/lean.

- Width/height: 49 lattice/event states plus 128 Sobol states, two unit actions
  each. Lattice heights include 21 and 22 beyond the default box to test the
  next cap regions.
- Width/lean: 70 lattice/event states plus 128 Sobol states, two unit actions
  each. Source widths span 1.8–4.6 and source lean spans −10 to +35 degrees.
- Integer width/height event cases and an almost-upright lean case are included.
- Ground truth is the original `grey_ones.render`, not the sparse updater.

Image-only chains repeatedly feed the updated 784 values into the next call:

| Slice | Schedules | Total unit updates | False/missed indices | Largest pixel error |
| --- | ---: | ---: | --- | ---: |
| Two height and two width steps | 6 | 24 | 0 / 0 | 1.55e-15 |
| Two width and ten lean steps | 66 | 792 | 0 / 0 | 5.33e-15 |

The largest endpoint disagreement across different orders is 6.66e-15.
The image-only chain never receives true coordinates; the separately stored
true coordinates are used only to render evaluation targets.

The equations establish the structural identities under their assumptions;
these samples verify implementation and numerical behavior. Neither this nor
an aggregate image score is a proof of correctness on arbitrary images.

## What five dimensions does and does not mean

Five controls do not mean five fixed pixel groups. A single continuous control
can move its affected region through many indices. The coordinate count and
the number of pixel patterns required by a fixed linear basis measure different
things. These overlap functions express index switching directly through
min/max and integer pixel locations. Code is evaluating the equations; there
is no mathematical representability barrier in this controlled example.

## Artifacts

- `01_width_height_indices.png`: current images and prospective masks after
  mixed width/height changes. Green outlines show returned indices.
- `02_width_lean_indices.png`: the same sparse overlap idea at different leans,
  showing both widening gains and lean gains/losses.
- `example_pixel_edits.csv` and `.json`: complete sparse indices and new values
  for four representative operations, including image-only input.
- `all_update_metrics.csv`: every one-step comparison and support count.
- `results.json`: all aggregate results, thresholds, assumptions, and chain checks.
- `width_height_trace.npz`: state, images, and carried prospective masks.

Reproduce from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/sparse_pixel_change_rules.py
```
