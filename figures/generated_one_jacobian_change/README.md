# How the Jacobian changes across the knob box

Built by `make_generated_one_jacobian_change.py`. Centre is the mid-box point
`(14.5, 14.5, 19.75, 3.2, 12.5)`, not the root (the root sits on the lower
`cx`/`cy` bounds). Steps are 10% of each knob's range, equal in both directions.

## `second_derivative_grid.png`

Top strip: the five columns of the Jacobian - what each knob does on its own.
Below: the 5x5 table of `d2 render / dpi dpj`, in normalised knob units. One
shared colour scale with a square-root stretch so the faint cells stay visible;
the number under each cell is its true size.

Sizes (normalised knob units):

```
             cx      cy   height   width    lean
  cx      14.87    3.97     1.46   19.59   33.35
  cy       3.97    0.93     0.00    5.11   14.07
  height   1.46    0.00     0.26    1.94    8.60
  width   19.59    5.11     1.94   26.02   45.96
  lean    33.35   14.07     8.60   45.96  137.85
```

- `lean/lean` = 137.85 dominates, five times the next largest.
- `cy/height` = exactly 0.00. The two ends translate independently.
- `height/height` = 0.26: height is nearly linear, which is why its grid line in
  `../generated_one_curvy_grid/` is nearly straight in 784 dimensions.
- The table is symmetric to **4.0e-15**, as mixed partials require. An earlier
  version smoothed the two directions by different amounts (10% vs 2%) and came
  out 2x asymmetric; on a piecewise function unequal smoothing is not harmless.

## What render is inside one piece

From `grey_ones.render`: `wy` (vertical coverage) and `ov` (horizontal overlap)
are both continuous piecewise-linear clips, so render is continuous and its
*derivative* jumps at breakpoints. Fitting inside a single kink-free gap, one
knob at a time (relative held-out error):

| knob | exact in | degree 1 | degree 2 | degree 3 |
| --- | --- | --- | --- | --- |
| cx | `cx` | 1.4e-12 | 1.0e-12 | 2.2e-12 |
| width | `width` | 4.0e-13 | - | - |
| height | `height` | 9.1e-14 | - | - |
| cy | `cy` | 3.5e-04 | 6.4e-13 | 6.0e-13 |
| lean | `tan(lean)` | 3.1e-14 | 3.7e-14 | 1.9e-14 |

So inside a piece render is **linear in `cx`, `width`, `height` and
`tan(lean)`, and quadratic in `cy`** (which enters both `wy` and `centre`).
Nothing above degree 2 exists in the interior - degree 3 adds nothing.

`tan(lean)` matters only once you are inside a piece. A 5-d affine fit over any
box straddles many pieces and shows no difference at all between `lean` and
`tan(lean)` (0.140 vs 0.140); the 1-d piece-interior test shows 3.1e-14.

## Kink counts along a single knob (others at mid-box)

`cx` 632, `width` 431, `height` 11, `lean` 1257 (counted in `tan(lean)`).
Counted by the difference order that vanishes inside a piece. The `cy` count is
not reliably measured here: `cy` is quadratic inside a piece, so it needs a
third difference, which was numerically unstable at this sampling.

## Does piecewise-linear work globally?

Along a generic straight line across the box, very well:

| pieces | 1 | 4 | 16 | 64 | 256 | 1024 | 4096 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| median rel error | 5.8e-1 | 1.1e-1 | 1.2e-2 | 8.3e-4 | 6.1e-5 | 4.4e-6 | 1.7e-8 |

In the full 5-d box it does not:

| cells | 1 | 32 | 243 | 1024 |
| --- | --- | --- | --- | --- |
| median rel error | 6.6e-1 | 3.4e-1 | 2.2e-1 | 1.6e-1 |

1024 cells is only 4 per axis. Matching the 1-d resolution of 64 pieces per
axis would need 64^5 = 1.1e9 cells. The obstacle is the fifth power, not the
jumps: a perfectly smooth function would need the same cell count.
