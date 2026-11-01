# The 2-knob sheet as a curvilinear grid

Built by `make_generated_one_curvy_grid.py`. Window: **height x lean**, with
`cx=14`, `cy=14`, `width=3.2` held at root-1 values. Height spans 14-20, which
extends `grey_ones.RANGES` (19.0-20.5); lean spans +-12, inside its default
range. No stroke in the window touches the canvas border (checked: border ink
is exactly 0.0), so nothing here is a clipping artifact.

## `curvy_grid.png` - the grid

Panel 2 is the baseline: the knob grid, straight and parallel by construction,
every line turning 0.0 deg. Panel 3 is the same grid in pixel space, drawn in
the 2-d plane that spreads the sheet out most (holds 77%); lines turn a median
of 47 deg and up to 147 deg. Panel 4 adds a third axis (84%) and shows the
sheet is a curved surface, not a bent plane.

## `why_it_curves.png` - the mechanism

A knob's direction is a 784-vector, so it is an image. The height direction is
supported only on the pixel rows holding the stroke's two ends. Rendered at
heights 15, 17 and 19 those three images **share no pixel**, so their pairwise
cosines are exactly `0.000`.

Panel A: the height direction stays identical (cosine 1.0000) until the end
crosses into the next pixel row at half a pixel, then drops to exactly 0. The
lean direction instead fades gradually and goes negative.

Panel B: this is why anchoring to a single 1 fails. The tangent plane at one 1
**saturates** - past about 1 px of height change it stops registering the
change at all, while the true distance keeps growing. An earlier version of
this figure projected onto that plane and produced a collapsed blob; the
best-flat-map view in `curvy_grid.png` replaced it.

## `what_survives.png` - projection-free readings

Panel 1: at each node, the polar curve `sqrt(d' g d)` over directions `d`,
with `g = J' J` in normalised knob units - how far one equal knob-step really
moves you, in every direction. Biggest step / smallest step across the sheet:
**3.1x**. This depends only on measured distances, not on any chosen view.

Panel 2: classical-MDS eigenvalue shares of the pairwise distance matrix:
`61.2, 15.4, 7.3, 4.1, 3.8, 3.0, 1.5 %`. Any flat map gets the first two. The
**7.3%** third share is the part no flat drawing can represent, so no faithful
flat map of this sheet exists - the same reason every flat map of the globe
distorts something.

## Caveats

- Panels 3 and 4 of `curvy_grid.png` are projections; the stated percentage is
  how much of the sheet each one holds.
- Finite secant steps are used throughout (`1e-3` px for height, `1e-2` deg for
  lean), not analytic derivatives. These were checked to be stable across step
  sizes from `1e-5` to `0.5` at both generic and root-aligned heights.
- The three-knob complement (`cx`, `cy`, `width`) is fixed, so this is a slice,
  not the whole five-parameter family.
