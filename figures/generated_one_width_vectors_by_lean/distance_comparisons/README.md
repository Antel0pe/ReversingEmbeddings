# Width and lean: distances and transferable vectors

The generated 1 has fixed center `(14,14)` and height `19.75 px`. Write `F(w,l)` for its 784 pixel coverage values at width `w` and lean `l`.
The normal root is `F(3.2,0)`. Width ranges 1.8–4.6 px; lean ranges −10°–35°.

## Three distinct comparisons

1. **Image gap:** `length(F(w,l)-F(w,0))`. How far apart are two 1s at the same width?
2. **Width-vector gap:** compare `F(w,l)-F(3.2,l)` with `F(w,0)-F(3.2,0)`.
3. **Lean-vector gap:** compare `F(w,l)-F(w,0)` with `F(3.2,l)-F(3.2,0)`.

The vector differences in 2 and 3 are identical. They both equal the mixed difference
`F(w,l)-F(w,0)-F(3.2,l)+F(3.2,0)`.
Subtracting each curve's own starting image matters: an image gap and a gap between vectors with different origins answer different questions.
All lengths are Euclidean over 784 values: square each entry, sum, and take the square root.

## Equal angular increments

`T(w,l) = F(w,l+1)-F(w,l)` always uses a +1° step.
We compare its length, its difference from `T(3.2,l)`, and its difference from `T(w,0)`.
Similar lengths do not imply similar vectors.
There is a local reuse interval: the 0-to-1-degree step matches across sampled widths 2.35-3.65 px, with vector gaps below 1e-6 after float32 storage and below 1e-12 before storage.
At widths 3.2 and 4.6 the same step has length 0.615 in both cases, but the two vectors are 90 degrees apart.
The canvas displacement at vertical coordinate `y` is `(14-y) * [tan(l+1)-tan(l)]` (angles in degrees), independent of width.
Equal degree steps are not exactly equal shear increments, and sampled pixel changes depend on edge positions and coverage overlaps.

## Four corners and commutativity

- `A = F(3.2,0)` normal root.
- `B = F(4.6,0)` max width only.
- `C = F(3.2,35)` max lean only.
- `D = F(4.6,35)` both maxima.

Known root vectors: `W = B-A`, `L = C-A`.
Recovered vectors: lean after widening is `D-B`; width after leaning is `D-C`.
Their changes satisfy `(D-B)-L = (D-C)-W = D-B-C+A`.
Both actual orders reach `D` exactly. Adding the original vectors predicts `B+C-A`; its error is the mixed difference.
Pixel-vector addition itself commutes. Its failure to match `D` is failure of a state-independent additive rule, not noncommutativity of those additions or of the known parameter setters.
The three images A, B, and C alone do not determine D; here the renderer supplies the fourth corner.
Distances alone cannot establish commutativity for an unknown learned direction field.

See `results.json` for measured corner lengths, angles, and checks, and the three CSV files for plotted data.
`neighbor_proximity.csv` additionally compares +0.1 px width against +1° lean, then +2.8/45 px width against +1° lean (equal fractions of the allowed ranges).
The sampled comparison uses starting widths 1.8–4.5 px and starting leans −10°–34°, so every endpoint remains in range.
These comparisons depend on specified step sizes; there is no unit-independent rule that all width neighbors are closer.

## Reproduce and scope

Run `OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/compare_width_lean_distances.py` from the repository root.
The script uses `grey_ones.render`, including its 16-subrow model and default float32 storage, and computes differences in float64.
The source grid has 57 widths × 91 leans; validation halves both spacings and checks every corresponding image.
The plots compare corresponding settings, not nearest distances between entire curves.
All claims apply to this sampled slice; no projection, decoder, or full-manifold equivalence is claimed.
