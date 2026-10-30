# Is there a linear map from the 5 knobs to the 784 pixels?

Question: fit one linear (affine) map `image = offset + A·(cx, cy, height, width, lean)`
on the generated-1 family, and ask whether it can reach 99.9999% accuracy.

Answer: no, and the obstruction is not the fit — it is the family.

## Scripts

- `prep.py` — fits the map and caches every number the figures use (`fig.npz`, in the scratchpad).
- `mkfig.py` — renders both figures from that cache.
- `lin5.py` — the headline fit plus the PCA ceiling, over the full box and over shrinking boxes around root 1.
- `why.py` — the three nonlinearity diagnostics (superposition, state-dependent knob columns, polynomial features).

Run from the repo root with `PYTHONPATH=. .venv/bin/python figures/linear_5d_to_784d/<script>.py`.

## Figures

**`linear_map_attempt.png`** — the best linear map and how it fails.
Least squares on 20,000 renders over the full default knob box, scored on 5,000 held out.

- Held-out R² = 0.5629; RMS pixel error 0.109 coverage; worst pixel error 1.15, wider than
  the whole 0→1 ink scale. 4.0% of predicted pixels are negative ink, 1.1% exceed full ink.
- Ceiling without fitting anything: the best 5-dimensional affine subspace through the data
  (top-5 SVD) holds 80.27% of the image variance. Any map whose last stage is linear in five
  numbers lands in some such subspace, so 19.7% of the variation is unreachable no matter how
  the five numbers are chosen or preprocessed. Reaching 99.9999% needs 221 fixed images.
- The same fit on shrinking patches around root 1 crosses 99.9999% only below ~0.3% of each
  knob's range. That is the tangent plane, not a description of the family.

**`linear_map_why_not.png`** — why, straight from the renderer, with nothing fitted.

- Superposition: `render(a) + render(b) − render(root)` vs `render(a + b − root)`. Over 2,000
  random knob pairs the mismatch is median 85%, mean 90%, worst 208% of the move's own size.
  A linear map scores 0% on every pair.
- Fixed columns: the same +11.25° lean step taken from a thin and a wide stroke gives patterns
  with cosine 0.26. Across 300 random starts per knob, median cosine 0.17–0.30 for all five knobs.
- Source: `tan(lean)`, the product `tan(lean)·(y − cy)`, and the `clip/min/max` coverage
  calculation. Polynomial features of the knobs (still least squares) reach only R² = 0.968 at
  degree 6, because no global polynomial has corners; the exact decoder is piecewise.

## Scope

Full default `RANGES` box, uniform sampling, `render(..., dtype=np.float64)` throughout.
Accuracy is held-out R² over all 784 coverage values, with RMS and worst-pixel errors reported
alongside. Shrinking-patch results are centered on root 1 `(14, 14, 19.75, 3.2, 0)`.
