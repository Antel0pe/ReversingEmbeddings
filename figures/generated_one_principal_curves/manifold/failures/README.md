# Failure analysis: learned generated-one manifold

All pixel diagnostics use the same 16,384 additional uniform points and existing fitted reconstructions. No model is retrained. Motion diagnostics use 128 interior Sobol starting settings, seed 948, each normalized knob fraction between 0.15 and 0.75.

## Pixel bounds and connectivity

Clipping removes 18.5447% of raw squared error. Capture changes from 99.879293% to 99.901678%. Values outside [0,1] contribute 20.1259% of raw error; clipping does not necessarily remove all error at such pixels.

After clipping, 85.7239% of residual squared error is on fractional source-coverage pixels, 8.4444% on empty source pixels, and 5.8317% on fully inked source pixels.

At cutoff 0.001, 99.7864% of bounds-anomaly pixels connect to the reference stroke region. Actual disconnected positive extra ink contributes 0.001803% of clipped error.

Bounds connectivity uses the union of reference ink >1e-6 and bounds anomalies beyond the selected cutoff, with eight neighbors. This measures spatial adjacency; negative values are not positive ink. For actual ink components, threshold the clipped prediction and select the component with greatest reference-ink overlap. Ghost errors count empty source pixels only. Results include cutoffs 1e-6, 1e-3, 0.01, 0.05 because tiny bridges change topology.

The main-component correction clips predictions, discards values at or below the cutoff, then keeps the main positive component. Its score includes any true ink lost through thresholding or splitting.

Clamping stored full-image nodes without refitting gives 99.658797% capture. Convex weights guarantee scalar bounds to roundoff, but clamping nodes destroys some interpolation cancellations. This ablation is not a constrained refit.

The old display assigns about 10% magenta tint even to violations barely beyond 1e-6, then increases tint with violation magnitude. Grayscale already clips for display. Magenta area is not proportional to reconstruction SSE.

## Direction tests

True pixel Jacobian J_true uses central differences at +/-0.001 of each physical knob range. Learned Jacobian J_grid uses the exact within-cell multilinear derivative at each encoded starting image. For clipped images, multiply derivative rows by the indicator that the raw decoded pixel lies strictly between zero and one. Clipping has derivative kinks at bounds; the test uses this local convention.

For each true knob direction compare (a) its corresponding learned axis with the best scalar scaling and (b) the best linear combination of all five learned axes. The latter is a least-squares oracle local calibration G: J_grid G approximates J_true. Off-diagonal G coefficients describe coordinate mixing. They are not automatically erroneous physical motion. All knobs and learned coordinates use normalized range fractions.

For finite 2.5%-range and 10%-range moves, hold the starting calibration G fixed, move the learned coordinates by G times the requested step, bound coordinates to [0,1], and decode. Compare the decoded image difference with the actual rendered knob-move difference. This is a diagnostic that uses the true tangent at the starting point, not an implemented physical knob controller. Finite-move mismatch also includes curvature and changes in the chart along the path.

A separate comparison independently projects both true endpoint images onto the learned manifold before differencing their reconstructions. It tests best-case representation of a change and must not be interpreted as prediction from the physical knob alone.

Direction error is the square root of total squared vector mismatch divided by total squared actual pixel change, pooled over 128 examples. This relative vector error is different from variance capture. Cosines and error on pixels with no actual change are also recorded.

90.0528% of baseline clipped reconstruction error on the 128 starting images is orthogonal to the five true knob tangents. This is a first-order local test; it does not compute exact distance to the full renderer family, and normal components can reflect curvature.

## Nonlinear validity and projection checks

Bounded two-start fits of the actual renderer leave 91.3749% of clipped error on 32 interior examples. On the four selected displayed cases, they leave 17.8318%. Large selected errors can therefore be mostly shifts to a different valid stroke, especially height, even though typical interior residuals are mostly shape mismatch. These cohorts are not a population-wide attribution. The optimizer is local, so nearest-family distances are upper bounds, not global certificates. The renderer is used solely as a diagnostic and does not enter the learned decoder.

On four selected cases, a broader projection diagnostic uses 32 nearest fitted-image starts, one normalized physical-knob start, and 32 Sobol coordinate-box starts, with up to 100 steps each. It barely reduces the largest raw residual (0.5912 to 0.5888). Several high-error fitted positions hit learned-coordinate bounds. This supports testing boundary-face coverage or chart extent; it does not prove a unique cause. No stored model or predictions were replaced by these diagnostics.

## Effort

The winning curve refinement path has 485 outer cycles and approximately 14.50 logged fitting minutes. The guided-to-fine grid path has 24 outer cycles and approximately 7.90 logged fitting minutes. Grid nodes: 15795; curve nodes: 2565.

Each grid outer cycle contains a large sparse node solve with conjugate-gradient iterations. Curve cycles use a different node system and coordinate projector. Counts are not equal units of compute. Times are historical fitting-loop logs, excluding diagnostic projection, rendering, initialization overhead, and discarded attempts.

## Interpretation and next experiments

The fitted grid stores unconstrained pixel values. Its reconstruction-and-bending objective does not impose scalar coverage bounds or coupled constraints for a valid straight stroke. The main residual is connected boundary mismatch, with both chart mixing and a local surface-direction mismatch. Disconnected islands make a negligible contribution to SSE. There is no stochastic noise source in decoding; error varies deterministically with coordinates. Concentration at fractional edges and repeatable tangent mismatch indicate structure, but these measurements alone do not identify a unique cause among interpolation, smoothing, or projection.

Focused next tests: fit nodes under bounds while refitting coordinates; refine cap/center axes where direction mismatch is largest; use actual small knob-move samples as an additional direction regularizer; and report shape validity and vector errors beside variance capture. Clipping and connectivity are useful constraints but do not certify membership in the generated family. These are proposed follow-up experiments, not changes performed here.

## Reproduce

```sh
.venv/bin/python analyze_generated_one_manifold_failures.py
.venv/bin/python make_generated_one_manifold_failure_view.py
```

pixels.json records full counts, SSE partitions, threshold sensitivity, node-clipping ablation, parameter and cell-phase error summaries, covering-cloud clipping results, and effort. motion.json records tangent and finite-change metrics, coordinate calibration, and scope limitations. Both NPZ files contain the displayed numerical examples.
