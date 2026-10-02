# Generated-one geometry: exact patches, sharp joins, and global span

Only the controlled five-knob family in `grey_ones.py` is studied here. No MNIST images, learned embedding, neighbor search, or neighbor-shell implementation was used.

Experiments began on 2026-09-30 at 23:18:16 UTC. All new files are in `claude/`. See `generated_geometry_results.json` for measured values and sampled states.

## Main findings

| Question | Result | Scope |
|---|---|---|
| Is a width-height formula patch exactly representable in 3D? | Yes, its mathematical affine span is at most three; all 500 sampled patches had rank three. | Between overlap-formula changes; before float32 rounding. |
| Are those patches flat sheets? | Generally no. All 500 tested patches had negative Gaussian curvature. | Smooth interiors of the tested width-height patches, not their sharp joins. |
| Does that make the whole width-height slice 3D? | No. One slice had numerical affine rank 77, stable across seeds and thresholds. | Centers fixed at 14.43, 14.62; lean fixed at 17.3 degrees. |
| Does the local five-dimensional tangent space vary continuously? | Not everywhere. A cap crossing a pixel-row boundary gives a 90-degree limiting principal angle. | A controlled boundary crossing; images remain continuous. |
| How many affine coordinates describe a full five-knob formula patch? | At most nine in suitable parameters. Survey: 1,977 rank-nine, 20 rank-eight, and 3 rank-seven patches. | 2,000 random states; this does not change the five-dimensional intrinsic dimension. |
| Can extra knobs shorten a path? | A feasible full-knob route was about 1.432% shorter than a lean-only route in one example. | Constructive numerical example; no claim of globally shortest path. |

The central distinction is between intrinsic dimension, tangent-space dimension, local affine dimension, and the affine span of an entire slice. They are different quantities.

## 1. Width and height give exact bilinear patches

With centers and lean fixed, horizontal overlap is piecewise affine in width and vertical overlap is piecewise affine in height. Their product and sum therefore give

\[
F(w_0+u,h_0+v)=A+B u+C v+Duv
\]

within each formula region. Here all four letters denote complete 784-component pixel arrays. No pixel dimension is discarded.

The region's image lies in `A + span(B,C,D)`, so an orthonormal basis of that span supplies an exact Euclidean 3D representation when the rank is three. The allowed points still form a two-dimensional surface in that space.

Across 500 randomly selected settings:

- All 500 patches had affine rank three.
- Maximum held-out image prediction error: 7.802e-15 in pixel L2.
- Maximum total-ink identity error: 7.106e-14.
- Median width interval: 0.004107 pixels.
- Median height interval: 0.08690 pixels.

The rectangular subdivision is safe but not necessarily minimal: it includes width events that matter at some allowed heights and can be irrelevant at others.

### Why the patches are saddles

Using local coordinates `u,v`, the second derivatives satisfy `F_uu = F_vv = 0` and `F_uv = D`. If `D_normal` is the part of D perpendicular to the tangent plane, then

\[
K=-\frac{\|D_{\mathrm{normal}}\|^2}
{\|F_u\|^2\|F_v\|^2-\langle F_u,F_v\rangle^2}.
\]

Thus every regular bilinear patch has nonpositive Gaussian curvature; it is strictly negative when its mixed response has a nonzero normal component. This is an algebraic statement, not a PCA interpretation.

All 500 tested patches were strictly negative, ranging from approximately -0.03463 to -0.00002146, with median -0.01555 in inverse squared pixel-L2 length. This is a sampled result, not a proof that every allowed state has strictly negative curvature.

This explains how individual straight width and height motions can jointly form a curved surface. It also means a curved patch generally cannot be flattened into an ordinary Euclidean 2D map while preserving its intrinsic local distances.

## 2. One whole width-height slice needs far more coordinates

Fix horizontal center 14.43, vertical center 14.62, and lean 17.3 degrees, and allow the full specified width and height ranges.

The event grid has 908 width intervals and 25 height intervals: 22,700 rectangles with 23,634 distinct grid vertices. Only 114 pixel coordinates vary across this slice; other coordinates are constant. Pixelwise monotonicity in width and height lets the extreme images identify those varying coordinates.

A basis built from 3,000 random images stabilized at rank 77. Another 1,000 held-out images had maximum reconstruction residual 7.758e-12.

Checking every one of the 23,634 grid vertices against that basis gave maximum residual 9.182e-12. Since every point inside each rectangle is a convex bilinear combination of its four vertices, the vertex check also controls the interior residual for the mathematical bilinear patch, subject to floating-point calculation error.

Two additional random seeds and thresholds 1e-6, 1e-8, and 1e-10 all returned rank 77.

This is strong numerical evidence and an exhaustive check of this slice's formula grid. It is not a symbolic proof of exact rank 77 over real arithmetic.

An important failed approach: building an unpivoted basis by visiting the grid in sorted order admitted spurious tiny directions and reported rank 86 at a 1e-10 threshold. Initializing with well-spread random states, reorthogonalizing twice, and checking seeds and tolerances removed that artifact.

The result does not mean the slice has 77 independent controls. It has two. It means preserving its full ambient Euclidean geometry requires a much larger affine space than the space containing each individual patch.

## 3. Continuous images can have sharply discontinuous tangent spaces

Use parameters

\[
(cx,cy,h,w,\mathrm{lean})=(14.43,14.2,19.6,3.2,17.3^\circ).
\]

The bottom cap is at `cy + h/2 = 24`, exactly a pixel-row boundary. Compare states with height `19.6 - epsilon` and `19.6 + epsilon`.

| Height half-step, pixels | Endpoint pixel L2 distance | Largest tangent principal angle |
|---:|---:|---:|
| 0.01 | 0.02126846 | 90 degrees |
| 0.001 | 0.002126846 | approximately 90 degrees |
| 0.0001 | 0.0002126846 | approximately 90 degrees |
| 0.00001 | 0.00002126846 | 90 degrees |
| 0.000001 | 0.000002126846 | 90 degrees |

Both sides had rank-five Jacobians. Four principal angles tend to zero; one tends to 90 degrees.

There is a direct explanation. In coordinates that move the bottom cap while holding the top and stroke centerline fixed, the cap response changes pixels in the old row just before the crossing, and pixels in the newly entered row just afterward. The new row was identically zero on the earlier side, including for every earlier tangent vector. The new cap direction is consequently orthogonal to the entire earlier tangent space.

At the boundary itself, do not claim one classical smooth tangent space. Use one-sided tangent spaces or a suitable tangent-cone description.

This is a geometric kink in the pixel representation. It does not make the images discontinuous or add a sixth continuous control.

## 4. A full five-knob formula patch has an exact nine-pattern description

Reparameterize the stroke using:

- top edge `t = cy - height/2`;
- bottom edge `b = cy + height/2`;
- lean slope `s = tan(lean)`;
- horizontal line intercept `a = cx + s*cy`;
- width `w`.

The stroke's horizontal center at sub-strip midpoint y is `a - s*y`.

Inside a fixed formula region, a sub-strip's vertical overlap is affine in t and b, while its horizontal overlap is affine in a, w, and s. Products are therefore quadratic, with only mixed vertical-horizontal terms.

There is just one partially covered sub-strip at each cap away from event boundaries. Let their fixed midpoints be `y_top` and `y_bottom`. Then the exact patch can be written with five linear-response patterns and four additional interaction patterns:

\[
F = A
+T\,dt+B\,db+X\,da+W\,dw+S\,ds
+U_t\,dt(da-y_{\mathrm{top}}ds)
+V_t\,dt\,dw
+U_b\,db(da-y_{\mathrm{bottom}}ds)
+V_b\,db\,dw.
\]

The letters A, T, B, X, W, S, U_t, V_t, U_b, V_b denote fixed pixel arrays for that patch. The nine nonconstant arrays establish an upper bound of nine on its affine dimension.

The nine displayed coefficients are constrained functions of five parameters. They are not nine new independent controls.

A survey of 2,000 settings found:

- affine rank 9: 1,977;
- affine rank 8: 20;
- affine rank 7: 3.

All observed lower-rank examples were narrow strokes. At a cap occupying only two pixel columns, the cap profile and its two horizontal responses can be linearly dependent, which explains how dimensions can disappear from this affine representation.

In 100 held-out small chart perturbations, the maximum reconstruction error was 2.377e-14.

This suggests using exact local polynomial charts to study this renderer, with explicit event boundaries and chart transitions, rather than inferring all structure from a sampled neighbor graph.

## 5. Which individual knobs are actually straight within a patch?

The formulas distinguish speed changes from direction changes.

- Horizontal center, width, and height individually give affine image paths while the active formulas stay fixed.
- Lean gives an affine path in lean slope. Using degrees instead makes its speed vary through the tangent function, but does not turn its direction within that formula region.
- Vertical center can give a quadratic image path even within a region, because it changes both the cap overlaps and the horizontal centerline intercept.

At the test state, changing vertical center by plus/minus 0.0001 pixels gave a nonzero normal second difference of about 1.237e-8. Equivalent tests of horizontal center, height, and width were at roundoff scale. Lean had a nonzero midpoint error due to its changing speed, but its normal second difference was only about 4.040e-15.

Thus new pixel participation is not the only way responses can change. Within-region interaction and coordinate-dependent speed must also be distinguished.

## 6. Paths: a modest positive result and a useful failure

A full-knob path was optimized between:

- start: (14.5, 14.5, 19.75, 4.4, -8 degrees);
- end: (14.5, 14.5, 19.75, 4.4, 33 degrees).

A lean-only path had length approximately 25.99808 in accumulated pixel L2. The found full-knob path had length approximately 25.62579, a 1.432% reduction. Its main change was temporarily reducing height to about 19.1095 pixels; width changed slightly as well.

Both endpoints are exact, and every intervening parameter segment stays inside the allowed box. Dense evaluation rendered 262,144 segments along the found route. Increasing resolution from 65,536 to 262,144 changed its estimated length by about 0.00001985, far smaller than the approximately 0.37229 advantage.

This establishes a numerical example of a valid shortcut. It does not establish that the route is globally shortest; the optimizer did not fully converge.

For two width-height endpoint pairs, a coarse discrete-energy optimization misleadingly favored unrestricted paths whose densely evaluated lengths were actually longer than the restricted alternatives. Straight pixel chords can cut across sharp joins. Those examples do not establish a shortcut and are retained as negative results.

The path module is experimental. Treat its discrete objective as a proposal generator and validate every candidate with dense rendered paths or stronger bounds.

## 7. Precision and validation limits

The original generator returns float32 images. Mathematical piecewise-polynomial statements here concern its underlying overlap formula before that final rounding.

At the baseline state, using a central-difference step of 1e-8 of each knob's range gave about 45.1% relative Jacobian error on float32 images, compared with the analytic derivative. A step of 1e-4 gave about 0.00450% error there. Smaller steps do not necessarily improve an estimate.

The analytic derivative was checked at 100 random states against float64 finite differences; the maximum per-column relative discrepancy was 2.719e-8. A separate literal JavaScript implementation of the original source loops matched all pixels of 100 test images exactly, both before and after float32 rounding.

During the original run, the local shell and Node REPL were unavailable because of an execution-environment mismatch. The numerical work then ran in the JavaScript tool runtime using the committed `grey_ones.py` coverage law, Git blob `d59685738e7303017503b2cd56bac0706c9dcfd9`. In the subsequent local-shell follow-up, the original Python renderer matched all twelve saved fixture images exactly, and a direct float64 implementation rounded to the same images for 400 random settings and all 32 corners. See `GENERATED_GEOMETRY_FOLLOWUP.md` for the additional experiments.

No figure is supplied: the numerical tables and complete arrays are the evidence. No uninspected scientific chart is being presented.

## Reproduction

Run the main checks from the repository root:

```sh
bun claude/generated_geometry_experiments.cjs
python claude/validate_generated_geometry_reference.py
```

The first writes `claude/generated_geometry_reproduced.json`. Both commands ran successfully in the follow-up. The second compared twelve saved float32 fixtures against the actual local Python generator, finding zero unequal pixels.

Files:

- `generated_geometry_core.cjs`: float64 renderer, vector operations, eigensolver, width-height regions and patches.
- `generated_geometry_analytic.cjs`: analytic Jacobian of the coverage formula.
- `generated_geometry_charts.cjs`: exact nine-pattern chart.
- `generated_geometry_experiments.cjs`: main reproducible experiment runner.
- `generated_geometry_paths.cjs`: experimental discrete path optimizer.
- `generated_geometry_results.json`: original results, raw settings, tangent survey, and path nodes.
- `generated_geometry_additional_results.json`: path refinement and individual-knob checks.
- `generated_geometry_reference_fixtures.json`: twelve float32 images for independent local comparison.
- `validate_generated_geometry_reference.py`: local Python comparison.

The next useful experiment would be a small exact-patch viewer that displays one width-height saddle, its pixel-response patterns, and the precise event at an adjoining patch. The full slice should retain its parameter map and measured geometry; forcing all 77 affine directions into an unlabeled 3D picture would undo the distinction established here.

