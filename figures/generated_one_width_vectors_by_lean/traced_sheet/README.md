# The sheet traced by the two curve families

Run `OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/draw_traced_width_lean_sheet.py` after the preceding sheet-view experiment.

The main object is the connected grid of actual rendered images. Blue curves vary width at fixed lean; orange curves vary lean at fixed width.
The 1,334 images span 29 widths (1.8-4.6 px) by 46 leans (-10 to 35 degrees), with center (14,14) and height 19.75 px fixed.
Every source image is checked against `grey_ones.render`. No PCA or pixel-based coloring is used for the main drawing.

The global-distance view reuses the preceding metric MDS fit to all raw image-pair distances.
Two new fits use those raw pair distances, relative short-path distances (immediate edges, cell diagonals, and two-tick path chords), actual width/lean crossing cosines, and triangle areas measured from 784D edge Gram matrices.
The objective has an analytic gradient checked against 12 central finite differences before fitting.
Each new fit has a 1,000-iteration limit; `results.json` records convergence and the objective weights. Finite optimization and initialization do not guarantee a global optimum.
An area regularizer discourages collapsed and inverted cells; residual orientation reversals are still counted and drawn red. The count does not detect all global self-overlap.

All source and drawn local crossing angles use the same available +0.1 px width and +1 degree lean intervals.
At upper boundaries the last increasing interval is translated to the same arrow origin for comparison. These are finite differences, not infinitesimal derivatives.
The two actual 784D vectors can be represented exactly as a pair of 2D arrows using their norms and angle. This local two-vector drawing does not claim the surrounding sheet is planar.
Sampled curve length sums the original consecutive image distances and compares them with the same curve's drawn segment lengths.
Curve bending is checked separately: median and 95th-percentile errors compare the angles between consecutive segments of each actual and drawn path. Small crossing-angle errors do not imply that the curve bends are preserved.
Map units use the original Euclidean coverage-distance scale. Source knob values label the paths, while fitted screen positions are approximate.

The old primary pixel inspector is preserved as `original_coordinates/pixel_inspector.html`. Its previous main URL now shows this geometric mesh.
The old inspector generator can overwrite that URL; rerun this generator to restore the main mesh if needed.
`traced_sheet_data.npz` preserves the original sampled images and all new display coordinates. No continuous decoder or exact lossless flattening is claimed.
