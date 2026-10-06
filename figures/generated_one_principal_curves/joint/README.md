# Five curves optimized together

The goal is minimum combined squared pixel reconstruction error with exactly five learned additive curves. Knob alignment is not required.

Each curve is a list of pixel-space nodes joined by straight interpolation. Reconstruction is a fitted intercept plus the sum of five curve values. No renderer call or physical stroke equation occurs in this decoder.

All 14,389 covering points participate in each fit, with equal weights. This is a finite approximation to the continuous five-knob range box, not every possible image. Scores use the mean of the evaluated cloud.

| Fit | Nodes per curve | Captured variation | Pixel RMS error |
| --- | ---: | ---: | ---: |
| Existing curves, positions refined | 65 | 97.662223% | 0.025803 |
| Joint fitting, 65 nodes | 65 | 98.569521% | 0.020184 |
| Joint fitting, 129 nodes | 129 | 99.182217% | 0.015261 |
| Joint fitting, 257 nodes | 257 | 99.489900% | 0.012053 |
| Joint fitting, 513 nodes | 513 | 99.733620% | 0.008710 |

At the matched 65-node resolution, the first curve alone drops from 90.005947% to 89.676554% capture while the combined five-curve score improves. Solo capture projects the image-minus-intercept onto curve 1 with all other curve contributions zero.


## Fitting choices

At 65 nodes, compare the previous greedy result, a known-knob additive initial fit, and a lean-informed initial path with residual starts. Knob values only choose starting arrangements; all positions are free during subsequent fitting.

Each cycle projects images onto the current segments, fits all five node sets jointly by least squares with a second-difference smoothness penalty, and projects again. Unlike the previous routine, it does not resample nodes to uniform arc length after each update; fractional node index defines interpolation. This avoids a lossy resampling step.

For 65-node fits the penalty schedule is 1, 0.1, 0.01. For higher resolution, insert interpolated nodes into the best previous curves and use 0.01 and 0.001 times the cube of relative node density. Finally polish the 513-node fit with penalties 0.0512 and 0.00512. Keep the lowest full-cloud reconstruction error. Histories record every completed update. The model intercept is learned jointly and curve constants are centered to resolve additive gauge freedom.

The encoder refines closest-point coordinates from two pixel-only starts: sequential projections and the coordinates of the nearest covering image. The generator is used to supply data and initial paths, then to render additional diagnostic points. It is absent from projection, node fitting, and decoding.

## Additional sampling checks

| Additional points | Existing curves | Best joint fit |
| --- | ---: | ---: |
| 16,384 additional uniform points | 97.613558% | 99.466107% |
| All 32 corners | 96.068974% | 99.857723% |
| 2,048 joint midpoints | 97.159178% | 99.388135% |

No fitting split is reserved. These additional points diagnose finite sampling resolution and were not used to choose the winning fit. All corners are also part of the covering lattice.

## Limits

Best found in the reported search; no certificate of a global optimum. More nodes increase shape freedom, so higher-resolution results are not same-complexity comparisons. Five additive curves remain a restricted decoder. High sampled capture is not an exact representation or a guarantee that arbitrary coordinate combinations yield valid strokes. Pixels outside [0, 1] are not clipped for scoring and are shown in magenta.

Use index.html for image and curve controls, results.json for measurements, curve_nodes.csv and intercept_pixels.csv for the literal best decoder. Preview nodes are rounded to six decimals for page size; fitting, scoring, and exported curves retain full precision. Run optimize_generated_one_joint_curves.py with phases 65, 129, 257, 513, polish, evaluate, then make_generated_one_joint_curve_view.py from the repository root.
