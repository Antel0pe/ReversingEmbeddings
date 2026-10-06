# Directly constructed knob curves

Assumption: the position on curve k is the actual value of knob k, normalized to its allowed range. The decoder is a fitted base plus five independently indexed pixel-space curves. Curve positions are not allowed to mix knobs.

## Same-cloud comparison

| Nodes per knob curve | Globally fitted fixed-coordinate capture |
| ---: | ---: |
| 65 | 80.464404% |
| 129 | 80.735084% |
| 257 | 81.613453% |
| 513 | 83.445548% |

The freely fitted five-curve model with 513 nodes captures 99.733620% on the same 14,389 images. Direct 513-node curves trail it by 16.288071 percentage points. The direct solve is linear least squares with a tiny numerical ridge; it does not suffer from curve-initialization local minima.

## Global additive optimum on complete grids

For every value of each knob, average the images across all combinations of the other four knobs. Subtract the overall mean. Those five lists of pixel-valued effects are the globally best additive curves on an equally weighted Cartesian grid. No iterative curve search or imposed smoothness is needed. Interpolating between those lists produces ordinary polylines.

Because the grid is a product distribution, the five centered effects are mutually orthogonal. The unexplained residual is orthogonal to every single-knob function. This proves optimality over all additive knob functions on the grid, not just the chosen spline basis.

| Values per knob | All joint images | Global optimal capture |
| ---: | ---: | ---: |
| 5 | 3,125 | 81.725790% |
| 9 | 59,049 | 81.479103% |
| 13 | 371,293 | 81.421732% |

Levels are the midpoints of equal intervals across each knob range. The grid optima are exact for those discrete distributions. Approximately 81.4% is a converging numerical estimate for the continuous independent-uniform box, not an exact universal percentage ceiling. Grid and pooled-cloud percentages use different distributions.

## Why infinitely fine knob curves cannot make the model exact

For any additive model, changing width and lean has zero mixed difference:

```text
model(both) - model(width only) - model(lean only) + model(start) = 0
```

The actual renderer gives mixed-difference L2 3.460099700 and maximum pixel interaction 0.500000030 for width +1 px and lean +15 degrees from the midpoint settings. The triangle inequality forces at least one of these four images to have image L2 reconstruction error at least 0.865024925 in every additive knob-indexed model. This conclusion is independent of node count, curve smoothness, or fitting algorithm.

## Scope of the conclusion

This impossibility result does not cover the learned free-coordinate curves: their positions can change together when a physical knob changes. Their observed scores remain 99.733620% on the fitting cloud and 99.466107% on additional uniform points. Their distance from a perfect score is 0.266380 and 0.533893 percentage points respectively. There is no certified global optimum for the free-coordinate five-curve model.

A finite set can also be memorized by a polyline visiting every image, with the other four contributions zero. That establishes only finite-data feasibility and does not establish a useful smooth five-coordinate representation of the continuous manifold.

Raw results: results.json. Constructed product-grid nodes: product_5.npz, product_9.npz, product_13.npz. Comparison: direct_comparison.png.
