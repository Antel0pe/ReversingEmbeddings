# Fitting a line through perpendicular points

`analyze_perpendicular_points_fit.py` · `analyze_perpendicular_points_grid.py` ·
output in `results.txt` and `grid_fineness.txt` · figure `in_between_points.png`

Four generated 1s at `cx` = 4, 10, 16, 22 share no pixel, so they are exactly
perpendicular and all six distances are equal (10.31). Fits use those four images
and their `cx` values. Misses are percent of the true image; a blank canvas is 100%.

| Fit, held-out `cx` in between | Average miss |
| --- | --- |
| stairs (nearest image) | 83.6% |
| straight lines between points | 78.1% |
| smooth curve per pixel | 87.5% |
| Gaussian kernel | 83.3% |
| neural net (train loss 4e-6) | 80.5% |
| best possible mix of the four images | 72.1% (98.2% at midpoints) |
| same smooth curve through ink positions | 13.9% (17.7% between pixels) |
| one shape slid at a fitted speed | 8.9% (11.3% between pixels) |
| either of the last two, pixels scrambled | 100.9%, 102.0% |

Ordinary fits return mixes of the training images, and the true in-between is
nearly perpendicular to every one of them. Scrambling keeps all distances and
angles, so the information these fits use lies in which pixels are neighbours.

Other results: a straight-line image-to-`cx` rule fitted on the four is exact on
them and misses by 4.74 px held out, while centre of ink misses by 0.008 px with
nothing fitted. Squared distance stops growing at 4 px (106.25). Blending needs
points about 0.25 px apart (1.3% miss). Blur 4 px gives a 12.0% fit but is 79.0%
from the sharp image. Lean: 24.7% per pixel, 10.1% on ink positions.

Five knobs, 1000 random examples, same third-degree formula: pixels 22.2%, ink
positions 11.0%. Giving the fit true positions leaves 2.02%, so the stall on the
28x28 grid is in the positions. On a 112x112 grid with 256 marks the total is
6.87%, with 4.28% left when only ink per row is fitted: rows do not handle the
stroke's top and bottom ends. Real MNIST was not tested.
