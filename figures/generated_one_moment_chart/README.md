# Five knobs from six linear measurements, in closed form

Built by `analyze_generated_one_moment_chart.py`. Numbers in `results.json`.

`figures/linear_5d_to_784d` ruled out linearity on the decoder side: the best
five-dimensional affine subspace holds only 80% of the image variance, so no map
whose last stage is linear in five numbers can describe this family. This run
puts the linearity on the **encoder** side instead, where it survives intact.

## The formula

Six fixed linear projections of the 784 coverage values -- total ink, and the
ink-weighted averages of x, y, x*x, y*y and x*y -- determine all five knobs
algebraically, with nothing fitted and no knowledge of the generator:

```
cx        = mean x
cy        = mean y
height    = sqrt(12 * var y)
tan(lean) = -cov(x, y) / var y
width     = sqrt(12 * (var x - tan(lean)^2 * var y))
```

The parallelogram is symmetric about its centre, so its centroid is exactly
`(cx, cy)`. Its vertical spread fixes the height; the tilt of its ink cloud
fixes the lean; whatever horizontal spread the lean does not account for is the
width. Variances are corrected by `pitch^2 / 12`, the variance of a pixel box
itself, which is the leading grid artefact.

## 1. Recovery over the default knob box

4000 random settings, 28x28, absolute error in each knob's own units:

| | cx | cy | height | width | lean |
| --- | ---: | ---: | ---: | ---: | ---: |
| worst | 0.04173 | 0.00637 | 0.05135 | 0.32849 | 0.30441 |
| median | 0.00040 | 0.00173 | 0.01211 | 0.00277 | 0.01504 |

One identity holds to machine precision rather than approximately: total ink
equals `width * height`, worst case **4.3e-14** over all 4000 settings. That is
how we know the renderer integrates area exactly and the grid is the only
approximation in play.

## 2. The residual is the ruler, not the method

Same closed form, same settings, finer grids over the same 0-to-28 canvas:

| Grid | Pixel pitch | cx | cy | height | width | lean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 28x28 | 1.000 | 0.03203 | 0.00594 | 0.04538 | 0.18651 | 0.23313 |
| 56x56 | 0.500 | 0.00862 | 0.00160 | 0.01111 | 0.01862 | 0.06967 |
| 112x112 | 0.250 | 0.00145 | 0.00039 | 0.00271 | 0.00446 | 0.00883 |
| 224x224 | 0.125 | 0.00017 | 0.00010 | 0.00077 | 0.00108 | 0.00274 |

Over an eightfold refinement the worst-case errors shrink 192x, 61x, 59x, 173x
and 85x. Pitch-squared predicts 64x. These are worst-of-200 statistics and so
noisy, but all five are in the pitch-squared regime or better, and nothing
plateaus. The formula is exact for the continuous shape; the error is the pixel
grid, and it is removable.

Note the contrast with `figures/generated_one_crease_origin`, where the same
refinement makes the pixel-space path *more* crumpled and its speed diverge as
one over the square root of the pitch. A well-behaved observable can converge
while the embedding it is computed from gets stiffer. Both are true.

## 3. The linear ceiling, for contrast

Best possible reconstruction of held-out images from a fixed linear subspace
fitted on 4000 training images, scored on 1500 held out:

| Rank | Variation left over |
| ---: | ---: |
| 5 | 44.49% |
| 10 | 33.38% |
| 20 | 22.54% |
| 50 | 8.44% |

Five flat directions in pixel space do not come close, and no choice of five
would, which is the `linear_5d_to_784d` result restated on this sample.

## 4. A decoder: warp one image

Recover the five knobs from a held-out image's moments, build the affine map
they imply, and apply it to a **single** stored picture, root 1:

| Decoder | Relative RMS |
| --- | ---: |
| recovered knobs, linear resampling | 16.76% |
| recovered knobs, cubic resampling | 13.65% |
| TRUE knobs, cubic (control) | **13.61%** |
| rank-5 linear subspace, for comparison | 44.49% |

The control is the point. Driving the same warp with the true knobs scores
13.61% against the recovered knobs' 13.65%, so essentially none of that error is
recovery error -- it is the cost of resampling a 28-pixel-wide picture by
interpolation. The encoder is not the bottleneck; the 28x28 grid is, again.

## What this does not show

The formula is specific to this shape. Moments invert this cleanly because the
family is an affine orbit of one convex, centrally symmetric parallelogram. Real
MNIST 1s are not that, and the same five moments would not invert for them. What
generalises is the shape of the method -- fixed linear measurements, then an
algebraic inversion, then a warped template as decoder -- not these five lines.

The affine structure was also assumed, not discovered. See the closing note in
`figures/generated_one_gradient_fields`.
