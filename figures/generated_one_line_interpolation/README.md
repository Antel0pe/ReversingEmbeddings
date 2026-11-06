# Five lines from the root 1: do they add up, and how many do you need?

Scripts: `analyze_generated_one_line_interpolation.py` (eight parts, runnable
separately), `make_generated_one_line_interpolation_figures.py`.
Full console output: `run_log.txt`. Numbers: `results.json`.

The question. Start at the **root 1** `(14.0, 14.0, 19.75, 3.2, 0.0)`. Hold four
dials still and turn the fifth up by **+1 of its own unit** (px, px, px, px,
degree). That gives five *lines* in pixel space: `line_k = render(root + e_k) -
render(root)`, each itself a 28x28 picture. If I add two lines, do I land on the
image that really has both dials changed? How many lines would it take to reach
anywhere in the family?

Every error below is a length divided by a length, in percent. Two normalisers
are used and always named: **% of the move** divides by `|true - root|`, so 0%
means the sum landed exactly and 100% means it was as wrong as not moving at
all; **% of the image** divides by `|true|`, the repo's usual decoder error.

## Short answers

| Question | Answer |
| --- | --- |
| Do two lines add to the joint change at +1? | No. 11-61% of the move, depending on the pair. All five at once: 63%. |
| Is that a fixed tax? | No. The miss is exactly second order (log-log slope **2.001**). At 5% of a unit it is 1.5% of the move; at 0.2% of a unit, 0.06%. |
| Do the five lines span the family? | Not remotely. A typical setting sits **80.7%** (median) of its own distance-from-root away from the 5-line flat. |
| Best possible five directions? | Still 33.1%. Five lines of any kind cannot do it. |
| Lines plus products of the dials? | 6 lines 40.0%, 21 lines 27.2%, 56 lines 19.6%, 126 lines 15.8% held out. Converges badly. |
| Best possible fixed pictures (PCA ceiling)? | 48 for 5%, **118 for 1%**, 196 for 0.1%. |
| Patches of five fresh lines? | Each patch is good to +-1.2% of each range; tiling the box needs ~1.3e8 patches, ~6.5e8 lines. |
| Is there a basis where five lines DO add up? | Yes: **warp space**, with `height * tan(lean)` as the fifth dial. Residual 5e-13%. |

## Part 1 - the five lines

| line | step | that step as % of range | length | as % of the root image |
| --- | --- | --- | --- | --- |
| cx | +1 px | 100% | 4.507 | 61.8% |
| cy | +1 px | 100% | 2.062 | 28.3% |
| height | +1 px | 67% | 0.922 | 12.7% |
| width | +1 px | 36% | 2.577 | 35.4% |
| lean | +1 deg | 2.2% | 0.615 | 8.4% |

"+1 on each dial" is not an even-handed step: it is the whole of `cx`'s MNIST
range and 2.2% of `lean`'s. That is why the misses below are large.

The five are genuinely independent (singular values 4.509, 2.585, 2.075, 0.853,
0.615; smallest/largest 0.136). Pairwise cosines are small; the largest is
cy-height at 0.335. At the root the **lean line is exactly orthogonal** to the
other four (cosines 0.000 to 1e-16): at `lean = 0` the stroke pivots about `cy`,
so the ink it gains above the centre it loses below, and that pattern is
orthogonal to every up-down-symmetric change.

## Part 2 - adding two lines

Two readings of "interpolate", both measured as % of the move:

- **corner** (add the moves): predict `render(root + e_i + e_j)` by `I0 + line_i + line_j`
- **midpoint** (average the images): predict `render(root + .5 e_i + .5 e_j)` by `(img_i + img_j)/2`

| pair | corner | midpoint | | pair | corner | midpoint |
| --- | --- | --- | --- | --- | --- | --- |
| cx+width | **60.5%** | 43.4% | | cx+lean | 19.1% | 62.3% |
| cy+height | 55.8% | 34.2% | | cy+lean | 14.2% | 20.2% |
| width+lean | 30.1% | 27.8% | | height+lean | 12.4% | 40.0% |
| cx+cy | 26.2% | 59.5% | | cx+height | 12.3% | 62.2% |
| cy+width | 21.4% | 28.5% | | height+width | **11.3%** | 30.8% |

All five dials at +1 together: **63.4%** of the move.

The worst pair, `cx+width`, is the one with the clearest reason: moving the
stroke right by a pixel and widening it by a pixel both rewrite the same two
edge columns, and the second change happens to ink the first change has already
moved. The best, `height+width`, touches two nearly disjoint sets of pixels (the
end caps and the side columns).

## Part 3 - the miss is second order, not a constant tax

Move all five dials by `t` units and remeasure the lines at that step:

| t | move length | miss, % of image | miss, % of the move |
| --- | --- | --- | --- |
| 2.0 | 8.732 | 108.4% | 90.5% |
| 1.0 | 5.716 | 49.7% | 63.4% |
| 0.25 | 1.829 | 4.43% | 17.6% |
| 0.05 | 0.379 | 0.077% | 1.49% |
| 0.01 | 0.076 | 0.0031% | 0.298% |
| 0.002 | 0.015 | 0.0001% | 0.059% |

Log-log slope of the absolute miss over t in [0.002, 0.05]: **2.001**. Over the
whole range including the big steps: 2.22, because steps of 0.25 and above are
past the local regime (and `t = 0.5` sits visibly above the power law, which is
a renderer crease, not noise). So the five lines are an exact local answer and a
bad global one, and the crossover is somewhere around a tenth of a unit.

## Part 4 - the five lines do not span the family

3000 settings drawn uniformly from the default range box; distance from the flat
through the root spanned by the five lines, as % of the move from the root:
median **80.7%**, p95 94.2%, max 95.4%. Pooled residual 85.6%.

Replacing the knob lines with the five *best* directions through the root (SVD
of the same 3000 moves) only gets to **33.1%**. Adding directions: 10 -> 24.5%,
20 -> 16.4%, 40 -> 8.1%, 70 -> 3.7%, 100 -> 1.9%. This is a property of the
family, not of the choice of lines.

## Part 5 - products of lines, and the PCA ceiling

Least squares from a polynomial in the five dials to the pixels, 4000 training /
2000 held-out settings, median error as % of the image:

| model | basis pictures | train | held out |
| --- | --- | --- | --- |
| degree 1 (the five lines + a mean) | 6 | 39.8% | **40.0%** |
| + all pair products | 21 | 27.1% | 27.2% |
| + all triple products | 56 | 19.4% | 19.6% |
| + all quadruple products | 126 | 15.3% | 15.8% |

For comparison, the best any *k* fixed pictures can do, with the *k* numbers read
off the target image instead of computed from the dials (an unbeatable ceiling):
5 -> 25.9%, 20 -> 13.6%, 40 -> 6.6%, 70 -> 2.9%, 100 -> 1.4%; **118 pictures for
1%**, 196 for 0.1%. Even that ceiling needs over a hundred, so the honest answer
to "how many lines" in pixel space is *about a hundred, and the five dials cannot
drive them*.

## Part 6 - patches of five fresh lines

A patch is a centre plus five lines measured there with step `h` (a fraction of
each dial's range); points are drawn uniformly in the +-h box around the centre.
95th-percentile error, % of the image, at five centres:

| h | the root | four other centres | pooled |
| --- | --- | --- | --- |
| 0.125 | 23.7% | 37-59% | 45.2% |
| 0.031 | 0.167% | 4.7-6.5% | 5.38% |
| 0.0078 | 0.010% | 0.33-0.70% | 0.46% |
| 0.00098 | 0.0001% | 0.007-0.017% | 0.010% |

Fitted power law `error = 4004% * h^1.87` (close to the second-order 2). To stay
under 1% a patch may be **+-1.2% of each range**, i.e. 42 patches per dial, so
tiling the full 5-dial box takes **1.3e8 patches = 6.5e8 lines**. At 5% it is 1.8
million patches; at 0.1%, 6.1e10. Five dimensions is what makes this hopeless,
not the amount of curvature.

The root is an unusually flat spot - about 40x better than the other centres at
the same `h`. It sits at `lean = 0`, where `tan` has no curvature, and at integer
`cx`, `cy`. Do not generalise from the root alone; that is why four random
centres are in every row.

## Part 7 - one dial at a time

Sweep one dial across its whole default range, others at the root, and ask how
well a single line holds the curve (error as % of the image):

| dial | range | single line | 2 lines | 3 lines | lines for 1% |
| --- | --- | --- | --- | --- | --- |
| cx | 14 to 15 | 16.3% | 2.3% | 0.00% | 3 |
| cy | 14 to 15 | 2.8% | 0.5% | 0.00% | 2 |
| height | 19 to 20.5 | 5.0% | 0.00% | 0.00% | 2 |
| width | 1.8 to 4.6 | 7.3% | 1.2% | 0.00% | 3 |
| lean | -10 to 35 | **74.1%** | 23.4% | 15.7% | **13** |

Lean is the troublemaker, and by a wide margin: one line keeps only a quarter of
its own sweep. Four dials are nearly straight over their MNIST ranges; the
leaning one is not, because the stroke sweeps across different pixels entirely.
A naive tensor guess from this table (3 x 2 x 2 x 3 x 13 = 468) is far above the
118 the family actually needs, so the family is not a product of its dials either.

## Part 8 - the five lines that do add up

Every setting is the root stroke pushed through one affine map of the canvas,
with no vertical shear:

    x = bx + sx*X + a12*Y        sx = width/3.2,  a12 = -tan(lean)*height/19.75
    y = by + sy*Y                sy = height/19.75

Fit five lines plus a constant from the dials to those five warp numbers over
2000 settings (relative residual):

| the five dials used | residual |
| --- | --- |
| `cx, cy, height, width, lean` as given | 3.86% |
| `cx, cy, height, width, height*tan(lean)` | **5.3e-13%** (machine zero) |

With the fifth dial written as `height * tan(lean)` the five warp lines add
**exactly**: the family is a genuinely flat 5-dimensional plane there, and
"line i + line j" lands on the corner with no error at all. The only nonlinearity
in the whole family was the `tan`, plus the fact that leaning tilts a stroke
whose length is itself a dial.

Turning those exact five numbers back into 28x28 pixels costs something, but it
is the template's cost, not the lines'. Warping a root template of pixel pitch
`p` and area-averaging back to 28x28:

| template | pitch | median error |
| --- | --- | --- |
| 28x28 | 1 px | 15.7% |
| 56x56 | 0.5 px | 3.74% |
| 112x112 | 0.25 px | 1.70% |
| 224x224 | 0.125 px | 0.27% |
| 448x448 | 0.0625 px | 0.11% |

The five warp lines are identical in every row; only the template changes. This
reproduces `figures/generated_one_warp_space` (flat to machine zero past the
fifth component) from the other direction, in closed form rather than by fitting,
and matches `figures/generated_one_moment_chart`'s finding that a warped-template
decoder is limited by resampling.

## Figures

- `lines_do_not_add.png` - the root, the five lines, the worst pair's corner test, and the second-order scaling.
- `how_many_lines.png` - lines needed in pixel space, patch tiling, and the warp-space result with its template cost.

## What this does not show

- Only the generated family. Real MNIST 1s are not affine warps of one stroke;
  see `figures/mnist_one_knob_search`.
- "Lines" here always means fixed pictures added with coefficients. Nothing is
  claimed about nonlinear decoders.
- Part 5's PCA ceiling needs its coefficients measured from the target image. It
  is a bound on representation, not a usable map from the dials.
- Part 8's exactness is a statement about warp coefficients. Pixels are recovered
  by resampling, which is separately measured and is not free at 28x28.
