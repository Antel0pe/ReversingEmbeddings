# The data picks its own five motions out of a generic library

Built by `analyze_generated_one_motion_discovery.py`. Numbers in `results.json`.

`figures/generated_one_gradient_fields` showed that every knob direction lives in
the six affine fields built from an image's own two spatial gradients. But that
experiment was handed the affine library. "The generator slides, stretches and
shears the plane" is most of the answer, so what it demonstrated was that the
affine hypothesis fits -- not a method someone could point at an unfamiliar
generator.

This experiment removes the hint. The library is **twelve** generic motions, and
nothing marks which are usable.

## The library, and what is withheld

Take an image's two spatial gradients, `d/dx` and `d/dy`, and weight each by six
position functions: `1`, `x`, `y`, `x^2`, `xy`, `y^2`. Twelve candidate motions.
Every polynomial warp of the plane up to quadratic is in there -- plain slides,
stretches, both shears, and six ways to bend.

A five-knob affine family with no vertical shear can only use five of them:

```
1*d/dx (cx)   x*d/dx (width)   y*d/dx (lean)   1*d/dy (cy)   y*d/dy (height)
```

That is the prediction under test, never an input. The seven unusable motions are
`x*d/dy` (vertical shear, which this family cannot do) and all six quadratics.

Position weights are measured from the **canvas** centre, not the stroke's, which
keeps the library label-free -- it needs no `(cx, cy)`. Moving the origin only
mixes `1*d/dx` with `x*d/dx` and so on, which stays inside the same span, so the
prediction is unaffected.

The analysis sees images and gradients computed from those images. Knob values
generate the dataset and are never used to fit or to score.

## 1. Forward selection, and where it stops helping

Add whichever single motion most reduces the median leftover on a neighbour
difference; repeat. 250 anchors, 8 close neighbours each, step scale 0.002 of
each knob's range. `smooth_ones`, exact gradients:

| Step | Motion chosen | Leftover | Predicted? |
| ---: | --- | ---: | --- |
| 1 | `y*d/dx` | 46.554% | yes (lean) |
| 2 | `x*d/dx` | 26.012% | yes (width) |
| 3 | `1*d/dx` | 9.753% | yes (cx) |
| 4 | `1*d/dy` | 4.744% | yes (cy) |
| 5 | `y*d/dy` | **0.667%** | yes (height) |
| 6 | `x^2*d/dy` | 0.666% | no |
| 7 | `y^2*d/dx` | 0.661% | no |
| 8 | `xy*d/dy` | 0.652% | no |

All five predicted motions are chosen first, then the curve plateaus. Rounds six
through eight buy 0.015 percentage points between them. Five alone leave 0.667%;
all twelve leave 0.629%, so the other seven are worth 0.04 points.

The count comes out of the data. Nothing supplied the number five.

## 2. The fit is exact in the limit, not just close

The leftover is measured by fitting a first-order model to a finite difference,
which carries an error proportional to the step. Shrink the steps:

| Step scale | Leftover, five predicted motions |
| ---: | ---: |
| 0.02 | 6.5918% |
| 0.002 | 0.6593% |
| 0.0002 | 0.0659% |

Ten times smaller step, ten times smaller leftover, twice over. That is ordinary
second-order error, so the five-motion model is exactly right rather than a good
approximation. A real missing ingredient would sit still as the step shrank.

## 3. Controls

`smooth_ones`, exact gradients, same pairs:

| Control | Leftover | Reading |
| --- | ---: | --- |
| five predicted motions | 0.667% | the result |
| drop the lean motion (`y*d/dx`) | 65.882% | the library cannot lose a member |
| motions from a **different** image | 90.947% | the fields must be the image's own |
| five random pictures | 99.728% | five directions in 784 explain nothing by chance |
| no density: nearest of a scattered sample, all twelve | 18.192% | see below |

## 4. Density is a real requirement of the method

The last control is a genuine limit, not a nuisance. Neighbour differences are
fitted to first order, so the pairs have to be close. Draw the sample scattered
across the box and take each image's nearest neighbours by pixel distance, and
those neighbours are far enough apart that even all twelve motions leave 18.192%.
The first version of this experiment failed exactly here.

So the method needs a dataset that is dense enough for local differences to be
small -- which a real collection of images generally is, but a sparse sweep is
not.

## 5. On the real renderer the vocabulary still comes out right

Same procedure on `grey_ones` coverage output. Coverage images have hard edges
and therefore no spatial derivatives, so they are blurred (sigma 1.0) and
differenced with `np.gradient`:

| Step | Motion chosen | Leftover | Predicted? |
| ---: | --- | ---: | --- |
| 1 | `y*d/dx` | 51.902% | yes (lean) |
| 2 | `x*d/dx` | 40.123% | yes (width) |
| 3 | `1*d/dx` | 28.785% | yes (cx) |
| 4 | `1*d/dy` | 26.067% | yes (cy) |
| 5 | `y*d/dy` | **23.728%** | yes (height) |
| 6 | `x^2*d/dx` | 23.374% | no |

Again all five first, in nearly the same order, and controls still fire: drop the
lean motion 71.591%, borrowed fields 79.193%, random pictures 99.726%.

But the leftover plateaus near 23.7% and does **not** shrink with the step
(23.432% / 23.133% / 23.213%). A floor.

## 6. The floor is mostly the gradient tooling, not the creases

The tempting reading is that the floor is crease noise, putting a permanent tax
on local analysis of the coverage renderer. It is not. Run the identical
blur-then-`np.gradient` pipeline on `smooth_ones`, which has **zero** creases:

| Pipeline | step 0.02 | step 0.002 | step 0.0002 |
| --- | ---: | ---: | ---: |
| `smooth_ones`, blur 1.0, `np.gradient` | 17.308% | 16.779% | 16.719% |
| `smooth_ones`, blur 2.0, `np.gradient` | 32.769% | 32.818% | 32.904% |
| `grey_ones`, blur 1.0, `np.gradient` | 22.842% | 22.959% | 23.015% |
| `grey_ones`, blur 2.0, `np.gradient` | 34.642% | 34.778% | 34.689% |
| `smooth_ones`, exact gradients | 6.592% | 0.659% | 0.066% |

A crease-free family shows a 16.7% floor under that pipeline. So most of the
coverage floor is the measuring equipment; the creases account for the gap
between 16.7% and 23.0%.

Heavier blur makes it worse for both, and there is a reason: blurring is a linear
smear over the picture, while every motion here multiplies a gradient by a
position weight. Smearing then weighting is not the same as weighting then
smearing, and the mismatch grows with the blur. Blurring to obtain gradients is
self-defeating past about one pixel.

## What this settles, and what it does not

Settled: the motion vocabulary is discoverable from unlabelled images. Given a
generic library, the data selects the right five, counts them as five, and the
model is exact in the limit. The affine assumption no longer has to be supplied,
which is the difference between reverse-engineering this family and having a
method.

Not settled: estimating spatial gradients from a 28x28 hard-edged coverage image.
That is the live obstacle, and it is tooling rather than geometry -- a local
smooth surface fit, or use of the sub-pixel edge information that coverage values
already carry, should move the 23.0% down toward the crease-only share near 17%.

Also unchanged: this family is an affine orbit of one convex template, which is
why a quadratic library suffices. Real MNIST 1s (`OnesManifold.ipynb`) are not,
and a richer library would be needed before any of this transfers to them.
