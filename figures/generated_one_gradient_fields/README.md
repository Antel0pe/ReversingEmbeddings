# All five knob directions are reweightings of two spatial gradients

Built by `analyze_generated_one_gradient_fields.py`. Numbers in `results.json`.

The question: at a given setting, can what a knob does to the image be built out
of ingredients taken from that image alone? If yes, the knobs' behaviour is
readable off a single picture, with no access to the generator and no parameter
labels -- which is the difference between reverse-engineering this family and
having a method.

## The two ingredients and the six fields

From one image, ask two questions of every pixel: how fast would its brightness
change if the whole picture slid sideways, and if it slid downward. Each answer
is itself a 784-number picture, `d/dx` and `d/dy`. Weight each by 1, by how far
right of `(cx, cy)` the pixel sits, and by how far below. Six fields:

```
d/dx      d/dy      x*d/dx      y*d/dx      x*d/dy      y*d/dy
```

Plain sliding moves every pixel alike; stretching moves far pixels more, so
"multiply by how far out" turns a slide into a stretch; leaning moves high
pixels more, so "multiply by how far up" turns a slide into a shear. Six fields
inside a 784-dimensional space is a severe constraint, so a tight fit is
informative rather than automatic. Three controls are below.

This requires `smooth_ones`, not `grey_ones`: coverage images have hard edges and
therefore no spatial derivatives. See `figures/generated_one_crease_origin`.

## 1. The fit, and whether the leftover is real

A knob's effect is measured by nudging it and subtracting. That is slightly
crooked: a before-and-after difference across a finite nudge carries an error
which theory says shrinks as the SQUARE of the nudge. So the leftover is
re-measured at four nudge sizes. Median over 25 settings spanning the default
box, as a fraction of the knob's own effect:

| Nudge | cx | cy | height | width | lean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1e-02 | 9.4e-03 | 7.1e-03 | 3.8e-03 | 2.9e-04 | 1.6e-04 |
| 1e-03 | 9.5e-05 | 7.1e-05 | 3.8e-05 | 2.9e-06 | 1.6e-06 |
| 1e-04 | 9.4e-07 | 7.1e-07 | 3.8e-07 | 2.9e-08 | 1.3e-08 |
| 1e-05 | 5.7e-09 | 4.2e-09 | 2.1e-09 | 4.4e-09 | 3.6e-09 |

A hundredfold smaller nudge shrinks the leftover by 9943x, 9964x, 10007x,
10122x and 12344x respectively, against 10000x expected if it is nudge-squared
measurement error. The last row flattens near 3e-09, the floating-point floor
where the arithmetic runs out of digits.

A real missing ingredient would have sat still while the nudge shrank. This one
tracked the nudge all the way down and then hit the machine. **There is no
leftover.** The six fields account for the five knobs completely.

## 2. Which fields each knob uses

Mean weight on each field, relative to that fit's largest weight:

| | d/dx | d/dy | x*d/dx | y*d/dx | x*d/dy | y*d/dy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cx | **1.0000** | 0 | 0 | 0 | 0 | 0 |
| cy | 0 | **1.0000** | 0 | 0 | 0 | 0 |
| height | 0 | 0 | 0 | 0.2484 | 0 | **1.0000** |
| width | 0 | 0 | **1.0000** | 0.2484 | 0 | 0 |
| lean | 0 | 0 | 0 | **1.0000** | 0 | 0 |

Writing `s = tan(lean)`, the warp algebra gives the closed forms exactly:

| Knob | Direction in pixel space |
| --- | --- |
| cx | `-d/dx` |
| cy | `-d/dy` |
| lean (via s) | `(y - cy) * d/dx` |
| height | `-((y - cy) / height) * (d/dy - s * d/dx)` |
| width | `-(((x - cx) + s * (y - cy)) / width) * d/dx` |

Height and width each carry a shear-coupling term, which is why both load
0.2484 on `y*d/dx`. That number is not arbitrary: in both rows the ratio of the
two weights is exactly `|s|`, and the mean of `|tan(lean)|` over the default
lean range of -10 to +35 degrees is about 0.25. The table is internally
consistent with the formulas.

The two pure identities are exact to the limit of measurement: `d(image)/d(cx)`
against `-d/dx` and `d(image)/d(cy)` against `-d/dy` both give cosine
1.00000000, median and worst case over all 25 settings.

## 3. The unused sixth field, and why there are five knobs

`x*d/dy` -- the downward gradient weighted by how far right a pixel sits -- is
vertical shear: tipping the 1 like a seesaw instead of shearing its top across.
This family cannot do that. The largest weight any knob puts on it is
**6.7e-07**, and dropping it from the fit costs nothing at all (bottom table).

That answers "why five?" structurally rather than empirically. Two spatial
gradients, three position weightings, six ways to build a motion. A general
affine map of the plane has six parameters. This generator uses five of them.

## 4. Controls

**Borrowed fields.** Fit a setting's knob direction using the six fields built
from a different image: median residual **0.8597**, against 3.8e-07 with the
image's own fields. The fields have to belong to the image.

**Dropped field.** Residual left when one field is removed, mean over settings:

| | d/dx | d/dy | x*d/dx | y*d/dx | x*d/dy | y*d/dy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cx | 0.8101 | 0 | 0 | 0 | 0 | 0 |
| cy | 0 | 0.8086 | 0 | 0 | 0 | 0 |
| height | 0 | 0 | 0 | 0.1296 | 0 | 0.4588 |
| width | 0 | 0 | 0.6492 | 0.2784 | 0 | 0 |
| lean | 0 | 0 | 0 | 0.5546 | 0 | 0 |

Each knob breaks when and only when a field it actually uses is removed, and the
`x*d/dy` column is zero throughout. The fit is not absorbing errors elsewhere.

## What this does not show

This is the decision-free family of `smooth_ones`, which differs from the
coverage renderer by about 15% relative RMS over the box. The result is a
statement about the five-knob affine family, not about coverage images, whose
hard edges have no spatial derivatives to build fields from.

It also assumes the right field library. Six affine fields were supplied, not
discovered; the result confirms that this family is an affine orbit, it does not
derive "affine" from the data. Doing that means fitting observed image-to-image
differences against a larger library and seeing which fields survive -- the
same machinery, run without being told the answer.
