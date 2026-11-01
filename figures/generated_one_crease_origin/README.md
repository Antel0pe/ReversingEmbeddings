# Where the creases come from, and why a finer grid makes them worse

`figures/linear_5d_to_784d` established that the exact decoder for the generated
1s is piecewise, and pointed at the `clip/min/max` coverage calculation as the
source. This run answers the next two questions: what exactly triggers each
crease, and whether refining the pixel grid smooths the path through pixel space.

Built by `analyze_generated_one_crease_origin.py`. Numbers in `results.json`.

## How a crease is detected

Creases are read off the DIRECTION of travel in pixel space, not off any single
pixel. Each step of a sweep gives a move vector in 784 dimensions; normalise it,
then measure how hard the direction kicks between consecutive steps. This is
basis-free and ignores how fast the path is moving. A step whose kick exceeds
0.01 is counted as a crease.

All sweeps vary `lean`, which moves both stroke edges across the most pixels,
starting from `(cx, cy, height, width, lean) = (14.4, 14.3, 19.9, 3.1, 8.0)` and
covering 0.2 degrees in 4001 steps, so one step is 5e-05 degrees.

## 1. Every crease is a decision flipping

A pixel's coverage is `clip(min(stroke_right, pixel_right) - max(stroke_left,
pixel_left), 0, 1)`. Each `min`, `max` and `clip` is a decision -- "whichever of
these two is smaller" -- and a decision has a corner. So a crease should occur
exactly when a stroke edge crosses a pixel-column boundary at some sub-row.
Those lean angles solve in closed form, with nothing rendered:

```
centre(y) = cx - tan(lean) * (y - cy)
edge at centre -/+ width/2 meets column boundary c  when
    tan(lean) = (cx -/+ width/2 - c) / (y - cy)
```

| | count |
| --- | ---: |
| decision flips predicted in the window, no rendering | 12 |
| creases actually observed | 12 |
| observed creases within one step of a prediction | 12 of 12 |

Worst gap between an observed crease and a predicted angle: **0.37 steps**, i.e.
2e-05 degrees. The mechanism is settled: creases are decision flips, and their
locations are computable in advance.

## 2. It is not pixels switching on

The intuition that a crease is a pixel going from zero to non-zero is wrong. The
single worst crease over all 784 pixels sits at pixel (row 6, col 14), whose
value **never leaves the range 0.992646 to 0.999037**. It is essentially
saturated with ink throughout, and never exactly zero anywhere in the sweep. Its
rate of change goes

```
-0.0183   -0.0183   -0.0183   -0.0193   -0.0238   -0.0274   -0.0274
```

a lurch of 101% of a typical rate inside one 5e-05 degree step. That crease is
`min(..., 1)` saturating -- the pixel running out of room to get any more inked
-- not `max(..., 0)` switching on. Being non-zero is no protection, and being
zero is not the cause: `clip` has a corner at both ends.

## 3. Refining the grid makes the embedding worse

| Grid | Creases | Total direction change | Largest kick | Path speed |
| --- | ---: | ---: | ---: | ---: |
| 14x14 | 3 | 0.0799 | 0.0261 | 0.431 |
| 28x28 | 12 | 0.2154 | 0.0164 | 0.613 |
| 56x56 | 17 | 0.4696 | 0.0166 | 0.874 |
| 112x112 | 24 | 1.5018 | 0.0119 | 1.258 |

Creases get more numerous, each individual kick does **not** shrink, and the
total crumpling of the path grows about sevenfold over an eightfold refinement.
Path speed -- the length of the move per degree of lean, in units that do not
depend on how many pixels there are -- grows by a factor of about 1.43 per
halving of the pixel pitch, which is one over the square root of the pitch. It
diverges. In the sharp-edged limit the map from knobs to image is not
differentiable at all.

This is the opposite of what the moment recovery does under the same refinement
(`figures/generated_one_moment_chart`, where error falls as pitch squared). The
two are not in conflict: a well-behaved observable can converge while the
embedding it is computed from gets stiffer.

## 4. A family with no decisions in it

`smooth_ones.py` renders the same five knobs with the hard edges replaced by
sigmoid ramps. It contains no `min`, no `max`, no `clip`, no branch: a pixel's
value is a product of exponentials of smooth functions of the knobs, so
derivatives of every order exist by construction rather than by approximation.
On the identical sweep:

| | creases | total direction change | largest kick |
| --- | ---: | ---: | ---: |
| coverage, 28x28 | 12 | 0.2154 | 1.6e-02 |
| decision-free, 28x28 | **0** | 0.0370 | **9.4e-06** |

Cost of the stand-in, stated plainly: it is not the coverage renderer. Over the
default knob box it differs by **14.66% relative RMS**, with ink correlation
0.9905. It is a smoothed model of the same five-knob family, not the same pixels.
What it buys is in `figures/generated_one_gradient_fields`: spatial derivatives
exist, and all five knob directions become reweightings of just two of them.

## What this does not show

The crease census is one sweep of one knob from one starting setting, chosen to
be generic. It establishes the mechanism and the direction of the trend with
grid fineness; it is not a count of creases over the whole box. The smooth
family's zero creases follow from its formula, so that row is a check on the
detector, not a discovery.
