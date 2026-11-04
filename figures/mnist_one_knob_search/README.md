# Looking for the knobs of real MNIST 1s

Built by `analyze_mnist_one_knob_search.py`, on `warp_charts.py`. Numbers in
`results.json`.

`figures/generated_one_motion_discovery` showed that unlabelled images can name
their own motions and count them, on the generated family. The question here is
the real one: given nothing but MNIST 1s, recover coordinates and an explicit
decoder from those coordinates to 784 pixels.

**The short answer.** The method works and is validated on known truth: on the
generated family it finds five coordinates, confirms five is enough, decodes
held-out images to **1.06%**, and -- after five labelled examples fix a
five-number gauge -- recovers the generator's OWN knobs to about a thousandth of
a pixel. On real MNIST 1s
there is no small knob set to find. Pure geometry saturates near 15%, two
model-free dimension estimators put the count at twelve or more without
converging, and the warp chart's advantage over plain PCA -- large when
coordinates are scarce -- disappears entirely once they are not.

Every MNIST claim below is paired with the generated family as a positive
control, because there the answer is known.

## 1. Why the earlier method cannot be used here

The motion-discovery experiment linearises, so it needs neighbours close enough
to be first order.

| set, 6742 images | nearest neighbour | 8th nearest |
| --- | ---: | ---: |
| generated 1s | 9.1% of the image | 15.3% |
| **real MNIST 1s** | **24.8% of the image** | 30.9% |

A 25% step is nowhere near first order. MNIST 1s are handwriting from thousands
of people and the set is far sparser in its own shape space than a sampled
generator is, so the warp is fitted directly here, by Gauss-Newton, which
handles pairs that are nowhere near each other.

## 2. What limits the decoder is the template, not the model

This capped every early result and is worth stating separately. Warp one
generated 1 onto another using the **exact** affine map between their true
knobs -- no fitting, no model error at all -- and vary only the resolution of
the source:

| source | leftover |
| --- | ---: |
| the 28x28 image | **16.42%** |
| rendered at 56x56 | 6.17% |
| rendered at 112x112 | 1.87% |
| rendered at 224x224 | **1.22%** |

(Each row area-averages 4x4 subsamples out, so the operator is identical
throughout. Bilinear point-sampling from 28x28 gives 14.58%, same conclusion.)

Read this as a reference point, not a bound. It is what the EXACT knobs achieve
under one particular resampling scheme, and a learned template can beat it by
absorbing some of the resampling blur -- which is what happens below, where five
fitted coordinates reach 1.06% with a 224^2 learned template.

So "five coordinates to 784 pixels" was never losing 15% to the model; it was
losing it to not knowing the template's sub-pixel structure. Supersampling the
OUTPUT does not help -- it is slightly worse -- because the missing information
is in the source. Hence `warp_charts` holds the template at 112x112 and
super-resolves it from the data: thousands of images sit at different sub-pixel
offsets, so splatting them all into the canonical frame recovers detail no
single 28x28 image carries. On the generated family that took five-coordinate
decoding from 19.5% to 2.93%.

## 3. Counting the knobs by saturation

Give the warp more and more freedom and watch when it stops paying.

GENERATED 1s (resampling floor 1.26%):

| warp freedom | params | held-out | PCA at same count |
| --- | ---: | ---: | ---: |
| affine, no vertical shear | 5 | **2.93%** | 27.09% |
| full affine | 6 | 2.93% | 25.43% |
| quadratic | 12 | 2.89% | 18.85% |
| cubic | 20 | 2.85% | 13.95% |
| quartic | 30 | 2.82% | 9.61% |
| quintic | 42 | 2.80% | 6.55% |

Flat. Thirty-seven extra parameters buy 0.13 points while PCA improves
fourfold, and five warp coordinates beat a twenty-coordinate PCA by nearly five
times. That is the count appearing as a saturation point, from data alone.

REAL MNIST 1s:

| warp freedom | params | held-out | PCA at same count | gain |
| --- | ---: | ---: | ---: | ---: |
| affine, no vertical shear | 5 | 28.28% | 39.18% | |
| full affine | 6 | 27.79% | 37.62% | +0.49 |
| quadratic | 12 | 21.17% | 31.34% | **+6.62** |
| cubic | 20 | 18.06% | 26.67% | +3.12 |
| quartic | 30 | 16.75% | 23.18% | +1.31 |
| quintic | 42 | 16.07% | 20.26% | +0.68 |

Never flat. Two things to read off it. The sixth parameter helps on MNIST and
bought *nothing* on the generated family -- real 1s do use vertical shear, which
`grey_ones` structurally cannot do, so the method is reporting a real difference
rather than imposing the generator's constraints. And the largest single gain
anywhere is affine to quadratic: **bending**. Real 1s curve, and that, not extra
strokes, is the thing five affine knobs cannot express.

Pushed further, to 90 warp numbers per image:

| degree | params | MNIST warp | PCA at same count | gain |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 42 | 16.07% | 20.26% | +0.68 |
| 6 | 56 | 15.69% | 17.75% | +0.38 |
| 7 | 72 | 15.30% | **15.61%** | +0.40 |
| 8 | 90 | 14.98% | **13.68%** | +0.32 |

An earlier draft of this README extrapolated the halving gains (6.62, 3.12,
1.31, 0.68) to an asymptote near 15%. That was wrong twice over and is
withdrawn. The decay stalls -- 0.38 then 0.40 -- so there is no clean geometric
tail to sum. And more importantly the question is ill-posed out there: **PCA
catches the warp model at 72 parameters and beats it at 90.** Past about forty
parameters the chart is no longer a geometric description of anything; it is a
flexible fit that happens to be parameterised by warps, scoring the same as a
generic linear basis. "Where does geometry saturate" has no answer in a regime
where geometry is no longer what is doing the work.

### Template resolution and round count together

Both were limiting, and both matter more than they look:

| model | template | rounds | coords | held-out |
| --- | ---: | ---: | ---: | ---: |
| generated 1s | 112^2 | 7 | 5 | 2.93% |
| generated 1s | 112^2 | 40 | 5 | 1.50% |
| **generated 1s** | **224^2** | **40** | **5** | **1.06%** |
| MNIST 1s | 112^2 | 40 | 5 | 27.68% |
| MNIST 1s | 224^2 | 40 | 5 | 27.65% |

Five coordinates reconstruct a held-out generated 1 to about one percent. The
same two improvements move MNIST by 0.03 points, which is the cleanest single
statement of the difference between the two families: **for the generated family
the error was my fit; for MNIST it is the family.**

The best structured model found for MNIST is a quadratic warp plus a cubic gain
field -- 22 coordinates, 18.27% at a 112^2 template and 18.32% at 224^2, so
resolution does nothing for MNIST at any coordinate count.

### The statistic that does mean something: parsimony

PCA is the optimal linear code at a given rank, so with enough components it has
to win. The informative number is how FEW parameters the structured model needs
to match it:

| family | warp coords | its error | PCA components needed to match | parsimony ratio |
| --- | ---: | ---: | ---: | ---: |
| generated 1s | 5 | 2.93% | about 70 | **~14x** |
| real MNIST 1s | 5 | 28.28% | about 17 | **~3.4x** |

Fourteen-fold against three-fold. That ratio is a direct measure of how much of
each family's variation really is one shape being pushed around.

### Are the higher-degree gains legitimate, or is the map tearing?

A fold -- the warp's Jacobian determinant going non-positive -- is the chart
flipping over itself rather than deforming, and gains bought that way are not
geometry. Measured two ways, because the distinction turns out to decide the
question:

| MNIST degree | held-out | folds anywhere on the canvas | folds ON THE INK | median min determinant on ink |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 21.17% | 3.6% | - | - |
| 3 | 18.06% | 11.9% | **1.1%** | 0.918 |
| 4 | 16.75% | 25.1% | **1.8%** | 0.873 |

Folding rises sharply with degree if you look at the whole canvas, and that
looks alarming -- at degree 4 a quarter of held-out images fold somewhere. But
restricted to pixels that actually carry ink it is 1-2%, with the determinant
sitting near 0.9. The folding lives in empty corners, where a high-degree
polynomial can thrash without touching a single lit pixel. That costs nothing
and is not the fit cheating, so the higher-degree gains stand.

Extended to degree 5: folds anywhere 30.6%, on the ink 2.7%, determinant 0.850.
The generated family folds in 0% of held-out images on the ink at every degree,
with determinants of 0.96 to 0.98.

This is the diagnostic `warp_charts.jacobian_determinants(..., ink=image)`
exists for; the whole-canvas number on its own is misleading.

One caveat in the other direction. Forcing fold-free warps everywhere by
strengthening the prior toward the affine chart does cost real accuracy -- at
degree 5, 16.38% at prior 1e-3 (16.2% folding) against 21.47% at prior 5e-2
(2.2% folding). But the prior is a blunt instrument: it restricts the warp
everywhere, including the on-ink flexibility that is doing legitimate work. The
on-ink measurement is the one that answers whether a fit is cheating, and it
says no.

## 4. Counting the knobs again, with no model at all

Two distance-only estimators, calibrated on sets whose dimension is known.

| set | TwoNN | MLE k=5 | k=10 | k=20 |
| --- | ---: | ---: | ---: | ---: |
| CONTROL 5-dim linear subspace of the 784 | 5.09 | 5.13 | 5.09 | 5.05 |
| CONTROL generated 1s (truly 5 knobs) | 4.64 | 4.59 | 4.53 | 4.42 |
| MNIST 1s, all | **11.92** | 10.71 | 9.88 | 9.14 |
| MNIST 1s, most stroke-like 25% | 10.30 | 9.06 | 8.02 | 7.09 |
| MNIST 1s, least stroke-like 25% | 11.16 | 10.07 | 9.55 | 9.02 |

Both controls come back right. But the estimate must be re-run at growing
sample size before it can be believed:

| set | TwoNN at n = 500 / 1k / 2k / 4k |
| --- | --- |
| generated 1s | 4.76, 4.58, 4.57, **4.57** (converged) |
| MNIST 1s, all | 9.17, 9.73, 10.83, **12.16** (still climbing) |
| MNIST 1s, stroke-like 25% | 8.03, 9.71, 10.30, 10.30 |

The generated family's estimate converges on five and more data does not move
it. MNIST's keeps rising, which is the signature of an estimate that is
data-starved rather than converged. So the honest statement is **at least twelve,
and 6742 images are not enough to pin down the real number** -- a lower bound
that has not stopped moving. Reported as "about twelve" it would be wrong.

Note too that even the most stroke-like quarter sits near ten, not five.

## 5. The subpopulation that really is a straight stroke

Selection is by overlap with the single parallelogram each image's own moments
imply -- a shape hypothesis using only `grey_ones.render`, independent of the
model under test. Training size is held fixed at 1011, because a smaller subset
otherwise gets a worse template and that alone could manufacture the trend.

| subset (1011 train, 674 test) | 5 warp | 20 warp | PCA 5 | PCA 20 |
| --- | ---: | ---: | ---: | ---: |
| all MNIST 1s | 27.13% | 17.26% | 38.04% | 26.17% |
| **most stroke-like 25%** | **18.13%** | **12.55%** | 33.42% | 21.45% |
| least stroke-like 25% | 49.75% | 34.57% | 55.98% | 42.72% |

A 2.75x spread, surviving the matched-size control. How well the five-knob model
describes an MNIST 1 is predicted by how much that 1 actually is a straight
stroke. The generated family's shape class is a real, identifiable
subpopulation of MNIST -- but note that even there five knobs leave 18%, against
2.93% on the synthetic family, and the dimension estimate for that same subset is
about ten. Straight-looking real 1s still carry curvature, thickness variation
and end-shape detail the five-knob family has no way to express.

## 6. What geometry cannot reach, priced against PCA

Warp coordinates plus appearance coordinates (PCA on the warp model's own
residuals), against plain PCA at matched total count:

| total coords | 5 warp + appearance | 20 warp + appearance | plain PCA |
| ---: | ---: | ---: | ---: |
| 5 | **28.25%** | | 39.18% |
| 20 | 22.28% (at 17) | 18.04% | 26.67% |
| 40 | 18.10% | 14.84% | 20.64% |
| 65 | 15.17% | 13.17% (at 60) | 16.50% |
| 105 | 11.87% | 10.75% (at 100) | 12.80% |
| 155 | 8.94% | **8.17%** | 9.18% |

The warp chart wins at every count, but watch the margin: eleven points ahead at
five coordinates, **one point ahead at 155**. That is the central verdict on
MNIST. The geometric structure is an excellent low-dimensional *summary* of
MNIST 1s and not a correct generative *description* of them. On the generated
family it was the correct description, which is precisely why it saturated there
at five.

## 6b. Do the coordinates GENERATE the family? (the sharpest test here)

Reconstruction error cannot see this. Fit coordinates, fit a distribution over
them, draw NEW coordinates, decode. A correct knob set produces only valid
members; a merely efficient basis produces ghosts. Precision is the distance
from each sample to the nearest real image; recall is the distance from each
real held-out image to the nearest sample, which catches a model that only
emits safe, typical members. Both are scored against the same reference half of
the real data, which sets the scale.

| source | stroke overlap | precision | recall |
| --- | ---: | ---: | ---: |
| GENERATED, real vs real (target) | 0.998 | 0.127 | 0.129 |
| GENERATED, new 5 warp coords | 0.998 | **0.098** | **0.129** |
| GENERATED, PCA 5 coords | 0.856 | 0.353 | 0.292 |
| GENERATED, PCA 20 coords | 0.807 | 0.409 | 0.326 |
| MNIST, real vs real (target) | 0.956 | 0.299 | 0.291 |
| MNIST, new 5 warp coords | **0.954** | 0.210 | 0.275 |
| MNIST, new 20 warp coords | 0.949 | **0.302** | **0.304** |
| MNIST, PCA 5 coords | 0.824 | 0.408 | 0.383 |
| MNIST, PCA 20 coords | 0.738 | 0.476 | 0.418 |

**This is the result that most changes the verdict on MNIST.** New warp
coordinates decode to images indistinguishable from real MNIST 1s, and at
twenty coordinates they match the real-data baseline on precision AND recall
(0.302/0.304 against 0.299/0.291) -- so they cover the real variety rather than
concentrating on the average 1. PCA samples are not 1s at all: overlap collapses
to 0.74 and they carry 9% negative ink, which a warped template can never
produce. PCA gets WORSE with more coordinates, the classic ghosting signature,
while reconstructing better.

So the warp coordinates of MNIST 1s are **valid but incomplete** knobs. Move
them and you stay inside the family -- they are real degrees of freedom of real
handwriting. But they do not span it, which is what the saturation and dimension
results say. The chart describes a genuine low-dimensional sub-family inside a
larger one. Section 6c pins down part of what it misses: ink density varying
along the stroke. Note that thickness variation along the stroke is NOT missing
-- that is already a warp.

## 6c. Can appearance explain what geometry cannot? Partly -- and not where I guessed

A warp of a fixed template cannot change how DARK the ink is, so an obvious
extension is to multiply the warped template by a smooth gain field written in
canonical coordinates. First, a correction to the obvious reading of the earlier
results: thickness varying ALONG the stroke is not the missing piece, because it
is already a warp -- a `u*v` term means "horizontal scale changes with height",
which is exactly why degree 2 bought the most.

**The confound, recorded because it nearly produced a false finding.** A gain
model also re-estimates the template, so comparing it against a shorter no-gain
fit credits the gain with extra template refinement. Held-out error with NO gain
at all, purely as a function of rounds:

| template rounds | generated | MNIST |
| ---: | ---: | ---: |
| 7 | 2.93% | 28.28% |
| 11 | **2.38%** | 28.07% |
| 15 | 2.08% | 27.94% |
| 25 | 1.72% | 27.73% |

The generated family's 2.38% at eleven rounds is exactly the number I had
credited to adding a darkness parameter. Worth zero.

**At matched rounds (11 for every row):**

| gain terms | generated (uniform coverage) | MNIST, 5 warp coords |
| ---: | ---: | ---: |
| 0 | 2.38% | 28.07% |
| 1 | 3.10% | 27.96% |
| 10 | 2.65% | **23.79%** |
| ten-term gain worth | **-0.26 points** | **+4.28 points** |

The control does not merely fail to benefit -- the gain is actively HARMFUL on
the generated family, which is what ten extra parameters should be when they
model nothing real. A 4.5-point swing between uniform ink and real handwriting. The fitted constant gain has an inter-quartile spread of
0.004 on the generated family against 0.029 on MNIST, stable across template
resolutions: real variation in handwriting, but the middle half of MNIST 1s
differ in overall darkness by about 1.5%, far too little to matter. The ten-term
spatial gain is worth about 4 points on MNIST and nothing (slightly negative) on
the generated family, whose coverage is uniform by construction.

And one more deflation, measured rather than assumed. The ten-term gain is
worth +4.28 points on top of a FIVE-coordinate affine warp but only +1.34 on
top of a twelve-coordinate quadratic warp:

| warp | 0 gain | 10 gain | gain worth |
| --- | ---: | ---: | ---: |
| 5 coords, affine | 28.07% | 23.79% | **+4.28** |
| 12 coords, quadratic | 20.58% | 19.24% | **+1.34** |

So most of what the gain appeared to add on top of the affine warp is really
geometry the affine warp could not do -- the quadratic warp gets there by
bending instead. Appearance and geometry are partly interchangeable in this
model, and the genuinely appearance-only part is about a point, not four.

What survives: there IS appearance structure in MNIST 1s that no warp of any
order can reach, it is **ink density varying along the stroke** rather than
overall darkness, and it is worth roughly a point. The control is what makes
even that statement safe -- a knob that improved both families equally would be
fixing my model, not describing handwriting.

## 7. What the failures look like -- a prediction that was wrong

A warp moves ink; it cannot create a stroke. So the natural guess was that the
failures carry a serif or a base bar that a plain stroke cannot grow. Tested:

| feature | best decile of fit | worst decile | correlation with error | share above 0.10 |
| --- | ---: | ---: | ---: | ---: |
| base bar | 0.001 | 0.033 | **+0.603** | 1% |
| top-left flag | 0.000 | 0.006 | +0.351 | 0% |

Right direction, wrong magnitude. Base bars correlate strongly with where the
model fails and the worst decile has 33x more of them, but only 1% of MNIST 1s
carry one worth a tenth of their ink. Serifs and bars are a real but **minor**
part of the failure, not the main cause. The saturation table names the main
cause instead: bending.

And MNIST 1s are not a few discrete hands:

| templates | held-out |
| ---: | ---: |
| 1 | 28.24% |
| 2 | 26.27% |
| 3 | 25.53% |
| 5 | 24.78% |
| 8 | 23.42% |

Smooth and gradual with no knee. A handful of styles would have closed most of
the gap and then flattened. This is a continuum, which rules out the
mixture-of-templates story.

## 8. What the recovered numbers mean

The five fitted warp numbers, correlated against independently measured
quantities:

| warp number | tracks most strongly | correlation |
| --- | --- | ---: |
| `zx` constant | centre x | -0.71 |
| `zx` per x | total ink / width | -0.90 / -0.59 |
| `zx` per y | **slant** | **+0.92** |
| `zy` constant | centre y | -0.61 |
| `zy` per y | height | -0.30 |

So the chart does land on recognisable quantities -- horizontal placement,
horizontal scale, slant, vertical placement, height -- with slant coming out
almost perfectly isolated. They are not identical to the generator's knobs, and
they do not need to be: any smooth invertible relabelling is as good.

## 9. The decoder, written out

For the generated family the whole answer is five numbers and one line, with `T`
the learned template:

```
image(x, y) = T( zx, zy )
     zx = a0 + a1 * (x - 14)/14 + a2 * (y - 14)/14
     zy = b0 +                    b2 * (y - 14)/14
```

Five free numbers per image; the absent `(x - 14)/14` term in `zy` is the
vertical shear the family does not use. Held-out accuracy 2.93% against a 1.26%
floor. The same form with 5 to 42 numbers gives 28.3% down to 16.1% on MNIST.

## 9b. From the unlabelled chart back to the generator's own knobs

The chart's five numbers are not the generator's five knobs; they are some
relabelling of them. That relabelling turns out to be exactly computable.

The fitted chart is `zx = a0 + a1*u + a2*v`, `zy = b0 + b2*v` with
`u = (x-14)/14`. The generator's own map is `u_gen = (W0/w)(x - cx + s(y - cy))`
and `v_gen = (H0/h)(y - cy)`. The canonical frame is a FIXED affine
reparameterisation of `(u_gen, v_gen)` -- the same for every image, because it
is only the frame the template happens to live in. Matching coefficients:

```
w  = Cw / a1
h  = Ch / b2
s  = (a2 - kappa * b2) / a1           (s = tan lean)
cy = 14 + 14 * (Zy - b0) / b2
cx = 14 + 14 * (Zx - a0 - a2 * (cy - 14) / 14) / a1
```

`Cw, Ch, kappa, Zx, Zy` depend only on the gauge, never on the image. So an
unlabelled chart pins the true knobs down to five numbers. Median absolute error
over 1500 held-out images, against knob ranges of 1, 1, 1.5, 2.8 px and 45 deg:

| labelled examples used | cx | cy | height | width | lean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **5** | 0.0014 | 0.0052 | 0.0091 | 0.0015 | 0.0088 |
| 20 | 0.0012 | 0.0042 | 0.0084 | 0.0015 | 0.0079 |
| 2500 | 0.0012 | 0.0042 | 0.0084 | 0.0015 | 0.0076 |
| worst case, full calibration | 0.0189 | 0.0178 | 0.0382 | 0.0280 | 0.1180 |
| CONTROL: chart rows shuffled | 0.289 | 0.291 | 0.436 | 0.799 | 12.91 |

About a thousandth of a pixel, or 0.1% of each knob's range. **Five labelled
examples are as good as 2500**, which is the point rather than a curiosity: the
gauge has five degrees of freedom, so five examples saturate it and more add
nothing. The shuffled control fails completely, so this is structure and not a
flexible fit.

That closes the loop. Unlabelled images give the chart and the decoder; five
labels name the axes.

## 9c. What the five MNIST coordinates ARE, in closed form

MNIST has no true knobs to check the inversion against, so the question becomes
whether the SAME closed-form expressions yield quantities that agree with
observables measured independently from the pixels by moments. Gauge calibrated
on 20 images; compared on 1500 held-out ones. The shuffled column pairs chart
rows with the wrong images, which is the scale of "no information".

| quantity | median error | data spread | shuffled | times better than chance |
| --- | ---: | ---: | ---: | ---: |
| slant | 0.2465 deg | 13.88 | 14.16 | **57x** |
| width | 0.1599 | 1.52 | 0.8696 | **5.4x** |
| centre x | 0.0663 | 0.305 | 0.3243 | **4.9x** |
| height | 0.3243 | 0.734 | 0.4434 | 1.4x |
| centre y | 0.2559 | 0.294 | 0.4406 | 1.7x |

The generated family, as a control where this must work: 0.0016 / 0.0043 /
0.0147 / 0.0720 / 0.0307 against spreads of 0.29 / 0.29 / 0.43 / 0.75 / 12.9.

So the five MNIST coordinates genuinely ARE slant, width and horizontal position
written in closed form -- and they barely determine height or vertical position.

There is a clean mechanism. A 1 is a long, nearly uniform vertical stroke.
Sliding it vertically, or lengthening it, changes the pixels very little,
because the stroke looks much the same all along its length. Its horizontal
position, thickness and lean change them a lot. So the vertical direction is
weakly identifiable FROM THE DATA, whichever method is used.

This independently reproduces `figures/generated_one_warp_space`, which found
from a completely different fit that "the vertical warp coefficients are not
equally visible, so a pixel-fitted warp is blind to `cy` (median error 0.467)".
Two unrelated experiments, same conclusion -- which is the best evidence
available that it is a property of 1s and not of either method.

## What this settles, and what it does not

Settled, for the generated family: the goal is met. Five coordinates, an
explicit decoder, 1.50% held-out against a 1.26% floor, the count confirmed by
saturation and by two model-free estimators, and the generator's own knobs
recovered to a thousandth of a pixel once five labels fix the gauge. No
parameter labels are used anywhere in the fitting or the scoring.

One methodological note worth carrying forward: the first draft of this README
reported 2.93%, which was simply an under-converged fit. Held-out error keeps
falling with template rounds (2.93% at 7, 2.08% at 15, 1.72% at 25, 1.50% at 40)
because each round sharpens the super-resolved template. `warp_charts.fit_chart`
now defaults to 25 rounds. Real MNIST barely moves over the same range
(28.28% to 27.73%), because there the error is the family, not the fit.

Settled, for MNIST: there is no small exact knob set. Three independent lines
agree -- geometry saturates near 15%, the dimension estimate exceeds twelve
without converging, and the warp chart's edge over PCA vanishes as coordinates
grow. MNIST 1s are not one shape pushed around by a few knobs, and no method
will find knobs that are not there.

Not settled: the real dimension of MNIST 1s. The estimate is a moving lower
bound, and settling it needs more data than MNIST's 6742 ones provide. The
geometry asymptote near 15% is an extrapolation from four decaying gains, not a
measurement; a free-form warp with many more parameters would test it directly.
