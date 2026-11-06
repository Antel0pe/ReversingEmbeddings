# Does pixel space know that two nearby pixels are nearby?

`analyze_pixel_metric_locality.py` (six parts, runnable separately) ·
`make_pixel_metric_locality_figures.py` · full console output in `results.txt`

## The complaint being tested

Treating each pixel as its own dimension makes every pixel exactly as far from
every other pixel. Two ink patterns that share no pixel are therefore
perpendicular — cosine exactly zero — whether they are neighbouring slivers
along the *same* edge, which are plainly the same gesture continued, or an
unrelated blob on the far side of the canvas. The pixel metric cannot tell those
two cases apart, and that same fact is where the infinite derivative of
`literature/multiscale-structure-of-non-differentiable-image-manifolds.md`
comes from.

Question: is there a representation that separates perpendicular-because-far
from perpendicular-because-adjacent?

## What was tried

Every linear candidate turns out to be the same move — **re-encode the picture,
then use ordinary Euclidean distance on the encoding** — and every linear,
translation-covariant, isotropic re-encoding is a reweighting of spatial
frequencies. So the whole family of linear answers is one function w(mu), mu =
squared spatial frequency, implemented in the DCT-II basis where the Neumann
Laplacian is exactly diagonal (hence exact at any grid fineness):

| name | w(mu) | in words |
| --- | --- | --- |
| `pixel` | 1 | every frequency equal — the baseline, no geometry |
| `blur s` | exp(−s²·mu/2) | discard detail finer than s |
| `H^-1/2`, `H^-1` | (kappa + mu)^−k | damp every scale at once, gently |
| `sharpen` | (kappa + mu)^+1/2 | control: the wrong direction |

One **nonlinear** candidate was also tested: optimal transport (debiased
Sinkhorn), where distance is how far the ink has to walk. That encodes
proximity by construction instead of by filtering, and destroys nothing.

Renderers: `coverage_grids.render_coverage` for the repo's own anti-aliased
object, and the same thresholded at 0.5 on a fine grid for a true cliff.

## Findings

**1. The pixel metric's notion of space is a step function (part 1).** One pixel
of ink against the same pixel d columns away: cos = 1.000 at d = 0 and 0.000 at
every d ≥ 1. One pixel over and half the canvas over score *identically*. Every
other encoding gives a decaying curve. Their "connection range" — the offset at
which cos first drops below 0.1 — is 1 px for `pixel`, 3 px for `blur 1px`,
13 px for `blur 4px`, 6 px for `H^-1/2`, 15 px for `H^-1`.

**2. The pixel metric scores a continued gesture and an unrelated blob the same
(part 2).** On binary ink at 224×224, take the width knob's increment over
[0, 1 px] and compare it with the same knob's increment over [1, 2 px] — the
gesture continued, abutting the first bands, sharing **0** pixels with them.

| partner | pixel | blur 1px | blur 4px | H^-1/2 | H^-1 | sharpen |
| --- | --- | --- | --- | --- | --- | --- |
| same knob, continued (want HIGH) | −0.000 | **+0.936** | +0.999 | +0.939 | +0.997 | −0.487 |
| the same bands 10 px away (want LOW) | −0.000 | +0.000 | +0.247 | +0.168 | +0.398 | −0.000 |
| a different knob | +0.151 | −0.233 | −0.744 | −0.345 | −0.678 | +0.393 |
| sign-scrambled, same pixels | +0.000 | +0.021 | −0.060 | −0.001 | −0.025 | −0.000 |
| white noise | −0.003 | +0.108 | +0.191 | +0.092 | +0.164 | −0.006 |

Separation score (the continued-gesture cosine minus the largest |cosine| among
the four wrong answers, averaged over width, lean and cx): **blur 2px +0.565**,
blur 1px +0.557, H^-1/2 +0.525, blur 0.5px +0.465, H^-1 +0.351, blur 4px +0.278,
**pixel −0.149**, sharpen −0.831. The pixel metric is *negative*: it ranks a
wrong answer above the right one. The best encodings have a connection range
near the stroke width (3.2 px) — too little and the continuation is missed, too
much and far-away bands start to count as related.

**3. The −1/2 law, reproduced here (part 3).** Speed ‖R(w+h) − R(w)‖/h on binary
ink: the pixel metric's log-log slope is **−0.500** for width and −0.53 for cx —
the paper's blow-up, to two decimals. Blur and H^-k give slopes of −0.02 to
−0.04 with the values visibly settling (blur 4px on width: 0.2918, 0.2960,
0.2980, 0.2989). `sharpen` gives −0.98, i.e. worse.

**4. The tangent plane exists in the re-encoded space and not in pixel space
(part 4).** Largest principal angle between the 5-knob plane at step h and at
h/2, binary ink at 448×448 (all five increments nonzero at every h shown):

| metric | 0.5→0.25 | 0.25→0.125 | 0.125→0.0625 |
| --- | --- | --- | --- |
| pixel | 55.9° | 56.5° | **53.4°** |
| blur 1px | 33.2° | 16.8° | 8.3° |
| blur 4px | 11.7° | 5.5° | **2.7°** |
| H^-1 | 20.7° | 10.8° | 5.6° |
| sharpen | 89.2° | 89.5° | 90.0° |

The pixel metric's angle is flat — the plane keeps rotating by the same amount
forever, which is what "no tangent plane" means operationally. Every re-encoding
halves the angle each time the step halves, i.e. converges linearly in h.

**5. The repo's own 28×28 coverage render already has a tangent plane
(part 4, second block).** The same measurement on `grey_ones`' coverage output
gives pixel-metric angles 38.5°, 26.8°, 3.6°, 0.0°, 0.0°. Below a normalised
step of about 0.06 it is exactly locally linear. This answers an open question
in the paper note: for steps that small the repo has been working on the
smoothed manifold F_s, not on F.

**6. Optimal transport is exact, and costs no detail (part 5).** Debiased
Sinkhorn on a cx translation returns **4.0000, 2.0000, 1.0000** for steps
4, 2, 1 px — the cost of moving the 1 by h is exactly h, with nothing blurred
away. Slope +1.00 against the pixel metric's +0.50.

**7. Below one pixel the blow-up is the SAMPLING GRID, not the sharp edges
(part 6).** OT's flat run of 1.0000 ends exactly at the pixel pitch, at every
fineness, and below it tracks 1/√h:

| grid | h=2 | h=1 | h=0.5 | h=0.25 | h=0.125 | h=0.0625 |
| --- | --- | --- | --- | --- | --- | --- |
| 28² (pitch 1) | 0.9999 | 0.9999 | 1.4142 | 1.9365 | 2.7386 | 3.8730 |
| 56² (pitch 0.5) | 0.9998 | 0.9997 | 0.9995 | 1.4142 | 2.0615 | 2.9580 |
| 112² (pitch 0.25) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.3070 | 1.7181 |
| 1/√h for reference | 0.7071 | 1.0000 | 1.4142 | 2.0000 | 2.8284 | 4.0000 |

Shifting a sampled picture by less than a pixel is not a translation of it at
all, it is a resampling, and no choice of metric can make it one. So there are
**two** separate causes of the blow-up and they need different fixes: sharp
edges, which OT removes for free above the pitch, and the sampling grid, which
nothing removes.

## Controls run

- `sharpen` — the same construction with the exponent's sign flipped. Fails
  every test (separation −0.831, speed slope −0.98, twist 90°), so the results
  are not an artefact of "any re-encoding helps".
- `white noise` and `sign-scrambled on the gesture's own pixels` — guard against
  an encoding that is only measuring support overlap. Both score near zero.
- `a different knob` at the same place, and `the same bands 10 px away` —
  separate "right place, wrong motion" from "right motion, wrong place".
- Live-direction counts in part 4, so a zero increment on a binary grid is never
  read as a 90° twist.
- Raw alongside debiased Sinkhorn, and eps held below every pitch in part 6.

## Traps recorded

- **A free coin flip is not a zero-mean control.** Scrambling signs over ~1300
  pixels leaves a ~3% mean offset; after heavy blurring almost nothing *but*
  that offset survives, which faked a sign-scramble score of **+0.896** and made
  blur look as if it could not distinguish a gesture from noise. Forcing the
  signs to balance exactly drops it to −0.060. Balance the signs.
- **Sinkhorn needs far more iterations than it appears to.** At 400 iterations
  the debiased divergence went *negative* at small steps and the curve plunged;
  at 4000 it is clean and matches 1/√h to three decimals. Convergence failure
  looks like a result.
- **A binary render has a hard floor at its own pitch.** Below it a knob step
  flips no pixel, the increment is exactly zero, and principal angles come back
  as a meaningless 90°. Count live directions before reading any such table.
- **Do not mix twist estimators.** The angles here are one-sided
  finite-difference angles between spans at h and h/2. The paper's 26.57° is the
  blur-ratio quantity of its Thm 4.1 and its ~30° is local PCA. Same behaviour,
  different numbers.

## What this does not show

The separation scores say an encoding can rank a continued gesture above an
unrelated one. They do **not** show that any of these encodings recovers the
five knobs, and no decoder or held-out reconstruction was run here. Blur buys
locality by destroying detail, which is the paper's own trade; OT avoids that
trade above the pitch but is a distance, not a representation, so it does not
hand you coordinates. The honest reading is that this measures *rulers*, not
coordinates — the repo's warp and motion-field work is where the coordinates
are.

## Figures

- `fig1_locality.png` — the complaint and the fix: the step function in space,
  the abutting bands of one gesture, and which encodings separate them.
- `fig2_derivative.png` — speed, tangent-plane twist, and optimal transport
  against the pixel pitch.
