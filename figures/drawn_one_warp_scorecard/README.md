# Warp scorecard: the MNIST warp method on drawn 1s whose knobs are known

Scripts (repo root):

| script | what it does |
| --- | --- |
| `analyze_drawn_one_warp_scorecard.py fit <family>` / `score` | fits 5 / 12 / 20 warp numbers (affine without vertical shear / quadratic / cubic warp of one super-resolved template, `warp_charts.fit_chart`, 25 rounds), then scores leftover, knob readability and leftover location |
| `analyze_drawn_one_ink_pieces.py <family> <warp count>` | adds 1-16 learned "ink pieces" in the template frame on top of a cached fit |
| `check_drawn_one_leftover_shape.py` | where the leftover sits across / along the stroke, worst-tenth share, sign flips |
| `check_mnist_one_worst_tenth.py` | what sets MNIST's worst-fitted tenth apart (measurements only) |
| `check_edge_roughness.py` | row-to-row roughness of stroke centre and width |
| `check_stroke_cross_section.py` | edge blur (grey share) and core darkness per family |
| `check_best_ninety.py` | leftover of each family's best-fitted 90% vs worst-fitted 10% |
| `make_drawn_one_warp_scorecard_figures.py` | the four figures (no MNIST images drawn) |

Every family: 3,000 training images, 1,500 held out, same settings. Fits run ~35 min each,
single-threaded; run the `fit` calls in parallel. Cached fits are `fit_<family>.npz`
(not needed to read the results; all numbers are in the JSON files).

Families (`knob_strokes`): `drawn5` (cx cy height width lean), `drawn8` (+ bend wiggle
taper), `drawn12` (+ bulge hook_len hook_ang bar_len; every member has a hook AND a bar),
`wobbly_0.4/0.8` (drawn12 + 8 smooth sideways/width wobbles, sine modes 2-5), `sharp5` and
`sharp12` (edge softness 0.18 px instead of 0.4, chosen so the share of ink in part-inked
pixels matches MNIST: 0.29/0.38/0.57 vs 0.26/0.40/0.57 at the 10th/50th/90th percentile;
chosen from that image statistic, never from the leftover), and `mnist` (first 4,500
training 1s).

## Predictions written before running (mine; the user made none) and outcomes

| prediction | outcome |
| --- | --- |
| the 12-knob family leaves much less than MNIST at 5 numbers | **Wrong.** 30.9% vs 27.2% (sharp version 35.9%) |
| cx, lean, width, bend, wiggle, taper read well | **Right at 20** (0.86-0.99); taper only 0.51 at 12, cx 0.57 at 5 |
| height and vertical position read weakly (the old "blind to cy") | **Wrong.** 0.92-0.96 at every count |
| hook and bar mostly end up in the leftover | **Partly.** 0.46-0.69 at 12 numbers, 0.76-0.95 at 20, 0.82-0.99 with ink pieces |

## 1. Leftover (held-out, relative pixel error)

| family | 5 | 12 | 20 | 20 + 16 pieces | PCA at 20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| drawn5 | 5.55 | 4.66 | 4.52 | - | 12.65 |
| sharp5 | 4.65 | 2.77 | 2.53 | 1.79 | 21.78 |
| drawn8 | 27.73 | 11.67 | 5.35 | 1.07 | 18.36 |
| drawn12 | 30.86 | 20.45 | 9.54 | 3.68 | 23.20 |
| wobbly_0.4 | 32.02 | 22.01 | 11.79 | 5.33 | 23.94 |
| wobbly_0.8 | 34.90 | 25.48 | 15.94 | 7.71 | 25.58 |
| sharp12 | 35.90 | 24.28 | 9.94 | 5.19 | 31.53 |
| **mnist** | **27.17** | **19.26** | **15.58** | **11.72** | 26.30 |

Share of the 20-number leftover removed by 16 pieces: drawn8 80%, drawn12 61%, wobbly 52-55%,
sharp12 48%, sharp5 29%, **MNIST 25%**. (MNIST with 12 warp + 16 pieces: 13.87%.)

## 2. Can the numbers be read back as the true knobs? (held-out R^2, quadratic ridge)

drawn12 (sharp12 within 0.05 except where noted): 5 warp numbers read cy 0.93, height 0.93,
width 0.84, lean 0.91 but cx only 0.57 (bend moves the ink centre). 12 numbers: the five
basics 0.89-0.98, bend 0.96, wiggle 0.76, taper 0.51, bulge 0.17, hook length 0.53, hook
angle 0.69, bar length 0.59. 20 numbers: all twelve 0.80-0.99. 20 + 16 pieces: all twelve
0.91-1.00 (sharp12 0.81-0.99). Warp beats PCA at the same count on every added knob (PCA 20:
bulge 0.63, hook angle 0.59). Five plain ink sums (no fitting) read vertical position at only
0.64 once a hook and bar are present; shuffled control negative everywhere.

**Consequence for `figures/mnist_one_knob_search` 9c.** There, MNIST's 5 warp numbers were
checked against moment-measured height and centre y and looked nearly blind (1.4x / 1.7x
better than chance). On the drawn 1s the warp reads both at 0.93 while the moment yardstick
itself reads cy at 0.64. The "blind to vertical position" verdict on MNIST is most likely a
bad yardstick, not a blind warp. (The degree-2 free-warp blindness in
`figures/generated_one_warp_space` is a different, unregularised setting and is not affected.)

## 3. Is MNIST's leftover the same kind of thing?

At 20 numbers, from the images alone:

| family | on centre line | worst tenth's share | sign flips down the stroke | grey ink share | best 90% | worst 10% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| drawn12 | 47% | 21% | 0.22 | 0.83 | 8.9 | 16.9 |
| wobbly_0.8 | 53% | 18% | 0.25 | 0.82 | 15.1 | 26.2 |
| sharp5 | 17% | 20% | 0.44 | 0.38 | 2.4 | 4.0 |
| sharp12 | 36% | 23% | 0.32 | 0.38 | 9.2 | 16.6 |
| **mnist** | **22%** | **45%** | **0.41** | **0.40** | **12.4** | **31.4** |

- Edge-heavy, often-flipping leftover is what SHARP ink does under resampling (sharp5 has it
  with no extra knobs). Not evidence of new structure.
- Smooth wobbles (wobbly) and row-to-row roughness do not match: MNIST's measured roughness is
  0.11 px (centre) / 0.19 px (width), smaller than drawn12's 0.15 / 0.58.
- What remains: MNIST's worst tenth fit twice as badly as any drawn family's (31.4 vs 16.6),
  and its best 90% still keep ~3 points more than sharp12 (12.4 vs 9.2).
- The worst tenth, by measurement (`mnist_worst_tenth.json`): NOT thicker (median row ink
  width 2.72 vs 2.83 px), but 57% have a centre line well off straight (row centres 0.56 vs
  0.26 px rms) and 43% have a few rows over 1.8x the median width (hook / serif / bar), against
  10% by chance for each; 31% are flagged by neither.
- MNIST is LOWER than the 12-knob family at 5 and 12 numbers (best 90%: 22.5 vs 34.6 at 5):
  a typical MNIST 1 is simpler than a drawn 1 that always has both a hook and a bar.

## Beliefs changed

1. "27.65% at five numbers means MNIST has no small knob set" -- **withdrawn**. A family with
   exactly 12 known knobs scores 30.9-35.9% there.
2. "The warp is blind to height / vertical position" -- **withdrawn for rich 1s**; the warp
   reads them at 0.93-0.96; the moment yardstick was the weak part.
3. "MNIST's extra leftover is hooks and bars" -- **mostly no**. Pieces that fix hooks and bars
   (61% removed on drawn12) remove 25% on MNIST.
4. "MNIST's leftover sits at the edges, so it is edge texture" -- **no**: sharp ink alone gives
   that shape.
5. **New**: MNIST beyond a 12-knob 1 = a tail of ~10% curved / serifed 1s that fit twice as
   badly as any drawn family's tail, plus ~3 points spread over the rest that neither more
   cubic warp nor 16 shared ink pieces remove.
