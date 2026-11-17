# Flattening generated 1s with shuffled pixels (rule B)

Scripts: `analyze_shuffled_pixel_flattening.py` (main) and `check_shuffled_flattening_box.py`
(follow-up). Numbers are in `results.json` and `box_check.json`.

The method gets 6,742 generated 1s as 784 numbers each, in a shuffled pixel order, and
nothing else. It learns which pixels are neighbours from which pixels get inked together,
optionally blurs along those learned neighbours, then flattens to 5 numbers with walking
distance (Isomap). The true knobs are used only for grading. The score is cross-validated
kNN R² in both directions: each knob from the 5 numbers, and each of the 5 numbers from
the knobs. To pass, every score must be at least 0.95.

## Predictions (written before running) and outcomes

| Prediction | Outcome |
| --- | --- |
| P1: learned neighbours right sideways, confused vertically with pixels along the stroke | **Right.** Only 34% of learned neighbours are true neighbours. The mean offset is 3.5 rows but 1.3 columns. |
| P2: the plain ruler finds lean, width and cx but misses cy and height | **Right.** cy 0.40, height 0.14 |
| P3: blurring on the true grid recovers cy and height | **Wrong.** cy 0.19-0.40, height 0.03-0.10 |
| P4: blurring on learned neighbours lands between P2 and P3 | **Wrong, it is worse.** At ~2 px, cy is 0.006 |

No configuration passed. The 4th and 5th numbers are bent copies of lean and width
(0.69-0.79 predictable from the knobs).

## Why: size, not ruler

Each knob's sweep across its full range moves the image by this much:

| knob | straight-line distance | distance along the sweep |
| --- | ---: | ---: |
| cx | 5.0 | 6.0 |
| cy | 2.5 | 2.8 |
| height | 1.8 | 1.8 |
| width | 6.4 | 8.4 |
| lean | 9.2 | 28.7 |

Lean's sweep is 16 times longer than height's, and it is bent: 28.7 along the sweep
against 9.2 straight. The flattener spends its spare numbers on lean's bends before it
reaches the small knobs. Blurring along the stroke also erases its ends, which is the
only place cy and height show, so the learned blur makes things worse.

Follow-up (plain ruler, same count, 5 numbers):

| box | cx | cy | height | width | lean | 5 numbers from knobs (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| default | 0.90 | 0.28 | 0.14 | 0.98 | 1.00 | 0.72 |
| lean narrowed to 10-13 deg | 0.99 | **0.92** | **0.82** | 1.00 | 0.88 | 0.94 |
| cy widened to 10-18 px | 0.69 | **0.99** | 0.00 | 0.96 | 1.00 | 0.81 |

Shrinking lean brings cy and height back with no ruler change at all. Widening cy brings
cy back, and height (still the smallest knob) stays lost.
