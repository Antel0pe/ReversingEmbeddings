# Curved components instead of straight ones (2026-11-22)

**Question (user):** one knob traces a *curve* through picture space, and PCA
needs many straight components to cover it. Can a method find ONE curved
component instead, either by fitting from a list of curve shapes, or by
combining PCA's straight components into fewer curved ones?

Everything here is on generated 1s (`grey_ones.render`, coverage renderer) with
ONE knob changing and the rest at the root 1, plus a plain circle as a sanity
check and (where the method allows) the full five-knob family. Methods get
shuffled pictures and no knob values, except the curve-shape list, which is told
the knob on purpose.

Miss = ‖rebuilt − true‖ / ‖true‖ over held-out pictures, in percent.
"Between" = trained on every other sweep position, tested on the ones in between.
"Outside" = trained on the middle third only, tested on the rest.

## Scripts (run from repo root with `.venv/bin/python curved_components/<script>`)

| Script | What it tests |
| --- | --- |
| `data.py` | shared test sets (circle, slide = cx 5→23, lean = −10→35°, five knobs) and scoring |
| `loops_and_rules.py` | PCA components pair into loops; one learned "turning" rule rebuilds unseen positions |
| `principal_curves.py` | Hastie–Stuetzle principal curve (one number per picture), plus a spacing sweep |
| `principal_polynomial.py` | Laparra-style principal polynomial analysis (bent PCA), degrees 2/3/5/9 |
| `vanishing_components.py` | equations that vanish on the data (plain-SVD version of VCA), in k PCA numbers |
| `curve_dictionary.py` | SINDy/symbolic-regression style: pick curve shapes (powers, waves, growth) from a list |
| `make_figure.py` | `results/curved_components_summary.png` |

PySR was not used (it needs Julia, which is not installed). `curve_dictionary.py`
is the same idea with a fixed list of shapes and greedy selection.

## Results

| Method | Slide | Lean | Verdict |
| --- | --- | --- | --- |
| PCA (straight) | <10% at 13 comps, <1% at 21 | <10% at 5, <1% at 14 | baseline |
| Principal curve, 1 number | 5.3% between; knob order 1.000 | 0.9%; order 1.000 | **yes, but only while pictures are close** |
| … "told the true order" control | 4.0% | 0.5% | the fitted curve ≈ joining the dots |
| Principal polynomial, degree 3 | <10% at 11, <1% at 20 | <10% at 3, <1% at 11 | **a little**: 1–4 fewer components |
| Vanishing equations | 3 PCA numbers, degree 3: 2 equations, leave 1 number free | 2 PCA numbers, degree 4: 2 equations, 1 free | **says "it is a curve"**, but only after PCA has thrown 58% / 22% away |
| Curve shapes from a list (told knob) | <10% at 14 | <10% at 5 | **no saving**: picks evenly spaced waves = the loops |
| Outside the middle third (median miss) | rule 18%, PCA-6 111%, list 539% | rule 20%, PCA-6 50%, list 67% | only the repeated rule carries on |
| Grid floor (exact move, same 28×28 grid) | 10% | 9.5% | what the rule's leftover is made of |

Details:

- **Circle sanity check.** Principal curve 1.7% with one number. Principal
  polynomial cannot help (70.7% at one bent component, same as PCA): seen from its
  main direction a loop is two-valued, so no polynomial predicts the rest.
  Vanishing equations find x² + y² − 1 exactly (one equation, 2e-15 held out).
- **Principal curve spacing.** Slide, training pictures 0.5 / 1 / 2 / 3 / 4 px
  apart: 5.0 / 18.2 / 29.6 / 44.9 / 63.6% (order breaks at 4 px, 0.80). Lean,
  1 / 2 / 4 / 6 / 8° apart: 0.8 / 1.3 / 4.0 / 7.0 / 10.9%. Fitted miss equals the
  "join the dots in the true order" control at every spacing from 1 px / 2° up,
  so the curve knows nothing between its pictures beyond a straight blend.
- **Principal polynomial on all five knobs** (2,000 train / 500 test):
  5 components 27.3% → 25.4% (degree 3), <10% at 29 → 26. Degrees 5–9 overfit.
- **Vanishing equations** are fragile: one degree too high and fake equations
  appear that leave "0 numbers free". Slide's first PCA pair is not a circle
  (best degree-2 equation 0.12 off zero; loop radius varies 14.5%).
- **Curve-shape list** first picks sin/cos at speeds 2.75, 4.0, 5.5, 6.75, 8.25 …
  (evenly spaced, i.e. the loop pairs). Each wave still carries its own picture-sized
  pattern, learned only where ink was seen, so it cannot carry on outside.
- **Loops and the rule on lean.** Lean's PCA pairs also loop (0.72, 1.67, 2.71
  turns, steady with tan(lean) r² ≈ 0.99). A per-row turning rule learned on the
  middle third misses 20.3% at unseen leans vs PCA-6 50.4%; spacing the training
  steps evenly in tan(lean) instead of degrees changes nothing (19.8%) because
  the grid floor (9.5%) dominates.

## What it means

A one-knob family IS one curve, and the methods that report "one number"
(principal curve, vanishing equations) do see that. None of them gives a short
*description* of the curve: the principal curve stores ~60 pictures, the
equations have dozens of terms in a few PCA numbers, and the shape list needs
as many wave patterns as PCA needs components. The only short description
found is the repeated rule ("turn every frequency at its own steady speed"),
which is the slide itself; it is the one that works past the training range.
