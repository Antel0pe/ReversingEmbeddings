# Matching families by patterns across multiple paths

The reference is the user's width/lean figure: paths at different widths have
related *whole distance profiles* as lean varies. The question here is whether
that kind of relationship can associate local SVD directions across states,
and keep five repeated walks on their expected families.
There is no target of linearly increasing distance or constant finite step size.
The original figure's numerical curves and physical labels are not supplied to
the selector.

## Result

One deterministic run, from the canonical upright root:

| Evaluation | Correct family | Correct family and sign |
| --- | ---: | ---: |
| Four training offsets on five context branches | 100 / 100 | 100 / 100 |
| Two fresh offsets on those branches | 50 / 50 | 50 / 50 |

These bundle comparisons come from **31 states around one root**, not 150
independent trials across the manifold. No association failures occurred in this
sample. Randomly shuffling and flipping the candidate vectors at every state
preserved all 150 signed selections.

The same rule was then applied repeatedly to five nine-hop walks. At each hop,
four branch offsets trained local profiles and the fresh 0.3 endpoint received
five jointly assigned families. All **45 / 45 executed moves** used the expected
family and sign. All **225 / 225 next-field selections** agreed with the physical
family and sign: 150 / 150 during the first six hops and 75 / 75 in the last
three. There were 225 additional local SVD observations, including the lookahead
branch starts. Templates are fitted locally at every hop; the last three hops
use the unchanged rule but are not blind prediction.

| Walk | Endpoint pixel L2 error versus pure control |
| --- | ---: |
| Lean | 0.0334 |
| Width | 0.1018 |
| Horizontal position | 0.0877 |
| Vertical position | 0.1156 |
| Height | 0.2433 |

The height walk reached height 22.089 (pure control: 22.065) while width drifted
from 3.2 to 3.127. Family tracking works here, but the selected SVD vectors are
still mixtures. No failure was observed within these particular nine-hop walks.

The root's five SVD axes correspond approximately to lean, width, horizontal
position, vertical position, and height. Their absolute cosines to the known
control directions are 0.999908, 0.999509, 0.999569, 0.999312, and 0.988154.
The lowest selected control cosine anywhere in this experiment is 0.984308.
Correct family labels therefore mean approximate direction agreement, not exact
recovery of the physical controls or their finite paths.

## What was done

1. Start at `(cx, cy, height, width, lean) = (14, 14, 19.75, 3.2, 0)`.
   Fit uncentered SVD to 256 equal-image-distance neighbors, at radius 0.0001,
   using float64 coverage values and the same fixed Sobol directions at each
   state. Keep five leading axes. Root signs are arbitrary and deterministic.
2. Make five context branches along the five root directions. Observe each at
   branch amounts 0.2, 0.4, 0.6, and 0.8. Branch amounts are calibrated by initial
   pixel speed, not by physical knob names. Finite distances are measured.
3. At each branch start, render five candidate path rays through its five local
   SVD directions, sampled at 33 corresponding ticks from -2.4 to +2.4.
   A ray has initial pixel speed 1 at its own start. These are finite probes;
   local SVD is not recomputed at every tick within a ray.
4. For each candidate pairing with a root family, measure three entire profiles:
   the pixel distance between paths at corresponding ticks; the difference
   between their displacement vectors after subtracting their own starts; and
   the angle between those displacement vectors. The angle at tick zero is
   undefined and excluded.
5. Render both orientations. Choose the orientation with the smaller mean
   squared displacement-vector mismatch over the whole curve. This sign choice
   is part of the method: it uses finite pixel changes, not physical labels.
6. Divide each profile by its observed start-image gap. Divide angles by 90
   degrees first. Give the three profile blocks equal squared-error weight.
   This asks whether an offset mostly changes a common profile's amplitude.
7. Enumerate all 120 one-to-one family assignments at the nearest offset.
   For each, Hungarian assignment finds the five jointly assigned families at
   every other offset that best match the nearest-offset profiles. Select the
   minimum total squared profile discrepancy. This solves the stated anchored
   objective exactly; it is not a global optimization over arbitrary curves.
8. Average the four selected normalized profiles for each family. Freeze these
   templates, then choose families at fresh offsets 0.3 and 0.7 with the same
   Hungarian rule. Only after all selections are frozen, inspect known controls.
9. For the repeated walks, center the same bundle procedure at the current
   state, using the moved family as the context branch. Its 0.3 endpoint is the
   next state. Orient and relabel all five axes there, then repeat. This includes
   cross-path evidence from the other four families in the joint assignment.
   All five trajectories freeze before their known-control evaluation.

All profiles are measured in the original 784 pixel values. No image segmentation,
edge detection, physical-knob classification, or tangent-cosine label matching
enters the selection cost. The finite displacement-mismatch profile and sign
rule do use pixel-vector relationships, as in the right panel of the reference.

## What the figures show

- `01_selected_bundles.png`: five selected family bundles, each at four offsets
  along another root direction. The middle/right curves show shared dips,
  plateaus, and bends, rather than distance along a single walk. Five of the
  25 scored family/context pairs are displayed. Contexts are chosen in advance
  to show width/lean and height/vertical-position interactions.
- `02_pure_control_bundles.png`: exact physical-family rays at the **same branch
  starts**, using the same local pixel-speed calibration. Small discrepancies
  expose SVD mixtures. Branch starts themselves can contain mixed changes.
- `03_validation.png`: all 25 family/context pairs, fresh-offset validation,
  and profile costs compared with correctly labeled axes and pure directions.
- `04_repeated_walks.png`: the five nine-hop walks, with pure-control references.
  Distance-from-root and turning-angle curves are **diagnostics**, not the
  fitting target. The fitting target remains the profiles across offset paths.

The shapes do not reproduce the reference image numerically: that figure used
center `(14.5, 14.5)` and a much longer lean sweep. Here the canonical root and
short local rays follow the requested local experiment.

## What seems promising, and what remains open

The shared-profile rule successfully picked the physical-family labels in this
local sample. The next-best nearest-offset assignment had a much higher cost
on every context branch. Dividing by the start gap helps reject incorrect
pairings: their displacement mismatch can remain large even as their starts
approach one another, rather than shrinking with the branch offset.

But profile agreement is approximate even for pure controls. Height and vertical
position produce substantially less coherent profiles than width/lean. Their
patterns include row-boundary changes and do not collapse to a single rescaled
curve. The correct family can win without the common-profile hypothesis being
exact.

Important limits:

- **Generator-assisted probes.** The renderer and its Jacobian supply valid
  moves: each requested SVD pixel direction is lifted to the closest parameter
  velocity, then that parameter ray is held fixed for the probe. This supplies
  a continuation rule that would have to be learned on an unknown manifold.
  The selector does not receive parameter values or physical family labels.
- **Limited walk length.** Five nine-hop walks were tested. Longer paths and
  other roots remain untested. This is not a controlled comparison with the
  previous matcher: the sign rule, amount of probe evidence, and matching
  objective have all changed. In particular, the finite-curve sign rule can
  address reversals that a single local dot product missed.
- **No recovered common knob numbers.** Corresponding ticks have equal initial
  pixel-speed calibration. They need not be equal physical knob increments.
- **Restricted candidates.** Only the five SVD axes are considered. Successful
  labeling does not solve the harder mixture problem or identify unique knobs.
- **A specific scaling assumption.** Profiles are assumed roughly proportional
  to observed start gap. This is motivated by the reference figure and tested
  here, not guaranteed for arbitrary manifolds. The nearest offset is the
  anchor; changing that choice could affect results.
- **Local, assisted sampling.** Default-range-shaped sampling and the common
  Sobol seed can influence the SVD axes. Both sides of the root's position
  boundaries are allowed, so 30 of 31 observed starts fall outside the default
  five-knob box, often through a tiny mixed displacement. All probe strokes
  remain positive and unclipped.
- **Interpolation check.** Fresh offsets lie between training offsets, along
  the same five context rays. This is not a new-root or extrapolation test.

## Reproduce and inspect

Run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/svd_cross_path_patterns.py
```

`results.json` records choices, assumptions, scores, and numerical checks.
`associations.csv` records every selected family and sign evaluation.
`paths_and_profiles.npz` retains the rendered candidate curves, descriptors,
assignments, templates, true-control curves, and fresh-offset curves.
`repeated_walks.npz`, `walk_associations.csv`, and `walk_diagnostics.json` retain
the trajectories, signed evaluations, and per-hop bundle costs.
The experiment uses one method and fixed settings; there is no method sweep.
