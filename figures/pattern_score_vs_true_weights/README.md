# Does the distance-and-angle score favor the true knob combinations?

**Not in general.** The original rule selected the correct physical-family
labels among five available SVD axes. It did not establish that the true
five-vector weights minimize its objective. This small scan finds explicit
mixed directions with better scores than the true combinations.

## Fair comparison and scope

The primary comparison preserves the previous experiment exactly: one root,
the same branch starts, the same SVD neighborhoods, 33 path ticks, four training
offsets, two fresh offsets, profile normalization, and cost weights. Most
importantly, every candidate uses the **same original root-reference paths**.

The earlier pure-control calibration plot used pure root-reference paths as
well as pure paths at the branch starts. Its cost therefore was not a direct
comparison against the original fixed SVD-reference objective. The present
scan holds that reference fixed, then changes it in a separate controlled
anchor diagnostic.

There are 20 family/context pairs with different path and context families:
five path families, each evaluated along four other root-direction branches.
Self-context pairs are retained in the saved arrays for reproducing the
original objective but excluded from the primary summary.

This is a local scan from the canonical root, not a proof about arbitrary
mixtures or other parts of the manifold. True directions are supplied only
to construct and evaluate the comparison candidates.

## The tested combinations

At each branch start, express both the original selected SVD axis and the
known true direction in its five-vector basis. Test:

```text
candidate weights = (1 - alpha) × original SVD weights
                    + alpha × true-knob weights
```

The grid is `-1, -0.5, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2`.
Alpha 0 is the original axis. Alpha 1 is the known true-knob projection.
Negative values move away from the true combination; values above 1 pass it.
The true direction's tiny component outside the five-vector span is projected
away. The same Jacobian lift as before realizes a valid parameter-ray motion,
normalized to initial pixel speed 1. At alpha 1, the realized directions differ
from the known true directions by approximately 1e-12 in this dataset.

The score still compares three complete profiles: corresponding-point image
distance, displacement-vector mismatch, and displacement-vector angle. Each
profile is divided by the observed start-image gap; angles are first divided
by 90 degrees. Costs measure differences from the nearest-offset profile.
No straight-line distance target or purity term is added.

For the aggregate family result, choose **one alpha per family**, shared
across all four other context branches. Minimize its summed training score.
All five selected directions remain independent: the smallest singular value
of the selected unit-direction frame is at least 0.979 for the original
reference and 0.888 for the pure reference. The counterexamples are not made
by collapsing the families into one direction.

## Original SVD root reference

In every one of the 20 cross-knob pairs, at least one tested mixture scores
better than the true combination. Different pairs can prefer different alphas.
The true combination beats the original axis in only 5 of 20 pairs.

When one alpha is shared across each family's four contexts:

| Family | Best tested alpha | True-combination score | Best-tested score | Score reduction versus true | Mean direction error versus true |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lean | -0.50 | 0.00830 | 0.00562 | 32.3% | 2.00% |
| Width | 0.25 | 0.08842 | 0.04034 | 54.4% | 2.34% |
| Horizontal position | -0.50 | 0.06617 | 0.04226 | 36.1% | 4.32% |
| Vertical position | 0.00 | 0.27165 | 0.20431 | 24.8% | 4.15% |
| Height | 0.25 | 3.04217 | 0.45380 | 85.1% | 10.89% |

Direction error means Euclidean error between unit pixel-change vectors, not
percentage change in a physical knob. Lean and horizontal position prefer
moving *away* from their true directions on this tested line of combinations.
Height shows the largest gap between score optimality and physical purity.

Fresh offsets use the mean training profile fitted for each candidate alpha,
with alpha chosen only on training data. The best-tested mixtures retain a
lower fresh-offset score than the true combinations for four of the five
families. Horizontal position does not: its fresh score is worse. Thus many
of the counterexamples survive this small interpolation check, but not all.

## Change only the root reference to pure directions

The candidate mixtures, branch starts, sampling, and scoring remain identical.
Only the five reference paths at the root are replaced with known pure-knob
paths. This is an anchor diagnostic, not a blind recovery procedure.

The preferred alpha moves closer to 1 for **all five families**:

| Family | Best alpha with original root | Best alpha with pure root | Pure-root score reduction versus true |
| --- | ---: | ---: | ---: |
| Lean | -0.50 | 0.75 | 6.95% |
| Width | 0.25 | 0.75 | 19.24% |
| Horizontal position | -0.50 | 0.50 | 33.22% |
| Vertical position | 0.00 | 1.00 | 0.00% |
| Height | 0.25 | 1.25 | 4.10% |

Here true combinations beat the original axes in 15 of 20 cross-knob pairs.
Nevertheless, some tested mixture still beats the true combination in 10 of
20 pairs. Four of the five aggregate family minima are still slightly mixed.
On fresh offsets, the optimized lean, width, and horizontal-position mixtures
retain their advantage; height does not. Vertical position is exactly the
true candidate and ties it.

For a descriptive correlation check within each pair, compare the ranks of
score and true-direction error over the ten tested alphas. The median Spearman
correlation across the 20 pairs is -0.236 with the original reference and
+0.964 with the pure reference. Positive correlation means lower scores tend
to accompany smaller true-direction errors. These are correlated, constructed
local comparisons, not population estimates or confidence intervals.

## Interpretation

The original objective encourages consistent continuation of the root's
chosen directions, including their mixtures. It is not a direct measure of
physical-knob purity. Changing only the root reference demonstrates strong
anchor dependence in this scan, especially for height, whose original root
axis needed the largest correction.

Even pure physical families need not be exact minimizers: their pixel distances
and angles can change with state, and the common-profile/proportional-scaling
assumption is only approximate. Corresponding ticks also use local pixel-speed
calibration rather than common physical-knob increments. These facts leave
room for a slightly mixed direction to improve the profile score.

The result supports family association but does not justify automatically
unmixing vectors by minimizing this score alone. A useful next experiment
would fit the root mixtures and branch mixtures jointly, with a shared rule
for their state dependence and consistent step calibration, then check fresh
joint moves. Even a consistent coordinate system would not uniquely identify
the original physical knobs without an additional selection principle.

## Verification and files

- Alpha 0 reproduces every original branch's five-family score within 1e-9.
- Alpha 1 with the pure root reference reproduces the previous pure-control
  calibration scores within 1e-7.
- All alpha 0 curves reproduce the saved candidate paths within 5e-12 per pixel.
- Every tested path remains a valid positive, unclipped stroke.
- The original whole-curve sign choice agrees with the orientation of every
  tested mixture under both references.

`score_vs_weights.png` shows score curves, best tested weights, and remaining
physical-direction error. `scores.csv` records the 20 pairs at every alpha
under both references. `scan_data.npz` includes all profiles and the diagonal
pairs. `results.json` records the aggregate and fresh-offset results.

Reproduce from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/check_pattern_score_vs_true_weights.py
```
