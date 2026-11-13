# The double-layer check (`verify.py`)

**Use it on every flattening.** It answers one question: *did the flattening put part of
the sheet on top of another part?* That covers a fold that came out still folded, a loop
squashed onto itself, and any other double layer the flattening created. It needs only
what you would have on real data: the 3-D dots and the flat result. It never looks at the
truth, so it works where the suite's scorer cannot (the generated 1s, or any real
dataset).

Every suite run already calls it. The terminal shows it as `layers: ...` next to
PASS/FAIL, and `results.json` stores it under `worst["double_layer"]`.

## How it works

Paper that is only bent, not stretched, can never come out **closer** when flattened.
Walking along the paper is never shorter than the straight line through space. So:

1. Resize the flat result so that ordinary nearby dots keep their 3-D spacing (median 1).
2. For each dot, take its 10 nearest dots **in the flat result**. Compare each one's real
   3-D distance with its flat distance. The dot's **ratio** is the worst of the ten:
   *how many times too close the flat result puts a neighbour.* An honest flattening gives
   about 1 everywhere.
3. **Flag** every dot whose ratio is above 2.
4. Group flagged dots that touch (nearest neighbours in 3-D or in the flat result) into
   **patches**. One big connected patch is the signature of a stacked area. Scattered
   single flags are not.

## Using it

```python
from unfolding_suite.verify import verify
v = verify(points, flat)        # points (n, 3) and flat (n, 2), same dot order
v["reading"]                    # "clean", "specks", "mild" or "stacked"
v["per_dot"]                    # each dot's ratio: colour the flat result by it to see WHERE
```

```bash
.venv/bin/python -m unfolding_suite.verify --run current
```

The command checks every saved layout of a suite run and prints each reading next to the
truth-based verdict, with a count of agreements and disagreements.

## Readings

| Reading | Rule | What it usually means | What to do |
| --- | --- | --- | --- |
| `clean` | No dot flagged | Nothing stacked that this check can see | Trust it, subject to the blind spots below |
| `specks` | Flags, but every patch is under 1% of the dots | Wobble at the scale of single dots, not a fold. One method leaves about 0.4% specks on every sheet | Fine on clean data. Above about 5% flagged, the data are noisy and small folds can hide here (see blind spots) |
| `mild` | A patch of 1% or more, but under 5% of dots in patches **and** typically under 2.5× too close | Mild squashing or stretched paper, not a fold | Usually fine. Look at where it is |
| `stacked` | A patch of 1% or more that is not mild | Part of the sheet lies on top of another part (or, rarely, a region is crushed hard) | Do not trust the flattening there. Colour by `per_dot` to find the area |

## The numbers

| Key | Meaning |
| --- | --- |
| `flagged_share` | Share of dots more than 2× too close |
| `stacked_share` | Share of dots in patches of 5 or more: the size of the problem. The terminal shows it as `stacked 34%` |
| `biggest_patch_share` | Share of dots in the single largest patch |
| `patches` | Number of patches of 5 or more dots. One big patch means one stacked area; many patches mean scattered damage |
| `flagged_ratio` | How many times too close the flagged dots typically are. The terminal shows it as `(7x)` |
| `typical_ratio` | Median over all dots. About 1.0-1.1 when the flattening is honest |

**How big is the problem?** `stacked_share` counts **both** layers of a double layer, the
one on top and the one underneath. So it is not the folded area. On bent sheets in the
reference run, it was 1.0-6.8 times the truth's folded share (median 1.8). Larger means
more is wrong, but don't convert one into the other.

**How far apart were the stacked parts?** `flagged_ratio` grows with how far apart in
space the stacked parts really are:

- **10-90×:** parts that are far apart in space were stacked, e.g. rolls laid on themselves,
  a squashed loop, or a projection onto a plane.
- **2-3×:** the stacked layers really are close in space, e.g. a thin fold pressed flat
  (layers under one dot spacing apart) or an unevenly stretched crumple.

On the folded results in the reference run it ranged from 2.5 to 89 (median 25).

## How well it works (calibration)

Every saved suite run was checked against the scorer's truth-based fold check. Rows are
saved in `runs/verify_calibration/`.

| Run | Truly folded: read `stacked` | Not folded: read `clean` / `specks` / `mild` / `stacked` |
| --- | --- | --- |
| `current` (all 24 sheets, reference) | 54 of 54 | 39 / 20 / 5 / 2 |
| `blind_holdout` | 15 of 15 | 31 / 19 / 3 / 0 |
| `explore_hops` | 44 of 44 | 65 / 2 / 0 / 1 |
| `noise_0.1` (3-D noise 0.1 dot spacings) | 3 of 3 | 6 / 6 / 0 / 0 |
| `tight_roll_16k_dots` | 1 of 1 | none |
| `noise_0.3` (3-D noise 0.3 dot spacings) | **6 of 15** | none |

- **Without heavy noise, it missed no fold:** 117 of 117 folded results read `stacked`.
- **False alarms (`stacked` but not folded): 3 results.** All three are projections onto a
  single flat plane that crush part of the sheet: `bowl / tangent_plane` (twice, 10-18% at
  2.4×) and `h_three_lobes / best_fit_plane` (60% at 6.4×, where the plane sees the lobes
  nearly edge-on). The scorer calls them unfolded because nothing flips, but the flat
  result really does put those dots several times too close. Read it as "crushed or stacked".
- **`mild`:** 8 results, all unfolded: one stretched, dented held-out sheet (`h_dents23`)
  under every method, and one plane projection of a cone. The `mild` thresholds were set on
  `current`, so its agreement there is not a fair test. `blind_holdout` agrees too, but it
  shares sheets with `current`.
- **Folds the size of single dots, under heavy noise:** at 0.3 dot spacings of noise, 9
  results with small, scattered real folds (0.3-2.2% folded) read `specks` with 5-22% of
  dots flagged. On clean data `specks` never went above 4.9% flagged.

The two folded sheets in `experiments/fold_flatten.py` (a flap folded flat, and a loop
that swings past the base) read `stacked` at gaps 0.02 and 0.05 under all four methods tried
(32-94% of dots, 3-29× too close). With a wide gap (0.10) they flatten correctly and read
`clean` or `specks`, except under `lle_k10`. That method warps parts of the sheet on its own
and reads `stacked` at 6-8% and about 2× on both sheets without folding them. This is
another case of a region squashed rather than stacked.

## Blind spots

1. **Very thin double layers look honest.** If two stacked layers are much closer than the
   dot spacing, the stacked flat result agrees with every 3-D distance. A flap folded flat
   at gap 0.005 (dot spacing 0.031) flags 0.1% of dots and reads `clean`. At gap 0.01, 9% are
   flagged. At 0.02, 33%. No check built only on distances can see this case.
2. **Heavy noise hides small folds** among the specks (above).
3. **It assumes the paper was bent, not stretched.** Uneven stretching can make honest
   areas `mild`, and it inflates `stacked_share` on crumples (22% stacked where 1.2% is
   folded).
4. **It only sees "too close".** It does not see tears, parts pushed too far apart, or a
   result that is the wrong shape. In the suite, the scorer still judges those.
5. **It looks at 10 neighbours.** Patches smaller than 5 dots are ignored.

## When it disagrees with the suite's verdict

| Suite says | Check says | Most likely |
| --- | --- | --- |
| folded | `stacked` | A real double layer. The usual case |
| folded | `clean` / `specks` | A very thin fold, or small folds in noisy data |
| passed | `stacked` | A region crushed without flipping, e.g. a plane projection |
| passed | `mild` | Stretched paper or mild squashing. Usually harmless |
