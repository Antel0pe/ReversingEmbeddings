# Unfolding test suite

A test bench for one question: **given only the 3-D positions of dots scattered over a
crumpled sheet, and told that the sheet is 2-D, can a method flatten it so that every
point of the sheet gets its own spot (one-to-one), with nothing folded over?**

Everything about the suite lives in this folder: code, test sheets, methods, saved runs,
figures and this guide. It is a toy for the repo's larger goal (recovering the five knobs
of the generated 1s from images alone; see the root `AGENTS.md`), but it does not import
anything from the rest of the repo.

## In one minute

- A **sheet** is the square of paper [-1, 1]² (side 2) bent or stretched into 3-D, sampled
  with 65 × 65 = 4,225 dots.
- A **method** is a Python function: 3-D dots in, one flat (x, y) position per dot out.
- A **test** is one sheet against one method, on several random seeds. It passes only if it
  passes on every seed.
- **Pass** means the flat result is certified one-to-one, smooth, and, for sheets that were
  never stretched, the original square to within a few percent.
- Every failed test gets a page that shows where the sheet went wrong and names the cause.

**Where things stand** (reference run `runs/current/`, seeds 100-102):

| Method | Design sheets (14) | Held-out sheets (10) |
| --- | ---: | ---: |
| `best_fit_plane` | 4 | 1 |
| `tangent_plane` | 3 | 0 |
| `isomap_k10` (plain walking) | 10 | 7 |
| `check_k10` (walking + link check) | 11 | 7 |
| `convex_edge` (link check + convex edge) | 11 | 7 |

Still unsolved: `tight_roll` at 4,225 dots (more dots fixes it) and the randomly twisted
crumples (nothing tried fixes them). Details under [Results so far](#results-so-far).

## Quick start

From the repo root, with the repo's virtual environment:

```bash
.venv/bin/python -m unfolding_suite.test_scoring
```

```bash
.venv/bin/python -m unfolding_suite --list
```

```bash
.venv/bin/python -m unfolding_suite --sheets s_curl roll --methods isomap_k10 check_k10 --seeds 1 --name try
```

1. Checks the scorer itself: 18 tests, about 20 seconds.
2. Lists every sheet and method.
3. A small run, about 30 seconds. Results go to `unfolding_suite/runs/try/`.

A full run (`--sheets all`, five default methods, three seeds) takes about 10 minutes.

## Folder map

| Path | What it is |
| --- | --- |
| `sheets.py` | The 14 design sheets and `make_instance`, which samples a sheet and hides its layout |
| `holdout.py` | 10 held-out sheets, kept out of method design for fair checks |
| `methods.py` | The method contract, registry and baseline methods; `layout_from_links` |
| `link_check.py` | The two methods that survived iteration: `check_k10` and `convex_edge` |
| `scoring.py` | The one-to-one certificate and every pass/fail threshold |
| `diagnose.py` | Names the cause of a failure, using evidence |
| `report.py` | Scorecard, galleries and failure pages |
| `run.py` | Command line (`python -m unfolding_suite`) |
| `test_scoring.py` | Tests of the scorer, built from layouts with known flaws |
| `experiments/` | One-off studies: `negative_space.py` + `negative_space_why.py` + `negative_space_score.py` (nearness-only, push-apart and air-probe methods, and why they fail; see below), `adaptive_hops.py` (an example plugin, rejected), `why_hops_jump.py` (side view of hops jumping layers), `make_twitter_graphic.py` |
| `runs/` | Saved runs, one folder each (listed under [Saved runs](#saved-runs)) |
| `figures/` | Pictures from `experiments/` |

## The problem and the contract

### What a method is given

`make_instance(sheet, grid=65, seed, noise=0)` samples a sheet and hides how it was made:

- The dots sit on a jittered grid on the paper. Jitter is under a quarter of a cell, and
  edge dots only slide along their edge, so the true outline is exactly the square.
- The 3-D positions are given a random rotation and shift, and their order is shuffled.
- `noise` adds random 3-D displacement, measured in grid spacings.

A method sees only `instance.points`. The suite keeps the true paper positions
(`instance.uv`) and the unhidden 3-D positions (`instance.xyz`) for scoring and diagnosis.
**A method must never use them.**

### What a method returns

```python
from unfolding_suite import method

@method("my_idea")
def my_idea(points, dim, rng):
    """One line describing the idea; it is shown by --list."""
    # points: (n, 3) shuffled, rotated 3-D dots.  dim: always 2.  rng: numpy Generator.
    return flat                     # (n, 2): where each point lands, in the SAME order as points
```

This list of pairs, each 3-D dot with its flat position, *is* the mapping between 2-D and
3-D. The suite knows which dots were neighbours on the paper, joins them into triangles,
and fills in the map between dots, so the list is all it needs to judge the whole sheet.

- Turning, mirroring, moving or resizing the answer does not matter.
- If a method raises an exception, that test is recorded as `crashed` and the run continues.
- A method may return `(flat, info)` instead, to improve the failure diagnosis. All keys are optional:

| Key | Meaning |
| --- | --- |
| `info["edges"]` | `(m, 2)` indices into `points`: the neighbour links the method relied on |
| `info["plane"]` | `(3, 2)` the plane's axes, for methods that project onto a plane |
| `info["captured"]` | Share (0 to 1) of the method's own distances that a flat layout can express |

`layout_from_links(points, links, dim)` does "shortest walk along these links, then the best
flat layout", so an idea about which dots to link takes a few lines (see `link_check.py`).

## Sheets

`--list` prints them all. Each sheet has a **family** and a **group**.

**Family**, which decides what counts as correct:

- `bent`: bent without stretching, like real paper. Lengths along the sheet equal lengths
  on the square, so the correct flat result is the original square.
- `stretched`: stretched while being shaped. No flat result can match the square, so only
  one-to-one and smoothness are judged.

**Group**, which decides what the sheet is for:

- `design` (14 sheets): used while developing methods. This is the default.
- `holdout` (10 sheets, names start `h_`): kept for fair checks. See [Fair checks](#fair-checks).

What the design sheets test:

| Sheet | Family | What makes it hard |
| --- | --- | --- |
| `flat`, `gentle_arc`, `wave` | bent | Nothing overhangs. Controls: everything sensible passes |
| `s_curl` | bent | Two curls whose free ends come close to the middle layer |
| `roll`, `diagonal_roll` | bent | A spiral with room between layers; the second is rolled at 45 degrees |
| `tight_roll` | bent | Its layers are closer together (0.017) than neighbouring dots (0.025) |
| `accordion` | bent | Four sharp creases |
| `cone` | bent | Bends along a fan of lines, not parallel ones |
| `bowl`, `dents` | stretched | Mild stretching, never overhangs |
| `uneven_roll` | stretched | A spiral stretched up to 5× along one direction, so dots are unevenly spaced |
| `crumple`, `hard_crumple` | stretched | Random twists of space: folds over and stretches unevenly. The sheet never passes through itself |

## How a result is judged

All thresholds are constants at the top of `scoring.py` and `diagnose.py`. Distances on
the paper are in paper units (side = 2), and "% of the side" means percent of 2.

### The one-to-one certificate

The suite joins the grid's neighbouring dots into triangles (it knows the grid; the method
does not) and looks at each triangle in the flat result.

- **Flipped triangle:** the triangle comes out mirror-image. That means the sheet has folded over there.
- **Outer edge crossing:** the paper's outline crosses itself in the flat result.

**If no triangle flips and the outline never crosses itself, the map filled in across each
triangle is one-to-one at every point of the paper, not just at the dots.** Both conditions
are needed: `test_wrapping_over_itself_with_no_flipped_triangle_is_caught` builds a layout
that lies over itself with every triangle the right way up.

The check runs on three mesh sizes: every dot, every 2nd and every 4th dot (`STRIDES`).
Triangles spanning several dot spacings cannot be flipped by small wobble, but a real fold
still flips them. That is how wobble is told apart from folding.

### Verdicts

| Verdict | Meaning | Passes? |
| --- | --- | --- |
| `certified` | No flips at any mesh size, no outline crossings | yes, if the other checks pass |
| `wobble only` | Flips only at the finest mesh, covering under 1% of the paper (`WOBBLE_LIMIT`) | yes, if the other checks pass |
| `folded` | Anything else | no |
| `crashed` | The method raised an exception or returned a bad array | no |

### The other checks

| Check | Rule | Constant |
| --- | --- | --- |
| Square | Bent sheets only. After the best turn, mirror, shift and resize: typical dot at most 3% of the side from its true place, worst at most 10% | `SQUARE_MEDIAN`, `SQUARE_WORST` |
| Torn / crushed | An edge more than 5× longer, or 5× shorter, than the edges around it. Up to 0.05% of edges (about 4) are allowed as isolated outliers | `TEAR`, `CRUSH`, `TEAR_TOLERANCE` |
| Roughness | How abruptly the stretching changes between neighbouring triangles, 95th percentile. At most 0.5 | `ROUGH_LIMIT` |
| Overlap share | Reported, not a pass rule. Of the flat area covered, the share covered more than once, counted on a fine grid. It catches far-apart parts landing together | none |

**Pass** = verdict `certified` or `wobble only`, and square (bent sheets only), and smooth
(no tear, no crush, roughness within limit). **A test passes only if every seed passes.**
Reports show the worst seed.

### Interpretation notes

- "Folded N%" in a scorecard cell is the larger of the folded share and the overlap share.
- Roughness and tears are measured against the paper's own grid, so on a strongly stretched
  sheet even a faithful flattening can look rough. On the crumples this is part of why
  they fail.
- Every threshold is a judgement call. Each was set by testing the scorer on layouts with
  known flaws and by looking at results.

## Reading the outputs

Each run writes `runs/<name>/`:

| File | How to read it |
| --- | --- |
| terminal | One line per test: PASS/FAIL, a short reason, seconds, and the first named cause |
| `scorecard.png` | Sheets down the side, methods across. Green passes on every seed; red shows what went wrong on the worst seed |
| `gallery_bent.png` | Each row: the crumpled input (coloured by where each dot sits on the paper) and every method's result, lined up with the original square (grey outline). Colour = how far each patch landed from its true place, pale to dark. Red = folded over. Thin lines = graph paper printed on the sheet |
| `gallery_stretched.png` | The same for stretched sheets, but results are coloured by the paper's own print, because the square is not their target. A smooth blend is good; speckle means parts landed together. Under each panel: the spread of short lengths against the 3-D sheet (1× = unchanged) |
| `failures/<sheet>__<method>.png` | One page per failed test. Left: where on the paper it went wrong (red = flipped, amber = landed on a far-away part), with an example pair A and B. Middle: the 3-D sheet, with any links that jump between layers in black. Right: the flat result, with A and B landing together. The header names the cause and its evidence |
| `results.json` | `settings` (the command's options) and `tests`. Each test, keyed `"<sheet> / <method>"`, holds `passed`, `seeds_passed`, `words` (the scorecard text), `worst` (every number for the worst seed) and `causes` |
| `layouts.npz` | The worst seed's flat result for each test, keyed `"<sheet> / <method>|<seed>"`. `--redraw` uses it to redraw galleries without rerunning |
| `comparison.json` | Only with `--against`: the tests that were `fixed`, `broken` or `still_failing` |

## Named causes of failure

`diagnose.py` names a cause only when there is evidence for it. It may use the truth,
because it explains failures and is not a method.

| Cause | Evidence |
| --- | --- |
| Hops jumped between layers | Some of the method's links join dots more than 0.3 apart on the paper (`FAR`), and the same layout step passes with perfect neighbours |
| Hops were lopsided | Over 1% of dots have links lying nearly along one line on the paper, and perfect neighbours pass |
| Hops: unknown reason | Perfect neighbours pass, but neither of the above shows |
| The layout step fails even with perfect hops | Given each dot's true paper neighbours, the same layout step still fails |
| The sheet turned past edge-on to the plane | Plane methods: part of the paper faces away from the plane |
| The plane shortens tilted parts | Plane methods: nothing folds, but the result is not square |
| Unexplained | None of the above. A new way to fail: add a check for it |

**Perfect hops** (`with_perfect_hops`) reruns the walking layout using each dot's 10
nearest neighbours *on the paper*. It separates the two stages of a walking method:
choosing which dots to link, and laying the result flat.

## Workflow

### Iterating on an idea

1. Write the idea as a method in its own file. `experiments/adaptive_hops.py` is an example.
2. Run it against the reference run's plain walking:

   ```bash
   .venv/bin/python -m unfolding_suite --plugin my_idea.py --methods my_idea --name my_idea --against current:isomap_k10
   ```

3. Read the printed `fixed`, `broken` and `still_failing` lists.
4. For each failure, open its page in `runs/my_idea/failures/`. Fix the named cause, or add
   a check to `diagnose.py` if it says `unexplained`.
5. If a new kind of crumpling matters, add a sheet (design group) and check it with
   `test_every_sheet_is_valid`.

Prefer simple methods. Add machinery only when a test still fails, and remove anything that
did not help.

### Fair checks

Results on the design sheets are optimistic, because methods were tuned on them.

- Before calling something better, run the held-out sheets on fresh seeds:

  ```bash
  .venv/bin/python -m unfolding_suite --sheets holdout --first-seed 100 --seeds 3 --name fair_my_idea
  ```

- Several settings of one idea are several methods. Report all of them, not the best.
- Save the first held-out run as it is, even if a bug shows up. `runs/blind_holdout/` is
  the example. If a method changes after you look at held-out results, those sheets are no
  longer a fair check for it: add new held-out sheets instead.

### Adding things

| To add | Do this |
| --- | --- |
| A method | `@method("name")` on a function with the contract above, in a plugin file or in `methods.py` |
| A sheet | `@sheet("name", "bent" or "stretched", "note", group="design")` on a function from `(n, 2)` paper positions to `(n, 3)`. Use `bent` only if it never stretches |
| A cause | A check in `diagnose.diagnose` that appends to `out["causes"]` only when its evidence is present |
| A scorer test | A `test_` function in `test_scoring.py`: build a layout whose flaw is known, assert the scorer catches it |

## Command line

`python -m unfolding_suite [options]`. It is the same as `python -m unfolding_suite.run`.

| Option | Meaning |
| --- | --- |
| `--list` | Print every sheet and method, then stop |
| `--sheets` | Sheet names, or the groups `design` (default), `holdout`, `all` |
| `--methods` | Method names. Default: `best_fit_plane tangent_plane isomap_k10 check_k10 convex_edge` |
| `--seeds`, `--first-seed` | How many seeds, starting where. Default: 2 seeds from 0. Use 100+ for fair checks |
| `--grid` | Dots per side. `grid - 1` must divide by 4 (65, 129, 257, 513) |
| `--noise` | 3-D noise, in grid spacings |
| `--name` | Output folder under `--out-root` (default `unfolding_suite/runs/`) |
| `--plugin` | A `.py` file that registers more methods or sheets |
| `--against` | Compare with a saved run, as `NAME` (same methods) or `NAME:METHOD` (one method from it) |
| `--redraw` | Redraw the galleries of run `--name` from its saved layouts. Pass the same `--plugin` if the run used one |

## Performance and limits

- `isomap_*` and `check_k10` store every pair of walking distances: about 3 seconds at
  4,225 dots, but memory grows with the square of the dot count (about 35 GB at 66,049).
  `convex_edge` walks only from the edge dots: 0.3 seconds at 4,225 dots and 30 seconds at 66,049.
- The dots are more evenly spread than real data. `--noise` adds 3-D noise, but there is
  no uneven-sampling option yet.
- "Certified" is about the map filled in across the triangles. It cannot see anything finer
  than the grid.
- Results are reproducible: the same seeds give the same numbers. Runs made before this
  was fixed (`blind_holdout`, `explore_hops`, `noise_*`) can differ slightly from a
  rerun for the walking methods, because their solver's starting guess depended on what ran earlier.

## Results so far

The two methods in `link_check.py` both start from a **link check**: keep each dot's 10
nearest links, but drop any link that points out of the sheet's fitted surface at either
end. A true link runs along the surface; a link that jumps to another layer leaves it.

- `check_k10`: walking along the checked links, then the usual flat layout.
- `convex_edge`: lay out only the edge dots, push them onto a convex outline, and put each
  inside dot at the average of its neighbours. No all-pairs step, so it scales.

Findings, with the run that shows each:

- **Why plain walking fails:** a hop longer than the gap between two layers lands on the
  other layer. On `s_curl` with 10 neighbours, 11 bad links out of 42,250 fold 9% of the
  paper. The link check fixes it. (`runs/explore_hops/`, `figures/why_hops_jump/`)
- **Choosing hop length by local flatness makes things worse.** The jumps happen where the
  sheet is flat, which is exactly where that idea allows long hops. (`experiments/adaptive_hops.py`)
- **`tight_roll` is a dot-count limit.** With `convex_edge` it folds 13% at 4,225 dots and
  8% at 16,641, and passes at 66,049 and 263,169 dots, about 0.2% off the square.
  (`runs/tight_roll_*_dots/`)
- **The twisted crumples are a layout limit.** Even perfect neighbours leave them folded.
  `check_k10` folds 1% of `crumple` and `hard_crumple`, and 5% and 10% of the two held-out crumples.
- **Noise:** at 0.1 grid spacings, 4 of 5 sheets still pass. At 0.3, none do. (`runs/noise_*`)

Tried and dropped, each by measurement: a stricter link angle (good links disappear before
bad ones), pruning links that share few neighbours, cutting links the layout disagrees
with, growing the surface outward from one patch, neighbour averaging, weighting short
distances, and placing the edge on a circle.

**Open problem:** a flattening step that cannot fold, for sheets stretched unevenly.

### Negative space experiments (2026-11-09)

Tested three ideas: the dots carry no notion of a sheet, nearest dots, directions and
distances cannot be trusted, and a method needs evidence of where the sheet is *not*.
Methods are in `experiments/negative_space.py`. Scores are in
`runs/negative_space/scores.json` (all 24 sheets, seeds 100-102, made by
`negative_space_score.py`, which skips failure pages). Truth-based diagnosis is in
`runs/negative_space/why.json` (`negative_space_why.py`, seed 0).

| Method | Design (14) | Held-out (10) | Why it fails |
| --- | ---: | ---: | --- |
| `nearby_only` (who is near whom, no lengths) | 0 | 0 | Even with the true neighbours it squashes the edges (4.3% off the square, edge triangles crushed), and it has no defence against jump links |
| `push_random` (UMAP-style random negatives) | 0 | 0 | Tears every sheet: pushing random pairs apart, with no target length, bunches the dots into clumps with gaps between them |
| `push_random_cold` (the same, from a random start) | 0 | 0 | Folds 94-100%: from a scrambled start the pushes settle into a scrunched tangle |
| `push_targeted` (push apart pairs that are close in the picture but far along the sheet) | 11 | 6 | Fixes nothing, and makes `crumple` worse (1% to 4%). On `crumple`, none of its 62 negatives is really far apart on the paper; they are squeezed parts. On `h_crumple29`, 469 of 584 are real overlaps, but pushing does not unflip a flipped piece |
| `air_probe` (drop links whose middle has no dot near it) | 0 | 0 | Cuts 21% of good links on unstretched sheets and 44-54% on stretched ones (stretched parts look like empty space). On `tight_roll` only 43% of jump links are removed, because the gap (0.017) is narrower than the dot spacing |
| `star_tutte` (non-crossing links, then `convex_edge`'s layout) | 10 | 7 | No better than `convex_edge`. Its links still cross on the crumples (4-205 crossing pairs) |
| `ring_circle(_star)` (outline ordered as a ring, put on a circle) | 2 | 1 | A circle cannot be the square. On crumples the found outline zigzags (8-21% of steps go backwards) and picks up false edge dots |

Key ceiling (truth-based): a layout where each inside dot is the average of its
neighbours, using the **true paper mesh** with the **true outline on a circle in its true
order**, passes all four crumples (certified, smooth). With the true 10-nearest links,
which cross each other, it still folds three of the four. On `crumple`, `convex_edge`'s
outline step puts 4.7% of edge dots out of order. So the crumples need two things the
dots do not give directly: links that never cross, and the outline in the right order.

### Pulling the paper apart (2026-11-09)

Question: would pulling a crumpled sheet apart, as you would by hand, make the hard sheets
easier? Code: `experiments/pull_apart.py` (part 1, part 2) and `pull_apart_figure.py`.
Results: `runs/pull_apart/*.json`. Figure: `figures/pull_apart/pull_apart.png`. The pull
strength (1e-3) was chosen by looking at `tight_roll`, `crumple` and `h_crumple29`, so the
held-out sheets are not a fair check for it.

- **Stretching the whole cloud along one of its main axes** (eigenvectors of its spread),
  then re-picking 10 nearest dots: fixes nothing. Along the longest axis it adds jump
  links (`s_curl` 12 to 519 at 2x). Along the thinnest axis at 2x it removes a few
  (`s_curl` 12 to 4, `h_crumple17` 56 to 24), but the sheet stays folded. A single stretch
  separates layers only where the paper faces the stretch direction. Where the paper runs
  along that direction, the stretch pulls its own dots apart.
- **Physical pull:** strings hold linked dots at their 3-D length while every dot is
  pushed away from the middle. The paper may pass through itself. The result is then
  dropped onto its best plane. With the paper's true neighbours as strings (a cheat
  control), this unrolls every roll, including `tight_roll`, which no method passes at
  4,225 dots. The third main axis's share of the spread falls from 2.3% to 0.0%. With
  strings chosen from the dots, any wrong string acts as a staple:
  - `s_curl` with plain nearest links: 12 staples, 35% folded.
  - With `checked_links`: `h_roll5` passes on 2 of 3 seeds (the third keeps 5 staples and
    folds 5%), and `tight_roll` keeps about 700-780 staples and folds 16-21%.
- **Crumples stall even with perfect strings:** 4-6% folded on `hard_crumple`. Pulling 3x
  harder does not change it. These sheets were stretched unevenly: 3-D link length runs
  from 0.4x to about 5x the paper length. No flat shape keeps those lengths, so extra paper
  puckers, and the puckers become folds.
- **Staples are not the tightest strings.** The two stapled layers move together, so the
  strain lands on good strings nearby. In none of 7 cases were the most stretched strings
  mostly staples.
- **Re-picking nearest dots after pulling** helps only when the strings were already clean
  (`h_roll5` 290 to 0 jump links, `h_crumple17` 56 to 0-1). With plain nearest strings it
  makes things worse (`s_curl` 12 to 537), because staples press layers together.

### Height split (2026-11-10)

Idea: paper has no thickness. Fit a plane, or a bowl, to a dot's K nearest dots, measure
each one's height above it, split the sorted heights at gaps wider than `gap` dot spacings,
and link the dot only to its own group. A whole other layer is dropped in one decision.
Code: `experiments/height_split.py` and `height_split_figure.py`. Scores:
`runs/height_split/scores.json`, all 24 sheets, seeds 100-102. Figure:
`figures/height_split/height_split.png`.

- **Ceiling (uses the truth):** on `tight_roll`, a perfect height cut separates the layers
  for 96-99% of dots in the outer turns, 71% where the bend radius is 0.09, and 0-5% in the
  innermost turn (radius about 2 dot spacings). Smaller neighbourhoods help a little: 89%
  separable overall at K = 12, against 61% at K = 40.
- **On its own:** 10/14 design and 7/10 held-out, the same as plain `isomap_k10` and worse
  than `check_k10`. It removes most jump links on `tight_roll` (8,745 down to 2,100-5,300,
  depending on settings), but the rest still fold it 24-28%. Every surviving jump on
  `s_curl` is at the paper's free edge, where the curl's end comes in at about 72 degrees.
  A layer arriving at an angle spreads its heights across 0, so it never forms a separate
  group. That case is the link check's strength.
- **Height split plus link check** (`both_K*`, a follow-up): passes `s_curl`, `roll` and
  `h_diag30`. `h_roll5` passes on 2 of 3 seeds, as `check_k10` does. `tight_roll` folds
  12-13% (`check_k10` 16%) and `h_crumple29` 7-8% (`check_k10` 10%), but there are no new
  passes. 665 of the 701 jumps left on `tight_roll` touch the paper's free end in the
  innermost turn. There the next layer wraps around the dot instead of lying at one height.
- **The walking layout tolerates almost no bad links:** 1 to 5 bad links fold `h_roll5` by
  2%, and 12 fold `s_curl` by 8%.

### Creased on one side, folded on the other (2026-11-10)

Question: if a sheet is creased (nothing overhangs) on one side and folded on the other, and
we flatten it by following nearby dots, where do we end up? And can that be seen from the
dots and the flat result alone? Code: `experiments/fold_flatten.py` (registers the sheets
below and an `lle_k10` method) and `fold_flatten_figure.py`. Results:
`runs/fold_flatten/*.json`, seeds 100-102. Figure: `figures/fold_flatten/fold_flatten.png`.
The sheets live in the experiment file, not `sheets.py`, so `runs/current` stays complete.

- **Sheets** (bent, so the answer is the square; bent along one direction, so a side view
  shows everything): three mild creases, then flat, then either `flap_gapXX` (a hairpin, and
  the rest lies flat over the base at a constant gap) or `loop_gapXX` (a 270-degree loop that
  swings past the base at its closest gap in one narrow strip, then curves away). Gaps 0.02,
  0.05, 0.10. Dot spacing is 0.031.
- **Pass/fail:** at gap 0.10 everything passes except `lle_k10` (its own 2-3% squareness
  error). At 0.02 and 0.05 every method fails every seed: `isomap_k10`, `check_k10` and
  `lle_k10` fold 44-55% on the flap and 28-47% on the loop; `convex_edge` folds 6-16% on the
  loop and 55-100% on the flap.
- **Where it ends up:** the creased part comes out flat and correct (0.3-0.9% off, never
  flipped) in every failing run. The flap is NOT squashed: it comes out at the right size and
  shape (size 1.05, 0.7% off on its own) but lies mirror-image on top of the base, i.e. the
  sheet is flattened still folded, like ironing a folded sheet. The loop is squashed flat
  onto itself (about 59% of it mirror-image) and its far end lands on the base.
- **Seeing it without the truth:** for each dot, take its 10 nearest dots in the flat result
  and ask how much closer they are there than in 3-D. Paper that was only bent can never
  come out closer when flattened. On the four controls that flatten correctly the median is
  1.0-1.1 and nothing is flagged above 2x. On the loop the ratio is 10-40x over the whole
  loop (58-67% of dots flagged, one connected patch), because the squashed loop stacks parts
  that are up to 0.34 apart in 3-D. On the flap the ratio is a flat plateau of about gap /
  spacing: about 10x at gap 0.05 (61-66% flagged, one patch), about 2x at 0.02 (33% flagged,
  speckled), 1.3x at 0.01 and 1.09x at 0.005 (0.1% flagged): blind. A thin flap folded flat
  is a self-consistent flat picture of the dots; nothing in the distances contradicts it.
  (`fold_flatten small`, `runs/fold_flatten/small_gaps.json`.)

### Finding folds with topology (2026-11-10)

Question: can topology, using only the dots, find the places where a dot's nearest dots are
on another layer (close in 3-D, far along the sheet)? And can it judge whether a
transformation of the cloud made that better? Code: `experiments/fold_topology.py` (parts
`truths`, `blobs`, `loops`, `judge`, `flatten`) and `fold_topology_figure.py`. Results:
`runs/fold_topology/`, detection on seed 100, flattening on seeds 100-102. Figure:
`figures/fold_topology/fold_topology.png`. Needs `gudhi`, which is not in `.venv`. It was run
with `uv pip install --target <dir> gudhi` and `PYTHONPATH=<dir>`.

- **A better truth for stretched sheets.** "Far on the paper" (`FAR` = 0.3) misses most
  shortcuts on the crumples. The new truth walks along the real 3-D sheet (rebuilt 4x finer,
  16 step directions) and calls a link a real shortcut if the walk is over 2x its straight
  length. On bent sheets the two truths agree exactly. On the crumples there are 36-2,616
  real shortcuts among the 10-nearest links, against 0-612 paper jumps. They sit in hairpin
  folds about as narrow as the dot spacing. (The coarse grid alone overstates walks on
  sheared patches, so it is not used.)
- **Loops that never fill in** (persistent homology, dimension 1; links filled with their
  triangles): on bent sheets it is exact. There are no false loops on the 15 bent sheets
  other than `tight_roll`. Every surviving loop on `s_curl` and `h_roll5` starts on a real
  shortcut, and the first one appears at exactly the true gap between layers (2.90 and 2.21
  dot spacings).
  It fails in two ways:
  - Where the paper is stretched, dots are too far apart for triangles to form. The hole
    looks like a fold: 75 false loops on `crumple`, 189 on `uneven_roll`. Measuring links in
    local dot spacing (Berry and Sauer's continuous nearest links) cuts the crumples' false
    loops by 55-80% but makes `uneven_roll` worse (702), because its spacing differs by
    direction.
  - Where layers are closer than about 1.5 dot spacings, triangles form across the gap as
    soon as along the paper, so the loop fills at once. Only 94 of `tight_roll`'s 8,745
    shortcuts, and 0-28 of each crumple's, start a long-lived loop.
- **Gap test on each dot's 40 nearest** (one clump or two, single linkage): on bent sheets
  shortcut dots score 1.9-2.6, and one-layer dots never score above 1.38. On the crumples,
  shortcut dots score 1.2-2.6, inside the 1-3.9 range that stretched one-layer dots reach.
  Requiring the second clump to lie above or below the dot (along the sheet's normal) drops
  those false alarms to 0-1.3%, but it also loses every `s_curl` shortcut. The curl's free
  edge arrives edge-on, beside the dot, not over it. The crumple hairpins are lost the same way.
- **As a judge of transformations** (`judge.json`; stretch along a main axis, or the
  pull-apart from above): it works when the layers stay apart and the spacing stays even.
  Pulling `s_curl` and `h_roll5` apart takes both true shortcuts and loops to 0. Stretching
  `s_curl` along its longest axis takes shortcuts 12 to 519 and loops 4 to 144. It is fooled
  both ways:
  - Stretching makes spacing uneven and creates false loops. `flat` stretched 2x has 0
    shortcuts and 15 loops.
  - Pressing layers together hides real folds. Pulled-apart `tight_roll` still has 7,924
    shortcuts, but local-spacing loops fall from 23 to 0.
- **As flattening methods** (`flatten.json`, 24 sheets, seeds 100-102):
  - `loopcut_knn`: delete the link that starts each never-filling loop, repeat, then walk.
    10/14 design, 7/10 held-out. It passes `s_curl` and `h_roll5`, which `check_k10` fails.
    On stretched sheets it cuts good links in thin places and folds them 11-72%.
  - `blob_k10`: link each dot only to its own clump. 10/14 design, 8/10 held-out (fails
    `tight_roll`, `uneven_roll` and the four crumples).
  - Not a fair held-out check: thresholds were set while looking at all 24 sheets.

### Saved runs

| Folder | What it is |
| --- | --- |
| `runs/current/` | Reference: all 24 sheets, the five default methods, seeds 100-102 |
| `runs/blind_holdout/` | The first held-out check, saved as it was. Two crashes in it led to the stranded-dot fix in `link_check.reattach_stranded` |
| `runs/explore_hops/` | Eight methods on the design sheets, including the rejected flatness hops |
| `runs/tight_roll_16k_dots/`, `_66k_dots/`, `_263k_dots/` | `convex_edge` on `tight_roll` with more dots |
| `runs/noise_0.1/`, `runs/noise_0.3/` | Five sheets with 3-D noise |
| `runs/fold_flatten/` | Creased-then-folded sheets, four methods, the closer-than-real check |
| `runs/fold_topology/` | Topology fold detectors against walk-along-the-sheet truth (`truth_*.npz`), the transformation judge, and five flattening methods |

## Background

The suite grew out of two earlier experiments, which stay in the repo root by repo convention:

- `analyze_generated_one_flat_plane.py` (notes in `figures/generated_one_flat_plane/README.md`)
  asked whether projecting the generated 1s onto one flat 5-D plane gives each 1 its own
  address. A plane taken at one 1 does not.
- `analyze_crumpled_sheet_unrolling.py` and `analyze_crumpled_sheet_followup.py` (notes in
  `figures/crumpled_sheet_unrolling/README.md`) moved to toy sheets. Projection fails on any
  sheet that folds over. Walking along the sheet works when hops stay on one layer. A fold
  test that uses badly shaped triangles at the edge invents folds, which is why the suite
  samples on a jittered grid.
