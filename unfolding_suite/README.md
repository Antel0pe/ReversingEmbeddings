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

### Saved runs

| Folder | What it is |
| --- | --- |
| `runs/current/` | Reference: all 24 sheets, the five default methods, seeds 100-102 |
| `runs/blind_holdout/` | The first held-out check, saved as it was. Two crashes in it led to the stranded-dot fix in `link_check.reattach_stranded` |
| `runs/explore_hops/` | Eight methods on the design sheets, including the rejected flatness hops |
| `runs/tight_roll_16k_dots/`, `_66k_dots/`, `_263k_dots/` | `convex_edge` on `tight_roll` with more dots |
| `runs/noise_0.1/`, `runs/noise_0.3/` | Five sheets with 3-D noise |

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
