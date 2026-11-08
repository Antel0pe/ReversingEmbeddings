# Unfolding test suite

A place to try ideas for flattening a crumpled 2-D sheet that sits in 3-D, and to see
exactly how and why each one fails.

Toy sheets only, for now. A sheet starts as the square [-1, 1]², gets bent or stretched
into 3-D, and a method has to flatten it again knowing only the 3-D dots and that the
sheet is 2-D.

## Run it

From the repo root:

```bash
.venv/bin/python -m unfolding_suite.test_scoring
```

```bash
.venv/bin/python -m unfolding_suite.run --name baseline
```

The first command checks the scorer itself. The second writes to
`figures/unfolding_suite/<name>/`:

| File | What it is |
| --- | --- |
| `gallery_bent.png` | Each row an input sheet, each column a method's result, lined up with the original square and coloured by distance from it |
| `gallery_stretched.png` | The same for stretched sheets, coloured by the paper's own print, because the square is not their target. Printed measure: how much short lengths changed |
| `scorecard.png` | Pass or fail for every sheet and method |
| `failures/<sheet>__<method>.png` | One page per failed test: where it failed, an example pair, and the cause |
| `results.json` | Every number |
| `comparison.json` | Only with `--against`: what was fixed and what broke |

## The loop for trying an idea

1. Write the idea as a function in its own file. `experiments/adaptive_hops.py` is a worked example.
2. Run it and compare with a saved run or with one method from that run:

   ```bash
   .venv/bin/python -m unfolding_suite.run --plugin my_idea.py --methods my_idea --name my_idea --against baseline:isomap_k10
   ```

3. Read the printed `fixed`, `broken` and `still_failing` lists.
4. Open the failure page for anything still failing. It names a cause, or says `unexplained`.
5. If it says `unexplained`, you have found a new way to fail. Add a check for it in `diagnose.py`.
6. If a new kind of crumpling matters, add a sheet in `sheets.py`.

## Adding a method

A method takes the dots and returns one flat position per dot, in the same order. That
list of pairs is the mapping between 2-D and 3-D, and it is all the scorer needs.

```python
import numpy as np
from unfolding_suite.methods import method

@method("my_idea")
def my_idea(points, dim, rng):
    # points: (n, 3), shuffled and at a random rotation. dim: always 2. rng: numpy generator.
    return flat        # (n, 2), same order as points
```

Scale, rotation and mirroring of the answer do not matter. If a method raises an error,
that test is recorded as `crashed` and the run continues.

To get better failure explanations, return `(flat, info)`. All of these are optional:

| Key | Meaning |
| --- | --- |
| `info["edges"]` | `(m, 2)` indices of the neighbour links the method relied on |
| `info["plane"]` | `(3, 2)` the axes of the plane, for methods that project onto one |
| `info["captured"]` | Share (0 to 1) of the method's own distances a flat layout can express |

`layout_from_links(points, links, dim)` in `methods.py` does the shortest-walk and flat
layout steps, so a new idea about which dots to link needs only a few lines.

## Adding a sheet

```python
from unfolding_suite.sheets import sheet

@sheet("my_sheet", "bent", "One line saying what it is.")
def my_sheet(uv):
    return xyz         # (n, 3) from (n, 2) paper positions
```

Use `"bent"` only if the sheet never stretches, because then the correct answer is the
original square. Use `"stretched"` otherwise. `test_every_sheet_is_valid` enforces this.

## What gets checked

The suite hides the dots' order and rotation from the method. It keeps the paper
positions for itself and joins neighbouring dots into triangles.

| Check | Meaning |
| --- | --- |
| **certified** | No triangle flips at any of three mesh sizes, and the outer edge never crosses itself. Then the map, filled in across the triangles, is one-to-one at every point of the paper, not only at the dots |
| **wobble only** | Flips appear only at the finest mesh and cover under 1% of the paper. Triangles spanning several dots stay correct, so this is small-scale noise. Counts as a pass |
| **folded** | Anything else |
| **Overlap share** | Of the flat area covered, the share covered more than once. Catches far-apart parts landing together |
| **Square error** | After the best turn, mirror and resize, how far dots land from their true place, as % of the side. Judged only on bent sheets: typical at most 3%, worst at most 10% |
| **Torn / crushed** | An edge more than 5× longer or shorter than the edges around it. Up to 0.05% of edges is allowed as isolated outliers |
| **Roughness** | How abruptly stretching changes between neighbouring triangles. Must be at most 0.5 |

A test passes only if it passes on every seed.

The outer-edge check is not optional. `test_wrapping_over_itself_with_no_flipped_triangle_is_caught`
builds a layout that lies over itself with every triangle the right way up.

## Causes the suite can name

Each cause needs evidence before it is named.

| Cause | Evidence |
| --- | --- |
| Hops jumped between layers | The method's links join dots far apart on the paper, and the same layout step passes with perfect neighbours |
| Hops were lopsided | Over 1% of dots have links lying nearly along one line on the paper, and perfect neighbours pass |
| The layout step fails even with perfect hops | Given neighbours from the true paper, the same layout still fails |
| The sheet turned past edge-on to the plane | Part of the paper faces away from the projection plane |
| The plane shortens tilted parts | Nothing folds, but the result is not square |
| Unexplained | None of the above. A new way to fail |

The perfect-neighbours rerun separates the two stages of a walking method: choosing the
hops, and laying the result flat.

## Limits

- Dots sit on a jittered grid. That is what makes the triangles well-shaped, but it is
  more even than real data. `--noise` adds 3-D noise; there is no uneven-sampling option yet.
- "Certified" is a statement about the map filled in linearly across triangles. It cannot
  see anything finer than the mesh.
- The pass thresholds were set by checking the scorer against layouts with known flaws
  and by looking at results on these sheets. They are judgement calls, listed at the top of `scoring.py`.
- Three strictness settings of an idea are three methods. Report all of them, because
  choosing the best on these same sheets flatters it.
