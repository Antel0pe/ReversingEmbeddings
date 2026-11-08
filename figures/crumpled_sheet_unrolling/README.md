# Flattening a 2-D sheet crumpled in 3-D: projection vs. walking along it

Built by `analyze_crumpled_sheet_unrolling.py` and `make_crumpled_sheet_figure.py`.
Numbers in `results.json`. Toy sheets only, no generated 1s.

Five sheets, each a square of paper (u, v) in [-1, 1]^2 bent into 3-D, 4,000 random
dots each. Both methods see only the 3-D dots and are told the sheet is 2-D.

- **Projection.** Pick a random dot, fit the flat plane touching the sheet there from
  its 20 nearest dots, drop every dot straight onto it. Done at 40 random dots, and
  also at the best of 3,000 plane tilts.
- **Walking along the sheet** (the user's fix; known as Isomap). Join each dot to its
  10 nearest, take distance as the shortest walk along those joins, then lay the dots
  flat keeping those distances as well as possible.

**Overlap** is the share of dots whose flat-picture neighbours include a dot from a
different part of the paper (more than 15% of the sheet's side away in true (u, v)).
**Tear** is the reverse: true neighbours landing far apart in the picture. True (u, v)
is used only to score.

| Sheet | Projection at a random dot (best / typical / worst of 40) | Best of 3,000 tilts | Walking: overlap | Walking: tear |
| --- | --- | ---: | ---: | ---: |
| Gently curved (bowl) | 0 / 0 / 0% | 0% | 0% | 0% |
| Rolled up | 65 / 70 / 79% | 32% | 0% | 0% |
| Crumpled | 26 / 40 / 60% | 26% | 0%* | 0.2% |
| Crumpled harder | 12 / 36 / 66% | 11% | 0.2%* | 0% |
| Tightly rolled | 80 / 87 / 92% | 70% | 85% | 0.2% |

\* Wrong as a verdict: see the follow-up below. These two layouts contain real folds over
1-2.4% of the paper that this dot-neighbour overlap test is too coarse to see.

A projection always gives every 3-D dot exactly one 2-D spot. What fails is the other
direction: on any sheet that folds over, one 2-D spot receives dots from several layers.

Walking fails only when its short steps jump between layers. On the tightly rolled
sheet the layers come within 1.1x the spacing between neighbouring dots, and 5.5% of
the steps jump. With 12,000 dots only 0.012% jump, but overlap is still 17%: a handful
of wrong steps is enough to wreck the layout. On every sheet that passed, no step
jumped layers. Joining fewer neighbours (4) leaves the dots in disconnected pieces on
three sheets, and on the passing crumpled sheets it causes small overlaps (2-5%).

The crumpled sheets are made by local twists of space. A twist can be undone exactly,
so the sheet folds over but never passes through itself, like real paper.

## Follow-up: why the unrolled sheets are not squares, and folds between the dots

Built by `analyze_crumpled_sheet_followup.py` and `make_crumpled_sheet_followup_figure.py`;
numbers in `followup_results.json`.

Every sheet starts as the same square, but my bending stretched most of them. Joining the
dots into triangles by their true (u, v) positions and comparing each triangle's 3-D area
with its paper area (real paper: exactly 1):

| Sheet | Stretch when made (5th-95th pct) | Folded share of paper, triangles from 4,000 / 1,000 / 250 dots |
| --- | --- | --- |
| Gently curved (bowl) | 1.02-1.30x | 0.01 / 0 / 0% |
| Rolled up | 2.42-6.18x | 0.25 / 0.11 / 0% |
| Crumpled | 0.93-3.71x | 1.95 / 1.77 / 2.35% |
| Crumpled harder | 0.83-3.54x | 1.30 / 1.01 / 1.14% |
| Tightly rolled | 1.67-2.87x | 48.8 / 46.1 / 42.6% |
| Rolled, no stretch (new) | 1.00x | 0.01 / 0 / 0% |
| S-curled, no stretch (new) | 1.00x | 29.3 / 27.5 / 23.8% (4,000 dots); 0.04% with 8,000 |

A fold is a triangle that comes out mirror-flipped in the flat layout. Only well-shaped
triangles (every angle at least 20 degrees) away from the paper's edge are counted, because
random dots make thin slivers at the edge and a sliver flips at the slightest wobble.
Counting slivers first gave 3.3% for the crumpled sheet and red wedges along its edge; that
was an artefact. Using every 4th or 16th dot makes each triangle span several dot spacings,
so wobble cannot flip it but a real fold still does. If no triangle flips and the outer edge
does not cross itself, the layout filled in across each triangle is one-to-one for every
point of the paper, not just the dots.

- Real paper (no stretch) comes back as the original square: dots sit a median 1.4% of the
  side from their true place, worst 3.7%, with no folds.
- The rolled-up sheet was stretched, but evenly enough to still be a flat sheet of a
  different shape (a long strip), so it unrolls without folds.
- The crumpled sheets were stretched unevenly. No flat shape matches them, and the
  compromise layout folds over 1-2.4% of the paper at every triangle size.
- The S-curl fails at 4,000 dots although only 0.1% of the hops jump layers. At 8,000 dots
  none jump and it comes back clean.
