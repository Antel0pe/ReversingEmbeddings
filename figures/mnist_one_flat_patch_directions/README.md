# How many perpendicular directions reach every neighbour of a 1?

Scripts: `analyze_mnist_one_flat_patch_directions.py` (parts `neighbours`, `zoom`),
`make_mnist_one_flat_patch_directions_figure.py`. Figure: `flat_patch_directions.png`.

Procedure: for a starting 1, take the arrows to its nearest k neighbours (k = 2..200)
and count the fewest perpendicular directions (uncentred SVD of the arrows) that reach
EVERY arrow with at most 10% of its length left over (also 25% and 5%). A flat
d-dimensional patch shows as a count that stops at d as k grows.

| family (13,533 images) | nearest neighbour | count at k = 20 | k = 200 |
| --- | --- | --- | --- |
| MNIST 1s (train + QMNIST test) | 23.6% of the image away | 20 | 85 |
| drawn 1s, 12 knobs | 27.6% | 19 | 67 |
| generated 1s, 5 knobs | 8.2% | 10 | 24 |
| generated 1s, 5 knobs, 200,000 images | 4.6% | 8 | 14 |

Zoom control (`zoom.json`): neighbours made on purpose by nudging knobs around a centre.
Generated 1s sit at exactly 5 for every k up to 200 once neighbours are within about 2%
of the image's size (5.5-6 at k = 150-200 at 2%; 9 at 6%; 23 at 20%). The 12-knob drawn
1s reach 12 only within about 0.6%. MNIST's closest neighbour is about ten times farther
away than the edge of the generated family's flat patch, and until about 25 neighbours each
one needs a direction of its own (count = k). So at the available density there is no
flat patch to measure, and this test cannot tell MNIST from a 12-knob family
(or from anything with even more knobs).

## Only the most crowded patches (`dense.json`, `dense_patches.png`)

All 13,533 images of a family are ranked by distance to their 10th nearest neighbour
(as a fraction of their own size). Centres are then drawn only from one slice of the
ranking; neighbours still come from the whole family.

| family | slice | nearest neighbour | count at k = 25 | k = 200 |
| --- | --- | --- | --- | --- |
| MNIST 1s | most crowded 1% | 16.6% | 25 | 87 |
| MNIST 1s | most crowded 5% | 17.7% | 25 | 88 |
| MNIST 1s | middle 10% | 23.6% | 25 | 86 |
| MNIST 1s | loneliest 5% | 47.5% | 24 | 77 |
| generated 1s, 5 knobs | most crowded 1% | 6.0% | 11 | 23 |
| generated 1s, 5 knobs | middle 10% | 8.0% | 11 | 24 |
| drawn 1s, 12 knobs | most crowded 1% | 22.3% | 23 | 68 |

Choosing crowded centres moves MNIST's nearest neighbour only from 24% to 17%, still
about eight times the ~2% inside which the generated family was flat, and the count is
unchanged (one direction per neighbour up to k = 25). Even the 5-knob family's most
crowded 1% (6.0%) is outside its own flat zone and does not settle at 5. The loneliest
centres need fewer directions only because all their neighbours lie in one far-off
crowd, so the arrows share one long common part (see `share` check in the recap).
