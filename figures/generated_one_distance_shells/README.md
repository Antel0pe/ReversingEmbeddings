# Equal-distance neighbors of five generated 1s

These figures use only the controlled five-knob renderer in `grey_ones.py`.
Each image has 28 by 28 ink-coverage values. White is 0 and black is 1.
The five settings are horizontal center, vertical center, height, width, and
lean angle. Positive lean tilts the top to the right.

For seed settings `p0` and candidate settings `p`, let `F(p)` be the rendered
784-pixel image. The distance used throughout is

`d(F(p0), F(p)) = sqrt(sum over all 784 pixels of (F(p0)[i] - F(p)[i])^2)`.

The requested neighbors form the level set

`{p in the allowed five-knob box : d(F(p0), F(p)) = 4.5}`.

This is a continuous set, generally a four-dimensional surface, so it cannot
be listed image by image. Five knobs describe five independent local changes;
they do not imply five points at a fixed distance. The figures show a finite,
diverse sample of that surface.

## How the sample was made

Five seeds were drawn from the full allowed knob ranges with NumPy random seed
`20260930`. For each seed, 2,048 deterministic directions were drawn on a
five-dimensional sphere in *range-normalized knob coordinates*. Each ray was
followed to the edge of the allowed knob box. A distance crossing was bracketed
on a nine-point radial grid and solved by bisection. If a ray reached radius
4.5, its first detected crossing was retained. Across all five seeds, 3,861
crossings were found. Re-rendered distances differed from 4.5 by at most
`0.0000035` pixel units. This is numerical equality to the chosen radius,
limited by float32 renderer output and bisection resolution.

Each seed figure displays 20 points selected greedily to maximize the smallest
pixel distance to the points already selected. That spreads out appearances;
it does not imply those 20 points are evenly spaced on the full surface. The
five exact input settings are printed under each displayed image. There is no
image-to-knob inversion: the settings are known because they generated the
images. The CSV records all sampled crossings, including those not displayed.

| Seed | Sampled crossings | Smallest distance between displayed neighbors | Figure |
| --- | ---: | ---: | --- |
| 1 | 619 | 2.40 | [Seed 1](seed_01_radius_4p5.png) |
| 2 | 1,067 | 2.42 | [Seed 2](seed_02_radius_4p5.png) |
| 3 | 561 | 2.36 | [Seed 3](seed_03_radius_4p5.png) |
| 4 | 332 | 2.08 | [Seed 4](seed_04_radius_4p5.png) |
| 5 | 1,282 | 2.53 | [Seed 5](seed_05_radius_4p5.png) |

## Is 4.5 a locally flat distance for this generator?

The earlier approximately 4.5-unit averaging cutoff in `OnesManifold.ipynb`
was measured on *real MNIST sample images*, with nearest-sample distance as its
acceptance test. It is a useful starting distance here, not a calibrated reach
for this generated family.

For 12 sampled seed-neighbor pairs from each 4.5-unit shell (60 pairs total),
the pixel average of the endpoint images was a median **1.209 pixel units**
from a numerically fitted valid generated image (90th percentile 1.306).
The generated path made by linearly changing all five settings had a median
length **1.251 times** the straight pixel-space endpoint distance. Path length
was approximated with 64 segments. This is one valid path, not a computed
shortest geodesic, and the fitted-image residual is an optimization result,
not a proof of the global nearest point.

For the same first seed, smaller radii give this scale comparison, with 12
sampled endpoint pairs per radius:

| Pixel radius | Median midpoint distance to a fitted valid generated 1 | Median direct-knob-path length / chord | Images |
| ---: | ---: | ---: | --- |
| 1.0 | 0.078 | 1.022 | [Radius 1.0](seed_01_radius_1p0_comparison.png) |
| 2.5 | 0.530 | 1.209 | [Radius 2.5](seed_01_radius_2p5_comparison.png) |
| 4.5 | 1.093 | 1.308 | [Radius 4.5](seed_01_radius_4p5.png) |

The seed-1 comparison suggests **1.0** as a better radius for inspecting
nearly straight local behavior. Radius 2.5 is an intermediate view, and 4.5
shows much broader shape changes. These sampled results do not establish a
global safety threshold for averaging. Pixel averaging generally leaves the
exact generated manifold even at small nonzero distances; “near” depends on
the tolerance chosen for the question.

The point counts above depend on the finite direction sample and how often a
ray reaches the chosen radius before the knob boundary. They are not counts
of all equal-distance images, nor estimates of surface area. A few radial
grids were nonmonotone, so later crossings along some rays may also exist.

Reproduce with `python make_generated_one_distance_shells.py`. The raw sampled
settings are in [sampled_radius_4p5_shells.csv](sampled_radius_4p5_shells.csv),
and machine-readable measurements are in [results.json](results.json).
