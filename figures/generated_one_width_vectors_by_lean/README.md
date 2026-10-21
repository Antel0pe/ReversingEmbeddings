# Width vectors across lean

Open `index.html` for the interactive comparison, or read the three numbered figures in order.

Center `(14,14)` and height `19.75 px` stay fixed. The root is `(14,14,19.75,3.2,0)`.
Sample 57 widths from 1.8 to 4.6 px every 0.05 px and 91 leans from -10° to 35° every 0.5°: 5,187 images.
The source is `grey_ones.render`, with 16 subrows and default float32 storage; subtract and measure in float64.

## What the vectors mean

At a fixed lean, **full width vector = image(width 4.6) − image(width 1.8)**.
Its 784 entries are the changes in pixel coverage between those two images.
Its length is `sqrt(sum(pixel_change**2))`.
The angle between two vectors is `acos(sum(a*b)/(length(a)*length(b)))`, in degrees.
For a direction-only comparison, lengths are divided out: 0° means the same direction; 90° means perpendicular.

`vector at lean − vector at upright` is a **second difference**: it describes how the width response changes, not direct ink gain or loss.
Its length measures the mismatch if the upright width vector were reused at the new lean.
The mixed difference is the same when computed as the change in the lean response across width.
Both parameter orders reach the same final rendered image; their intermediate pixel changes differ.

The full width vector sums the 56 consecutive width-step vectors exactly. It does not encode the curved intermediate path.
The sampled width-path length is the sum of their L2 lengths; dividing by the full-vector length gives 1 for a straight path without backtracking.

## Measured results

- Full-vector length ranges 6.243–6.569.
- At lean 35°, its direction is 74.12° from upright.
- A finite 1° lean change turns the full width vector by 7.30–10.27° over the sampled centers.
- Sampled widening path / endpoint distance ranges 1.229–1.348; widening is not a straight pixel-space line.
- Angle from upright need not grow monotonically with lean, even though neighboring vectors keep changing. See `whole_vectors.csv`.

## Checks and limits

Refined grid: 113 widths × 181 leans (20,453 images), half the spacings.
Every matching image is identical. The largest relative path-length change after halving the width spacing is 0.2677%.
On the finer lean grid, finite 1° comparisons range 7.213–10.272°.
Ink mass agrees with `height * width` to 8.8e-07; all width steps add nonnegative ink; their sum agrees with each endpoint vector to 0 per pixel.
Float32 versus float64 endpoint vectors differ by at most 3e-08 per pixel, retaining the same 16-subrow coverage model.

This is the requested width/lean slice at fixed center and height. It is a finite sampled experiment, not an exact map of the continuous region or the full five-knob manifold.
The 0.05-pixel and 1° vectors are finite differences, not analytic derivatives.
The direct vector images show all 784 entries; the scalar plots summarize them without projecting them.
Colors share explicit scales and have no clipped values. Differences from real MNIST are outside this experiment.

## Reproduce and reuse

Run `OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/width_vectors_by_lean.py` from the repository root.
`samples.npz` stores all images, settings, and full width vectors.
`whole_vectors.csv` and `small_width_steps.csv` store the measurements; `results.json` stores settings and verification results.
The standalone interactive viewer embeds the actual float32 samples and computes comparisons from them; it needs no server or network.
