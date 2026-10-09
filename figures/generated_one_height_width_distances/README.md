# Generated-one height/width distance sweep

Baseline `(cx, cy, height, width, lean)`: `(14.0, 14.0, 19.75, 3.2, 0.0)`. The center is the geometric center of the canvas; normal height/width use the default range midpoints unless overridden.

Widths: 1.7–4.7 px (31 values). Heights: 18.25–21.25 px (31 values). Step: 0.1 px. These are experimental ranges outside part of the default box. All strokes fit the canvas.

The x-axis varies width; each line holds height fixed. Every y-value measures distance to the SAME original image. There is no separate height-vector calculation. Lines connect consecutive sampled distances and do not preserve the full 784D location or direction.

`samples.npz` stores `points[width_index, height_index, pixel_index]`, row-major flattened from 28×28 images, plus the five parameters, axes, baseline, and distances. `distances.csv` stores all knob offsets and distances.

Reproduce from the repo root:

```sh
MPLCONFIGDIR=/tmp/generated-one-mpl .venv/bin/python experiments/height_width_distances_20261008.py --x width --width 3.2 --width-span 1.5 --height-span 1.5 --step 0.1
```
