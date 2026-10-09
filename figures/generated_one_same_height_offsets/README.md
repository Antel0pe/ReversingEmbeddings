# Same-height width offsets

View `same_height_width_offsets.png`; SVG is the same figure.

Each line fixes width. X is height change from 19.75px. Y is `norm(image(width, height) - image(3.2, same_height))` over all 784 coverage values. The reference changes height at each x, and its own curve is identically zero. Other widths remain at nonzero distance even at the base height.

Baseline: `(14, 14, 19.75, 3.2, 0)`. Both height and width vary ±1.5px in 0.1px steps. Saved images come from `figures/generated_one_height_change_comparison/samples.npz`. Distances were checked against independent exact rectangle coverage.

Reproduce from the repo root after generating those samples:

```sh
MPLCONFIGDIR=/tmp/generated-one-mpl .venv/bin/python experiments/plot_same_height_width_offsets_20261008.py
```
