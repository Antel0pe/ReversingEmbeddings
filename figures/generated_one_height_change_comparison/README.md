# Height-change comparison across widths

View `height_change_comparison.png` (SVG is the same figure).

Baseline `(cx, cy, height, width, lean) = (14, 14, 19.75, 3.2, 0)`. Width and height vary ±1.5px, every 0.1px: 31 widths × 31 heights. Center and lean remain fixed.

At each width w, the finite height-change vector is `D(w,x) = render(w,h0+x) - render(w,h0)`. Plot `norm(D(w,x) - D(w0,x))`, with `h0=19.75` and `w0=3.2`. Each line holds width fixed; x is signed height change. The reference is the normal-width root, not the mean. Vectors are unnormalized, so this measures differences in both magnitude and pixel pattern. All curves equal zero at x=0, and the reference is identically zero.

Thinner and wider widths appear in two panels with matching axes to improve readability. Some curves overlap because equal scalar distances do not identify the full pixel-change pattern.

`samples.npz` stores points, parameters, full height changes, reference height changes, and distances. Array order is width × height × row-major pixel. `distances.csv` contains every plotted distance. Coverage bounds, zero cases, integrated area, and an independent rectangle-coverage formula were verified.

Reproduce from the repo root:

```sh
MPLCONFIGDIR=/tmp/generated-one-mpl .venv/bin/python experiments/compare_height_changes_across_widths_20261008.py
```
