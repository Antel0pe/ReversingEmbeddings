"""Export the same-height width comparison as a Markdown pixel-coefficient table."""

from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "figures/generated_one_height_change_comparison/samples.npz"
OUT = ROOT / "figures/generated_one_same_height_offsets/pixel_values_and_changes.md"
SCALE = 1_000_000


def number(units):
    return f"{units / SCALE:.6f}".rstrip("0").rstrip(".") or "0"


def index_sets(indices):
    """Compact exact index sets into per-column runs with explicit stride 28."""
    columns = defaultdict(list)
    for i in indices:
        columns[i % 28].append(i)
    runs = []
    for values in columns.values():
        first = last = values[0]
        for value in values[1:]:
            if value == last + 28:
                last = value
            else:
                runs.append((first, last))
                first = last = value
        runs.append((first, last))
    return [f"E[{first}]" if first == last else f"E[{first}:{last}:28]"
            for first, last in sorted(runs)]


def reference_cell(values):
    grouped = defaultdict(list)
    for i, value in enumerate(values):
        if value:
            grouped[int(value)].append(i)
    terms = []
    for value, indices in sorted(grouped.items()):
        for basis in index_sets(indices):
            terms.append(f"`{number(value)}·{basis}`")
    return "**Reference X_h; d = 0**<br>" + " +<br>".join(terms)


def changed_cell(reference, current, distance):
    grouped = defaultdict(list)
    for i, (a, b) in enumerate(zip(reference, current)):
        if a != b:
            grouped[(int(a), int(b-a))].append(i)
    terms = []
    for (a, x), indices in sorted(grouped.items()):
        sign = "+" if x > 0 else "−"
        for basis in index_sets(indices):
            terms.append(f"`{number(a)}·{basis} {sign} {number(abs(x))}·{basis} = {number(a+x)}·{basis}`")
    return f"**d = {distance:.6f}**<br>" + "<br>".join(terms)


def main():
    with np.load(SOURCE) as saved:
        points = saved["points"].astype(np.float64)
        widths = saved["widths"]
        heights = saved["heights"]
        offsets = saved["height_offsets"]
        baseline = saved["baseline"]
    root = int(np.argmin(abs(widths-baseline[3])))
    display = np.rint(points*SCALE).astype(np.int64)
    distances = np.linalg.norm(points-points[root:root+1], axis=-1)
    assert np.abs(display/SCALE-points).max() <= 0.5/SCALE
    # Every entry can be reconstructed by inheriting its reference coefficients
    # and replacing precisely the coefficients included in that cell.
    for i in range(len(widths)):
        for j in range(len(heights)):
            reference = display[root, j]
            current = display[i, j]
            reconstructed = reference.copy()
            changed = current != reference
            reconstructed[changed] += (current-reference)[changed]
            assert np.array_equal(reconstructed, current)

    header = """# Pixel values and changes: each width versus the normal width at the same height

This is the pixel-level version of `same_height_width_offsets.png`.
Rows hold **width fixed**; columns hold **height fixed**. Both are sampled in 0.1 px steps.
The center stays at `(14, 14)` and lean stays at `0°`.
Base height is 19.75 px; reference width is 3.2 px. There are 31 widths and 31 heights.

## How to read a cell

- `E[i]` is the 784D basis vector for pixel **i**: that coordinate is 1 and all other coordinates are 0.
- Pixel numbering starts at **0** in the top-left: `i = 28 × row + column`, with rows and columns 0–27. The last pixel is 783.
- `a_i` is the reference image's pixel coefficient at this column's height.
- `x_i` is the **signed increase/decrease** from that coefficient to the row's width.
- The new coefficient is `a_i + x_i`. Coverage is 0 for no ink and 1 for full ink.

The pixel contribution is **`a_i·E[i] + x_i·E[i] = (a_i + x_i)·E[i]`**.
The complete image is `Y = X_h + Σ_i x_i·E[i]`, where `X_h` is the normal-width image at the SAME height.
The scalar change `x_i` multiplies its basis vector; it is not added separately to a vector.

Example: `0.6·E[152] − 0.05·E[152] = 0.55·E[152]` means pixel 152 had coverage 0.6 in the reference,
lost 0.05, and now has coverage 0.55. Positive changes gain ink; negative changes lose ink.

## Grouping and precision

`E[first:last:28]` means `E[first] + E[first+28] + … + E[last]`, including both endpoints.
For example, `E[152:628:28]` identifies pixels 152, 180, 208, …, 628.
Every pixel in a group has the same reference coefficient, signed change, and new coefficient.
This groups repeated values without averaging different pixels.

The **3.2 px reference row** gives each column's full image as a sparse basis expansion; unlisted coefficients there are zero.
Other rows list **every changed coefficient**. Unlisted coefficients in those rows stay equal to the reference row in that column.
Terms in a nonreference cell are separate coefficient updates; apply them to `X_h` to obtain `Y`.
Thus the table specifies all 784 coefficients of every image, with repeated and unchanged values abbreviated.

Coefficients are rounded to six decimal places. Signed changes are computed between those displayed coefficients,
so the displayed arithmetic agrees exactly. Maximum rounding error per coefficient is 0.0000005.
`d` is the graph's Euclidean distance `‖Y − X_h‖₂`, computed from the original renderer values and displayed to six decimals.
Distance has no sign, while each pixel update does.

## Table

The table is wide: 31 height columns. For a vertical comparison, choose one height column and compare width rows.
For a height sweep, follow one width row across columns. The center column is height change 0, actual height 19.75 px.

"""
    lines = [header]
    labels = [f"Δh {offset:+.1f} px<br>h {height:.2f} px" for offset, height in zip(offsets, heights)]
    lines.append("| Width (px) / height → | " + " | ".join(labels) + " |")
    lines.append("| --- | " + " | ".join(["---"]*len(heights)) + " |")
    for i, width in enumerate(widths):
        cells = []
        for j in range(len(heights)):
            cells.append(reference_cell(display[root, j]) if i == root else
                         changed_cell(display[root, j], display[i, j], distances[i, j]))
        label = f"**{width:.1f} (reference)**" if i == root else f"**{width:.1f}** (Δw {width-baseline[3]:+.1f})"
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    lines.append("\nSource images: `figures/generated_one_height_change_comparison/samples.npz`. "
                 "Reproduce from the repo root with "
                 "`.venv/bin/python experiments/write_same_height_pixel_table_20261008.py`.\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print("Created:", OUT.relative_to(ROOT))
    print("Bytes:", OUT.stat().st_size)
    print("Table shape:", len(widths), "width rows x", len(heights), "height columns")
    print("Verified every image reconstructs from the reference plus displayed updates.")


if __name__ == "__main__":
    main()
