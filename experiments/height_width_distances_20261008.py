"""Plot directly rendered height/width samples against one fixed original 1.

Run from the repo root:
    MPLCONFIGDIR=/tmp/generated-one-mpl .venv/bin/python experiments/height_width_distances_20261008.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grey_ones import KNOBS, RANGES, render


def offsets(span, step):
    count = int(round(span / step))
    if not np.isclose(count * step, span):
        raise ValueError("Each half-span must be an integer multiple of the step")
    return np.arange(-count, count + 1, dtype=np.float64) * step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", choices=["height", "width"], default="width")
    parser.add_argument("--width", type=float, default=round(float(RANGES[3].mean()), 10))
    parser.add_argument("--width-span", type=float, default=1.5)
    parser.add_argument("--height-span", type=float, default=1.5)
    parser.add_argument("--step", type=float, default=0.1)
    args = parser.parse_args()
    if min(args.step, args.width_span, args.height_span) <= 0:
        parser.error("Step and half-spans must be positive")

    baseline = RANGES.mean(axis=1)
    baseline[:2] = 14.0  # Geometric center of the canvas [0, 28] on both axes.
    baseline[3] = args.width
    baseline[4] = 0.0
    dw = offsets(args.width_span, args.step)
    dh = offsets(args.height_span, args.step)
    widths, heights = baseline[3] + dw, baseline[2] + dh
    if widths.min() <= 0 or heights.min() <= 0:
        parser.error("Every sampled width and height must be positive")
    if widths.max() > 28 or heights.max() > 28:
        parser.error("Centered strokes must fit within the 28 by 28 canvas")
    if min(len(dw), len(dh)) < 25:
        parser.error("Use at least 25 samples per knob")

    parameters = np.tile(baseline, (len(widths), len(heights), 1))
    parameters[:, :, 3] = widths[:, None]
    parameters[:, :, 2] = heights[None, :]
    points = render(parameters.reshape(-1, 5)).reshape(len(widths), len(heights), 784)
    original = render(baseline).reshape(784)
    distances = np.linalg.norm(points.astype(np.float64) - original.astype(np.float64), axis=-1)
    wi, hi = len(dw) // 2, len(dh) // 2
    assert np.array_equal(points[wi, hi], original)
    assert distances[wi, hi] == 0.0
    assert np.isfinite(points).all() and points.min() >= 0 and points.max() <= 1
    assert np.allclose(np.diff(widths), args.step) and np.allclose(np.diff(heights), args.step)
    # At zero tilt the stroke is a rectangle: integrated coverage equals area.
    assert np.allclose(points.sum(axis=-1, dtype=np.float64), widths[:, None] * heights[None, :], atol=1e-5)

    out = ROOT / "figures" / "generated_one_height_width_distances"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "samples.npz", points=points, parameters=parameters,
                        original=original, baseline=baseline, widths=widths,
                        heights=heights, distances=distances)
    with (out / "distances.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["width_change_px", "height_change_px", "width_px", "height_px", "distance_from_original"])
        for i, width in enumerate(widths):
            for j, height in enumerate(heights):
                writer.writerow([dw[i], dh[j], width, height, distances[i, j]])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    x_height = args.x == "height"
    x = dh if x_height else dw
    fixed = widths if x_height else heights
    curves = distances if x_height else distances.T
    fixed_name = "Width" if x_height else "Height"
    fixed_baseline = baseline[3] if x_height else baseline[2]
    central = wi if x_height else hi
    norm = Normalize(fixed[0], fixed[-1])
    cmap = plt.get_cmap("coolwarm")

    fig = plt.figure(figsize=(13.2, 7.3), facecolor="white")
    fig.text(0.055, 0.95, "How far does each generated 1 move from the original?", fontsize=18, weight="bold")
    fig.text(0.055, 0.91, f"{args.x.capitalize()} changes on x; one connected line per fixed {fixed_name.lower()}. All distances use the same original.", fontsize=11)
    image_ax = fig.add_axes([0.065, 0.48, 0.18, 0.33])
    image_ax.imshow(original.reshape(28, 28), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    image_ax.set_title("Start: original 1", fontsize=12, pad=12)
    image_ax.set_xticks([])
    image_ax.set_yticks([])
    fig.text(0.055, 0.45, "28 × 28 image = 784 coverage values\nWhite: 0 (no ink); black: 1 (full ink)", fontsize=10, va="top")
    fig.text(0.055, 0.35,
             f"Fixed center: (14, 14) px\nFixed lean: 0°\nOriginal height: {baseline[2]:g} px\nOriginal width: {baseline[3]:g} px", fontsize=11, va="top", linespacing=1.6)
    fig.text(0.055, 0.18,
             f"Width: {widths[0]:g}–{widths[-1]:g} px ({len(widths)} values)\nHeight: {heights[0]:g}–{heights[-1]:g} px ({len(heights)} values)\nBoth sampled every {args.step:g} px", fontsize=10, va="top", linespacing=1.5)

    ax = fig.add_axes([0.34, 0.22, 0.52, 0.60])
    for i, value in enumerate(fixed):
        if i != central:
            ax.plot(x, curves[i], color=cmap(norm(value)), linewidth=0.8, alpha=0.65)
    ax.plot(x, curves[central], color="#181818", linewidth=2.3,
            label=f"Original {fixed_name.lower()}: {fixed_baseline:g} px")
    ax.scatter([0], [0], color="black", s=40, zorder=5)
    ax.annotate("Original (0, 0)", xy=(0, 0), xytext=(-110, 20), textcoords="offset points", fontsize=10)
    ax.axvline(0, color="#888888", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bottom=-0.3)
    ax.set_xlabel(f"{args.x.capitalize()} change from original (pixels)", fontsize=12)
    ax.set_ylabel("Euclidean distance over 784 coverage values", fontsize=11)
    ax.set_title(f"{len(fixed)} lines × {len(x)} sampled points per line", fontsize=12, pad=14)
    ax.grid(alpha=0.18)
    ax.legend(loc="upper right", fontsize=10)
    cb_ax = fig.add_axes([0.89, 0.22, 0.018, 0.60])
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax, label=f"Fixed {fixed_name.lower()} per line (pixels)")
    fig.text(0.34, 0.12, "At x = 0, colored lines still differ in the other knob; only the black line reaches the original.", fontsize=10)
    fig.text(0.055, 0.055,
             "Measurement: ‖render(new knobs) − render(original knobs)‖₂. Lines connect samples; they are not paths drawn in 784D.", fontsize=10)
    fig.text(0.055, 0.025, "Controlled height/width slice, extending beyond default ranges. The graph shows distance, so pixel direction is hidden.", fontsize=9, color="#555555")
    fig.savefig(out / "distance_curves.png", dpi=160)
    fig.savefig(out / "distance_curves.svg")
    plt.close(fig)

    summary = {
        "baseline_knobs": dict(zip(KNOBS, baseline.tolist())),
        "x_axis": args.x + " change from original, in pixels",
        "line_per": fixed_name.lower(),
        "step_px": args.step,
        "width_range_px": [float(widths[0]), float(widths[-1])],
        "height_range_px": [float(heights[0]), float(heights[-1])],
        "width_samples": len(widths), "height_samples": len(heights),
        "total_images": len(widths) * len(heights),
        "distance": "Euclidean over 784 coverage values; float64 subtraction of float32 render outputs",
        "verification": "Baseline exact; finite coverage in [0, 1]; requested spacing; integrated coverage agrees with rectangle area",
        "understanding_check": {
            "object": "Each sampled point measures one directly rendered 784D image against the fixed original",
            "axes_and_color": "Signed knob change in pixels; Euclidean coverage distance; color names the fixed knob in pixels",
            "starting_case": "Original image shown first; black curve and zero-distance marker identify the baseline",
            "visible_result": "Only the unchanged baseline has zero distance; curves show distances through the sampled sweep",
            "scope": "Center and lean fixed; height and width sampled; distance hides pixel direction; experimental ranges stated",
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "README.md").write_text(
        "# Generated-one height/width distance sweep\n\n"
        f"Baseline `(cx, cy, height, width, lean)`: `{tuple(baseline.tolist())}`. "
        "The center is the geometric center of the canvas; normal height/width use the default range midpoints unless overridden.\n\n"
        f"Widths: {widths[0]:g}–{widths[-1]:g} px ({len(widths)} values). "
        f"Heights: {heights[0]:g}–{heights[-1]:g} px ({len(heights)} values). Step: {args.step:g} px. "
        "These are experimental ranges outside part of the default box. All strokes fit the canvas.\n\n"
        f"The x-axis varies {args.x}; each line holds {fixed_name.lower()} fixed. "
        "Every y-value measures distance to the SAME original image. There is no separate height-vector calculation. "
        "Lines connect consecutive sampled distances and do not preserve the full 784D location or direction.\n\n"
        "`samples.npz` stores `points[width_index, height_index, pixel_index]`, row-major flattened from 28×28 images, "
        "plus the five parameters, axes, baseline, and distances. `distances.csv` stores all knob offsets and distances.\n\n"
        "Reproduce from the repo root:\n\n```sh\n"
        "MPLCONFIGDIR=/tmp/generated-one-mpl .venv/bin/python experiments/height_width_distances_20261008.py "
        f"--x {args.x} --width {args.width:g} --width-span {args.width_span:g} --height-span {args.height_span:g} --step {args.step:g}\n```\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
