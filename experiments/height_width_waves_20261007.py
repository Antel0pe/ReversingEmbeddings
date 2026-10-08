"""Render a width/height sweep using grey_ones, and export figures and raw distances.

Run from the repository root:
    .venv/bin/python experiments/height_width_waves_20261007.py
"""

from pathlib import Path
import csv
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grey_ones import KNOBS, RANGES, render


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    bounds = np.asarray([RANGES[k] for k in KNOBS] if isinstance(RANGES, dict)
                        else RANGES, dtype=np.float64)
    if bounds.shape != (5, 2):
        raise ValueError("Expected five knob ranges in (cx, cy, height, width, lean) order")
    reference = bounds.mean(axis=1)
    reference[4] = 0.0
    widths = reference[3] + np.arange(101, dtype=np.float64) / 100
    if widths[-1] > bounds[3, 1] + 1e-12:
        raise ValueError("Reference width plus one exceeds the allowed width range")
    heights = np.linspace(bounds[2, 0], bounds[2, 1],
                          int(round((bounds[2, 1] - bounds[2, 0]) / 0.01)) + 1)
    heights = np.unique(np.r_[heights, reference[2]])
    parameters = np.tile(reference, (len(widths) * len(heights), 1))
    parameters[:, 3] = np.repeat(widths, len(heights))
    parameters[:, 2] = np.tile(heights, len(widths))
    images = render(parameters).reshape(len(widths), len(heights), 784).astype(np.float64)
    width_references = np.tile(reference, (len(widths), 1))
    width_references[:, 3] = widths
    same_width = render(width_references).reshape(len(widths), 784).astype(np.float64)
    root = render(reference).reshape(784).astype(np.float64)
    height_vectors = images - same_width[:, None, :]
    fixed_root_distance = np.linalg.norm(images - root, axis=2)
    height_distance = np.linalg.norm(height_vectors, axis=2)
    vector_change = np.linalg.norm(height_vectors - height_vectors[0:1], axis=2)

    # Interpolate actual height vectors, then compare with directly rendered values.
    coarse = np.arange(0, len(widths), 10)
    interpolated = np.empty_like(height_vectors)
    for a, b in zip(coarse[:-1], coarse[1:]):
        t = np.arange(b - a + 1, dtype=np.float64) / (b - a)
        interpolated[a:b + 1] = ((1 - t[:, None, None]) * height_vectors[a]
                                 + t[:, None, None] * height_vectors[b])
    interpolation_residual = interpolated - height_vectors
    interpolation_distance = np.linalg.norm(interpolation_residual, axis=2)

    output = ROOT / "figures" / "generated_one_height_width_waves"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "distances.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["width_increment_px", "width_px", "height_px",
                         "distance_from_fixed_root", "height_vector_length",
                         "height_vector_change_from_reference_width",
                         "coarse_width_interpolation_vector_error"])
        for wi, width in enumerate(widths):
            for hi, height in enumerate(heights):
                writer.writerow([width-reference[3], width, height,
                                 fixed_root_distance[wi, hi], height_distance[wi, hi],
                                 vector_change[wi, hi], interpolation_distance[wi, hi]])
    summary = {
        "reference_knobs": dict(zip(KNOBS, reference.tolist())),
        "width_step_px": 0.01,
        "height_step_px": float(heights[1]-heights[0]),
        "width_samples": len(widths), "height_samples": len(heights),
        "distance": "Euclidean over 784 coverage values; float64 differences of renderer outputs",
        "interpolation": "Piecewise linear height-vector interpolation from width steps of 0.1; validated at 0.01",
        "maximum_interpolation_vector_error": float(interpolation_distance.max()),
        "maximum_interpolation_pixel_error": float(np.abs(interpolation_residual).max()),
        "status": "Generated from grey_ones.render; explanatory understanding check remains required before delivery",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    norm = Normalize(heights[0], heights[-1])
    cmap = plt.get_cmap("coolwarm")
    step = widths-reference[3]
    for name, distances, description in [
        ("fixed_root", fixed_root_distance, "Distance from the fixed root image"),
        ("height_vectors", height_distance, "Height-vector length at the same width"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True,
                                 layout="constrained")
        for ax, smaller, title in zip(axes, [True, False],
                                     ["Shorter than the reference", "Taller than the reference"]):
            ids = np.flatnonzero(heights <= reference[2]+1e-10 if smaller
                                 else heights >= reference[2]-1e-10)
            for a, b in zip(ids[:-1], ids[1:]):
                ax.fill_between(step, distances[:, a], distances[:, b],
                                color=cmap(norm((heights[a]+heights[b])/2)), alpha=0.8,
                                linewidth=0)
            # Fine sampling supplies the fill; height curves every 0.1 remain legible.
            line_ids = [i for i in ids if abs((heights[i]-heights[0])/0.1-
                                            round((heights[i]-heights[0])/0.1)) < 1e-7]
            line_ids = sorted(set(line_ids + [int(ids[0]), int(ids[-1])]))
            for i in line_ids:
                ax.plot(step, distances[:, i], color=cmap(norm(heights[i])), linewidth=1)
            ax.set_title(title)
            ax.set_xlabel("Width increase from reference (pixels)")
            ax.set_ylabel("Euclidean image distance")
            ax.grid(alpha=0.2)
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                     label="Height (pixels)")
        fig.suptitle(description + "\nStraight reference: height %.2f, width %.2f; width sampled every 0.01 px"
                     % (reference[2], reference[3]), fontsize=13)
        fig.savefig(output / (name + ".png"), dpi=180)
        fig.savefig(output / (name + ".svg"))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    mesh = ax.pcolormesh(step, heights, vector_change.T, shading="auto", cmap="viridis")
    ax.axhline(reference[2], color="white", linestyle="--", linewidth=1)
    ax.set_xlabel("Width increase from reference (pixels)")
    ax.set_ylabel("Height (pixels)")
    ax.set_title("How the height vector changes with width\n"
                 "Distance between height vectors, relative to the vector at reference width")
    fig.colorbar(mesh, ax=ax, label="Euclidean vector difference")
    fig.savefig(output / "vector_changes.png", dpi=180)
    fig.savefig(output / "vector_changes.svg")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
