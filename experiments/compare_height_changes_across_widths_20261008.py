"""Compare finite height-change vectors at each width with the normal-width root.

Run from the repo root:
    MPLCONFIGDIR=/tmp/generated-one-mpl .venv/bin/python experiments/compare_height_changes_across_widths_20261008.py
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
    baseline = RANGES.mean(axis=1)
    baseline[:2] = 14
    baseline[3] = round(float(baseline[3]), 10)
    baseline[4] = 0
    steps = np.arange(-15, 16, dtype=np.float64) * 0.1
    widths = baseline[3] + steps
    heights = baseline[2] + steps
    middle = 15
    parameters = np.tile(baseline, (31, 31, 1))
    parameters[:, :, 3] = widths[:, None]
    parameters[:, :, 2] = heights[None, :]
    points = render(parameters.reshape(-1, 5)).reshape(31, 31, 784)
    # Each width starts from ITS OWN image at the original height.
    height_changes = points.astype(np.float64) - points[:, middle:middle+1, :].astype(np.float64)
    root_height_changes = height_changes[middle]
    differences = height_changes - root_height_changes[None, :, :]
    distances = np.linalg.norm(differences, axis=-1)

    assert np.isfinite(points).all() and points.min() >= 0 and points.max() <= 1
    assert np.array_equal(distances[:, middle], np.zeros(31))
    assert np.array_equal(distances[middle], np.zeros(31))
    assert np.allclose(points.sum(axis=-1, dtype=np.float64), widths[:, None] * heights[None, :], atol=1e-5)
    # Independent scalar rectangle calculation for every plotted distance.
    edges = np.arange(28)
    def coverage(center, sizes):
        return np.maximum(0, np.minimum(center + sizes[:, None]/2, edges+1)
                          - np.maximum(center - sizes[:, None]/2, edges))
    horizontal = coverage(baseline[0], widths)
    vertical = coverage(baseline[1], heights)
    analytic = (np.linalg.norm(horizontal-horizontal[middle], axis=-1)[:, None]
                * np.linalg.norm(vertical-vertical[middle], axis=-1)[None, :])
    max_check_error = float(np.abs(distances-analytic).max())
    assert np.allclose(distances, analytic, atol=1e-6)

    out = ROOT / "figures" / "generated_one_height_change_comparison"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "samples.npz", baseline=baseline, parameters=parameters,
                        points=points, height_changes=height_changes,
                        root_height_changes=root_height_changes, distances=distances,
                        widths=widths, heights=heights, height_offsets=steps)
    with (out / "distances.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["width_px", "height_change_px", "height_px", "distance_between_height_changes"])
        for i, width in enumerate(widths):
            for j, height in enumerate(heights):
                writer.writerow([width, steps[j], height, distances[i, j]])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    norm = Normalize(vmin=widths[0], vmax=widths[-1])
    cmap = plt.get_cmap("coolwarm")

    fig = plt.figure(figsize=(15, 8), facecolor="white")
    fig.text(.045, .955, "Does the same height change produce the same pixel change at different widths?", fontsize=18, weight="bold")
    fig.text(.045, .915, "Each line holds width fixed. At each x, compare its height-change vector with the normal-width root's height-change vector.", fontsize=11)
    img_ax = fig.add_axes([.055, .51, .16, .30])
    img_ax.imshow(points[middle, middle].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    img_ax.set_title("Reference: normal root 1", fontsize=12, pad=13)
    img_ax.set_xticks([])
    img_ax.set_yticks([])
    fig.text(.045, .465, "28 × 28 = 784 coverage values\nWhite: no ink (0); black: full ink (1)", fontsize=10, va="top")
    fig.text(.045, .365, "Center: (14, 14) px; lean: 0°\nBase height: 19.75 px\nReference width: 3.2 px", fontsize=11, va="top", linespacing=1.6)
    fig.text(.045, .25, "31 widths: 1.7–4.7 px\n31 heights: 18.25–21.25 px\nBoth sampled every 0.1 px", fontsize=10, va="top", linespacing=1.6)

    ymax = float(distances.max()) * 1.08
    for rect, ids, title in [
        ([.30, .25, .27, .56], range(middle), "Thinner widths: 1.7–3.1 px"),
        ([.62, .25, .27, .56], range(middle+1, 31), "Wider widths: 3.3–4.7 px"),
    ]:
        ax = fig.add_axes(rect)
        for i in ids:
            ax.plot(steps, distances[i], color=cmap(norm(widths[i])), linewidth=1.35)
        ax.plot(steps, distances[middle], color="black", linewidth=2.2,
                label="Reference width: 3.2 px")
        ax.axvline(0, color="#777777", linewidth=.8, linestyle="--", zorder=0)
        ax.scatter([0], [0], s=25, color="black", zorder=4)
        ax.set_title(title, fontsize=12, pad=13)
        ax.set_xlabel("Height change from base height (pixels)", fontsize=10, labelpad=10)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-.035 * ymax, ymax)
        ax.set_xticks(np.arange(-1.5, 1.6, .5))
        ax.grid(alpha=.2)
        ax.legend(loc="upper center", fontsize=9, framealpha=.95)
        if rect[0] == .30:
            ax.set_ylabel("Distance between height-change vectors (784D L₂)", fontsize=10)
    cb_ax = fig.add_axes([.925, .25, .016, .56])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax)
    cb.set_ticks([1.7, 2.2, 2.7, 3.2, 3.7, 4.2, 4.7])
    cb.set_label("Fixed width per line (pixels)", fontsize=10)
    fig.text(.30, .16, "All curves meet at x = 0: no height change at any width. The reference stays at y = 0 across all heights.", fontsize=10)
    fig.text(.30, .125, "Matching axes in both panels. Equal distances can overlap; distance alone hides the pixel-change direction.", fontsize=10)
    fig.text(.045, .075, "Height-change vector at width w:  Δ(w, x) = image(w, 19.75 + x) − image(w, 19.75).", fontsize=11)
    fig.text(.045, .04, "Plotted y = ‖Δ(w, x) − Δ(3.2, x)‖₂. Reference is the normal-width root, not an average across widths.", fontsize=11)
    fig.text(.045, .012, "Controlled generated-one slice; center and lean fixed. Experimental ranges extend beyond parts of the default box.", fontsize=9, color="#555555")
    fig.savefig(out / "height_change_comparison.png", dpi=160)
    fig.savefig(out / "height_change_comparison.svg")
    plt.close(fig)

    summary = {
        "baseline_knobs": dict(zip(KNOBS, baseline.tolist())),
        "width_samples": 31, "height_samples": 31,
        "step_px": .1,
        "x": "Signed height change from 19.75px",
        "line": "Fixed width",
        "height_change": "image(width, base_height+x) - image(width, base_height)",
        "y": "Euclidean distance between that height change and the height change at reference width 3.2px, at the same x",
        "reference": "Normal-width root; no averaging or vector normalization",
        "verification": {"all_widths_at_zero_height_change": 0,
                         "reference_curve_all_heights": 0,
                         "maximum_distance_error_vs_rectangle_coverage": max_check_error},
        "understanding_check": {
            "object": "Each mark compares two finite 784D height-change vectors at the same signed height offset",
            "axes_colors": "Height offset in pixels; raw L2 vector difference; colors identify width; all stated on figure",
            "baseline": "Reference image shown first; reference width is black at zero; all curves zero at x=0",
            "visible_evidence": "Curve separation shows a width-dependent difference in height changes; two panels use identical axes",
            "scope": "Fixed center and lean; sampled height/width slice; scalar distance hides direction; overlapping equal distances stated",
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "README.md").write_text(
        "# Height-change comparison across widths\n\n"
        "View `height_change_comparison.png` (SVG is the same figure).\n\n"
        "Baseline `(cx, cy, height, width, lean) = (14, 14, 19.75, 3.2, 0)`. "
        "Width and height vary ±1.5px, every 0.1px: 31 widths × 31 heights. Center and lean remain fixed.\n\n"
        "At each width w, the finite height-change vector is `D(w,x) = render(w,h0+x) - render(w,h0)`. "
        "Plot `norm(D(w,x) - D(w0,x))`, with `h0=19.75` and `w0=3.2`. "
        "Each line holds width fixed; x is signed height change. The reference is the normal-width root, not the mean. "
        "Vectors are unnormalized, so this measures differences in both magnitude and pixel pattern. "
        "All curves equal zero at x=0, and the reference is identically zero.\n\n"
        "Thinner and wider widths appear in two panels with matching axes to improve readability. "
        "Some curves overlap because equal scalar distances do not identify the full pixel-change pattern.\n\n"
        "`samples.npz` stores points, parameters, full height changes, reference height changes, and distances. "
        "Array order is width × height × row-major pixel. `distances.csv` contains every plotted distance. "
        "Coverage bounds, zero cases, integrated area, and an independent rectangle-coverage formula were verified.\n\n"
        "Reproduce from the repo root:\n\n```sh\n"
        "MPLCONFIGDIR=/tmp/generated-one-mpl .venv/bin/python experiments/compare_height_changes_across_widths_20261008.py\n```\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
