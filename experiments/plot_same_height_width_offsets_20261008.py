"""Plot the saved width sweeps against the normal-width image at each height.

Run after compare_height_changes_across_widths_20261008.py:
    MPLCONFIGDIR=/tmp/generated-one-mpl .venv/bin/python experiments/plot_same_height_width_offsets_20261008.py
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "figures" / "generated_one_height_change_comparison" / "samples.npz"
OUT = ROOT / "figures" / "generated_one_same_height_offsets"


def main():
    with np.load(SOURCE) as saved:
        points = saved["points"].astype(np.float64)
        widths = saved["widths"]
        heights = saved["heights"]
        steps = saved["height_offsets"]
        baseline = saved["baseline"]
    wi = int(np.argmin(abs(widths-baseline[3])))
    hi = int(np.argmin(abs(steps)))
    distances = np.linalg.norm(points-points[wi:wi+1], axis=-1)
    assert np.array_equal(distances[wi], np.zeros(len(heights)))
    assert np.all(distances[np.arange(len(widths)) != wi, hi] > 0)
    assert np.isfinite(distances).all()
    # Independent exact rectangle coverage check (all strokes have zero lean).
    edges = np.arange(28)
    def coverage(center, sizes):
        return np.maximum(0, np.minimum(center+sizes[:, None]/2, edges+1)
                          - np.maximum(center-sizes[:, None]/2, edges))
    horizontal = coverage(baseline[0], widths)
    vertical = coverage(baseline[1], heights)
    expected = (np.linalg.norm(horizontal-horizontal[wi], axis=-1)[:, None]
                * np.linalg.norm(vertical, axis=-1)[None, :])
    error = float(np.abs(expected-distances).max())
    assert np.allclose(expected, distances, atol=1e-6)

    norm = Normalize(widths[0], widths[-1])
    cmap = plt.get_cmap("coolwarm")
    fig = plt.figure(figsize=(15, 8), facecolor="white")
    fig.text(.045, .955, "How far is each width from the normal-width 1 at the same height?", fontsize=19, weight="bold")
    fig.text(.045, .915, "One line per fixed width. At every x, both images have height 19.75 + x pixels; the reference width is always 3.2 pixels.", fontsize=11)
    img_ax = fig.add_axes([.055, .51, .16, .30])
    img_ax.imshow(points[wi, hi].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    img_ax.set_title("Reference at x = 0", fontsize=12, pad=13)
    img_ax.set_xticks([])
    img_ax.set_yticks([])
    fig.text(.045, .465, "28 × 28 = 784 coverage values\nWhite: no ink (0); black: full ink (1)", fontsize=10, va="top")
    fig.text(.045, .365, "Center: (14, 14) px; lean: 0°\nBase height: 19.75 px\nReference width: 3.2 px", fontsize=11, va="top", linespacing=1.6)
    fig.text(.045, .25, "31 widths: 1.7–4.7 px\n31 heights: 18.25–21.25 px\nBoth sampled every 0.1 px", fontsize=10, va="top", linespacing=1.6)
    ymax = float(distances.max()) * 1.08
    for rect, ids, title in [
        ([.30, .25, .27, .56], range(wi), "Thinner widths: 1.7–3.1 px"),
        ([.62, .25, .27, .56], range(wi+1, len(widths)), "Wider widths: 3.3–4.7 px"),
    ]:
        ax = fig.add_axes(rect)
        for i in ids:
            ax.plot(steps, distances[i], color=cmap(norm(widths[i])), linewidth=1.35)
        ax.plot(steps, distances[wi], color="black", linewidth=2.2, label="Reference width: 3.2 px")
        ax.axvline(0, color="#777777", linewidth=.8, linestyle="--", zorder=0)
        ax.set_title(title, fontsize=12, pad=13)
        ax.set_xlabel("Height change from base height (pixels)", fontsize=10, labelpad=10)
        ax.set_xlim(steps[0], steps[-1])
        ax.set_ylim(-.035*ymax, ymax)
        ax.set_xticks(np.arange(-1.5, 1.6, .5))
        ax.grid(alpha=.2)
        ax.legend(loc="upper center", fontsize=9)
        if rect[0] == .30:
            ax.set_ylabel("Distance to normal-width image at same height (784D L₂)", fontsize=10)
    cb_ax = fig.add_axes([.925, .25, .016, .56])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax)
    cb.set_ticks([1.7, 2.2, 2.7, 3.2, 3.7, 4.2, 4.7])
    cb.set_label("Fixed width per line (pixels)", fontsize=10)
    fig.text(.30, .16, "At x = 0 the widths still differ, so colored curves remain above zero. The black reference stays at zero.", fontsize=10)
    fig.text(.30, .125, "Matching axes in both panels. Some curves overlap; equal distances can hide different pixel patterns.", fontsize=10)
    fig.text(.045, .075, "For each x: X = image(width 3.2, height 19.75 + x); Y = image(line's width, height 19.75 + x).", fontsize=11)
    fig.text(.045, .04, "Plotted y = ‖Y − X‖₂ over all 784 coverage values. The normal-width reference changes height along with each line.", fontsize=11)
    fig.text(.045, .012, "Controlled generated-one slice; center and lean fixed. Experimental ranges extend beyond parts of the default box.", fontsize=9, color="#555555")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "same_height_width_offsets.png", dpi=160)
    fig.savefig(OUT / "same_height_width_offsets.svg")
    plt.close(fig)
    summary = {
        "source_samples": str(SOURCE.relative_to(ROOT)),
        "baseline_knobs": baseline.tolist(),
        "formula": "norm(image(width, height) - image(3.2, same_height))",
        "x": "Height change from 19.75px", "line": "Fixed width",
        "width_samples": len(widths), "height_samples": len(heights),
        "maximum_error_vs_exact_rectangle_coverage": error,
        "understanding_check": {
            "object": "Each mark compares two 784D images with equal height and different widths",
            "axes_and_colors": "Height offset in pixels; raw L2 coverage distance; color identifies width",
            "reference": "Normal-width source image shown first; same-height reference curve black at zero",
            "evidence": "Nonreference curves remain above zero at base height and vary modestly across height",
            "scope": "Fixed center and lean; sampled slice; distance hides pixel pattern; overlap stated",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (OUT / "README.md").write_text(
        "# Same-height width offsets\n\nView `same_height_width_offsets.png`; SVG is the same figure.\n\n"
        "Each line fixes width. X is height change from 19.75px. "
        "Y is `norm(image(width, height) - image(3.2, same_height))` over all 784 coverage values. "
        "The reference changes height at each x, and its own curve is identically zero. "
        "Other widths remain at nonzero distance even at the base height.\n\n"
        "Baseline: `(14, 14, 19.75, 3.2, 0)`. Both height and width vary ±1.5px in 0.1px steps. "
        "Saved images come from `figures/generated_one_height_change_comparison/samples.npz`. "
        "Distances were checked against independent exact rectangle coverage.\n\n"
        "Reproduce from the repo root after generating those samples:\n\n```sh\n"
        "MPLCONFIGDIR=/tmp/generated-one-mpl .venv/bin/python experiments/plot_same_height_width_offsets_20261008.py\n```\n"
    )
    print("Plot: " + str((OUT / "same_height_width_offsets.png").relative_to(ROOT)))
    print("Maximum verification error: %.3g" % error)


if __name__ == "__main__":
    main()
