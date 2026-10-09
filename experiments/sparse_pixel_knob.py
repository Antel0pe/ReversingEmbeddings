"""Build a sparse 784-pixel knob and plot every requested setting.

Run from the repository root: .venv/bin/python experiments/sparse_pixel_knob.py
"""

from pathlib import Path
import csv
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grey_ones import render

PIXELS = np.array([5, 32, 234, 423, 664])
OUT = ROOT / "figures" / "sparse_pixel_knob"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(784, dtype=np.int64)
    mask[PIXELS] = 1
    k = np.arange(11)
    # Integer tenths encode the exact decimal construction; floats are for plotting.
    tenths = k[:, None] * mask[None, :]
    images = tenths / 10.0
    vector = mask / 10.0
    other = mask == 0
    differences = np.diff(images, axis=0)
    assert np.all(tenths[:, other] == 0)
    assert np.all(np.diff(tenths, axis=0)[:, PIXELS] == 1)
    assert np.all(np.count_nonzero(images[1:], axis=1) == 5)
    assert np.all(differences[:, other] == 0)
    np.testing.assert_allclose(differences[:, PIXELS], 0.1, rtol=0, atol=2e-16)
    max_error = float(np.max(np.abs(differences[:, PIXELS] - 0.1)))

    np.save(OUT / "vector.npy", vector)
    np.save(OUT / "settings_0_to_10.npy", images)
    with (OUT / "all_pixel_values.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pixel_index", "row", "column", "per_step_change"] +
                        [f"knob_{value}" for value in k])
        for index in range(784):
            writer.writerow([index, index // 28, index % 28,
                             f"{mask[index] / 10:.1f}"] +
                            [f"{value / 10:.1f}" for value in tenths[:, index]])
    verification = {
        "indexing": "zero-based, row-major",
        "selected_pixels": PIXELS.tolist(),
        "knob_settings": k.tolist(),
        "nonzero_pixels_at_each_positive_setting": [5] * 10,
        "other_pixels_max_absolute_value": float(np.max(np.abs(images[:, other]))),
        "other_pixels_max_absolute_increment": float(np.max(np.abs(differences[:, other]))),
        "exact_integer_tenths_increment_check": "passed for all 5 pixels and all 10 steps",
        "float64_max_absolute_increment_error": max_error,
    }
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")

    source = render([14.5, 14.5, 19.75, 3.2, 12.5])[0]
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    fig.text(.5, .965, "Can one knob grow exactly five pixels by 0.1 per step?",
             ha="center", fontsize=21, weight="bold")
    fig.text(.5, .928, "Yes: x(k) = k v, where v has 0.1 at indices 5, 32, 234, 423, 664 and 0 at all other indices.",
             ha="center", fontsize=12)
    fig.text(.5, .897, "28 × 28 image = 784 pixel values. Intensity 0 = no ink; 1 = full ink. Indices start at 0 and run across each row.",
             ha="center", fontsize=11, color="#444444")

    ax = fig.add_axes([.055, .63, .245, .22])
    ax.imshow(source, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("1. Familiar object: a generated 1", fontsize=12, pad=10)
    for index in PIXELS:
        row, col = divmod(int(index), 28)
        ax.plot(col, row, "o", ms=9, mfc="none", mec="#138363", mew=1.5)
        offset = (9, -3) if index == 5 else (-23, -12) if index == 32 else (7, 2)
        ax.annotate(str(index), (col, row), xytext=offset, textcoords="offset points",
                    fontsize=9, color="#08704d")
    ax.set_xlabel("Green circles locate your five pixel indices.\nReference only; this image is not added below.", fontsize=10)
    ax.set_ylabel("Pixel row (downward)", fontsize=10)

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    for left, setting, title in [(.38, 0, "2. Starting case: x(0) = 0"),
                                 (.705, 10, "3. After 10 steps: x(10) = 10v")]:
        ax = fig.add_axes([left, .63, .245, .22])
        im = ax.imshow(images[setting].reshape(28, 28), cmap="RdBu_r", norm=norm,
                       interpolation="nearest")
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel("Pixel column (rightward)\n" +
                      ("All 784 values = 0" if setting == 0 else "Five values = 1.0; all other 779 values = 0"),
                      fontsize=10)
    for ax in fig.axes:
        ax.set_xticks([0, 7, 14, 21, 27])
        ax.set_yticks([0, 7, 14, 21, 27])
        ax.tick_params(labelsize=9)
    cax = fig.add_axes([.72, .545, .215, .014])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
    cb.ax.tick_params(labelsize=9)
    cb.set_label("Pixel change: blue loses ink; red gains ink", fontsize=9, labelpad=2)

    ax = fig.add_axes([.065, .155, .335, .30])
    ax.plot(k, images[:, PIXELS[0]], "o-", color="#c94e26", lw=2.5, ms=5,
            label="All five selected pixels (identical curves)")
    ax.plot(k, np.max(np.abs(images[:, other]), axis=1), "s--", color="#16724e", lw=2,
            ms=4, label="Maximum |value| across the other 779 pixels")
    ax.set_title("4. Every selected pixel rises by 0.1", fontsize=12, pad=10)
    ax.set_xlabel("Knob value k (0 is the baseline)", fontsize=11)
    ax.set_ylabel("Pixel intensity", fontsize=11)
    ax.set_xticks(k)
    ax.set_yticks(np.arange(0, 1.01, .1))
    ax.set_ylim(-.04, 1.08)
    ax.grid(alpha=.2)
    ax.legend(loc="upper left", fontsize=9, framealpha=.95)

    ax = fig.add_axes([.545, .155, .43, .30])
    values = np.vstack([images[:, PIXELS].T, np.max(np.abs(images[:, other]), axis=1)])
    ax.imshow(values, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
    for row in range(6):
        for col in range(11):
            ax.text(col, row, f"{values[row, col]:.1f}", ha="center", va="center",
                    fontsize=10, color="white" if values[row, col] >= .7 else "#222222")
    ax.set_xticks(np.arange(11), k)
    ax.set_yticks(np.arange(6), [f"Pixel {i}" for i in PIXELS] + ["Other 779:\nmax |value|"])
    ax.tick_params(axis="y", length=0, labelsize=10)
    ax.set_xlabel("Knob value k (0 → 10)", fontsize=11)
    ax.set_title("5. Values at every setting: 0.1, 0.2, …, 1.0", fontsize=12, pad=10)

    fig.text(.5, .075, "Verified on all 784 pixels at all 11 settings: other pixels and their increments are exactly zero.",
             ha="center", fontsize=11, color="#16724e")
    fig.text(.5, .049, f"Selected increments are exactly one tenth by construction; float64 subtraction error ≤ {max_error:.2g}.",
             ha="center", fontsize=10, color="#444444")
    fig.text(.5, .022, "This is a sparse line in pixel space. The isolated pixels are not a stroke in the five-parameter generated-one family. No values are clipped.",
             ha="center", fontsize=10, color="#444444")
    fig.savefig(OUT / "sparse_pixel_knob.png", dpi=150)
    fig.savefig(OUT / "sparse_pixel_knob.svg")
    plt.close(fig)
    (OUT / "README.md").write_text(
        "# Sparse pixel knob\n\n"
        "Pixel indices are zero-based and row-major. Define a 784-vector v with "
        "v[i] = 0.1 for i in {5, 32, 234, 423, 664} and zero elsewhere. "
        "Then x(k) = k v, starting at x(0) = 0. Settings 1–10 give selected "
        "intensities 0.1–1.0; the other 779 values remain zero.\n\n"
        "For an existing image b, b + k v instead keeps other pixels at their "
        "original values. Selected pixels may exceed 1 if they start above 0; "
        "clipping would break the constant-increment requirement. This sparse "
        "direction is a pixel-space knob, not one of the five generator knobs.\n\n"
        "The plot uses the default midpoint generated image as a spatial reference "
        "only. The measured sweep starts at the zero vector. Values use integer "
        "tenths before conversion to float64; verification.json reports float "
        "roundoff separately. all_pixel_values.csv contains all 784 coordinates, "
        "and vector.npy contains the per-step vector.\n\n"
        "Reproduce: `.venv/bin/python experiments/sparse_pixel_knob.py`\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
