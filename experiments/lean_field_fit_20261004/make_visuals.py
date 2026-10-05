"""Build the explanatory figure and a self-contained measured-path viewer."""
import base64
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def main():
    metrics = json.loads((OUT / "metrics.json").read_text())
    paths = [dict(np.load(OUT / f"path_{i}.npz")) for i in range(6)]
    metadata = {"metrics": metrics, "paths": [], "angles": paths[0]["lean_degrees"].tolist(),
                "widths": [float(p["width"]) for p in paths]}
    buffers = []
    offset = 0
    for p in paths:
        entries = {}
        for key, value in p.items():
            if key in ("width", "lean_degrees"):
                continue
            flat = np.asarray(value, dtype="<f4").ravel()
            entries[key] = {"offset": offset, "length": len(flat)}
            buffers.append(flat.tobytes())
            offset += len(flat)
        metadata["paths"].append(entries)
    encoded = base64.b64encode(gzip.compress(b"".join(buffers), mtime=0)).decode("ascii")
    template = (HERE / "viewer_template.html").read_text()
    html = template.replace("__METADATA__", json.dumps(metadata)).replace("__PATHS__", encoded)
    assert "__METADATA__" not in html and "__PATHS__" not in html
    (OUT / "viewer.html").write_text(html)

    p = paths[2]
    k = int(np.argmin(abs(p["lean_degrees"] - 12.5)))
    fig = plt.figure(figsize=(16.5, 10), facecolor="white")
    fig.text(.045, .95, "Can following a fitted lean field reproduce the true image?", fontsize=21, weight="bold")
    fig.text(.045, .893,
             f"Generated 1s: 28 × 28 ink coverage values, white = 0 and black = 1. Width {float(p['width']):.4f} px is fixed.\n"
             "Every reconstructed path starts from the actual −10° image. Center (14.5, 14.5) px and height 19.75 px are fixed.",
             fontsize=11)
    images = [p["actual_images"][0], p["actual_images"][k],
              p["polynomial_degree_5_images"][k], p["piecewise_edge_rule_images"][k]]
    titles = ["1. Shared starting image", "2. Actual image after leaning",
              "3. Degree-5 polynomial field", "4. Boundary-aware field"]
    for i, (image, title) in enumerate(zip(images, titles)):
        ax = fig.add_axes([.05 + .235 * i, .575, .185, .24])
        ax.imshow(image, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        out = (image < 0) | (image > 1)
        rgba = np.zeros((28, 28, 4)); rgba[out] = [145/255, 72/255, 175/255, 1]
        ax.imshow(rgba, interpolation="nearest")
        ax.set_axis_off()
        ax.set_title(title, fontsize=12, pad=12)
        error = float(np.linalg.norm(image - p["actual_images"][k])) if i else 0
        caption = "Lean −10° · exact start" if i == 0 else (
            f"Lean 12.5° · L2 error {error:.4g}\nMax pixel error {abs(image-p['actual_images'][k]).max():.3g}")
        ax.text(.5, -.11, caption, ha="center", va="top", transform=ax.transAxes, fontsize=10)
    fig.text(.045, .463,
             "Purple marks raw predicted pixels outside [0, 1], including faint errors. Predictions are never clipped for measurement.\n"
             "A close image can still be invalid. Image L2 error compares all 784 pixels with the target renderer; 0 means exact agreement.", fontsize=10)

    axes = [fig.add_axes([.065, .165, .255, .235]), fig.add_axes([.385, .165, .255, .235]),
            fig.add_axes([.71, .165, .255, .235])]
    models = ["polynomial_degree_1", "polynomial_degree_3", "polynomial_degree_5", "piecewise_edge_rule"]
    colors = ["#a9a5a0", "#ce8b5c", "#b63b35", "#2563a6"]
    labels = ["Linear", "Cubic", "Degree 5", "Edge rule"]
    for model, color, label in zip(models, colors, labels):
        err = np.linalg.norm(p[f"{model}_images"] - p["actual_images"], axis=(1, 2))
        axes[0].plot(p["lean_degrees"], err, color=color, label=label, lw=1.6)
    axes[0].set(title="Image error along the same lean path", xlabel="Lean (degrees)", ylabel="Whole-image L2 error")
    axes[0].legend(fontsize=8, ncol=2)
    for key, color, label in [("piecewise_edge_rule_images", "#2563a6", "Edge rule"),
                              ("oracle_integrated_images", "#16705e", "Measured-arrow integration")]:
        err = np.linalg.norm(p[key] - p["actual_images"], axis=(1, 2))
        axes[1].plot(p["lean_degrees"], err, color=color, label=label, lw=1.6)
    axes[1].set(title="Same error, zoomed to reveal drift", xlabel="Lean (degrees)", ylabel="Whole-image L2 error")
    axes[1].legend(fontsize=8)
    controls = [metrics["full_space_control"]["models"][m]["relative_arrow_error"]["mean"] * 100 for m in models]
    axes[2].bar(range(4), controls, color=colors)
    axes[2].set(title="Control: all five settings vary", ylabel="Mean relative arrow error (%)",
                xticks=range(4), xticklabels=["Linear", "Cubic", "Deg. 5", "Edge"])
    for i, val in enumerate(controls):
        axes[2].text(i, val + max(controls)*.025, f"{val:.2f}%", ha="center", fontsize=9)
    axes[2].set_ylim(0, max(controls)*1.19)
    for ax in axes:
        ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
    fig.text(.045, .063,
             "425 training states; 703 held-out slice states + 4 boundary corners. Numerical path scores use 901 samples; charts show 91 stored frames.\n"
             "Full-space control: 256 new random states. Polynomials only see width and lean; the edge rule uses all five known settings and supplied geometry.\n"
             "The edge rule fits one scalar; degree 5 stores 21 × 784 coefficients. This is field fitting, not discovery of the generator or proof of exact membership.",
             fontsize=10)
    fig.savefig(OUT / "comparison.png", dpi=150)
    plt.close(fig)
    print("Wrote results/viewer.html and results/comparison.png")


if __name__ == "__main__":
    main()
