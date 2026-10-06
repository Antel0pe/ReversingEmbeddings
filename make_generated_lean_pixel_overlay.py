"""Render width-labelled lean differences with the actual grey_ones renderer.

Run from the repository root: .venv/bin/python make_generated_lean_pixel_overlay.py
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

from grey_ones import RANGES, render

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures/generated_lean_pixel_overlay"
WIDTHS = np.array([1.8, 3.2, 3.3, 3.4, 4.6])
LEAN = np.arange(-10, 35.01, .5)
FIXED = np.array([14.5, 14.5, 19.75])
SELECTED = [0, 1, 4]
EPS = 1e-7
CMAP = LinearSegmentedColormap.from_list("ink_change", ["#2166ac", "#ffffff", "#cf312b"])


def frames():
    p = np.array([[*FIXED, w, t] for w in WIDTHS for t in LEAN])
    assert np.all((p >= RANGES[:, 0]) & (p <= RANGES[:, 1]))
    x = render(p).reshape(len(WIDTHS), len(LEAN), 28, 28)
    assert x.dtype == np.float32 and x.min() >= 0 and x.max() <= 1
    # No part of these strokes touches the canvas boundary.
    assert np.all(x[:, :, [0, -1], :] == 0) and np.all(x[:, :, :, [0, -1]] == 0)
    assert np.max(np.abs(x.sum((2, 3)) - WIDTHS[:, None] * FIXED[2])) < 2e-5
    return x


def measure(d):
    support = np.abs(d) > EPS
    counts = support.sum(0)
    return dict(norms=np.linalg.norm(d.reshape(len(d), -1), axis=1).tolist(),
                changed_pixels=support.sum((1, 2)).tolist(),
                union_pixels=int((counts > 0).sum()), shared_pixels=int((counts > 1).sum()),
                triple_pixels=int((counts == 3).sum()),
                opposite_sign_pixels=int(((d > EPS).any(0) & (d < -EPS).any(0)).sum()),
                max_pixel_change=float(np.max(np.abs(d))))


def slots_image(reference, d, scale, cell=24):
    """One raster cell per real pixel; three width slots and white gutters.

    Reference is context only. Values are never added to reference or each other.
    """
    image = np.repeat(np.repeat(1 - reference[:, :, None] * .78, cell, 0), cell, 1)
    image = np.repeat(image, 3, 2)
    lane = cell // len(d)
    for y in range(28):
        for x in range(28):
            if np.any(np.abs(d[:, y, x]) > EPS):
                for k in range(len(d)):
                    image[y*cell:(y+1)*cell, x*cell+k*lane] = 1
                    image[y*cell:(y+1)*cell, x*cell+(k+1)*lane-1] = 1
            for k in range(len(d)):
                if abs(d[k, y, x]) > EPS:
                    rgb = CMAP(.5 + .5 * d[k, y, x] / scale)[:3]
                    image[y*cell+1:(y+1)*cell-1, x*cell+k*lane+1:x*cell+(k+1)*lane-1] = rgb
    return image


def figure(x):
    start = int(np.flatnonzero(LEAN == 0)[0])
    end = int(np.flatnonzero(LEAN == 5)[0])
    a, b = x[SELECTED, start].astype(float), x[SELECTED, end].astype(float)
    d = b - a
    ref = x[1, start]
    m = measure(d)
    scale = m["max_pixel_change"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(14, 10), facecolor="white")
    fig.text(.045, .96, "Where does the lean-change pattern move as width changes?", fontsize=18, weight="bold")
    fig.text(.045, .912, "Generated 28 × 28 coverage images. Lean 0° → +5° (top moves right). Center (14.5, 14.5) px and height 19.75 px stay fixed.\n"
             "Every colored value = leaned image − its OWN upright image. Red gains ink; blue loses ink. Width alone distinguishes the three layers.", fontsize=10)
    ax = fig.add_axes([.048, .64, .15, .21])
    ax.imshow(ref, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest", extent=[0, 28, 28, 0])
    ax.set_title("1  Standard upright reference\nwidth 3.2 px", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(.5, -.1, "White = 0 ink; black = 1 ink", ha="center", va="top", transform=ax.transAxes, fontsize=9)
    for k, left in enumerate([.30, .52, .74]):
        ax = fig.add_axes([left, .64, .16, .21])
        ax.imshow(d[k], cmap=CMAP, vmin=-scale, vmax=scale, interpolation="nearest", extent=[0, 28, 28, 0])
        ax.set_title(f"2  Width {WIDTHS[SELECTED[k]]:.1f} px\nits lean-change vector", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(.5, -.1, f"L2 amount = {m['norms'][k]:.3f}\n{m['changed_pixels'][k]} pixels change", ha="center", va="top", transform=ax.transAxes, fontsize=9)
    ax = fig.add_axes([.06, .15, .36, .40])
    tiled = slots_image(ref, d, scale)
    ax.imshow(tiled, extent=[0, 28, 28, 0], interpolation="nearest")
    ax.set_title("3  All three widths on the same pixel grid", fontsize=12, pad=12)
    ax.set(xlabel="Horizontal pixel position (right →)", ylabel="Vertical pixel position (down →)")
    ax.set_xticks([0, 7, 14, 21, 28]); ax.set_yticks([0, 7, 14, 21, 28])
    # Real source-pixel boundaries; display subdivisions are not extra data pixels.
    for pos in range(29):
        ax.axvline(pos, color="#b5b5b5", lw=.22, alpha=.45)
        ax.axhline(pos, color="#b5b5b5", lw=.22, alpha=.45)
    ax.plot([11, 18, 18, 11, 11], [5, 5, 10, 10, 5], color="#7653a6", lw=1.7)
    ax = fig.add_axes([.50, .34, .21, .21])
    ax.imshow(tiled[5*24:10*24, 11*24:18*24], extent=[11, 18, 10, 5], interpolation="nearest")
    ax.set_title("Enlarged: top edge, same pixels", fontsize=11, pad=12)
    ax.set_xticks(range(11, 19)); ax.set_yticks(range(5, 11))
    for pos in range(11, 19): ax.axvline(pos, color="#aaa", lw=.5)
    for pos in range(5, 11): ax.axhline(pos, color="#aaa", lw=.5)
    ax.set(xlabel="Horizontal pixel position", ylabel="Vertical pixel position")
    fig.text(.50, .275, "Within EACH pixel: left slot = width 1.8,\ncenter = 3.2, right = 4.6 px. White gutters\nseparate layers without moving their pixels.", fontsize=10, va="top")
    fig.text(.76, .51, f"{m['shared_pixels']} of {m['union_pixels']} changed pixels\nare shared by multiple widths.\n\nThe edge pattern shifts with width.\nSome changes occupy the same\npixel even at these wide spacings.\n\nBlack/gray underlay is always\nthe standard width-3.2 reference;\nit is context, not another delta.", fontsize=10, va="top", linespacing=1.35)
    cax = fig.add_axes([.51, .17, .40, .018])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-scale, scale), cmap=CMAP), cax=cax, orientation="horizontal")
    bar.set_ticks([-scale, 0, scale], labels=[f"−{scale:.3f}: loses ink", "0", f"+{scale:.3f}: gains ink"])
    fig.text(.51, .12, "One shared color scale for all widths; no per-layer normalization.", fontsize=9)
    fig.text(.045, .048, "One mark is one pixel's total coverage change over 5°, not a pixel trajectory. Slot positions only identify widths.\n"
             "This shows three samples of the width/lean slice, not the full five-knob family. Flattening each delta gives the same 784D vector.", fontsize=10)
    fig.savefig(OUT / "overlay.png", dpi=160)
    plt.close(fig)
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x = frames()
    result = figure(x)
    with (OUT / "step_metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["width_px", "start_lean_deg", "end_lean_deg", "step_deg", "change_l2", "average_change_l2_per_degree", "net_ink_change"])
        for wi, width in enumerate(WIDTHS):
            for end in LEAN:
                start = 0.
                d = x[wi, np.flatnonzero(LEAN == end)[0]].astype(float) - x[wi, np.flatnonzero(LEAN == start)[0]].astype(float)
                n = float(np.linalg.norm(d))
                writer.writerow([width, start, end, end-start, n, "" if end == start else n/abs(end-start), float(d.sum())])
    # Float32 inputs converted to float64 before subtraction preserve the actual
    # renderer values, including cancellations, without quantizing the data.
    sparse = []
    for width_frames in x:
        row = []
        for image in width_frames:
            flat = image.ravel()
            ids = np.flatnonzero(flat)
            row.append([ids.tolist(), flat[ids].astype(float).tolist()])
        sparse.append(row)
    data = dict(widths=WIDTHS.tolist(), lean=LEAN.tolist(), images=sparse,
                fixed=FIXED.tolist(), support_threshold=EPS)
    template = (ROOT / "experiments/lean_pixel_overlay/viewer.html").read_text()
    assert template.count("__RENDERED_DATA__") == 1
    (OUT / "index.html").write_text(template.replace("__RENDERED_DATA__", json.dumps(data, separators=(",", ":"))))
    # Actual decoder check: sparse payload must recover every float32 sample.
    for wi in range(len(WIDTHS)):
        for ti in range(len(LEAN)):
            ids, values = sparse[wi][ti]
            recovered = np.zeros(784, np.float32)
            recovered[ids] = values
            assert np.array_equal(recovered.reshape(28, 28), x[wi, ti])
    start, end = np.flatnonzero(LEAN == 0)[0], np.flatnonzero(LEAN == 5)[0]
    d = x[SELECTED, end].astype(float) - x[SELECTED, start].astype(float)
    assert np.max(np.abs(x[SELECTED, start].astype(float) + d - x[SELECTED, end])) == 0
    assert np.max(np.abs(d.sum((1, 2)))) < 2e-6
    result.update(widths_px=WIDTHS[SELECTED].tolist(), start_lean_deg=0, end_lean_deg=5,
                  fixed_center_px=FIXED[:2].tolist(), height_px=float(FIXED[2]),
                  renderer="grey_ones.render, float32, 16 subrows; subtraction in float64",
                  validation=dict(rendered_frames=int(len(WIDTHS)*len(LEAN)), sparse_roundtrip="exact", endpoint_reconstruction="exact", no_canvas_clipping=True))
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "README.md").write_text("""# Lean pixel changes across widths

Open `index.html` for the interactive view, or `overlay.png` for the initial comparison.
Regenerate from the repository root with `.venv/bin/python make_generated_lean_pixel_overlay.py`.

The earlier scalar curves are in `../generated_lean_width_comparison/width_sweep/`.
This experiment shows their underlying pixel-change patterns directly.

Center is (14.5, 14.5) px and height is 19.75 px. The default widths are 1.8,
3.2, and 4.6 horizontal pixels; nearby-width mode uses 3.2, 3.3, and 3.4.
All settings are inside the generator's default ranges. Positive lean moves
the top rightward. Start/end lean can be selected on a 0.5-degree grid from
-10 through +35 degrees. Every displayed image was rendered by `grey_ones.render`;
the viewer reads those samples and does not reimplement or interpolate the renderer.

For each width w, the displayed vector is `render(w, end) - render(w, start)`
with the other three knobs fixed. Each width uses its own starting image.
Subtracting the standard width-3.2 image from every endpoint would combine
width and lean changes, a different question. Black/gray is the standard
width-3.2 image at the selected start lean, used only as context.

Red is positive coverage change and blue is negative. All widths share one
symmetric scale, either the largest absolute change among the three layers
or a fixed +/-1. In per-degree mode the scale uses coverage change per degree.
The full-image panels contain one signed value per real pixel. The shared
grid divides each displayed cell into labelled width slots with white gutters:
these subdivisions identify widths and are not extra spatial coordinates.
Same-cell overlap is measured and reported; it is not removed. Blend mode
averages signed values among selected nonzero layers, so cancellation can
occur; purple borders explicitly mark pixels where multiple layers change.
Hover or tap a shared-grid pixel to read the individual values in all modes.
Hiding a width keeps the common color scale and width-slot locations fixed.

The vector contains total changes, not time or instantaneous velocity.
Dividing by `end-start` gives the average signed pixel response per degree.
Its norm is chord amount per degree, not path length per degree. For a
small step it approximates a one-sided local derivative; pixel events can
change that derivative. Normalizing to unit length removes overall magnitude.
The page reports amount and amount/absolute-step separately. Zero-step
response per degree is undefined and is shown explicitly.

This is a sampled width/lean slice; no global manifold or exact tangent
alignment claim follows from the visual similarity. Source coverage values
are float32; subtraction and measurements use float64. Values below 1e-7
are not drawn as colored marks and are excluded from support counts; raw
hover values and norms still use all values. Sparse data recovers every
sample exactly. Checks also cover image bounds, ink conservation, no
canvas clipping, and exact reconstruction of the endpoint by start + delta.
The common-scale endpoint colors do not clip the selected data.
""")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
