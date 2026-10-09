"""Check separable pixel coefficients and explain the centered, untilted slice."""

from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grey_ones import render

OUT = ROOT / "figures/generated_one_height_width_relationship"


def ramp(x):
    return np.clip(x, 0, 1)


def predict(heights, widths):
    d = np.abs(np.arange(28)+.5-14)-.5
    rows = ramp(np.asarray(heights)[:, None]/2-d)
    cols = ramp(np.asarray(widths)[:, None]/2-d)
    return rows[:, :, None]*cols[:, None, :]


def main():
    with np.load(ROOT / "figures/generated_one_height_change_comparison/samples.npz") as saved:
        points = saved["points"].astype(np.float64).reshape(31, 31, 28, 28)
        widths, heights, baseline = saved["widths"], saved["heights"], saved["baseline"]
    # Extract each factor directly from full-coverage interior rows/columns.
    horizontal = points[:, 15, 14, :]
    vertical = points[15, :, :, 13]
    product = horizontal[:, None, None, :]*vertical[None, :, :, None]
    profile_error = float(np.abs(product-points).max())
    hh, ww = np.meshgrid(heights, widths)
    formula_error = float(np.abs(predict(hh.ravel(), ww.ravel()).reshape(points.shape)-points).max())
    maskw = (widths > 2) & (widths < 4)
    maskh = (heights > 18) & (heights < 20)
    dh, dw = np.meshgrid(heights[maskh]-19.75, widths[maskw]-3.2)
    design = np.column_stack([np.ones(dh.size), dh.ravel(), dw.ravel(), (dh*dw).ravel()])
    measured = points[maskw][:, maskh].reshape(-1, 784)
    coefficients = np.linalg.lstsq(design, measured, rcond=None)[0]
    fit_error = float(np.abs(design@coefficients-measured).max())

    rng = np.random.default_rng(20261008)
    test_h = rng.uniform(heights.min(), heights.max(), 1024)
    test_w = rng.uniform(widths.min(), widths.max(), 1024)
    parameters = np.tile(baseline, (1024, 1))
    parameters[:, 2], parameters[:, 3] = test_h, test_w
    predictions = predict(test_h, test_w)
    # Renderer is used only as held-out ground truth, not inside predict().
    truth = render(parameters).astype(np.float64)
    residual = predictions-truth
    test_pixel_error = float(np.abs(residual).max())
    test_image_error = float(np.linalg.norm(residual.reshape(-1, 784), axis=-1).max())
    assert max(profile_error, formula_error, fit_error, test_pixel_error) < 1e-6

    OUT.mkdir(parents=True, exist_ok=True)
    results = {
        "scope": "Center (14,14), lean 0; width 1.7-4.7px, height 18.25-21.25px",
        "max_pixel_error_measured_profile_product": profile_error,
        "max_pixel_error_position_formula_on_grid": formula_error,
        "max_pixel_error_bilinear_fit_in_one_region": fit_error,
        "pixel_124_fitted_coefficients_constant_dh_dw_dhdw": coefficients[:,124].tolist(),
        "held_out_joint_samples": 1024,
        "held_out_max_pixel_error": test_pixel_error,
        "held_out_max_784D_error": test_image_error,
        "seed": 20261008,
        "understanding_check": {
            "object": "One untilted generated image, its one-dimensional factors, and one pixel's coefficient",
            "axes": "Width and height in pixels; factors are fractional coverage in [0,1]; curves identify pixel columns/rows",
            "baseline": "Normal image first; baseline settings shown and dashed on both factor plots",
            "evidence": "Row/column factors multiply; worked pixel product shown; joint held-out error stated",
            "scope": "Fixed center, zero tilt, explicit bilinear region; whole formula clips; no claim about all five knobs",
        },
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2)+"\n")

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    fig.text(.045, .955, "A pixel coefficient = its height factor × its width factor", fontsize=21, weight="bold")
    fig.text(.045, .915, "Centered, untilted generated 1s: height controls the rows; width controls the columns. The two factors multiply.", fontsize=12)
    img_ax = fig.add_axes([.06, .38, .19, .39])
    img_ax.imshow(points[15, 15], cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    img_ax.add_patch(Rectangle((11.5,3.5), 1, 1, fill=False, edgecolor="#ca3c2a", linewidth=2))
    img_ax.set_title("Start: normal root 1", fontsize=13, pad=15)
    img_ax.set_xticks([])
    img_ax.set_yticks([])
    fig.text(.045, .34, "Center (14,14); lean 0°\nHeight 19.75 px; width 3.2 px\nWhite = 0 ink; black = 1 ink\nMarked pixel: row 4, column 12 (i = 124)", fontsize=10.5, va="top", linespacing=1.7)

    axw = fig.add_axes([.34, .39, .27, .38])
    axh = fig.add_axes([.68, .39, .27, .38])
    colors = ["#4276ad", "#ca3c2a", "#539761"]
    dense_w = np.linspace(1.7,4.7,301)
    dense_h = np.linspace(18.25,21.25,301)
    d = np.abs(np.arange(28)+.5-14)-.5
    for c, color in zip([11,12,13], colors):
        axw.plot(dense_w, ramp(dense_w/2-d[c]), color=color, linewidth=2,
                 label=f"Column {c}")
        axw.scatter(widths[::3], horizontal[::3,c], color=color, s=15, zorder=3)
    for r, color in zip([3,4,5], colors):
        axh.plot(dense_h, ramp(dense_h/2-d[r]), color=color, linewidth=2,
                 label=f"Row {r}")
        axh.scatter(heights[::3], vertical[::3,r], color=color, s=15, zorder=3)
    for ax, x0, label, title in [
        (axw, 3.2, "Width (pixels)", "Column factor A_c(w)"),
        (axh, 19.75, "Height (pixels)", "Row factor B_r(h)"),
    ]:
        ax.axvline(x0, color="#888888", linestyle="--", linewidth=1)
        ax.set_ylim(-.04,1.09)
        ax.set_title(title, fontsize=13, pad=13)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Coverage factor (0–1)", fontsize=11)
        ax.grid(alpha=.2)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.18), ncol=3,
                  fontsize=9, frameon=False, handlelength=1.1, columnspacing=.8)
    axw.set_xlim(1.7,4.7)
    axh.set_xlim(18.25,21.25)
    fig.text(.34, .31, "Dots: measured factors from the images. Lines: clipped linear formulas. Dashed: baseline settings.", fontsize=10)
    fig.text(.34, .265, "Marked pixel:  B₄(h) = clip(h/2 − 9);   A₁₂(w) = clip(w/2 − 1).", fontsize=12)
    fig.text(.34, .225, "At the baseline:  0.875 × 0.6 = 0.525 pixel coverage.", fontsize=12, weight="bold")
    fig.text(.045, .145, "For this pixel, while 18 ≤ h ≤ 20 and 2 ≤ w ≤ 4:", fontsize=12)
    fig.text(.045, .10, "coefficient = 0.525 + 0.3·Δh + 0.4375·Δw + 0.25·Δh·Δw", fontsize=16, weight="bold")
    fig.text(.045, .055, f"Checked on 1,024 new joint height/width settings: maximum pixel error {test_pixel_error:.2g} (float32 precision).", fontsize=11)
    fig.text(.045, .018, "clip(z) clamps to [0,1]. Δh = h − 19.75; Δw = w − 3.2. Rows/columns start at 0. Fixed center and zero tilt throughout.", fontsize=10, color="#555555")
    fig.savefig(OUT / "pixel_coefficient_product.png", dpi=160)
    fig.savefig(OUT / "pixel_coefficient_product.svg")
    plt.close(fig)

    note = r"""# A compact mathematical relationship for height and width

This describes our **centered (14,14), untilted** generated-one slice. It does not claim to cover moving centers or nonzero lean.
The inputs are `h = 19.75 + Δh` and `w = 3.2 + Δw`. Pixels use zero-based row `r`, column `c`, and flattened index `i = 28r + c`.

## The pattern found in the images

Every image is an outer product of a vertical profile and a horizontal profile:

\[
I_{r,c}(h,w)=B_r(h)A_c(w).
\]

Profiles extracted from the images themselves reconstruct the sampled images with maximum pixel error PROFILE_ERROR.
At a fixed width, the same column profile repeats down the stroke, scaled by the row's factor. At a fixed height, the same row profile repeats across the stroke, scaled by the column's factor.
This is matrix rank one for each nonempty image, not a claim that the entire 784D family is one linear direction.

## A formula for any pixel coefficient

Define a clipped ramp and each pixel interval's distance from the center:

\[
C(z)=\min(1,\max(0,z)),\qquad d_k=|k+\tfrac12-14|-\tfrac12.
\]

Then the coefficient of basis vector \(E_i\), where \(i=28r+c\), is

\[
\boxed{a_i(\Delta h,\Delta w)=
C\!\left(\frac{19.75+\Delta h}{2}-d_r\right)
C\!\left(\frac{3.2+\Delta w}{2}-d_c\right).}
\]

The full image is \(I=\sum_{i=0}^{783}a_i E_i\).
This equation uses pixel position and the two inputs directly. It contains no renderer call, knob inversion, or lookup of a stored joint image.

Each factor is zero until an edge reaches that row/column, grows with slope 1/2 while coverage is partial, then stays at one.
Mirror pixels have the same factors: rows r and 27−r match, and columns c and 27−c match.

## One pixel, worked out

For **row 4, column 12, pixel i = 124**:

\[
a_{124}(h,w)=C(h/2-9)C(w/2-1).
\]

At our baseline this is \(0.875\times0.6=0.525\).
Within \(18\le h\le20\) and \(2\le w\le4\), the clipping is inactive for this pixel, so:

\[
\boxed{a_{124}=0.525+0.3\Delta h+0.4375\Delta w+0.25\Delta h\Delta w.}
\]

The signed change from the original coefficient is the same expression without the constant 0.525.
The interaction term says that a wider pixel footprint changes how much ink a height change adds or removes.
Outside those intervals, use the clipped formula; continuing this polynomial across an edge transition gives incorrect coefficients.
More generally, in a region between coverage transitions, every pixel has the form
\(a_i=\alpha_i+\beta_i\Delta h+\gamma_i\Delta w+\eta_i\Delta h\Delta w\).
The coefficients change when a row or column enters/leaves partial coverage.

## Why the distance curves have matching shapes

At any matched height:

\[
\|I(h,w)-I(h,w_0)\|_2=\|B(h)\|_2\,\|A(w)-A(w_0)\|_2.
\]

Thus each width's curve is the SAME height function multiplied by a width-dependent constant.
For cumulative height changes \(D_w(h)=I(h,w)-I(h_0,w)\):

\[
\|D_w(h)-D_{w_0}(h)\|_2
=\|B(h)-B(h_0)\|_2\,\|A(w)-A(w_0)\|_2.
\]

That explains the earlier collapse at the base height and the shared curve shapes. The vertical change pattern is shared; width controls its horizontal footprint.

## Check and scope

The formula was checked against all 961 existing images and **1,024 new jointly varying height/width settings**, drawn continuously off the original grid in width 1.7–4.7 px and height 18.25–21.25 px.
Maximum held-out pixel error: TEST_PIXEL_ERROR. Maximum held-out 784D image error: TEST_IMAGE_ERROR.
The renderer supplied ground truth only; predictions used the displayed mathematical formula.
The agreement is within float32 storage precision.

The relationship follows from the separable coverage of this axis-aligned stroke. It is a mathematical description of this slice, not an independently learned model of all five knobs.
For nonzero tilt, the horizontal profile generally depends on the row, so this particular outer-product relationship no longer holds.

Figure: `pixel_coefficient_product.png` (SVG is the same figure).
Numerical results: `results.json`.
Reproduce from the repo root:

```sh
MPLCONFIGDIR=/tmp/generated-one-mpl .venv/bin/python experiments/analyze_height_width_pixel_relationship_20261008.py
```
"""
    note = note.replace("PROFILE_ERROR", f"{profile_error:.3g}")
    note = note.replace("TEST_PIXEL_ERROR", f"{test_pixel_error:.3g}")
    note = note.replace("TEST_IMAGE_ERROR", f"{test_image_error:.3g}")
    (OUT / "README.md").write_text(note)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
