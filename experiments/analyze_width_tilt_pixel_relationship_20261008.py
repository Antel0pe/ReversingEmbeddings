"""Study mathematical pixel relationships in the fixed-center width/tilt slice."""

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

OUT = ROOT / "figures/generated_one_width_tilt_relationship"
BASE = np.array([14, 14, 19.75, 3.2, 0.], dtype=np.float64)


def positive_linear_area(z0, z1, length):
    """Exact integral of the positive part of a linear function on an interval."""
    both_positive = (z0 >= 0) & (z1 >= 0)
    crosses = (z0 > 0) ^ (z1 > 0)
    change = np.abs(z1-z0)
    triangle = np.divide(np.maximum(z0,z1)**2, 2*change,
                         out=np.zeros_like(z0), where=crosses & (change > 0))
    return length*np.where(both_positive, (z0+z1)/2, np.where(crosses, triangle, 0))


def predict(widths, angles):
    """Continuous pixel area from analytic hinge integrals; no sampled subrows."""
    w = np.asarray(widths)[:, None, None]
    slope = np.tan(np.deg2rad(angles))[:, None, None]
    rows = np.arange(28)[None, :, None]
    cols = np.arange(28)[None, None, :]
    lo = np.maximum(rows, 4.125)
    hi = np.minimum(rows+1, 23.875)
    length = np.maximum(0, hi-lo)
    def clipped_linear_area(a):
        z0 = a-slope*(lo-14)
        z1 = a-slope*(hi-14)
        return (positive_linear_area(z0,z1,length)
                - positive_linear_area(z0-1,z1-1,length))
    return clipped_linear_area(14+w/2-cols)-clipped_linear_area(14-w/2-cols)


def ground_truth(widths, angles):
    parameters = np.tile(BASE, (len(widths), 1))
    parameters[:, 3], parameters[:, 4] = widths, angles
    return render(parameters).astype(np.float64)


def main():
    widths = 3.2 + np.arange(-15, 16)*.1
    angles = np.arange(-20, 21)*.5
    theta, ww = np.meshgrid(angles, widths)
    samples = ground_truth(ww.ravel(), theta.ravel()).reshape(31,41,28,28)
    predicted = predict(ww.ravel(), theta.ravel()).reshape(samples.shape)
    grid_error = float(np.abs(samples-predicted).max())
    assert np.isfinite(predicted).all()
    assert predicted.min() > -1e-12 and predicted.max() < 1+1e-12
    assert np.array_equal(samples, samples[:, ::-1, :, ::-1])
    assert np.array_equal(samples, samples[:, :, ::-1, ::-1])

    # Fit one pixel in a single coverage region; retain the interaction as a diagnostic.
    local_w = np.linspace(2.8,3.6,17)
    local_theta = np.linspace(-1,1,17)
    tt, local_ww = np.meshgrid(local_theta,local_w)
    slope = np.tan(np.deg2rad(tt.ravel()))
    dw = local_ww.ravel()-3.2
    design = np.column_stack([np.ones(dw.size),dw,slope,dw*slope])
    local_samples = ground_truth(local_ww.ravel(),tt.ravel())[:,5,15]
    fit = np.linalg.lstsq(design,local_samples,rcond=None)[0]

    rng = np.random.default_rng(20261008)
    test_w = rng.uniform(1.7,4.7,1024)
    test_theta = rng.uniform(-10,10,1024)
    truth = ground_truth(test_w,test_theta)
    residual = predict(test_w,test_theta)-truth
    max_pixel = float(np.abs(residual).max())
    max_image = float(np.linalg.norm(residual.reshape(-1,784),axis=-1).max())
    rmse = float(np.sqrt(np.mean(residual**2)))
    assert max_pixel < 1e-4
    test_local_w = rng.uniform(2.8,3.6,1024)
    test_local_theta = rng.uniform(-1,1,1024)
    local_truth = ground_truth(test_local_w,test_local_theta)[:,5,15]
    local_prediction = .6+.5*(test_local_w-3.2)+8.5*np.tan(np.deg2rad(test_local_theta))
    local_error = float(np.abs(local_prediction-local_truth).max())
    assert local_error < 1e-6
    rank_errors = {}
    for angle in [0,5,10]:
        image = samples[15,int(round((angle+10)/.5))]
        singular = np.linalg.svd(image,compute_uv=False)
        rank_errors[str(angle)] = float(np.sqrt(np.sum(singular[1:]**2)/np.sum(singular**2)))

    OUT.mkdir(parents=True,exist_ok=True)
    results = {
        "baseline_knobs": BASE.tolist(),
        "width_range_px": [1.7,4.7], "width_step_px": .1, "width_samples": 31,
        "tilt_range_degrees": [-10,10], "tilt_step_degrees": .5, "tilt_samples": 41,
        "grid_max_pixel_error_continuous_area": grid_error,
        "held_out_joint_samples": 1024,
        "continuous_area_held_out_max_pixel_error": max_pixel,
        "continuous_area_held_out_max_784D_error": max_image,
        "continuous_area_held_out_pixel_rmse": rmse,
        "pixel_155_local_fit_constant_dw_slope_dw_times_slope": fit.tolist(),
        "local_held_out_samples": 1024, "pixel_155_local_held_out_max_error": local_error,
        "best_rank_one_relative_image_errors_at_width_3p2_by_tilt_degrees": rank_errors,
        "mirror_symmetry_exact_on_grid": True,
        "scope": "Center (14,14), height 19.75 fixed; width horizontal; slope=tan(tilt)",
        "precision_note": "Continuous area is exact for the ideal stroke. Renderer uses 16 midpoint subrows; agreement is approximate, not float32-exact globally.",
        "understanding_check": {
            "object": "Untilting/tilting and widening the same stroke; one marked pixel's joint coefficient surface",
            "axes_colors": "Heatmap axes are width and tilt; color is absolute pixel coverage 0-1",
            "baseline": "Normal root first; all source images labeled; baseline point marked on heatmap",
            "evidence": "Pixel heatmap shows coupled inputs; local equation and held-out numerical errors shown",
            "scope": "Center and height fixed; local affine region explicitly bounded; global continuous model precision stated",
        },
    }
    (OUT / "results.json").write_text(json.dumps(results,indent=2)+"\n")
    np.savez_compressed(OUT / "samples.npz", points=samples.astype(np.float32),
                        widths=widths,tilt_degrees=angles,baseline=BASE)

    fig = plt.figure(figsize=(15,8),facecolor="white")
    fig.text(.04,.95,"Width sets the stroke's span; tilt shifts that span differently in each row",fontsize=18,weight="bold")
    fig.text(.04,.905,"Fixed center (14,14), height 19.75 px. Positive tilt moves the top rightward. Red square marks pixel 155 (row 5, column 15).",fontsize=11)
    cases = [(3.2,0),(3.2,10),(4.7,10)]
    for k,(width,angle) in enumerate(cases):
        ax = fig.add_axes([.045+.195*k,.435,.16,.34])
        image = ground_truth([width],[angle])[0]
        ax.imshow(image,cmap="gray_r",vmin=0,vmax=1,interpolation="nearest")
        ax.add_patch(Rectangle((14.5,4.5),1,1,fill=False,edgecolor="#cf432a",linewidth=2))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Normal root" if k==0 else ("Tilt changes" if k==1 else "Width also changes"),fontsize=12,pad=13)
        ax.text(.5,-.11,f"w = {width:g} px; tilt = {angle:g}°\npixel 155 = {image[5,15]:.3f}",ha="center",va="top",transform=ax.transAxes,fontsize=10,linespacing=1.5)
    fig.text(.045,.315,"All images: white = 0 coverage; black = 1 coverage.\nThe same marked pixel is shown in all three images.",fontsize=10,linespacing=1.5)
    ax = fig.add_axes([.69,.425,.235,.36])
    mesh = ax.pcolormesh(widths,angles,samples[:,:,5,15].T,cmap="viridis",vmin=0,vmax=1,shading="nearest")
    ax.scatter([3.2],[0],s=65,facecolor="white",edgecolor="black",zorder=3)
    ax.annotate("Root: 0.6",xy=(3.2,0),xytext=(8,-22),textcoords="offset points",fontsize=9,color="white")
    ax.set_title("Coefficient of pixel 155",fontsize=12,pad=13)
    ax.set_xlabel("Width (pixels)"); ax.set_ylabel("Tilt (degrees)")
    cax = fig.add_axes([.943,.425,.012,.36])
    fig.colorbar(mesh,cax=cax,label="Pixel coverage (0–1)")
    fig.text(.69,.32,"Color: measured coverage for each width/tilt pair.\n31 widths × 41 tilts; white dot marks the root.",fontsize=9.5,linespacing=1.5)
    fig.text(.045,.245,"Shared rule: horizontal shift at vertical position y = (14 − y)·tan(tilt).",fontsize=14,weight="bold")
    fig.text(.045,.185,"Worked pixel 155, while 2.8 ≤ width ≤ 3.6 px and −1° ≤ tilt ≤ 1°:",fontsize=12)
    fig.text(.045,.14,"coefficient = 0.6 + 0.5·Δwidth + 8.5·tan(tilt)",fontsize=18,weight="bold")
    fig.text(.045,.09,f"Local formula: maximum held-out pixel error {local_error:.2g}. Full continuous-area formula: {max_pixel:.2g} across 1,024 joint settings.",fontsize=11)
    fig.text(.045,.045,"Global differences reflect the renderer's 16-subrow approximation; the continuous-area equation describes the ideal pixel coverage.",fontsize=10)
    fig.text(.045,.015,"Δwidth = width − 3.2 px. Tilt is in degrees: slope = tan(π·tilt/180). Center and height stay fixed throughout.",fontsize=9.5,color="#555555")
    fig.savefig(OUT / "width_tilt_pixel_relationship.png",dpi=160)
    fig.savefig(OUT / "width_tilt_pixel_relationship.svg")
    plt.close(fig)

    note = r"""# Pixel coefficients as functions of width and tilt

The root is `(cx, cy, height, width, lean) = (14, 14, 19.75, 3.2, 0)`.
Center and height stay fixed. Width varies **1.7–4.7 px every 0.1 px** (31 widths).
Tilt varies **−10° to +10° every 0.5°** (41 tilts). Positive tilt moves the top rightward.
These are experimental settings; the width range extends slightly beyond the usual defaults.
Pixels use zero-based row `r`, column `c`, and number `i = 28r + c`. Coefficients mean fractional ink coverage, from 0 to 1.

![Width and tilt pixel relationship](width_tilt_pixel_relationship.png)

## Main result: one shared profile with a row-dependent shift

For height and width with zero tilt, a row factor and a column factor multiply.
Tilt adds a dependence of horizontal position on vertical position.
Use slope \(s=\tan(\pi\theta/180)\) for tilt \(\theta\) in degrees.
At vertical position \(y\), the stroke's horizontal center is

\[
\boxed{m(y)=14+(14-y)s.}
\]

This is a shear: points near the center barely move, and points near the ends move further.
At \(y=5.5\), a slope change shifts the center by \(8.5s\); at \(y=22.5\), the shift is \(-8.5s\).
Width changes the horizontal span around that shifted center. The same horizontal coverage profile applies at every vertical position, translated by this shared shift rule.

## A local coefficient equation, checked from image values

For pixel **155**, row 5, column 15, while width is 2.8–3.6 px and tilt is −1° to +1°:

\[
\boxed{a_{155}(w,\theta)=0.6+0.5(w-3.2)+8.5\tan(\pi\theta/180).}
\]

The coefficient's signed change from the root is \(0.5\Delta w+8.5s\).
Pixel 155 sits on the right edge in this region. Width adds coverage with slope 1/2.
The tilt coefficient 8.5 is its row midpoint's distance from the vertical center, with the appropriate sign.
Its mirrored left-edge pixel at row 5, column 12 has the opposite tilt sign:

\[
a_{152}=0.6+0.5\Delta w-8.5s.
\]

At row 22, the right-edge tilt sign also reverses: \(a_{631}=0.6+0.5\Delta w-8.5s\).
Thus tilt gains ink on one side and loses it on the other, with the amount controlled by row position.
This relationship is linear in **slope s**, not exactly linear in angle in degrees.

I fit pixel 155 from 289 rendered width/tilt pairs using terms `[1, Δw, s, Δw·s]`.
The fitted coefficients were FIT_COEFFICIENTS. The expected values are `[0.6, 0.5, 8.5, 0]`.
The width/slope interaction coefficient is at numerical-noise scale in this particular coverage region.
On **1,024 fresh local joint settings**, the explicit equation's maximum pixel error was LOCAL_ERROR.
This affine equation stops applying when an edge leaves the pixel or the pixel saturates; do not extrapolate it across the entire sweep.

## General local rule by pixel position

The fixed stroke occupies vertical coordinates \(T=4.125\) through \(U=23.875\).
For row r, use \(\ell=\max(r,T)\), \(u=\min(r+1,U)\), visible row length \(v=\max(0,u-\ell)\), and midpoint \(\bar y=(\ell+u)/2\).
When the right edge stays inside column c over the whole visible row segment and the left edge stays to its left:

\[
a_{r,c}=v\,[14+w/2-c+(14-\bar y)s].
\]

When the left edge stays inside the column and the right edge stays to its right:

\[
a_{r,c}=v\,[c+1-14+w/2-(14-\bar y)s].
\]

Fully covered pixels have coefficient v; untouched pixels have coefficient zero.
The coefficient patterns are therefore determined by pixel position and which coverage case the pixel occupies.
Transitions occur when an edge crosses a pixel boundary. Rows near the ends cross boundaries sooner for the same tilt change.

## A full coefficient equation across transitions

The horizontal edges are

\[
L(y)=14-w/2-s(y-14),\qquad R(y)=14+w/2-s(y-14).
\]

Define the clipped ramp \(C(z)=\min(1,\max(0,z))\). Then the **ideal continuous-area** pixel coefficient is

\[
\boxed{a_{r,c}(w,s)=\int_\ell^u
\big[C(R(y)-c)-C(L(y)-c)\big]\,dy.}
\]

Rows with no visible interval have coefficient zero. The full image is \(\sum_{r,c}a_{r,c}E_{28r+c}\).
This combines one-dimensional edge responses using pixel position, width, and slope; it does not invert knobs or call a renderer to predict a pixel.

The integral also has a closed form. Let

\[
G(z)=\tfrac12\big([z]_+^2-[z-1]_+^2\big),\quad [z]_+=\max(0,z),
\]

\[
q_\ell=\ell-14,\quad q_u=u-14,\quad A=14+w/2-c,\quad B=14-w/2-c.
\]

For \(s\ne0\):

\[
\boxed{a_{r,c}=
\frac{G(A-sq_\ell)-G(A-sq_u)-G(B-sq_\ell)+G(B-sq_u)}{s}.}
\]

At \(s=0\), take the limit:

\[
a_{r,c}=v\,[C(A)-C(B)].
\]

The implementation integrates positive linear pieces as exact triangles/trapezoids, avoiding cancellation near zero slope. It does not sample subrows.
This is a mathematical area relationship, not an independently learned global model; the local regression is a separate image-data check.

## Verification and the renderer's precision

The continuous formula was checked on all **1,271 grid images** and **1,024 fresh jointly varying width/tilt settings**, with both inputs continuous and off the original grid.
Predictions used the analytic formula; `grey_ones.render` supplied ground truth only.

| Check | Result |
| --- | ---: |
| Maximum pixel error on the grid | GRID_ERROR |
| Maximum pixel error on fresh joint settings | MAX_PIXEL |
| Maximum 784D image error on fresh joint settings | MAX_IMAGE |
| Pixel RMSE on fresh joint settings | RMSE |
| Local pixel 155 maximum error | LOCAL_ERROR |

The global formula is exact for ideal continuous pixel areas. **It is not bit-for-bit exact for the existing renderer**, which uses 16 midpoint subrows per pixel row.
Within a wholly linear coverage case, midpoint averaging is exact up to float32 storage; across edge transitions, numerical quadrature explains the small global discrepancy.
This distinction matters: the height/width equation agreed at float32 precision everywhere because untilted coverage separated exactly.

## What differs from the height/width result

One row factor times one common column factor no longer reconstructs the tilted images: different rows have differently shifted profiles.
For width 3.2, the best rank-one approximation gives these relative image errors (residual Frobenius norm divided by image Frobenius norm):

| Tilt | Best rank-one relative error |
| --- | ---: |
| 0° | RANK_0 |
| 5° | RANK_5 |
| 10° | RANK_10 |

The replacement relationship is **a shared horizontal profile translated by a row-dependent shear**, followed by coverage integration.
Within one stable edge case, coefficients are affine in width and slope; a single global affine coefficient equation does not cover edge crossings.
Symmetry also survives: \(a_{r,c}(w,\theta)=a_{27-r,27-c}(w,\theta)\), and reversing tilt mirrors the image horizontally.
The horizontal tilt-reversal identity held exactly on this sampled float32 grid.

This is one analysis step for the fixed-center, fixed-height width/tilt slice. It does not establish coordinates for all five knobs.

Files: `width_tilt_pixel_relationship.png` (SVG is the same figure), `results.json`, and `samples.npz`.
Reproduce from the repository root:

```sh
MPLCONFIGDIR=/tmp/generated-one-mpl .venv/bin/python experiments/analyze_width_tilt_pixel_relationship_20261008.py
```
"""
    replacements = {
        "FIT_COEFFICIENTS": "`["+", ".join(f"{x:.9g}" for x in fit)+"]`",
        "LOCAL_ERROR": f"{local_error:.3g}", "GRID_ERROR": f"{grid_error:.3g}",
        "MAX_PIXEL": f"{max_pixel:.3g}", "MAX_IMAGE": f"{max_image:.3g}", "RMSE": f"{rmse:.3g}",
        "RANK_0": f"{rank_errors['0']:.3g}",
        "RANK_5": f"{100*rank_errors['5']:.2f}%", "RANK_10": f"{100*rank_errors['10']:.2f}%",
    }
    for key,value in replacements.items():
        note = note.replace(key,value)
    (OUT / "README.md").write_text(note)
    print(json.dumps(results,indent=2))


if __name__ == "__main__":
    main()
