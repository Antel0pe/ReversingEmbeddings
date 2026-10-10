"""Fit a width movement equation from labeled images on one fixed slice."""

from pathlib import Path
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grey_ones import render  # Validation ground truth only; not used by the fitted rule.

OUT = ROOT / "figures/generated_one_width_movement"


def fit_rule(widths, images):
    secants = np.diff(images,axis=0)/np.diff(widths)[:,None]
    jumps = np.linalg.norm(np.diff(secants,axis=0),axis=1)
    knots = widths[1:-1][jumps > 1e-4]
    reference = float(widths[len(widths)//2])
    design = np.column_stack([np.ones(len(widths)), widths-reference,
                              *[np.maximum(0,widths-k) for k in knots]])
    coefficients = np.linalg.lstsq(design,images,rcond=None)[0]
    # Calibrate an observable image coordinate from width labels, not geometry.
    ink_fit = np.linalg.lstsq(np.column_stack([np.ones(len(widths)),widths]),
                             images.sum(axis=1),rcond=None)[0]
    return {"knots":knots,"coefficients":coefficients,"ink_fit":ink_fit,
            "reference_width":np.array(reference)}


def pixel_changes(image, width_change, model):
    """Input image and requested width change; output 784 signed pixel changes."""
    image = np.asarray(image,dtype=np.float64).reshape(784)
    intercept, scale = model["ink_fit"]
    width = (image.sum()-intercept)/scale
    coefficients = model["coefficients"]
    result = width_change*coefficients[1].copy()
    for j,knot in enumerate(model["knots"]):
        hinge_change = max(0,width+width_change-knot)-max(0,width-knot)
        result += hinge_change*coefficients[j+2]
    return result


def main():
    source = ROOT / "figures/generated_one_width_tilt_relationship/samples.npz"
    with np.load(source) as saved:
        widths = saved["widths"]
        zero_tilt = int(np.argmin(abs(saved["tilt_degrees"])))
        images = saved["points"][:,zero_tilt].reshape(len(widths),784).astype(float)
    model = fit_rule(widths,images)
    assert np.allclose(model["knots"],[2,4])
    rng = np.random.default_rng(20261009)
    initial_widths = rng.uniform(widths.min(),widths.max(),1024)
    final_widths = rng.uniform(widths.min(),widths.max(),1024)
    settings = np.tile([14,14,19.75,3.2,0.],(1024,1))
    settings[:,3] = initial_widths
    initial = render(settings).reshape(-1,784).astype(float)
    settings[:,3] = final_widths
    truth = render(settings).reshape(-1,784).astype(float)
    changes = np.array([pixel_changes(image,float(w1-w0),model)
                        for image,w0,w1 in zip(initial,initial_widths,final_widths)])
    predictions = initial+changes
    residual = predictions-truth
    max_pixel = float(abs(residual).max())
    max_image = float(np.linalg.norm(residual,axis=1).max())
    assert max_pixel < 1e-6
    # Repeated movements must update state, including moves across both knots.
    loop_image = images[15].copy()
    for amount in [1.2,-2.6,1.4]:
        loop_image += pixel_changes(loop_image,amount,model)
    loop_error = float(abs(loop_image-images[15]).max())
    assert loop_error < 1e-6

    controls = {}
    for label,settings in [
        ("same slice",[14,14,19.75,3.2,0.]),
        ("tilt changed to 10 degrees",[14,14,19.75,3.2,10.]),
        ("height changed to 20.5",[14,14,20.5,3.2,0.]),
        ("horizontal center changed to 14.5",[14.5,14,19.75,3.2,0.]),
    ]:
        before = render(settings).reshape(784).astype(float)
        after_settings = settings.copy(); after_settings[3] += .1
        after = render(after_settings).reshape(784).astype(float)
        difference = before+pixel_changes(before,.1,model)-after
        controls[label] = {"maximum_pixel_error":float(abs(difference).max()),
                           "image_l2_error":float(np.linalg.norm(difference))}
    directions = [model["coefficients"][1]+model["coefficients"][2:2+j].sum(axis=0)
                  for j in range(3)]
    active_columns = [np.flatnonzero(abs(v.reshape(28,28)).max(axis=0)>1e-5).tolist()
                      for v in directions]

    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT / "model.npz",**model)
    results = {
        "training": "31 labeled width images, 1.7-4.7px every 0.1; other knobs fixed",
        "prediction_inputs": "Current image and width change, plus learned fixed model parameters",
        "knots_detected_from_image_secants":model["knots"].tolist(),
        "observed_total_ink_vs_width_intercept_slope":model["ink_fit"].tolist(),
        "active_columns_in_three_segments":active_columns,
        "held_out_moves":1024,"held_out_max_pixel_error":max_pixel,
        "held_out_max_image_l2_error":max_image,
        "returned_loop_max_pixel_error":loop_error,
        "frozen_model_controls":controls,
        "scope":"One fixed width curve, center (14,14), height19.75, tilt0. Not a field over all other settings.",
        "understanding_check":{
            "object":"Source image followed by three learned width direction vectors displayed as +0.1px changes",
            "axes_colors":"Pixel rows and columns; common signed coverage scale; red gains ink and blue loses ink",
            "start":"Root image first; active width intervals labeled on each movement panel",
            "evidence":"Active columns move outward at learned switches; held-out coefficient error stated",
            "scope":"Single fixed slice; phase crossings handled; frozen rule does not generalize to other tilts/positions",
        },
    }
    (OUT / "results.json").write_text(json.dumps(results,indent=2)+"\n")

    fig = plt.figure(figsize=(14,7.5),facecolor="white")
    fig.text(.045,.95,"One width curve has three simple pixel-change directions",fontsize=21,weight="bold")
    fig.text(.045,.90,"Learned from 31 width-labeled images. Center, height, and tilt stay fixed; prediction uses the current image and requested width change.",fontsize=10.5)
    panels = [(images[15].reshape(28,28),"Start: normal root, width 3.2",None)]
    for vector,title in zip(directions,["Width 1.7–2 px","Width 2–4 px","Width 4–4.7 px"]):
        panels.append((.1*vector.reshape(28,28),title,"change"))
    for j,(image,title,kind) in enumerate(panels):
        ax = fig.add_axes([.055+.235*j,.40,.185,.35])
        if kind:
            im = ax.imshow(image,cmap="RdBu_r",vmin=-.05,vmax=.05,interpolation="nearest")
            caption = "Rate × 0.1 px (no switch crossed)\nColumns " + ", ".join(map(str,active_columns[j-1])) + " gain ink"
        else:
            ax.imshow(image,cmap="gray_r",vmin=0,vmax=1,interpolation="nearest")
            caption = "28 × 28 coverage values\nWhite = 0 ink; black = 1 ink"
        ax.set_title(title,fontsize=11,pad=13)
        ax.set_xticks([0,7,14,21,27]); ax.set_yticks([0,7,14,21,27]); ax.tick_params(labelsize=8)
        ax.set_xlabel("Pixel column",fontsize=9)
        if j==0: ax.set_ylabel("Pixel row",fontsize=9)
        ax.text(.5,-.25,caption,ha="center",va="top",transform=ax.transAxes,fontsize=9.5,linespacing=1.5)
    cax = fig.add_axes([.35,.235,.55,.018])
    cb = fig.colorbar(im,cax=cax,orientation="horizontal",ticks=[-.05,0,.05])
    cb.set_label("Pixel change: blue loses ink; red gains ink. Common scale for all three movement panels.",fontsize=9)
    fig.text(.045,.16,"Within each interval: ΔIᵢ = Δwidth × vᵢ. At width 2 or 4, switch directions and continue from the updated image.",fontsize=12)
    fig.text(.045,.10,f"1,024 unseen width moves, including interval crossings: maximum pixel error {max_pixel:.2g}.",fontsize=12)
    fig.text(.045,.055,"Active interior pixels gain 0.05 for a +0.1 px step; top/bottom partial-row pixels gain 0.04375. Other pixels stay unchanged.",fontsize=10)
    fig.text(.045,.018,"Scope: center (14,14), height 19.75, tilt 0; width 1.7–4.7 px. This fitted rule is not valid for arbitrary images or other knob settings.",fontsize=9.5,color="#555555")
    fig.savefig(OUT / "width_movement_equation.png",dpi=160)
    fig.savefig(OUT / "width_movement_equation.svg")
    plt.close(fig)

    note = r"""# A width movement equation learned from image differences

This first test covers one width curve: center `(14,14)`, height 19.75, tilt 0 remain fixed.
Width ranges from 1.7 to 4.7 px. These fixed settings describe the source data; they are not inputs to the learned prediction function.

![Learned width movement directions](width_movement_equation.png)

## What was learned

I used 31 width-labeled images sampled every 0.1 px. Consecutive pixel differences divided by the width step revealed three constant movement vectors, with switches at widths **2 and 4**.
The switches were detected where the observed secant directions changed, not supplied from the renderer's geometry.
A piecewise-linear model then fitted all 784 coefficient curves from those image values and width labels.

The prediction function uses **current image I and requested width change δ**, together with the learned fixed model parameters.
It does not call the renderer, infer centers/tilts/edges, or construct a parallelogram.

## The per-pixel direction equation

Let pixel `i = 28r + c`, with rows and columns numbered from zero. The learned common row factors are

\[
b_r=\begin{cases}
0.875,&r=4\text{ or }23,\\
1,&5\le r\le22,\\
0,&\text{otherwise}.
\end{cases}
\]

The three learned directions, in coverage change per pixel of width, are

\[
v_{0,i}=\tfrac12 b_r\,\mathbf1_{c\in\{13,14\}},\quad
v_{1,i}=\tfrac12 b_r\,\mathbf1_{c\in\{12,15\}},\quad
v_{2,i}=\tfrac12 b_r\,\mathbf1_{c\in\{11,16\}}.
\]

Here the indicator is 1 for a listed column and 0 otherwise. These support patterns and values were read from the measured pixel differences.
The local width direction is

\[
V_i(I)=\begin{cases}
v_{0,i},&1.7\le\hat w(I)<2,\\
v_{1,i},&2<\hat w(I)<4,\\
v_{2,i},&4<\hat w(I)\le4.7.
\end{cases}
\]

At the two switching widths, the left and right derivatives differ. The finite-step equation below handles either direction and avoids choosing an ambiguous derivative.

## Reading the current curve coordinate from the image

The width labels also revealed a linear relation between total ink and width:

\[
\sum_i I_i\approx a+m w,\qquad a\approx0,\quad m\approx19.75.
\]

Both a and m were fitted from the image data. Therefore

\[
\hat w(I)=\frac{\sum_i I_i-a}{m}.
\]

This is a calibrated observable coordinate on this one curve. The slope numerically equals its fixed height, but no height value is supplied to prediction.
This relation is not a width estimator for arbitrary images: changing height changes the calibration.
If the current width is already known, use it directly instead of estimating it from I.

## One finite-step equation, including crossings

Let \([z]_+=\max(0,z)\), \(w=\hat w(I)\), and let δ be the requested width change. Define

\[
H_t(w,\delta)=[w+\delta-t]_+-[w-t]_+.
\]

Then every signed pixel change is

\[
\boxed{\Delta I_i=\delta v_{0,i}
+H_2(w,\delta)(v_{1,i}-v_{0,i})
+H_4(w,\delta)(v_{2,i}-v_{1,i}).}
\]

The output image is \(I'_i=I_i+\Delta I_i\).
The hinge terms allocate the correct portions of a movement to the appropriate direction. They handle widening, narrowing, and crossings without microscopic numerical steps.
The compact v vectors above are the rounded learned pattern; `model.npz` retains the actual least-squares fitted coefficients used for reported validation.

For example, pixel 152 is row 5, column 12. At width 3.2, a +0.1 move gives ΔI=+0.05.
At width 1.8, the same requested move gives that pixel ΔI=0; different columns are active there.
For a move from 3.9 to 4.2, that pixel gains 0.05 on the way to 4.0 and then stops changing; the next columns gain the rest.
Thus the same knob has different pixel effects depending on the current state.

## Validation

I tested 1,024 fresh start/end width pairs drawn continuously from 1.7–4.7 px; both endpoints were off the training grid.
The endpoints include widening, narrowing, and movement across either or both switches.
`grey_ones.render` was used only to provide validation ground truth, not in the prediction rule or the fit from saved images.

| Check | Maximum error |
| --- | ---: |
| Held-out pixel coefficient | MAX_PIXEL |
| Held-out 784D image L2 | MAX_IMAGE |
| Return loop 3.2 → 4.4 → 1.8 → 3.2, per pixel | LOOP_ERROR |

This agrees with the original renderer to float32 precision on this slice.

## What remains unsolved

This is a field along **one curve**, not a width field over the whole five-knob family.
Passing a tilted or shifted image to the same frozen model is not valid. To test this limitation, I reused the rule for a +0.1 width move:

| Initial context | Image L2 prediction error |
| --- | ---: |
| Same upright slice | CONTROL_SAME |
| Tilt 10° | CONTROL_TILT |
| Height 20.5 | CONTROL_HEIGHT |
| Horizontal center 14.5 | CONTROL_CENTER |

The more general challenge is to learn how the width direction changes across those other states; the successful one-curve model does not solve that problem by itself.
For now, we have an explicit, data-fitted per-pixel movement equation and a held-out reconstruction check for one knob with the others fixed.

## Files and reproduction

- `model.npz`: fitted pixel coefficient functions, knots, and image-coordinate calibration.
- `results.json`: numerical validation and scope.
- `width_movement_equation.png`: explanatory figure; SVG is the same figure.

From the repo root:

```sh
MPLCONFIGDIR=/tmp/generated-one-mpl .venv/bin/python experiments/learn_width_pixel_movement_20261009.py
```

The function `pixel_changes(image, width_change, model)` in that script performs prediction. Add its output to the input image to obtain the new image.
"""
    replacement = {"MAX_PIXEL":f"{max_pixel:.3g}","MAX_IMAGE":f"{max_image:.3g}","LOOP_ERROR":f"{loop_error:.3g}",
                   "CONTROL_SAME":f"{controls['same slice']['image_l2_error']:.3g}",
                   "CONTROL_TILT":f"{controls['tilt changed to 10 degrees']['image_l2_error']:.3g}",
                   "CONTROL_HEIGHT":f"{controls['height changed to 20.5']['image_l2_error']:.3g}",
                   "CONTROL_CENTER":f"{controls['horizontal center changed to 14.5']['image_l2_error']:.3g}"}
    for key,value in replacement.items(): note = note.replace(key,value)
    (OUT / "README.md").write_text(note)
    print(json.dumps(results,indent=2))


if __name__ == "__main__":
    main()
