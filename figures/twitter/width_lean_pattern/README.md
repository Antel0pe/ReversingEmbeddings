# Width-dependent lean: two images for one post

Attach `01_pixels.png` and `02_pattern.png` to `post.txt`. These replace the earlier, subtler pixel example and the graph containing width normalization. Both files are 1800 by 1200 pixels.

## 1. An exaggerated illustration of which pixels change

Widths are 1.8 and 10.0 pixels, 5.56 times apart. Both synthetic strokes tilt from upright to 20 degrees, with center (14.5, 14.5) and height 19.75 px fixed. Width 10 is deliberately outside the default 1.8-to-4.6-pixel range, as stated on the image. The renderer accepts it, and no ink reaches the frame. The line graph retains the original widths 3.2 to 4.2 pixels.

Signed change maps show after-minus-before coverage, on the common scale -1 to +1: red loses ink and blue gains it. Source strokes use dark for zero and white for full coverage. The two full 784-pixel change vectors have an angle of 89.995 degrees. This compares their pixel responses, not the physical stroke angles; both physical angles are 20 degrees.

## 2. Distance patterns and local direction angles

All ten curves use width increases +0.1 through +1.0 pixels from normal width 3.2. Center and height remain fixed.

The distance panel uses 901 lean samples from -10 to 35 degrees. For each width, subtract its own upright image to obtain a lean-change vector. Then subtract the width-3.2 vector at matching lean and take the Euclidean length over all 784 pixels. Upright-subtracted mismatch is zero at lean zero by construction, even though the upright images themselves differ.

The angle panel uses 881 lean samples from -9.5 to 34.5 degrees. At each lean, measure the centered 0.1-degree change: image(lean + 0.05) minus image(lean - 0.05). Compare its unit direction with normal width at the same lean. Zero means aligned pixel directions; 90 degrees means perpendicular. For width +1, the largest sampled angle is 80.5668 degrees at lean -3.5 degrees. Near upright, the tested directions align. Small width offsets tend to produce smaller directional differences in this experiment, but angle profiles are not exactly proportional to width increases.

The distance graph describes accumulated displacement from upright; the angle graph describes a local move. Neither is a projection of the image paths. Repeated scalar shapes do not imply identical pixel directions or an exact representation of the full five-knob family.

## Sources and verification

The figures re-render with `grey_ones.render`. Original sources are `figures/generated_lean_width_comparison/README.md`, `width_sweep/profiles.csv`, `width_sweep/summary.csv`, and `width_sweep/pattern_analysis.json` in that experiment directory. Source peak angles and distance-fit results reproduce the saved measurements.

`verification.json` records the settings, numeric checks, overlap checks, and figure understanding check. `plot_data.npz` stores image changes and all scalar profiles. The old normalization analysis remains in the numeric data, but no normalized graph is shown. `preview.png` is a phone-size contact sheet, not a third attachment.

From the repository root:

```sh
.venv/bin/python figures/twitter/width_lean_pattern/make_figures.py
```

Use `--distance-only` if a single distance panel is preferred.

## Alt text

**Image 1.** A thin 1.8-pixel stroke and a deliberately exaggerated 10-pixel-wide stroke both tilt from upright to 20 degrees. Blue marks gained ink and red marks lost ink. The wide stroke's edge changes occur in different columns, visibly separated from the thin stroke's changes. The pixel-change vectors are nearly perpendicular. The exaggerated width is outside the default range.

**Image 2.** Ten colored curves show width increases from +0.1 to +1 pixel. Left: differences between lean-change vectors have similar shapes whose amplitudes grow with width. Right: the directions of small lean changes align near upright, but show repeated peaks elsewhere. Width +1 reaches an 80.6-degree difference from the normal-width response. Colors use the same width scale in both panels.
