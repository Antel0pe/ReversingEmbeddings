# A compact mathematical relationship for height and width

This describes our **centered (14,14), untilted** generated-one slice. It does not claim to cover moving centers or nonzero lean.
The inputs are `h = 19.75 + Δh` and `w = 3.2 + Δw`. Pixels use zero-based row `r`, column `c`, and flattened index `i = 28r + c`.

## The pattern found in the images

Every image is an outer product of a vertical profile and a horizontal profile:

\[
I_{r,c}(h,w)=B_r(h)A_c(w).
\]

Profiles extracted from the images themselves reconstruct the sampled images with maximum pixel error 6.02e-08.
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
Maximum held-out pixel error: 2.98e-08. Maximum held-out 784D image error: 1.94e-07.
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
