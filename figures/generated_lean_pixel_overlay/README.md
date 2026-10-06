# Lean pixel changes across widths

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
