# Nine SVD directions around the root 1

Twitter attachment: `svd_nine_directions.png` (2000 × 1500 pixels).
`svd_nine_directions.svg` is the vector version; `preview.png` is an 800-pixel inspection copy.

The figure shows the actual nine saved SVD vectors, in decreasing singular-value order,
from `figures/root_one_local_svd/bases_and_samples.npz`. It does not substitute analytic
knob derivatives or manually chosen interaction maps for the fitted vectors.

The source matrix has 512 rows: each is a synthetic neighbor image minus the canonical
root image, flattened to 784 coverage values. Every neighbor is at Euclidean image
distance 0.001 from the root. All five controls vary, with position sampling explicitly
extended to allow left/up moves from the root. Coverage retains float64 precision.

Percentages are `100 * sigma_i**2 / sum(sigma_j**2)`. These are uncentered displacement
energy shares about the specified root. Mean-centered PCA explained variance would
require a different fit; the displayed percentages are not that statistic. Values are
specific to this controlled neighborhood and its sampling design.

The first five vectors closely match tilt, width, horizontal position, vertical
position, and height. Labels were assigned after fitting using known derivatives;
small mixtures remain. Their absolute derivative cosines range from 0.98836 to
0.999997. This does not establish uniquely identified physical controls.

After removing their tiny tangent components, vectors 6–9 lie in the four-dimensional
mixed-interaction normal span, with unit-vector residuals below 7e-9. They are mixtures
of interaction patterns, not one named knob pair per vector. The supporting analytic
diagnostic is `figures/generated_one_manifold_transport/`.

All nine displayed vectors have unit length and share one symmetric signed color
scale. Red indicates positive ink change, blue negative, and white zero. The first
five signs are chosen to align with increasing the corresponding physical control;
SVD signs are otherwise arbitrary. The interaction vectors are tiny in energy, even
though their displayed normalized maps are as large as the dominant vectors.

Reproduce from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python figures/twitter/svd_nine_directions/make_figure.py
```

The script checks saved singular values against directly projected training energy,
orthonormality, shell distances, knob associations, interaction-span agreement,
held-out reconstruction, text overlaps, and text bounds. `verification.json` records
the measurements and figure understanding check. The full-size and 800-pixel images
were visually inspected before delivery.

Alt text: An upright synthetic 28-by-28 stroke is the starting image. All 512 fitting
neighbors are at Euclidean pixel distance 0.001. Five red-and-blue SVD maps resemble
tilt, width, sideways movement, vertical movement, and height, capturing 66.65%,
18.48%, 11.55%, 2.05%, and 1.27% of squared pixel change. Four further maps capture
interaction effects with shares of 5.76e-8%, 2.84e-8%, 6.65e-9%, and 4.28e-9%.
All maps show unit vectors on the same color scale, so the tiny interaction patterns
are visible despite their very small energy.

The simplified attachment labels the percentages “Variation captured” to avoid
calling uncentered squared changes mean-centered variance. The main figure keeps
only the distance, root, group labels, nine vectors, percentages, and color legend.
