# Edge alignment of the five direction families

Run `.venv/bin/python analyze_generated_one_edge_alignment.py` from the repository root.

## What the earlier experiment actually learned

The large SVD libraries were fixed linear pixel patterns, with coefficients obtained by projecting an already-observed direction. They were not newly discovered piecewise motion functions. Existing analytic renderer derivatives supplied the physical knob directions. Two pixelwise nonlinear links between families were established: width is half the absolute horizontal-center direction, and height is half the absolute vertical-center direction after removing tan(lean) times horizontal-center direction. These do not reduce the five-dimensional local tangent rank: absolute value is nonlinear, not a linear dependency between direction vectors.

## Alignment experiment

384 training images, 128 validation images for a separate state-only predictor, 256 original random tests, 243 original near-edge tests, 512 fresh random tests and 1024 fresh near-edge tests. The seed is 20261018. All five knobs vary. Images use float64 coverage before float32 storage. Alignment settings are recovered from source pixels by the pre-existing `invert_grey.invert`, using the known straight-stroke geometry.

Raw pixels store zero alignment scalars. Centerline registration stores three: horizontal center, vertical center and lean. It applies reversible integer shifts to each pixel row and a reversible integer vertical shift. Boundary registration stores five: the two centers, height, width and lean. It splits the raster at the centerline and vertical center into four corner patches. For each half-row it aligns the corresponding side edge by an integer column shift; rows are measured from the corresponding cap. Each patch has 14 rows and 8 columns, with halos and zero padding. Values are neither interpolated nor scaled. The reverse mapping returns every original nonzero coefficient exactly before reduction.

For one direction, the code length is five geometry numbers plus k learned basis coefficients, with the physical knob label known. At an exact event, the requested one-sided derivative also supplies one discrete side bit per known knob. For all five directions together at a smooth state, the geometry is shared: five geometry numbers plus the sum of the five basis sizes = 399 continuous numbers at the fitted ranks. The decoder also stores a fixed set of shared basis vectors. Floor/ceil decisions, row shifts and quadrant masks are deterministic functions of the five geometry numbers; no per-row displacement list, unknown cluster label or uncounted phase list is supplied.

## Sharing rules across knob labels

The earlier full raw height span lies within the width span, and the earlier full raw horizontal and lean spans coincide numerically. Unit basis-vector projection residuals are around 1e-14. These are spans pooled across different source images; they do not equate the knobs at a single image.

The two nonlinear identities also let us store only the aligned horizontal, height and lean fields. Width is half the absolute horizontal field; vertical is tan(lean) times horizontal plus the signed cap field obtained from height. This uses 208 total scalars including geometry on smooth samples, versus 399 for five separately encoded fields. The identities are supplied mathematical rules, not fitted state-to-coefficient functions. Results are in `shared_three_field_decoder`.

## Limits that matter

The SVD encoder is supplied the actual vector. This is compression, not a state-only motion predictor. We separately fit an RBF predictor from five image-derived settings to the aligned maps, selecting bandwidth and regularization on validation images only. Its held-out errors are reported separately and remain far above numerical precision.

The five geometry numbers are the existing physical coordinates. Together with the known renderer they already specify the image and its derivatives. We do not claim to have discovered a novel five-coordinate description. This experiment measures how much coordinate registration simplifies a generic shared pixel-pattern library.

Integer alignment leaves fractional pixel positions, the sixteen subrow crossings, cap placement within a row and changing angle-dependent speeds. A reduced rank cannot by itself establish which remaining effect causes each pattern. The result is specific to this tested registration, not a lower bound on every possible nonlinear alignment.

## Joint directions and finite moves

At smooth fresh states we combine all five reconstructed directions with random simultaneous knob increments and compare to the same linear combination of the true Jacobian columns. This checks instantaneous joint moves. A finite change is different: true first-order predictions themselves deviate from rendered endpoints as steps cross events or follow curvature. Both true and reconstructed first-order errors are recorded across normalized step sizes 0.00001 through 0.1. No near-exact finite-step decoder is claimed.

## Events and precision

Five exact boundary/intersection states are tested with both one-sided physical directions. Limits are obtained by extrapolating analytic derivatives from that side and checked against one-sided rendered moves. A single source-side registration chart fails for three of 50 vectors, with worst relative error 1.0. Choosing the chart on the requested side of each event reduces the worst error to 2.073e-15, without adding learned patterns. This uses one additional side-selection bit for each requested one-sided direction; the knob identity is already known. Grid-rounding ties use tolerance 1e-10. This repair was motivated by the first five event states; a separate event audit is reported in `additional_event_audit`. These side derivatives are not averaged together or treated as a unique smooth Jacobian. Some sides leave the default range box; rendering allows these infinitesimal probes.

Float32 source images are tested separately for their effect on alignment decisions while keeping the target direction in float64. This isolates geometry/branch selection sensitivity, not the error of taking finite differences of rounded images.

## Artifacts

- `index.html`: illustrated report and interactive source/direction/registered/reconstructed comparison.
- `01_registration.png`, `02_basis_counts.png`, `03_reconstruction.png`, `04_event_charts.png`: explanatory figures.
- `results.json`: all measured summaries, predictor errors, event checks and finite-step checks.
- `curves.csv`: raw and registered reconstruction curves.
- `arrays.npz`: source images, directions, inferred geometry, aligned maps and learned bases.
