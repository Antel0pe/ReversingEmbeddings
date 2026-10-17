# Data-only reconstruction experiment: fixed success contract

This protocol is written before fitting. Five controls are an allowed target,
not an inferred dimension claim. Exact recovery of the original controls is
not required. No previous strip encoder, inverse, decoder, derivatives,
parameter ranges, or rendering grammar may enter the learner.

## Allowed inputs and general assumptions

Condition A: finite raw pixel coverage arrays, known image array coordinates,
and the target of five understandable coordinates.
Condition B: the same arrays plus one five-number coordinate row per image.
The five supplied columns have no names or units. No interventions are given.
The learner uses only training observations and a separate validation set.

Allowed generic priors: nonnegative ink weights; spatial moments through
second order; numerical column standardization; a locally smooth function
approximated by polynomials through degree three. These priors are explicit
and not neutral, but do not state straightness, horizontal caps, constant
width, subrow sampling, clipping formulas, or physical meanings of columns.
Image geometry is used only to define ordinary spatial moments.

## One route, two information conditions

A: compute six generic measurements (ink mass, two centroids, two variances,
and covariance). Select five of these six using held-out validation pixel
reconstruction, using the same decoder throughout. Their meanings stay the
same as the original measurements. Do not call a projection or lossless
example lookup an explanatory mechanism.
B: supply the five coordinate values directly to the same decoder. Report
this as conditional decoding, not discovered coordinates. A data-only inverse
from six moments to these coordinates is checked separately.

The shared decoder is a local polynomial least-squares rule: select nearby
training observations in standardized coordinate distance, fit a low-degree
polynomial, and evaluate it. First select degree/neighborhood size on validation
only. Examine training counts 512,4096,32768 to determine data-resolution
behavior; this is not a broad model benchmark. A global polynomial at the
same selected degree is a compactness control. No trigonometric, strip,
edge-overlap, subrow or renderer-specific features are supplied.

## Success before fitting

A claimed numerical family explanation must meet BOTH max absolute pixel
error <=2e-6 and maximum Euclidean image error <=1e-5 on held-out float32
observations. Target values are unchanged from the earlier experiment.
No clipping or denoising is allowed to hide prediction errors.

Test unseen full joint states, a complete lattice including corners, near
faces, renderer cap/side transitions, and independent joint paths. The
oracle alone constructs the special boundaries. Distinguish isolated smooth
patches from the whole family. Repeat local-patch checks before/after float32
storage to separate interpolation error and quantization.

Any candidate five-coordinate map must be evaluated for numerical rank,
collisions, and whether coordinate changes and interpolated coordinates
correspond to family members. Bounds of five measured coordinates are not
assumed to define a valid rectangular control domain.

Explanation requires a short explicit coordinate interpretation and a compact
mechanism. A decoder retaining all training images is an interpolator even
if accurate; polynomial coefficient tables do not alone establish meaningful
generative laws. Report retained data and coefficient counts. We will not
call an empirical approximation, finite testing, or low variance residual an
exact mechanism. A failure must identify whether the coordinates, global
formula, model simplicity, domain, or numerical tolerance failed.

The evaluation driver can use the existing generator and settings ONLY after
models and hyperparameters are frozen. Discovery imports no renderer and
reads no ground-truth evaluation files. This is software information isolation;
the experiment designer already knows the family, so this is not a claim of
human-level blind experimental design.
