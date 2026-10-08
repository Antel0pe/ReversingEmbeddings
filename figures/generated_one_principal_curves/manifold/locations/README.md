# Error location and local bending

The fine fitted grid is unchanged. Full-cloud results use cached pixel-only projections for 16,384 uniform Sobol images, seed 740. No fit, refit, model selection or output correction beyond the explicitly labelled clipping is performed.

Image L2 is Euclidean pixel distance over all 784 ink-coverage values. RMS image error = sqrt(mean per-image squared error). Concentration percentages sum squared error, not image counts or visible magenta area.

## Where errors concentrate

| Group | Share of images | Share of clipped squared error | RMS image L2 |
| --- | ---: | ---: | ---: |
| Within 5% of any physical knob limit | 40.95% | 54.72% | 0.16655 |
| Away from all 5% physical limits | 59.05% | 45.28% | 0.12616 |
| At a fitted decoder-coordinate boundary | 4.97% | 11.42% | 0.21845 |

The worst 10% of images contribute 35.25% of clipped squared error, leaving 64.75% across the other 90%. All pairwise knob maps use ten equal physical-range bins per axis and average over the other three knobs. They do not display the entire five-dimensional geometry. Fitted-chart boundary contact is an association, not proof of a failed closest-point search.

The original 14,389 fitting images show a weaker elevation near limits: clipped RMS 0.09381 near versus 0.08545 away. That cloud includes boundary and lattice oversampling, so it is not the same distribution as the uniform diagnostic cloud.

## Local bending check

The same 1,024 images are drawn uniformly without replacement from cloud points at least 2.5% away from every physical knob limit, seed 1008. The cohort therefore excludes extreme boundary points. Five single-knob paths and five fixed random joint paths use unit directions in normalized setting space. Symmetric probe scales are 0.005 and 0.02, with all endpoints within the legal box.

Before-to-center and center-to-after pixel vectors define a turn angle. Half their second image difference is the midpoint chord displacement. A local tangent span comes from central finite differences at 0.001 range fraction and a rank-filtered SVD (relative singular cutoff 1e-6). Projecting the chord displacement off this span removes tangent acceleration. The normal bending proxy is twice the normal displacement magnitude divided by the squared mean endpoint-to-center image distance. Ten-path averages and maxima are proxies at a finite scale, not a global curvature tensor.

| Probe scale | Mean normal bending vs clipped error rank correlation | Highest-bending decile error share | Mean normal bending vs scalar overshoot rank correlation |
| ---: | ---: | ---: | ---: |
| 0.005 | 0.1407 | 11.60% | 0.0529 |
| 0.02 | 0.2360 | 11.25% | 0.1319 |

Scalar overshoot energy is sum squared distance of raw learned pixels to [0,1]. It differs from reconstruction error and from error reduction after clipping.

Results show elevated error near physical knob limits and only modest interior curvature associations. No causal share is attributed to finite grid resolution, node approximation, regularization, sampling density or projection. The worst boundary cases are not directly tested by the interior bending cohort. Pixel-row cap alignment bins do not show a simple monotonic increase toward row boundaries.

In the images, 85.72% of clipped squared error is on partially covered target pixels. The three-row bands centered on each cap row contribute 38.98%; these bands include one row before and after the pixel row containing each cap. Image locations and positions in the five-knob parameter box are distinct questions.

The strong-bending/small-error and weak-bending/large-error examples are deliberate counterexamples within the high/low bending deciles at scale 0.02; they are selected for small/large reconstruction errors, not representatives of average group errors.

Reproduce from the repository root: `.venv/bin/python analyze_generated_one_error_location.py`, then `.venv/bin/python make_generated_one_error_location_view.py`. Read `results.json` and the NPZ files for all numerical definitions, cohort indices and measured features.
