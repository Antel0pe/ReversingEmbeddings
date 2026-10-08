# Reference-image midpoint interpolation

Two endpoints are actual fitting-cloud images. Pixel average = (image A + image B) / 2. True midpoint = render((knobs A + knobs B) / 2). The unchanged fine five-dimensional learned manifold reconstructs this true midpoint with the same pixel-only multistart projection as the failure analysis. No training or model selection occurs.

Image L2 error is Euclidean distance over 784 coverage pixels. RMS image L2 squares each image error, averages, and takes a square root. Pixel RMS = RMS image L2 / 28. Average-image error is distance to a defined physical-knob midpoint, not the minimum distance to any valid stroke.

The fitting cloud has 14,389 images: 10,240 uniform, 3,125 five-point-per-axis lattice, 1,024 near-face. There was no fixed distance-4 image selection. All fitting images were used by the prior fit.

Pair diagnostic: seed 1007, 192 unique nearest-reference pairs sampled from 1,024 uniform reference-index probes; 128 randomly selected reference pairs in each of five conditional pixel-distance bands; 24 adjacent original lattice pairs per knob. Endpoint pairs are sampled from the existing discrete cloud, not an exhaustive pair analysis. Single-knob lattice steps are one quarter of the physical range.

| Pair group | N | Mean endpoint distance | Average RMS image error | Learned clipped RMS image error |
| --- | ---: | ---: | ---: | ---: |
| Nearest reference pairs | 192 | 0.5787 | 0.05063 | 0.08722 |
| Adjacent lattice: cx | 24 | 1.4550 | 0.10921 | 0.12967 |
| Adjacent lattice: cy | 24 | 0.7553 | 0.06364 | 0.08932 |
| Adjacent lattice: height | 24 | 0.4058 | 0.05022 | 0.12451 |
| Adjacent lattice: width | 24 | 1.9748 | 0.22874 | 0.25745 |
| Adjacent lattice: lean | 24 | 5.3369 | 1.76958 | 0.33174 |
| Distance 1 | 128 | 1.2514 | 0.18067 | 0.11029 |
| Distance 2 | 128 | 2.1156 | 0.36236 | 0.12376 |
| Distance 3 | 128 | 3.0437 | 0.63307 | 0.13014 |
| Distance 4 | 128 | 4.0076 | 1.02067 | 0.12095 |
| Distance 5 | 128 | 4.9898 | 1.57496 | 0.12586 |

## Fixed-target spacing check

The same 128 distance-4 true midpoints and physical-knob path directions are held fixed. Shrink both endpoint offsets symmetrically, render exact additional reference endpoints, and average. No fitting cloud or model is changed. At each row the learned clipped RMS error is 0.12095.

| Endpoint offset fraction | Mean pixel distance | Average RMS image error |
| ---: | ---: | ---: |
| 1 | 4.00762 | 1.02067 |
| 0.5 | 2.24610 | 0.34854 |
| 0.25 | 1.17394 | 0.11106 |
| 0.125 | 0.59745 | 0.03371 |
| 0.0625 | 0.30063 | 0.00971 |

Averaging exact endpoint images creates bounded edge blur and incorrect intermediate coverage. It cannot cause negative or above-one pixels. The learned manifold interpolates 32 learned grid nodes per cell, not two original references. Nearby-reference averaging error is smaller than learned error, so short chords alone are insufficient to explain every learned residual. Node approximation, grid resolution, smoothing, boundary behavior and projection are not causally apportioned by this experiment.

For distance-4 pairs, average error mean 1.0001, RMS 1.0207, median 1.0234, p95 1.2674. Learned clipped RMS on the same true midpoints is 0.1210. The 8.44 ratio refers to RMS image L2; the squared-error ratio is its square.

Mean fitted endpoint chart-coordinate decoding is separately recorded. Since the chart has been fitted and is warped relative to physical knobs, this is a different target operation and does not include closest-point refinement.

Display examples include typical nearby and distance-4 averaging errors, smallest and largest distance-4 errors in the finite sample, and typical averaging errors for each single-knob lattice move. Representative means selected by proximity to median image L2 error; no claim about population extrema.

Reproduce from the repository root: `.venv/bin/python analyze_generated_one_midpoint_interpolation.py`, then `.venv/bin/python make_generated_one_midpoint_view.py`. Read `results.json` for complete metric definitions and `pairs.npz` for all pairs, endpoints, true midpoints, predictions and fitted coordinates.
