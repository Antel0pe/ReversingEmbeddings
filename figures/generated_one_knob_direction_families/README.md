# Five knob direction families

A reproducible exploration of the known physical derivative for each knob across the generated straight-one family. Real MNIST is not used.

Run from the repository root: `.venv/bin/python analyze_generated_one_knob_direction_families.py`.

Open `index.html` for the illustrated report and interactive reconstruction inspector. `results.json` holds all measured summaries, `arrays.npz` holds settings, images, full raw derivatives and fitted bases, and `reconstruction_errors.csv` holds reconstruction curves.

## Sampling and precision

Seed 20261017. All five controls vary independently and uniformly across `grey_ones.RANGES`. There are 384 training images, 128 validation images for nonlinear-model choices, and 256 untouched random test images. An additional 243-point grid uses normalized levels 0.0000001, 0.5000001234, and 0.9999999 on every knob. It probes near edges without landing exactly on known cap kinks. This is finite sampled coverage, not a guarantee for the whole range.

A direction is the analytic derivative per positive unit of its knob: one pixel for centers, height and width; one degree for lean. These are instantaneous directions, not finite-step differences. Derivatives use the float64 coverage calculation before default float32 storage. At nonsmooth clipping/subrow events a unique two-sided derivative may not exist; exact-event one-sided derivatives must be treated separately. Twenty-four random states per knob were checked against tiny central rendered moves.

## Global SVD

Rows are direction vectors, columns are 784 pixel coefficients. Uncentered SVD uses a linear space through zero; centered PCA uses the training mean plus a basis. Unit-vector SVD separates orientation from speed. All reconstruction errors use true held-out vectors, without clipping. Relative error = vector reconstruction L2 error / true vector L2 length. Near-exact target = worst relative error at most 1e-10. Training rank counts singular values above the largest times 1e-10. Neither rank nor variance retention is called intrinsic dimension.

| Knob | Training rank | Patterns for worst test error ≤1% | Patterns for ≤1e-10 | Median pairwise cosine |
| --- | ---: | ---: | ---: | ---: |
| cx | 220 | not reached | not reached | 0.184 |
| cy | 225 | not reached | not reached | 0.166 |
| height | 74 | not reached | not reached | 0.133 |
| width | 226 | not reached | not reached | 0.272 |
| lean | 220 | not reached | not reached | 0.023 |

## Coverage refinement and fresh audit

The original 384 fitted states miss rare configurations: even their full nonzero SVD spans do not reconstruct every test vector. Arbitrary zero-singular-value basis completions are excluded. We added 4096 independent scrambled Sobol states, 2048 beta(0.18, 0.18) edge-biased states, and a 3125-point grid at normalized levels 0.000001, 0.23, 0.47, 0.71, 0.999999. The total enriched training set has 9653 states. After this refinement we ran a fresh audit of 512 uniform random states and 1024 grid states at normalized levels 0.00001, 0.37, 0.68, 0.99999. This fresh audit was not used to fit or choose the basis.

| Knob | Enriched rank | Worst fresh random error | Worst fresh edge error |
| --- | ---: | ---: | ---: |
| cx | 260 | 2.128e-15 | 2.019e-15 |
| cy | 265 | 1.964e-15 | 2.068e-15 |
| height | 95 | 2.072e-15 | 2.168e-15 |
| width | 267 | 2.037e-15 | 2.132e-15 |
| lean | 260 | 2.715e-15 | 2.370e-15 |

The rank counts match simple support and row-total constraints. A renderer-box envelope gives 283 possible pixels across 23 rows, and 96 cap pixels across 6 rows. Horizontal and lean changes preserve ink in every row: 283 - 23 = 260. Height changes add equal total ink at the top and bottom: 96 - 1 = 95. Vertical changes have zero row totals except at the six cap rows, whose combined total is zero: 283 - (17 + 1) = 265. Width changes add the same ink amount in each of the 17 fully covered interior rows: 283 - 16 = 267. These bounds match the observed ranks. The envelope is computed directly from parameter-box extrema for each renderer subrow; the independence/rank lower bound is numerical.

All five enriched spans reconstruct the fresh audit to approximately float64 precision. This is a sampled result, not a proof of full-box completeness or minimum rank. The initial test set was used diagnostically during the coverage refinement; the fresh audit is reported separately. The second figure compares the initial and enriched fits.

## Other reductions and prediction

The comparison includes centered PCA, SVD on unit vectors, a union of 12 local affine PCA patches, and RBF kernel PCA with a nonlinear RBF inverse. Kernel PCA is fitted on unit directions; its inverse predicts full raw directions, so changing magnitude can remain a source of error. The kernel uses the median training pairwise squared distance; inverse bandwidth and regularization are chosen using only validation states. A local PCA patch is selected by the closest training unit-vector centroid; the stated dimension is per patch and excludes its discrete label and the stored collection of bases. These methods consume an observed vector, not just its source image. They have explicit decoders and held-out errors.

Starting-state predictors are a separate supervised test: linear/quadratic settings regression, RBF settings regression, and ridge regression directly from all source-image pixels. The first three receive the five true settings. Hyperparameters use only validation images. Their errors are in `results.json`; they do not count as discovery of unknown knobs.

## Exact relationships

Across the default box, the stroke is wide enough that its left and right edge contributions cannot share a pixel within one image row: minimum width minus maximum absolute slope is greater than one pixel. Horizontal shift changes those edge contributions with opposite signs; increasing width changes each by half as much with positive signs. Therefore, pixel by pixel, `width direction = abs(horizontal-center direction) / 2`.

Vertical center motion consists of `tan(lean) * horizontal-center direction` plus signed top/bottom cap motion. Removing the horizontal part leaves the caps. They occupy separate rows, so `height direction = abs(vertical-center direction - tan(lean) * horizontal-center direction) / 2`. These statements follow the renderer formulas away from events; they were checked numerically across all sampled states. They link actual physical derivatives, not individual SVD components.

Largest relative width-identity error: 0.000e+00. Largest relative height-identity error: 6.793e-17.

## Other questions

Pairwise cosines retain the positive-knob sign. The mean length of unit directions measures alignment: 1 means all identical; near 0 means cancellation. The figure histogram uses a deterministic subsample of pairs, while reported quantiles use every unordered pair among 768 random states.

Paired setting interventions replace one knob at a time at 96 starting states, recomputing all five directions. They measure which settings influence each direction; the sensitivity entry is the median relative vector change. This is a change to a direction pattern, not a direct ink change.

The common-target probe fits a point closest in least-squares distance to training tangent lines, and measures misses at new random images. It also finds the common positive unit-length step that most reduces source-image spread. Neither tangent-line extrapolation nor small projected distances prove an endpoint on the curved family. A common nonlinear future endpoint is a different question.

At the canonical root, one-sided direction checks are recorded separately in `root_kink_check`. A separate upright integer-column event has disagreeing one-sided horizontal/width directions, recorded in `lateral_pixel_event_checks`. The analytic routine averages cap-side derivatives at exact cap events; interpreting that average as a unique derivative would be misleading.

## Figures

- `01_direction_examples.png`: source images and five physical directions, with unit-normalized maps and raw speeds.
- `02_reconstruction.png`: worst held-out and near-edge error versus SVD basis size.
- `03_alignment.png`: cosine distributions without low-dimensional projection.
- `04_exact_links.png`: direct pixel-map evidence for the two exact family relationships.

Limit: the renderer and known control labels are supplied. This experiment studies their direction families; it does not infer the physical knobs from unlabeled images or build a global learned image decoder.
