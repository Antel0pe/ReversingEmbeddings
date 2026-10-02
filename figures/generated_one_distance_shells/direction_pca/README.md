# PCA of directions from seed 1

Each direction is the 784-vector `neighbor image - seed image`. All endpoints
are the existing sampled generated 1s at pixel distance approximately 4.5.
The seed has settings `(14.4173683584, 14.3665941296, 19.4545119198,
2.7230292799, -2.9233148527)` in the order horizontal center, vertical center,
height, width, and lean degrees.

## Measured dimensions

| Set | Ordinary PCA: 95% variance | Ordinary PCA: all numerical variance | Variance in 5 PCs |
| --- | ---: | ---: | ---: |
| 20 neighbors previously displayed | 8 | 19 | 87.1182% |
| All 619 sampled neighbors | 8 | 133 | 89.9005% |

Ordinary PCA includes one shared mean direction in addition to the listed
variable coefficients. Its mean can be added to the seed to obtain the common
reconstruction origin. Twenty centered observations have rank at most 19;
these 20 attain that bound. The mean direction is not zero.

If every reconstructed displacement must instead be a linear combination of
vectors based at the seed, with no extra mean offset, use uncentered SVD:

| Set | 95% of total squared direction length | All numerical variation |
| --- | ---: | ---: |
| 20 displayed neighbors | 8 | 20 |
| 619 sampled neighbors | 4 | 133 |

The second table measures total squared displacement rather than variance
about the mean. The 619 directions have a strong common component: the mean
direction has length 3.740 while each direction has length about 4.5. The
direction sample and the 20-point diversity selection give different weights
to different parts of the shell. These percentages do not describe a uniform
surface measure on the full continuous shell.

## Precision and reconstruction checks

The renderer computes coverage in float64 and stores float32 images. For the
main table, the identical 16-subrow coverage rule was evaluated while keeping
float64 results. Casting these results to float32 matched the original
renderer bit for bit for all 619 neighbors and the seed.

Numerical rank uses `max(matrix dimensions) * float64 epsilon * largest
singular value`. For centered PCA of 619 directions, the rank threshold is
about `7.53e-12`; the 133rd singular value is `9.79e-4`, and the 134th is
`1.43e-13`. This large gap supports the numerical rank of 133 for this sample.
Keeping 133 components reconstructs all 619 float64 directions with maximum
pixel error below `3.3e-15`.

The stored float32 images, analyzed in float64, have numerical rank 160.
Their extra 27 axes have singular values at the scale of pixel-storage
rounding. Thus literal "100%" depends on whether storage rounding is counted.
These are numerical ranks of finite samples, not a proof of the linear rank
of the entire continuous shell.

Keeping 8 PCs gives RMS direction reconstruction errors of 0.747 for the 20
displayed neighbors and 0.558 for all 619. The worst direction errors are
1.099 and 2.242 respectively. A 95% variance target minimizes total squared
error; it does not ensure every neighbor is reconstructed accurately or is
still a valid generated 1.

## How the numbers are found

1. Flatten each image into its 784 pixel values. Subtract the seed image to
   form a matrix `D`, one row per direction. Do not standardize pixel columns:
   that would change the pixel-distance metric.
2. For ordinary PCA subtract the mean row, making `X = D - mean(D)`.
3. Compute the singular value decomposition `X = U S V^T`.
4. Square the singular values. Their normalized cumulative sum is the
   explained-variance fraction. The first count reaching 0.95 is the 95%
   answer. Count singular values above the stated numerical threshold for
   the numerical all-variance answer.
5. A direction's PCA coordinates are `(direction - mean) @ V_k`.
   Reconstruct with `mean + coordinates @ V_k^T`, then add the seed image
   if the endpoint image is wanted.

Centering also cancels the fixed seed subtraction. Ordinary PCA therefore
has the same spectrum as PCA of the endpoint images. Uncentered SVD keeps
the seed as the origin: apply the SVD directly to `D`.

## An exact five-variable description

Let `F` denote the known renderer, `p0` the seed's five knob settings, and
`delta_p` the five changes in those settings. Then

`direction(delta_p) = F(p0 + delta_p) - F(p0)`.

This is an exact nonlinear description using five values per direction and
the shared renderer. Applying it to all 619 saved neighbors gives zero pixel
error relative to the existing float32 images. The five numbers are obtained
directly by subtracting the known seed settings from each endpoint's settings.
They are not five coefficients multiplying fixed pixel vectors. At finite
distance, different controls and their combinations can change which pixels
gain or lose ink. Five fixed PCA patterns cannot encode all those changes.

The fixed-radius condition imposes one relation on the five knob changes, so
the regular parts of the shell locally have four independent coordinates.
That does not imply four or five fixed linear axes can contain the shell.
Five local tangent vectors describe infinitesimal changes; the vectors here
are finite seed-to-endpoint displacements.

The CSV contains the exact five knob changes and the first eight PCA scores
for the 619-point PCA fit. `pca_bases.npz` saves the mean and basis for both
centered fits. `results.json` records spectra, thresholds, and errors.
Reproduce with `python analyze_generated_one_direction_pca.py`.
