# Acquisition-quality control, before physical evaluation

The primary blind run is frozen and physical labels have not been evaluated.
Its very small patches still have appreciable family response scatter. Before
interpreting that scatter as curvature or failure to discover coordinates,
measure frame variability at a completely unchanged image.

At the two preselected sources, acquire ordinary SVD/PCA frames using 192
Gaussian rays, 512 scrambled Sobol rays, and 2048 scrambled Sobol rays. Use
12, 12 and 6 independent seeds respectively. Reference axes come from pooled
covariance of the 2048-ray observations. Compare axis rotations, off-family
response of the reference frame, and subspace residuals. No original physical
directions, readers or coordinate labels enter this diagnosis.

As an acquisition-only comparison, replace frames at exactly the original
states for root and second-source patches at radii 0.005 and 0.25 with 512-ray
scrambled Sobol frames. Keep image train/validation/test sets unchanged. Repeat
the fixed joint and quadratic scalar-function methods, with the same ridge
grid and validation-image selection, for both SVD and PCA. Keep both outcomes;
do not replace the primary run. Freeze this run too before physical evaluation.
This follow-up is triggered by blind response scatter, not physical results.
