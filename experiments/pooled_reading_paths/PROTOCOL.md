# Frozen reading paths and direction PCA

This is a post-discovery evaluation. Do not refit the frozen encoders in
`figures/pooled_family_coordinates/`. Verify both existing freeze manifests
before and after evaluation.

Compare the 192-random and 512-balanced SVD fixed-weight readings at the two
radius-0.25 patches. Counts and sampling designs both differ. For each chart,
use its source plus the first eight held-out frame states, with no favorable
selection. The five scalar readings are the actual learned 5-by-784 pixel
matrix plus offsets, not the reverse-engineered geometric readers.

## Coordinate paths

On the valid generated-image family solve for images whose five frozen readings
equal the start readings plus a change in exactly one reading. Use the known
renderer and its derivative as an evaluation-only inverse solver. This tests
the geometry of the learned coordinates, not a newly discovered decoder or an
image-cloud-only navigation method. Original control values are never a target
for encoder changes. Retain failures, poor conditioning and excursions outside
the original training patch and original parameter box.

Use 17 ticks per path and half-spans 0.005, 0.05, 0.25 and 1.0 in first-order
pixel-distance units. Compare two calibrations: one reading step size shared
across all starts (unit image speed at the chart source), and unit initial image
speed separately at every start. Report the difference so speed normalization
cannot silently manufacture distance agreement.

For each reading compare each held-out path's complete pairwise pixel-distance
matrix with the source path. Report relative matrix error, and shape-only error
after normalizing matrix norms. Different coordinate families need not have
different distance signatures: measure all cross-family source pairs as well.
The original default-box indicator is descriptive, not a solver constraint;
root neighborhoods explicitly allow two-sided position changes. Require positive
height/width and the full stroke inside the 28-by-28 canvas. Incomplete or
nonconvergent paths are failures, not omitted silently.

## Actual mappings and PCA

After freeze, fit simple linear and cubic original-control maps on the existing
training set; assess the held-out set and export every coefficient. Report
standardized linear mixing matrices separately at each source and acquisition.
No coordinate reading is relabeled to improve the results.

For each anonymous vector family, center the aligned fitting vectors and run PCA
on their variation. Report variance magnitude, components for 95%/99% training
variation, and held-out mean-only/1/2/5-component vector errors. Also check the
algebraic result of applying PCA again to the full PCA score array: absent
truncation or another transformation it cannot create new information. A PCA of
five orthonormal vectors at one state has a centered rank of four and equal
nonzero singular values, so it is not evidence for four physical factors.

All path and mapping results are local or explicit stress tests, not a global
manifold certificate. Save source-relative paths only in shareable artifacts.
