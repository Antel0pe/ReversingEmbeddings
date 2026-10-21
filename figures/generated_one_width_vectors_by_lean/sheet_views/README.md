# Four drawings of the width/lean sheet

Run `OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/unfurl_width_lean_sheet.py` from the repository root.
Center (14,14) and height 19.75 px are fixed. Width 1.8-4.6 px and lean -10 to 35 degrees span the sheet.

The ordered parameter rectangle supplies connectivity, but its knob values are not used as display positions in these maps.
PCA2 and PCA3 use actual centered pixel projection coefficients. PCA is fitted on 5,187 equally spaced width/angle samples.
The displayed mesh has 1,334 vertices (29 widths by 46 leans). Uniform controls do not imply uniform surface-area weighting.
Metric MDS2 minimizes the sum of squared differences between displayed pair distances and the original image distances.
The unfolding uses the same stress objective with graph path distances as its target.
Each MDS objective uses two initializations and up to 700 monotone-majorization iterations; it is a fitted result, not a certified global optimum.

Graph edges connect only local parameter offsets (16 directions) on a 57 by 91 grid.
An edge's weight includes a newly rendered midpoint, measuring two short image-space segments instead of one chord.
Shortest graph distances approximate travel constrained to the sheet. A 113 by 181 grid checks resolution at 49 or fewer landmarks against every displayed vertex.
This check leaves finite-direction graph bias unresolved; these distances are not certified exact continuous geodesics.

Reported stress is sqrt(sum((displayed_distance-target_distance)^2)/sum(target_distance^2)) across all distinct pairs, in the original distance units without rescaling.
The local edge ratios compare displayed mesh edges to original pixel chords, not to a nonlinear decoder metric.
Opposite-orientation triangles flag local planar foldovers; they do not enumerate all global self-overlap.
PCA reconstruction uses its linear decoder and 512 fresh joint held-out points. MDS maps have no decoder here.
Exact source images are retrieved from stored samples in the viewer; that lookup does not validate continuous reconstruction.

`07_why_curves_bunch.png` compares radial distance to a fixed upright image with distance travelled along the lean curve.
It also compares actual consecutive one-degree gaps with increments in the scalar distance-from-upright reading.
The renderer uses tan(lean) and clipped horizontal overlap, not an exponential spacing schedule. Coverage is continuous and piecewise affine in width and tan(lean) at fixed vertical extent.

See `results.json` for measurements, `sheet_data.npz` for arrays and distances, and `view_coordinates.csv` for all plotted coordinates.
