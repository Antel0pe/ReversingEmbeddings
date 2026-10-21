# Original 784D values attached to the unwrapped sheet

Run `OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/inspect_original_sheet_coordinates.py` from the repository root, after generating `sheet_views/sheet_data.npz` with the preceding sheet experiment.

Each of 1,334 map vertices stores its full original 784-value float32 coverage image, without projecting or rounding those values.
The two map numbers are an address for a point on the sheet. The 784 original numbers are the vector attached to that address.
All arrays are checked bit-for-bit against `grey_ones.render` at their corresponding width/lean settings.
The gzip/base64 transfer is checked for a byte-identical round trip.
The displayed unwrapped positions come from the earlier local-graph-distance MDS map. Their distance distortion and discretization checks are in `sheet_views/results.json`.

The inspector shows the full coordinate array as a 28 by 28 numeric table, supports selecting any of the 784 pixel coordinates, and exports the complete unrounded vector for any selected point as JSON.
An original coordinate means a pixel's coverage value in [0,1], not the row or column number of that pixel.
Indices start at zero, with `index = 28 * row + column`.
The map color and contour lines show the selected original coordinate as a function over the sheet.
Between sampled vertices those fields are linearly interpolated; this is not an exact continuous decoder.
The full-vector display can be rounded for readability. The selected-coordinate readout and export use the original unrounded numbers.

Nothing about storing the 784 values makes the layout distance preserving. Distinct images can share the same value of one original coordinate; the full vector distinguishes them.
These statements apply to this fixed-center, fixed-height finite sample of the width/lean family.
