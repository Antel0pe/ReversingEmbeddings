# Sparse pixel knob

Pixel indices are zero-based and row-major. Define a 784-vector v with v[i] = 0.1 for i in {5, 32, 234, 423, 664} and zero elsewhere. Then x(k) = k v, starting at x(0) = 0. Settings 1–10 give selected intensities 0.1–1.0; the other 779 values remain zero.

For an existing image b, b + k v instead keeps other pixels at their original values. Selected pixels may exceed 1 if they start above 0; clipping would break the constant-increment requirement. This sparse direction is a pixel-space knob, not one of the five generator knobs.

The plot uses the default midpoint generated image as a spatial reference only. The measured sweep starts at the zero vector. Values use integer tenths before conversion to float64; verification.json reports float roundoff separately. all_pixel_values.csv contains all 784 coordinates, and vector.npy contains the per-step vector.

Reproduce: `.venv/bin/python experiments/sparse_pixel_knob.py`
