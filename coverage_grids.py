"""grey_ones.render's coverage maths on a pixel grid of any fineness.

`grey_ones.render` is fixed at 28x28, which is the right default: it is what
MNIST is. Two experiments need to vary the grid instead, to separate properties
of the five-knob family from properties of the ruler used to measure it:

  - `analyze_generated_one_crease_origin` asks whether refining the grid smooths
    the path through pixel space (it does not).
  - `analyze_generated_one_moment_chart` asks whether the error in recovering
    the knobs from image moments is a limit of the method or of the grid (it is
    the grid, and it falls as the square of the pixel pitch).

The canvas always spans coordinates 0 through 28, so `n_pix=56` means pixels of
pitch 0.5 over the same picture, not a bigger picture. Horizontal coverage stays
exact; `sub` sub-rows per pixel row handle the vertical direction, as in
`grey_ones`. Returns float64, since both callers do numerical-rank work.
"""

import numpy as np

from grey_ones import N_PIX, SUB


def render_coverage(P, n_pix=N_PIX, sub=SUB, batch=16):
    """P: (n, 5) settings -> (n, n_pix, n_pix) coverage images."""
    P = np.atleast_2d(np.asarray(P, np.float64))
    pitch = 28.0 / n_pix
    out = np.empty((len(P), n_pix, n_pix))
    sub_lo = np.arange(n_pix * sub) * (pitch / sub)
    sub_mid = sub_lo + 0.5 * pitch / sub
    cols = np.arange(n_pix) * pitch
    for s in range(0, len(P), batch):
        cx, cy, h, w, lean = [P[s:s + batch, k][:, None] for k in range(5)]
        top, bot = cy - h / 2, cy + h / 2
        wy = np.clip(np.minimum(sub_lo + pitch / sub, bot) - np.maximum(sub_lo, top),
                     0, pitch / sub) * (sub / pitch)
        centre = cx - np.tan(np.radians(lean)) * (sub_mid - cy)
        L, R = centre - w / 2, centre + w / 2
        ov = np.clip(np.minimum(R[:, :, None], cols + pitch)
                     - np.maximum(L[:, :, None], cols), 0, pitch) / pitch
        out[s:s + batch] = (ov * wy[:, :, None]).reshape(
            len(L), n_pix, sub, n_pix).mean(2)
    return out
