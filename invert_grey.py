"""Read the five knobs straight off a grey-1 image, in closed form. No search, no fitting loop.

It works because of three facts about the renderer (grey_ones.py):

  1. Every horizontal slice of the stroke is exactly `width` long, so the ink in a
     row is  width * (fraction of that row lying between top and bottom).
     A FULL row therefore sums to exactly `width`.
  2. The partial first and last rows sum to width * (their covered fraction), which
     pins `top` and `bottom` exactly -- and with them cy = (top+bot)/2, height = bot-top.
  3. Pick a pixel boundary k that sits inside the stroke for the whole row. The ink
     to its left is exactly  k - (left edge),  averaged over the row -- so the left
     edge at the row's middle is  k - (ink left of k),  and the centre is that plus
     width/2. (The ink-weighted average of pixel centres is NOT exact: a partly
     covered edge pixel drags it toward its own middle.) The centres of the full
     rows lie on a straight line in `row`: slope -tan(lean), value cx at cy.

    knobs = invert(image)        # (cx, cy, height, width, lean)
"""
import numpy as np

COLS = np.arange(28) + 0.5


def invert(img, full_tol=1e-3):
    I = np.asarray(img, np.float64).reshape(28, 28)
    rows = I.sum(1)
    inked = np.where(rows > 1e-9)[0]
    r0, r1 = inked[0], inked[-1]
    width = np.median(rows[r0 + 1:r1])                  # fact 1
    top = (r0 + 1) - rows[r0] / width                   # fact 2
    bot = r1 + rows[r1] / width
    cy, height = (top + bot) / 2, bot - top
    full = np.arange(r0 + 1, r1)
    full = full[np.abs(rows[full] - width) < full_tol * max(width, 1)]
    rough = (I[full] * COLS).sum(1) / rows[full]        # close enough to choose k
    k = np.round(rough).astype(int)
    left_ink = np.array([I[r, :kk].sum() for r, kk in zip(full, k)])
    centre = (k - left_ink) + width / 2                 # fact 3
    slope, icpt = np.polyfit(full + 0.5 - cy, centre, 1)
    return np.array([icpt, cy, height, width, np.degrees(np.arctan(-slope))])
