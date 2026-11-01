"""A decision-free stand-in for the grey-1 generator.

`grey_ones.render` computes exact ink coverage, so every pixel is a `clip` of a
`min` minus a `max`. Each of those is a decision -- "whichever of these two is
smaller" -- and a decision has a corner. Sweep a knob and the corners arrive as
creases: the rate of change of a pixel lurches, thousands of times across the
default box. `figures/generated_one_crease_origin` measures where they come from.

This module renders the same five-knob family with no decisions at all. The
parallelogram's hard edges become sigmoid ramps, so a pixel's value is a product
of exponentials of smooth functions of the knobs. Derivatives of every order
exist everywhere, by construction rather than by approximation.

Parameter order is `grey_ones`': (cx, cy, height, width, lean), lean in degrees.

The construction is deliberately an affine warp of ONE template, so the family
is exactly an orbit of a five-dimensional group acting on pixel POSITIONS:

    v = (ROOT_HEIGHT / height) * (y - cy)
    u = (ROOT_WIDTH / width) * (x - cx + tan(lean) * (y - cy))
    image(x, y) = template(u, v)

Two translations, an x-scale, a y-scale and one shear. A general affine map of
the plane has six parameters; the sixth, vertical shear, is the one this family
does not use. That is why there are five knobs and not six.

The blur is what makes the family differentiable in x and y, which is what makes
`figures/generated_one_smooth_surrogate` able to express all five knob
directions as reweightings of just two spatial gradients.

Cost of the stand-in: it is NOT the coverage renderer. Over the default knob box
it differs from `grey_ones.render` by roughly 15% relative RMS (the exact figure
is measured and recorded in both figure READMEs). It is a smoothed model of the
same five-knob family, not the same pixels.
"""

import numpy as np

N_PIX = 28
ROOT_SETTINGS = np.array([14.0, 14.0, 19.75, 3.2, 0.0])
ROOT_WIDTH, ROOT_HEIGHT = 3.2, 19.75
TAU = 0.35                      # ramp width, in root-template pixels


def _sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


def template(u, v, tau=TAU):
    """The root 1, soft-edged, in its own centred coordinates."""
    return (_sigmoid((u + ROOT_WIDTH / 2) / tau)
            * _sigmoid((ROOT_WIDTH / 2 - u) / tau)
            * _sigmoid((v + ROOT_HEIGHT / 2) / tau)
            * _sigmoid((ROOT_HEIGHT / 2 - v) / tau))


def sample(p, x, y, tau=TAU):
    """One setting, evaluated at arbitrary coordinate arrays.

    Needed for spatial derivatives: shift the sample points, not the shape.
    """
    cx, cy, height, width, lean = p
    shear = np.tan(np.radians(lean))
    v = (ROOT_HEIGHT / height) * (y - cy)
    u = (ROOT_WIDTH / width) * (x - cx + shear * (y - cy))
    return template(u, v, tau)


def pixel_centres(n_pix=N_PIX):
    """Sample points of an n_pix grid spanning the same 0..28 canvas."""
    step = 28.0 / n_pix
    centres = (np.arange(n_pix) + 0.5) * step
    return (np.broadcast_to(centres, (n_pix, n_pix)).copy(),
            np.broadcast_to(centres[:, None], (n_pix, n_pix)).copy())


def render(P, n_pix=N_PIX, tau=TAU, dtype=np.float64):
    """P: (n, 5) settings -> (n, n_pix, n_pix) images. Mirrors grey_ones.render."""
    P = np.atleast_2d(np.asarray(P, np.float64))
    x, y = pixel_centres(n_pix)
    out = np.empty((len(P), n_pix, n_pix), dtype=dtype)
    for i, p in enumerate(P):
        out[i] = sample(p, x, y, tau)
    return out


def spatial_gradients(p, n_pix=N_PIX, tau=TAU, step=1e-4):
    """d(image)/dx and d(image)/dy, by central differences in POSITION.

    These exist only because the edges are soft; the coverage renderer has none.
    """
    x, y = pixel_centres(n_pix)
    d_dx = (sample(p, x + step, y, tau) - sample(p, x - step, y, tau)) / (2 * step)
    d_dy = (sample(p, x, y + step, tau) - sample(p, x, y - step, tau)) / (2 * step)
    return d_dx, d_dy


def affine_fields(p, n_pix=N_PIX, tau=TAU):
    """The six directions an affine warp can push an image in, at setting p.

    Built from nothing but the image's own two spatial gradients, weighted by 1,
    by how far right of centre each pixel sits, and by how far below centre.
    Returned as (6, n_pix*n_pix) in the order
        d/dx, d/dy, x*d/dx, y*d/dx, x*d/dy, y*d/dy
    with x, y measured from (cx, cy). Index 4, x*d/dy, is the vertical shear
    this family lacks; no knob should load on it. The other five are what
    cx, cy, lean, height and width turn out to be made of.
    """
    d_dx, d_dy = spatial_gradients(p, n_pix, tau)
    x, y = pixel_centres(n_pix)
    off_x, off_y = x - p[0], y - p[1]
    return np.stack([d_dx, d_dy,
                     off_x * d_dx, off_y * d_dx,
                     off_x * d_dy, off_y * d_dy]).reshape(6, -1)
