"""Transformations applied to 1s BEFORE a decoder learns them, each with its undo.

Every transform t has forward(X) -> (Y, side) and inverse(Y, side) -> X, where
`side` is any per-image numbers set aside (they count toward the total).
Images are (n, 784) float in [0, 1].

  pose   straighten and centre: measure each image's centre, slant, height and
         width from its ink (5 numbers, set aside), resample into a standard
         upright, centred, standard-size pose; undo by resampling back.
  blur   Gaussian blur, sigma 1 px (no undo: error is scored against the
         ORIGINAL, so lost detail counts against it).
  deslant  like pose but only slant and left-right centre are removed (2 numbers).
  sdf    signed distance to the stroke edge (positive inside); undo with one
         fitted S-curve from distance back to ink.
"""
import numpy as np
from scipy import ndimage

H0, W0, C0 = 18.0, 4.0, 14.0     # standard pose: height, width, centre
UP = 2                           # canonical grid is 56x56 (half-pixel spacing)
G = np.arange(28) + 0.5


def moments(X):
    I = X.reshape(-1, 28, 28).astype(np.float64)
    m = I.sum((1, 2)) + 1e-9
    xs, ys = G[None, None, :], G[None, :, None]
    cx = (I * xs).sum((1, 2)) / m; cy = (I * ys).sum((1, 2)) / m
    dx, dy = xs - cx[:, None, None], ys - cy[:, None, None]
    vxx = (I * dx * dx).sum((1, 2)) / m; vyy = (I * dy * dy).sum((1, 2)) / m
    vxy = (I * dx * dy).sum((1, 2)) / m
    s = vxy / vyy
    h = np.sqrt(12 * vyy); w = np.sqrt(np.maximum(12 * (vxx - vxy ** 2 / vyy), 0.25))
    return np.stack([cx, cy, s, h, w], 1)


def _canon_grid():
    g = (np.arange(28 * UP) + 0.5) / UP
    return np.meshgrid(g, g, indexing="xy")   # u (x-like), v (y-like)


def _lite(P):
    """'deslant' variant: undo only slant and left-right position (2 numbers);
    vertical position, height and width are left as drawn."""
    Q = P.copy(); Q[:, 1] = C0; Q[:, 3] = H0; Q[:, 4] = W0
    return Q


def upsample_forward(X):
    """Control: the same 56x56 resampling as pose, but changing nothing."""
    P = np.tile([C0, C0, 0.0, H0, W0], (len(X), 1))
    return pose_forward(X, P)[0], P


def deslant_forward(X):
    P = _lite(moments(X))
    return pose_forward(X, P)[0], P


def pose_forward(X, P=None):
    P = moments(X) if P is None else P
    U, V = _canon_grid()
    out = np.empty((len(X), (28 * UP) ** 2), np.float32)
    for n, (cx, cy, s, h, w) in enumerate(P):
        y = cy + (V - C0) * h / H0
        x = cx + s * (y - cy) + (U - C0) * w / W0
        im = X[n].reshape(28, 28).astype(np.float64)
        out[n] = ndimage.map_coordinates(im, [y - 0.5, x - 0.5], order=3, mode="constant").ravel()
    return np.clip(out, 0, 1), P


def pose_inverse(Y, P):
    yy, xx = np.meshgrid(G, G, indexing="ij")
    out = np.empty((len(Y), 784), np.float32)
    for n, (cx, cy, s, h, w) in enumerate(P):
        v = C0 + (yy - cy) * H0 / h
        u = C0 + (xx - cx - s * (yy - cy)) * W0 / w
        c = Y[n].reshape(28 * UP, 28 * UP).astype(np.float64)
        out[n] = ndimage.map_coordinates(c, [v * UP - 0.5, u * UP - 0.5], order=3, mode="constant").ravel()
    return np.clip(out, 0, 1)


def blur_forward(X, sigma=1.0):
    return np.stack([ndimage.gaussian_filter(x.reshape(28, 28), sigma).ravel() for x in X]).astype(np.float32), None


SDF_LO, SDF_HI = -16.0, 6.0
SDF_T = 0.45                     # S-curve softness, fitted once on MNIST train (see fit_sdf_t)


def sdf_forward(X):
    out = np.empty_like(X, dtype=np.float32)
    for n, x in enumerate(X):
        b = x.reshape(28, 28) > 0.5
        d = ndimage.distance_transform_edt(b) - ndimage.distance_transform_edt(~b)
        d = np.where(b, d - 0.5, d + 0.5)  # edge sits between pixels
        out[n] = ((np.clip(d, SDF_LO, SDF_HI) - SDF_LO) / (SDF_HI - SDF_LO)).ravel()
    return out, None


def sdf_inverse(Y, _=None, t=None):
    d = Y * (SDF_HI - SDF_LO) + SDF_LO
    return (1 / (1 + np.exp(-d / (t or SDF_T)))).astype(np.float32)


def fit_sdf_t(X):
    Y, _ = sdf_forward(X)
    ts = np.linspace(0.1, 1.5, 57)
    err = [np.linalg.norm(sdf_inverse(Y, t=t) - X) for t in ts]
    return float(ts[int(np.argmin(err))])
