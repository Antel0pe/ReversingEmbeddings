"""Paths between images: intensity averaging vs optimal transport.

Averaging two images cross-fades them (both strokes go translucent). Optimal
transport moves the ink instead, so strokes stay solid and slide. See
OnesManifold.ipynb Part III.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

_YY, _XX = np.mgrid[0:28, 0:28]
COORD = np.stack([_XX.ravel(), _YY.ravel()], 1).astype(np.float64)


def to_particles(img, N=800, seed=0):
    """Image -> N roughly equal-mass ink particles at sub-pixel positions.

    Inverse-CDF sampling of the intensity field, with jitter inside each pixel.
    N controls fidelity: round-trip error is ~1.8 at N=200, ~1.1 at N=800.
    """
    w = img.ravel().astype(np.float64)
    w = w / w.sum()
    idx = np.clip(np.searchsorted(np.cumsum(w), (np.arange(N) + 0.5) / N), 0, 783)
    rng = np.random.default_rng(seed)
    return COORD[idx] + rng.uniform(-0.5, 0.5, size=(N, 2))


def splat(pos, mass, shape=(28, 28)):
    """Particles -> image, spreading each particle bilinearly over 4 pixels."""
    out = np.zeros(shape)
    x, y = pos[:, 0], pos[:, 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    for dx in (0, 1):
        for dy in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            w = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
            ok = (xi >= 0) & (xi < 28) & (yi >= 0) & (yi < 28)
            np.add.at(out, (yi[ok], xi[ok]), mass * w[ok])
    return out


def transport_plan(imgA, imgB, N=800):
    """Match A's ink particles to B's by minimising total squared travel."""
    pa, pb = to_particles(imgA, N, seed=1), to_particles(imgB, N, seed=2)
    C = ((pa[:, None, :] - pb[None, :, :]) ** 2).sum(2)
    r, c = linear_sum_assignment(C)
    return pa, pb[c]


def ot_path(imgA, imgB, K=9, N=800):
    """Displacement interpolation: every grain of ink slides along a straight line."""
    pa, pb = transport_plan(imgA, imgB, N)
    mA, mB = imgA.sum() / N, imgB.sum() / N
    return np.array([splat((1 - t) * pa + t * pb, (1 - t) * mA + t * mB)
                     for t in np.linspace(0, 1, K)])


def linear_path(imgA, imgB, K=9):
    """Plain intensity averaging -- the cross-fade."""
    return np.array([(1 - t) * imgA + t * imgB for t in np.linspace(0, 1, K)])


def w2(pa, pb):
    """RMS ink displacement between two particle sets (a Wasserstein-2 distance)."""
    C = ((pa[:, None, :] - pb[None, :, :]) ** 2).sum(2)
    r, c = linear_sum_assignment(C)
    return np.sqrt(C[r, c].mean())


# ---------------------------------------------------------------------------
# Warp-field paths: move the whole image in sync, instead of moving ink grains
# independently. A single smooth displacement field u(x,y) is applied to every
# pixel, so neighbours travel together and the image cannot tear.
# ---------------------------------------------------------------------------

from scipy.ndimage import map_coordinates, gaussian_filter

_GY, _GX = np.mgrid[0:28, 0:28].astype(np.float64)


def warp(img, u):
    """Resample img through displacement field u (shape (2,28,28): uy, ux).

    Sub-pixel motion is handled by bilinear resampling -- the same operation that
    put the grey levels into MNIST in the first place.
    """
    return map_coordinates(img, [_GY - u[0], _GX - u[1]], order=1,
                           mode="constant", cval=0.0)


def demons(A, B, sigmas=(3.0, 2.0, 1.5, 1.0, 0.8), iters=150, step=1.5):
    """Find a smooth u with warp(A, u) ~= B (Thirion's demons, coarse-to-fine).

    Smoothing u after every update is the regulariser: it is what forces
    neighbouring pixels to move together rather than independently.
    """
    u = np.zeros((2, 28, 28))
    for sigma in sigmas:
        for _ in range(iters):
            Aw = warp(A, u)
            diff = B - Aw
            gy, gx = np.gradient(Aw)
            denom = gx**2 + gy**2 + diff**2 + 1e-6
            u[0] -= step * diff * gy / denom
            u[1] -= step * diff * gx / denom
            u = gaussian_filter(u, (0, sigma, sigma))
    return u


def warp_path(imgA, imgB, K=9, **kw):
    """One-directional: push A along its own displacement field toward B."""
    u = demons(imgA, imgB, **kw)
    return np.array([warp(imgA, t * u) for t in np.linspace(0, 1, K)])


def morph_path(imgA, imgB, K=9, **kw):
    """Symmetric morph: warp both ends to the shared intermediate shape, then blend.

    The blend is between images that are already aligned, so it does not ghost.
    """
    u = demons(imgA, imgB, **kw)
    v = demons(imgB, imgA, **kw)
    out = []
    for t in np.linspace(0, 1, K):
        out.append((1 - t) * warp(imgA, t * u) + t * warp(imgB, (1 - t) * v))
    return np.array(out)


def bridge_path(A_img, B_img, theta, maxleg=6, maxK=64):
    """Insert warped frames between two images until every consecutive hop < theta.

    Warps A toward B, doubling the number of frames until no hop exceeds theta.
    If one warp field does not arrive (demons leaves a residual), it re-solves from
    where it got to and continues, up to `maxleg` legs.

    Returns (frames, arrived). `frames` excludes A and B themselves.

    Known artefact: each leg resamples the previous leg's output, so blur compounds
    across legs. Composing the displacement fields and always warping the original
    A would avoid this.
    """
    cur = np.asarray(A_img, dtype=np.float64)
    B = np.asarray(B_img, dtype=np.float64)
    prev = cur.ravel()
    out = []
    for _ in range(maxleg):
        u = demons(cur, B)
        K = 2
        while K <= maxK:
            fr = [warp(cur, t * u) for t in np.linspace(0, 1, K + 1)[1:]]
            seq = [prev] + [f.ravel() for f in fr]
            if max(np.linalg.norm(seq[i+1] - seq[i]) for i in range(len(seq)-1)) < theta:
                break
            K *= 2
        out.extend(fr)
        prev = fr[-1].ravel()
        cur = fr[-1]
        if np.linalg.norm(prev - B.ravel()) < theta:
            return out, True
    return out, False


def compose(u1, u2):
    """Field for 'apply u1, then u2', so one warp of the ORIGINAL image replaces two.

    warp(warp(A,u1),u2)(x) = A(x - u2(x) - u1(x - u2(x))), so the total displacement is
    U = u2 + u1 resampled at (x - u2). Resampling a *smooth field* is nearly lossless;
    resampling a *sharp image* is not. That asymmetry is the whole point.
    """
    yq, xq = _GY - u2[0], _GX - u2[1]
    return np.array([u2[0] + map_coordinates(u1[0], [yq, xq], order=1, mode="nearest"),
                     u2[1] + map_coordinates(u1[1], [yq, xq], order=1, mode="nearest")])


def warpc(img, u, order=3):
    """Warp with cubic resampling, clipped to [0, 1].

    Cubic interpolation overshoots near sharp edges (ringing), which without the upper
    clip produces pixel values above 1.0 -- brighter than any real MNIST pixel can be.
    """
    return np.clip(map_coordinates(img, [_GY - u[0], _GX - u[1]], order=order,
                                   mode="constant", cval=0.0), 0.0, 1.0)


def bridge_path2(A_img, B_img, theta, maxleg=6, maxK=64, order=3, init=None):
    """bridge_path, but every frame is ONE warp of the original A.

    Legs accumulate into a composed displacement field instead of re-warping the
    previous leg's output, so interpolation blur does not compound.
    """
    A = np.asarray(A_img, dtype=np.float64)
    B = np.asarray(B_img, dtype=np.float64)
    U = np.zeros((2, 28, 28)) if init is None else np.asarray(init, dtype=np.float64).copy()
    prev = A.ravel()
    out = []
    for _ in range(maxleg):
        cur = warpc(A, U, order)
        u = demons(cur, B)
        K = 2
        while K <= maxK:
            fields = [compose(U, t * u) for t in np.linspace(0, 1, K + 1)[1:]]
            fr = [warpc(A, F, order) for F in fields]
            seq = [prev] + [f.ravel() for f in fr]
            if max(np.linalg.norm(seq[i+1] - seq[i]) for i in range(len(seq)-1)) < theta:
                break
            K *= 2
        out.extend(fr)
        U = fields[-1]
        prev = fr[-1].ravel()
        if np.linalg.norm(prev - B.ravel()) < theta:
            return out, True
    return out, False


def ot_field(A_img, B_img, N=800, sigma=1.5):
    """Displacement field from the optimal-transport matching, smoothed into a warp.

    `demons` is a local gradient method: if a stroke in A has no overlap with where it
    must end up in B, the image gradient carries no signal and the field stays ~0.
    Optimal transport solves the correspondence *globally* (an assignment problem), so it
    finds long-range motion. Smoothing the per-grain map turns it into a coherent field,
    usable on its own or -- better -- as an initialisation for `demons`.
    """
    pa, pb = transport_plan(np.asarray(A_img, float), np.asarray(B_img, float), N)
    d = pb - pa
    num = np.zeros((2, 28, 28)); den = np.zeros((28, 28))
    xi = np.clip(np.round(pa[:, 0]).astype(int), 0, 27)
    yi = np.clip(np.round(pa[:, 1]).astype(int), 0, 27)
    np.add.at(num[1], (yi, xi), d[:, 0])
    np.add.at(num[0], (yi, xi), d[:, 1])
    np.add.at(den, (yi, xi), 1.0)
    num = gaussian_filter(num, (0, sigma, sigma))
    den = gaussian_filter(den, (sigma, sigma))
    return num / np.maximum(den, 1e-3)
