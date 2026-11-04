"""Warp charts: low-dimensional coordinates for an image family, from data only.

A chart here is a polynomial map from image coordinates into a shared canonical
frame, plus one template living in that frame. An image is decoded by warping
the template, so `n` warp numbers per image ARE the low-dimensional coordinates
and the decoder is explicit:

    image(x, y)  =  template( zx(x, y; theta),  zy(x, y; theta) )

`figures/generated_one_motion_discovery` established that the five generated-1
knobs are exactly the affine warps of one template. That experiment linearised,
which needs neighbours close enough to be first order. Real MNIST 1s are far
too sparse for that (`figures/mnist_one_knob_search` measures it), so this
module fits the warp directly instead, by Gauss-Newton, and so handles pairs
that are nowhere near each other.

TEMPLATE RESOLUTION IS THE BINDING CONSTRAINT, not the model. Warping one
generated 1 onto another through the EXACT affine map between their true knobs
still leaves ~15% if the source is 28x28, and ~1.3% if it is 224x224. So the
template is held at `G` x `G` and super-resolved from the data: thousands of
images sit at different sub-pixel offsets, and splatting them all into the
canonical grid recovers detail no single 28x28 image carries.

Nothing here uses parameter labels. Moments supply the initial affine chart;
everything after that is driven by image residuals.
"""

import numpy as np
from scipy.ndimage import map_coordinates

N_PIX = 28
G = 112                     # canonical template resolution
SC = G / N_PIX
SUB = 4                     # output subsamples per pixel per axis (area average)
CEN = 14.0                  # canvas centre; the warp basis is scaled about it

_R, _C = np.mgrid[0:N_PIX, 0:N_PIX].astype(np.float64)
PIX_X, PIX_Y = _C + 0.5, _R + 0.5
_s = (np.arange(SUB) + 0.5) / SUB
_o = np.add.outer(np.arange(N_PIX), _s).ravel()
SAMP_X, SAMP_Y = np.meshgrid(_o, _o)


def terms(deg):
    """(power of u, power of v) pairs: 1, u, v, u^2, uv, v^2, u^3, ..."""
    return [(i, d - i) for d in range(deg + 1) for i in range(d, -1, -1)]


def basis(x, y, deg):
    u, v = (x - CEN) / CEN, (y - CEN) / CEN
    return [(u ** i) * (v ** j) for i, j in terms(deg)]


def model(deg, drop_vertical_shear=False):
    """A warp family: (degree, terms zx may use, terms zy may use).

    `drop_vertical_shear` removes every term in which the VERTICAL map depends
    on x. At degree 1 that leaves five parameters -- the generated-1 family's
    exact shape, where y never depends on x.
    """
    nb = len(terms(deg))
    iy = [k for k, (i, _) in enumerate(terms(deg)) if i == 0] if drop_vertical_shear \
        else list(range(nb))
    return (deg, list(range(nb)), iy)


def n_par(m):
    return len(m[1]) + len(m[2])


def zmap(theta, m, x, y):
    deg, ix, iy = m
    B = basis(x, y, deg)
    nx = len(ix)
    return (sum(theta[k] * B[j] for k, j in enumerate(ix)),
            sum(theta[nx + k] * B[j] for k, j in enumerate(iy)))


def moment_chart(I):
    """Five numbers per image from six linear projections of the pixels.

    Centre (2) plus the upper-triangular factor of the second-moment matrix (3).
    The zero sits in the (y, x) slot, i.e. "y does not depend on x" -- the same
    restriction the generated family obeys. Nothing is fitted.
    """
    tot = I.reshape(len(I), -1).sum(1)
    f = lambda W: (I * W).reshape(len(I), -1).sum(1) / tot
    cx, cy = f(PIX_X), f(PIX_Y)
    vxx = f(PIX_X * PIX_X) - cx * cx
    vyy = np.maximum(f(PIX_Y * PIX_Y) - cy * cy, 1e-9)
    vxy = f(PIX_X * PIX_Y) - cx * cy
    d = np.sqrt(vyy)
    b = vxy / d
    a = np.sqrt(np.maximum(vxx - b * b, 1e-9))
    return np.stack([cx, cy, a, b, d], 1)


def init_from_moments(I, m):
    """Initial affine chart from the moments, written in the scaled basis."""
    P = moment_chart(I)
    P0 = P.mean(0)
    U0 = np.array([[P0[2], P0[3]], [0.0, P0[4]]])
    deg, ix, iy = m
    out = np.zeros((len(I), n_par(m)))
    for i, p in enumerate(P):
        Ai = U0 @ np.linalg.inv(np.array([[p[2], p[3]], [0.0, p[4]]]))
        vx = {0: P0[0] + Ai[0, 0] * (CEN - p[0]) + Ai[0, 1] * (CEN - p[1]),
              1: Ai[0, 0] * CEN, 2: Ai[0, 1] * CEN}
        vy = {0: P0[1] + Ai[1, 0] * (CEN - p[0]) + Ai[1, 1] * (CEN - p[1]),
              1: Ai[1, 0] * CEN, 2: Ai[1, 1] * CEN}
        out[i] = [vx.get(j, 0.0) for j in ix] + [vy.get(j, 0.0) for j in iy]
    return out


def decode(T, TH, m):
    """Coordinates -> 784 pixels: warp the template, then area-average."""
    out = np.empty((len(TH), N_PIX, N_PIX))
    for i, th in enumerate(TH):
        zx, zy = zmap(th, m, SAMP_X, SAMP_Y)
        v = map_coordinates(T, [zy * SC - 0.5, zx * SC - 0.5], order=1,
                            mode="constant", cval=0.0)
        out[i] = v.reshape(N_PIX, SUB, N_PIX, SUB).mean((1, 3))
    return out


def splat(vals, TH, m):
    """Weighted scatter of 28x28 values into the canonical grid, then average.

    Used two ways: on the images themselves it super-resolves the template; on
    residuals it is the Richardson step that sharpens it.
    """
    num = np.zeros((G, G))
    den = np.zeros((G, G))
    for r, th in zip(vals, TH):
        zx, zy = zmap(th, m, SAMP_X, SAMP_Y)
        gx, gy = zx * SC - 0.5, zy * SC - 0.5
        v = np.repeat(np.repeat(r, SUB, 0), SUB, 1)
        x0 = np.floor(gx).astype(int)
        y0 = np.floor(gy).astype(int)
        fx, fy = gx - x0, gy - y0
        for dy in (0, 1):
            for dx in (0, 1):
                w = ((fx if dx else 1 - fx) * (fy if dy else 1 - fy)).ravel()
                yy, xx = (y0 + dy).ravel(), (x0 + dx).ravel()
                ok = (yy >= 0) & (yy < G) & (xx >= 0) & (xx < G)
                np.add.at(num, (yy[ok], xx[ok]), (w * v.ravel())[ok])
                np.add.at(den, (yy[ok], xx[ok]), w[ok])
    return np.where(den > 1e-8, num / np.maximum(den, 1e-9), 0.0)


def refine(T, TH, I, m, steps=2, damp=3e-2, prior=1e-3, TH0=None):
    """Gauss-Newton on the warp numbers, against the template's own gradients.

    `prior` pulls weakly toward the starting chart, which keeps high-degree
    warps from folding to chase ink; `jacobian_determinants` checks that.
    """
    Ty, Tx = np.gradient(T)
    deg, ix, iy = m
    npar = n_par(m)
    TH = TH.copy()
    if TH0 is None:
        TH0 = TH.copy()
    B = basis(SAMP_X, SAMP_Y, deg)
    for _ in range(steps):
        for i in range(len(TH)):
            th = TH[i]
            zx, zy = zmap(th, m, SAMP_X, SAMP_Y)
            yy, xx = zy * SC - 0.5, zx * SC - 0.5
            val = map_coordinates(T, [yy, xx], order=1, mode="constant", cval=0.0)
            gx = map_coordinates(Tx, [yy, xx], order=1, mode="constant", cval=0.0) * SC
            gy = map_coordinates(Ty, [yy, xx], order=1, mode="constant", cval=0.0) * SC
            r = (I[i] - val.reshape(N_PIX, SUB, N_PIX, SUB).mean((1, 3))).ravel()
            J = np.empty((N_PIX * N_PIX, npar))
            for k, j in enumerate(ix):
                J[:, k] = (gx * B[j]).reshape(N_PIX, SUB, N_PIX, SUB).mean((1, 3)).ravel()
            for k, j in enumerate(iy):
                J[:, len(ix) + k] = (gy * B[j]).reshape(
                    N_PIX, SUB, N_PIX, SUB).mean((1, 3)).ravel()
            H = J.T @ J
            sc = max(np.trace(H), 1e-12) / npar
            H += (damp + prior) * sc * np.eye(npar)
            g = J.T @ r + (prior * sc) * (TH0[i] - th)
            try:
                TH[i] = th + np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                pass
    return TH


def fit_chart(I, m, train, test, rounds=25):
    """Alternate: sharpen the template, then refit every image's coordinates.

    `rounds` matters more than it looks. Held-out error on the generated family
    keeps falling well past the point it appears to settle:

        rounds   7      15     25     40
        error    2.93%  2.08%  1.72%  1.50%     (floor 1.26%)

    so a seven-round fit understates a chart by a factor of two. The default is
    25; pass more when the number is the point. Real MNIST 1s barely move
    (28.28% to 27.79% from 7 to 21 rounds), because there the error is the
    family, not the fit. A long schedule cannot overfit the reported number:
    the template comes from the training images only, and the test images'
    coordinates are fitted against it.
    """
    THtr = init_from_moments(I[train], m)
    THte = init_from_moments(I[test], m)
    T = splat(I[train], THtr, m)
    for _ in range(rounds):
        T = T + splat(I[train] - decode(T, THtr, m), THtr, m)
        THtr = refine(T, THtr, I[train], m)
        THte = refine(T, THte, I[test], m)
    return T, THtr, THte


def jacobian_determinants(theta, m, res=14, ink=None):
    """Determinant of the fitted map. Non-positive anywhere = a folded warp.

    `ink`, a 28x28 image, restricts the grid to pixels that actually carry ink.
    That distinction matters: a high-degree polynomial warp can thrash in empty
    corners without touching a single lit pixel, which costs nothing and is not
    the fit cheating. Folding ON the stroke would invalidate it. Measured both
    ways, MNIST folds in up to a quarter of held-out images somewhere on the
    canvas but only ~1-2% on the ink, with the ink-restricted determinant
    sitting near 0.9.
    """
    deg, ix, iy = m
    if ink is not None:
        g = np.arange(N_PIX) + 0.5
        X, Y = np.meshgrid(g, g)
        mask = ink > 0.10
    else:
        g = np.linspace(2, 26, res)
        X, Y = np.meshgrid(g, g)
        mask = None
    h = 1e-3
    zx1, zy1 = zmap(theta, m, X + h, Y)
    zx0, zy0 = zmap(theta, m, X - h, Y)
    ax, ay = (zx1 - zx0) / (2 * h), (zy1 - zy0) / (2 * h)
    zx1, zy1 = zmap(theta, m, X, Y + h)
    zx0, zy0 = zmap(theta, m, X, Y - h)
    bx, by = (zx1 - zx0) / (2 * h), (zy1 - zy0) / (2 * h)
    det = ax * by - ay * bx
    return det[mask] if mask is not None and mask.any() else det


def rel_error(A, B):
    return float(np.linalg.norm(A - B) / np.linalg.norm(A))


# ---- model-free dimension counting, for comparison with a fitted chart ----

def neighbour_distances(V, kmax=20):
    sq = (V ** 2).sum(1)
    D = sq[:, None] + sq[None, :] - 2 * V @ V.T
    np.fill_diagonal(D, np.inf)
    return np.sort(np.sqrt(np.maximum(D, 0)), axis=1)[:, :kmax]


def twonn(d, trim=0.9):
    """Facco et al.: dimension from the ratio of 2nd to 1st neighbour."""
    mu = d[:, 1] / np.maximum(d[:, 0], 1e-12)
    mu = np.sort(mu[np.isfinite(mu) & (mu > 1 + 1e-9)])
    n = len(mu)
    F = np.arange(1, n + 1) / (n + 1)
    keep = F < trim
    x, y = np.log(mu[keep]), -np.log(1 - F[keep])
    return float((x @ y) / (x @ x))


def mle_dim(d, k):
    """Levina-Bickel maximum likelihood dimension from the first k neighbours."""
    r = np.maximum(d[:, :k], 1e-12)
    inv = (np.log(r[:, k - 1:k]) - np.log(r[:, :k - 1])).mean(1)
    return float(1.0 / np.mean(inv))

# ---- geometry times appearance: image = gain(z) * template(z) ----
#
# A warp of a fixed template cannot change how DARK the ink is. Note what it
# CAN already do: thickness varying along the stroke is a warp, because a `u*v`
# term means "horizontal scale changes with height" -- which is why degree 2
# buys the most on MNIST. So the genuinely new freedom is a multiplicative gain
# field, written in CANONICAL coordinates so it attaches to the stroke ("this
# part of the stroke is darker") rather than to the canvas.
#
# The model is LINEAR in the gain coefficients, so fitting them per image is an
# exact small least-squares -- no optimiser and no new failure mode.
#
# Measured, at matched template rounds: one gain term (overall darkness) is
# worth ~0 on both families, and the fitted constant gain has an inter-quartile
# spread of only 0.004 on the generated family and 0.029 on MNIST -- real
# variation in handwriting, but far too small to matter. Ten terms (a cubic
# spatial gain) are worth 4.14 points on MNIST and 0.13 on the generated
# family, whose coverage is uniform by construction. So the real missing
# freedom is ink density varying ALONG the stroke, not overall darkness.


def gain_basis(zx, zy, n_terms):
    u, v = (zx - CEN) / CEN, (zy - CEN) / CEN
    return [(u ** i) * (v ** j) for i, j in terms(3)][:n_terms]


def gain_design(T, theta, m, n_terms):
    """Per-image design matrix: each gain basis element times the warped T."""
    zx, zy = zmap(theta, m, SAMP_X, SAMP_Y)
    t = map_coordinates(T, [zy * SC - 0.5, zx * SC - 0.5], order=1,
                        mode="constant", cval=0.0)
    return np.stack([(b * t).reshape(N_PIX, SUB, N_PIX, SUB).mean((1, 3)).ravel()
                     for b in gain_basis(zx, zy, n_terms)], 1)


def fit_gains(T, TH, I, m, n_terms):
    coef = np.empty((len(TH), n_terms))
    rec = np.empty_like(I)
    for i, th in enumerate(TH):
        A = gain_design(T, th, m, n_terms)
        c, *_ = np.linalg.lstsq(A, I[i].ravel(), rcond=None)
        coef[i] = c
        rec[i] = (A @ c).reshape(N_PIX, N_PIX)
    return coef, rec


def splat_gained(resid, TH, coef, m, n_terms):
    """Re-estimate the template given the gains: undo the local gain first."""
    num = np.zeros((G, G))
    den = np.zeros((G, G))
    for r, th, c in zip(resid, TH, coef):
        zx, zy = zmap(th, m, SAMP_X, SAMP_Y)
        g = sum(ci * b for ci, b in zip(c, gain_basis(zx, zy, n_terms)))
        weight = np.abs(g)
        val = (np.repeat(np.repeat(r, SUB, 0), SUB, 1)
               * np.sign(g) / np.maximum(np.abs(g), 0.15))
        gx, gy = zx * SC - 0.5, zy * SC - 0.5
        x0 = np.floor(gx).astype(int)
        y0 = np.floor(gy).astype(int)
        fx, fy = gx - x0, gy - y0
        for dy in (0, 1):
            for dx in (0, 1):
                w = (((fx if dx else 1 - fx) * (fy if dy else 1 - fy))
                     * weight).ravel()
                yy, xx = (y0 + dy).ravel(), (x0 + dx).ravel()
                ok = (yy >= 0) & (yy < G) & (xx >= 0) & (xx < G)
                np.add.at(num, (yy[ok], xx[ok]), (w * val.ravel())[ok])
                np.add.at(den, (yy[ok], xx[ok]), w[ok])
    return np.where(den > 1e-8, num / np.maximum(den, 1e-9), 0.0)


def fit_chart_gained(I, m, train, test, n_terms, rounds=11):
    """Alternate template, coordinates and gains. Compare against `fit_chart`
    at the SAME `rounds`: the gain models otherwise get extra template
    refinement, which is worth more than the gain itself and was the confound
    that made a one-term gain look valuable."""
    THtr = init_from_moments(I[train], m)
    THte = init_from_moments(I[test], m)
    T = splat(I[train], THtr, m)
    for _ in range(rounds):
        coef, rec = fit_gains(T, THtr, I[train], m, n_terms)
        T = T + splat_gained(I[train] - rec, THtr, coef, m, n_terms)
        THtr = refine(T, THtr, I[train], m)
        THte = refine(T, THte, I[test], m)
    return T, THtr, THte
