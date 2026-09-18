"""Curves through pixel space: geodesics along the 1-manifold vs straight lines.

A path between two 1s can be followed two ways -- hopping between real neighbours
(along the manifold) or interpolating pixel values (straight through the space).
The first slides ink; the second cross-fades. See OnesManifold.ipynb.

A curve through 784 dimensions is not as unwieldy as it sounds: projected onto its
own principal axes it is nearly flat (3 axes hold ~93%), so it plots on an ordinary
3-D graph and a cubic in that basis needs only twelve coefficients.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


def neighbour_graph(pts, radius):
    """Weighted graph joining points closer than `radius`."""
    s = (pts ** 2).sum(1)
    D = np.sqrt(np.maximum(s[:, None] + s[None, :] - 2 * pts @ pts.T, 0))
    np.fill_diagonal(D, 0)
    D = (D + D.T) / 2
    return D, csr_matrix(np.where((D <= radius) & (~np.eye(len(pts), dtype=bool)), D, 0))


def geodesic(G, D, a, b):
    """Node sequence of the shortest path from a to b through the graph, or None."""
    _, pred = dijkstra(G, indices=[a], directed=False, return_predecessors=True)
    pred = pred[0]
    p = [b]
    while p[-1] != a:
        nxt = pred[p[-1]]
        if nxt < 0:
            return None
        p.append(int(nxt))
    return p[::-1]


def arclength_param(P):
    """Parameterise a polyline by cumulative arc length, normalised to [0, 1]."""
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return s / s[-1]


def fit_curve(P, degree=3, n_axes=3):
    """Fit a polynomial curve to a path, in the path's own principal axes.

    Returns (coeffs, basis, mean, t, captured) where `coeffs` is (degree+1, n_axes) --
    twelve numbers for a cubic in three axes -- and `captured` is the fraction of the
    path's variance those axes hold.
    """
    t = arclength_param(P)
    mu = P.mean(0)
    Z = P - mu
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    basis = Vt[:n_axes]
    captured = (S[:n_axes] ** 2).sum() / (S ** 2).sum()
    C = Z @ basis.T
    coeffs, *_ = np.linalg.lstsq(np.vander(t, degree + 1, increasing=True), C, rcond=None)
    return coeffs, basis, mu, t, captured


def eval_curve(coeffs, basis, mu, ts):
    """Evaluate the fitted curve back in full pixel space. ts>1 extrapolates.

    Extrapolation fails quickly and the higher the degree the faster: a cubic is
    already off the manifold by t=1.1. To keep walking, take a small step along the
    tangent and re-project onto the data instead.
    """
    ts = np.atleast_1d(ts)
    return np.vander(ts, coeffs.shape[0], increasing=True) @ coeffs @ basis + mu


def frenet(curve, ts):
    """Tangent, normal and curvature along a sampled curve, in the ambient space.

    The magnitudes (curvature) plot on an ordinary 2-D graph. The directions are
    vectors in pixel space, so for image data each one *is* an image -- which is how
    you show which way a curve bends without leaving the original coordinates.
    """
    d1 = np.gradient(curve, ts, axis=0)
    d2 = np.gradient(d1, ts, axis=0)
    speed = np.linalg.norm(d1, axis=1, keepdims=True)
    T = d1 / np.maximum(speed, 1e-12)
    N = d2 - (d2 * T).sum(1, keepdims=True) * T
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    kappa = ((d2 * N).sum(1)) / np.maximum(speed[:, 0] ** 2, 1e-12)
    return T, N, kappa


def straight_line(A, B, k):
    """The other route: interpolate pixel values. Shorter, and further from real data."""
    return np.array([(1 - t) * A + t * B for t in np.linspace(0, 1, k)])


# ---------------------------------------------------------------------------
# Arcs between two fixed images.
#
# Three points pin a quadratic, so with A and B held fixed the whole family of
# arcs between them is parameterised by one vector: where the halfway point sits.
# Splitting that vector into a magnitude (how much it bulges) and a direction
# (which way) separates two effects that look similar and are not:
#
#   bulge SIZE      controls how much the stroke rotates rather than cross-fades.
#                   More bulge always removes ghosting -- monotonically, out to
#                   20x -- but always costs distance from real data. A single arc
#                   can buy sharpness or proximity, never both. The manifold gets
#                   both, which is the evidence that it is not an arc at all.
#
#   bulge DIRECTION is a needle. The perpendicular space has 782 dimensions; of
#                   200 random directions none came within 4x of the manifold's
#                   own. Rotating away from it by 45 degrees already doubles the
#                   distance to real data.
# ---------------------------------------------------------------------------


def arc_through(A, B, M, ts):
    """Quadratic Bezier from A to B passing through M at t=0.5.

    The control point that achieves this is 2M - (A+B)/2, so `M` is literally
    "where the curve is halfway". M = (A+B)/2 gives the straight line.
    """
    P1 = 2 * np.asarray(M) - (np.asarray(A) + np.asarray(B)) / 2
    ts = np.atleast_1d(ts)[:, None]
    return (1 - ts) ** 2 * A + 2 * ts * (1 - ts) * P1 + ts ** 2 * B


def bulge_of(A, B, M):
    """Split 'where the halfway point sits' into (magnitude, unit direction).

    Only the component perpendicular to the chord matters -- a component along
    the chord just re-times the curve without changing its shape.
    """
    chord = np.asarray(B) - np.asarray(A)
    e = chord / np.linalg.norm(chord)
    dev = np.asarray(M) - (np.asarray(A) + np.asarray(B)) / 2
    dev = dev - np.dot(dev, e) * e
    h = np.linalg.norm(dev)
    return h, dev / max(h, 1e-12)


def ghosting(frames, lo=0.2, hi=0.8, ink=0.05):
    """Fraction of the ink sitting at half strength -- the cross-fade signature.

    A real stroke is mostly 0 or mostly 1. An image made by averaging two strokes
    is full of mid-greys. Calibration: a real MNIST 1 scores ~0.34 (its own
    anti-aliasing), the manifold route ~0.39, a straight line ~0.54.
    """
    frames = np.clip(np.atleast_2d(frames), 0, 1)
    out = []
    for f in frames:
        v = f[f > ink]
        out.append(0.0 if not len(v) else float(((v >= lo) & (v <= hi)).mean()))
    return np.array(out)
