"""A ladder of generators for REAL (grey) MNIST 1s.

Each rung is a small set of named knobs that draws an image. The claim tested:
a 1 is any image within a small distance (the tube radius) of the rung's surface.

    rung 'straight'  cx, cy, lean, half-length, half-width, edge softness      (6)
    rung 'bent'      + bend: the stroke's centreline is a parabola             (7)
    rung 'flag'      + a top flag: length and direction                        (9)
    rung 'foot'      + a base bar: half-length                                 (10)

Ink from several strokes combines as a soft union a + b - ab, so nothing exceeds 1.
"""
import numpy as np
from scipy.optimize import least_squares

YY, XX = np.mgrid[0:28, 0:28].astype(np.float64)
PX, PY = XX.ravel(), YY.ravel()

KNOBS = {
    "straight": ["cx", "cy", "lean", "halflen", "halfwidth", "softness"],
    "bent":     ["cx", "cy", "lean", "halflen", "halfwidth", "softness", "bend"],
    "flag":     ["cx", "cy", "lean", "halflen", "halfwidth", "softness", "bend", "flaglen", "flagdir"],
    "foot":     ["cx", "cy", "lean", "halflen", "halfwidth", "softness", "bend", "flaglen", "flagdir", "footlen"],
}
LO = {"cx": 0, "cy": 0, "lean": -1.3, "halflen": 2, "halfwidth": 0.3, "softness": 0.2,
      "bend": -0.25, "flaglen": 0, "flagdir": np.radians(80), "footlen": 0}
HI = {"cx": 27, "cy": 27, "lean": 1.3, "halflen": 16, "halfwidth": 6, "softness": 4,
      "bend": 0.25, "flaglen": 9, "flagdir": np.radians(220), "footlen": 7}
INIT_EXTRA = {"bend": 0.0, "flaglen": 1.0, "flagdir": np.radians(140), "footlen": 1.0}


def seg_dist(ax, ay, bx, by):
    """Distance from every pixel centre to the segment a-b."""
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    t = np.clip(((PX - ax) * vx + (PY - ay) * vy) / max(L2, 1e-9), 0, 1)
    return np.hypot(PX - (ax + t * vx), PY - (ay + t * vy))


def soft(d, r, s):
    return np.clip((r - d) / abs(s) + 0.5, 0, 1)


def union(a, b):
    return a + b - a * b


def centreline(p, n=12):
    """Points along the (possibly bent) stroke from top to bottom."""
    cx, cy, th, L = p["cx"], p["cy"], p["lean"], p["halflen"]
    ux, uy = np.sin(th), np.cos(th)            # th=0: vertical, u points DOWN the image
    nx, ny = uy, -ux                           # perpendicular
    t = np.linspace(-L, L, n + 1)
    k = p.get("bend", 0.0)
    off = k * (t * t - L * L / 3)              # parabola, zero-mean offset so the centre stays put
    return cx + t * ux + off * nx, cy + t * uy + off * ny


def render(pv, rung):
    p = dict(zip(KNOBS[rung], pv))
    r, s = p["halfwidth"], p["softness"]
    xs, ys = centreline(p)
    d = np.min([seg_dist(xs[i], ys[i], xs[i + 1], ys[i + 1]) for i in range(len(xs) - 1)], axis=0)
    img = soft(d, r, s)
    if "flaglen" in p and p["flaglen"] > 1e-6:
        fx, fy = xs[0] + p["flaglen"] * np.cos(p["flagdir"]), ys[0] + p["flaglen"] * np.sin(p["flagdir"])
        img = union(img, soft(seg_dist(xs[0], ys[0], fx, fy), r, s))
    if "footlen" in p and p["footlen"] > 1e-6:
        img = union(img, soft(seg_dist(xs[-1] - p["footlen"], ys[-1], xs[-1] + p["footlen"], ys[-1]), r, s))
    return img


def init_straight(v):
    I = v.reshape(28, 28); m = I.sum()
    cx, cy = (XX * I).sum() / m, (YY * I).sum() / m
    dx, dy = XX - cx, YY - cy
    C = np.array([[(dx * dx * I).sum(), (dx * dy * I).sum()],
                  [(dx * dy * I).sum(), (dy * dy * I).sum()]]) / m
    w, V = np.linalg.eigh(C); ax = V[:, 1]
    if ax[1] < 0: ax = -ax
    L = np.sqrt(3 * w[1])
    return [cx, cy, np.arctan2(ax[0], ax[1]), L, max(m / (4 * L), 0.6), 1.0]


def fit(v, rung, p0=None):
    """(params, residual norm). p0: params of a lower rung to start from."""
    v = np.asarray(v, np.float64)
    names = KNOBS[rung]
    if p0 is None:
        p0 = init_straight(v)
    p0 = list(p0) + [INIT_EXTRA[k] for k in names[len(p0):]]
    lo = np.array([LO[k] for k in names]); hi = np.array([HI[k] for k in names])
    p0 = np.clip(p0, lo + 1e-6, hi - 1e-6)
    r = least_squares(lambda p: render(p, rung) - v, p0, bounds=(lo, hi), xtol=1e-6, ftol=1e-6)
    return r.x, float(np.linalg.norm(r.fun))


def jacobian(pv, rung, eps=1e-3):
    """Numerical 784 x k Jacobian: each column is the image of 'turning one knob'."""
    pv = np.asarray(pv, float)
    J = np.empty((784, len(pv)))
    for j in range(len(pv)):
        d = np.zeros(len(pv)); d[j] = eps
        J[:, j] = (render(pv + d, rung) - render(pv - d, rung)) / (2 * eps)
    return J


def _fit_row(args):
    v, rung, p0 = args
    return fit(v, rung, p0)


def fit_all(X, rung, P0=None, procs=4):
    from multiprocessing import Pool
    args = [(X[i], rung, None if P0 is None else P0[i]) for i in range(len(X))]
    with Pool(procs) as pool:
        out = pool.map(_fit_row, args, chunksize=16)
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])
