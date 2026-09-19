"""Straight 1s: selection, a validity check for new images, and PCA-guided augmentation.

Used by StraightOnes.ipynb. Everything is in raw 784-d pixel space, [0, 1] per pixel.

    stroke_fit(v)     best-fitting ideal straight stroke (a soft-edged capsule) and the
                      L2 distance between it and v -- the "how far from a straight 1" score
    Checker           decides whether a NEW image counts as a straight 1
    pca_stats(Z)      eigenvalues + how many PCs reach 80/90/95/99% of variance
    augment(...)      greedily add valid 1s that shrink the 95% PC count
"""

import numpy as np
from scipy.ndimage import binary_dilation, label, map_coordinates
from scipy.optimize import least_squares

YY, XX = np.mgrid[0:28, 0:28].astype(np.float64)
PX, PY = XX.ravel(), YY.ravel()

THETA = 1.89          # mean nearest-neighbour distance among all MNIST 1s (OnesManifold Part II)
STRAIGHT_CUT = 2.0    # max distance from the best-fitting ideal straight stroke
FRACS = (0.80, 0.90, 0.95, 0.99)


# ---------------------------------------------------------------------------
# The ideal straight stroke
# ---------------------------------------------------------------------------

def render_stroke(p):
    """Soft-edged capsule: centre (cx, cy), lean th (0 = vertical), half-length L,
    half-width r, edge softness s. Returns a flat 784 vector in [0, 1]."""
    cx, cy, th, L, r, s = p
    ux, uy = np.sin(th), np.cos(th)
    dx, dy = PX - cx, PY - cy
    t = np.clip(dx * ux + dy * uy, -L, L)
    d = np.hypot(dx - t * ux, dy - t * uy)
    return np.clip((r - d) / abs(s) + 0.5, 0, 1)


def _init(v):
    I = v.reshape(28, 28); m = I.sum()
    cx, cy = (XX * I).sum() / m, (YY * I).sum() / m
    dx, dy = XX - cx, YY - cy
    C = np.array([[(dx * dx * I).sum(), (dx * dy * I).sum()],
                  [(dx * dy * I).sum(), (dy * dy * I).sum()]]) / m
    w, V = np.linalg.eigh(C); ax = V[:, 1]
    if ax[1] < 0: ax = -ax
    L = np.sqrt(3 * w[1])
    return [cx, cy, np.arctan2(ax[0], ax[1]), L, max(m / (4 * L), 0.6), 1.0]


_LO, _HI = [0, 0, -1.3, 2, 0.3, 0.2], [27, 27, 1.3, 16, 6, 4]


def stroke_fit(v):
    """(params, residual): residual = ||v - best straight stroke||, same units as THETA."""
    v = np.asarray(v, np.float64)
    if v.sum() < 1:
        return np.array(_init(np.ones(784))), np.inf
    p0 = np.clip(_init(v), np.array(_LO) + 1e-6, np.array(_HI) - 1e-6)
    r = least_squares(lambda p: render_stroke(p) - v, p0, bounds=(_LO, _HI))
    return r.x, float(np.linalg.norm(r.fun))


def n_pieces(v, thr=0.2):
    return int(label(np.asarray(v).reshape(28, 28) > thr)[1])


def tv_per_ink(Q):
    """Total variation / ink: sharp strokes score high, blurred ones low."""
    I = np.asarray(Q).reshape(-1, 28, 28)
    tv = np.abs(np.diff(I, axis=1)).sum((1, 2)) + np.abs(np.diff(I, axis=2)).sum((1, 2))
    return tv / np.maximum(I.sum((1, 2)), 1e-9)


def grey_frac(Q):
    """Fraction of inked pixels that are mid-grey (0.2-0.8): the blur signature."""
    Q = np.asarray(Q).reshape(len(Q), -1) if np.ndim(Q) > 1 else np.asarray(Q)[None]
    ink = Q > 0.2
    return ((Q > 0.2) & (Q < 0.8)).sum(1) / np.maximum(ink.sum(1), 1)


def ghost_ink(Q):
    """Ink more than 2 px away from the stroke (pixels > 0.3). Real 1s have almost none;
    a faint second stroke (the cross-fade ghost) or resampling ripple shows up here."""
    Q = np.atleast_2d(Q)
    out = np.empty(len(Q))
    for i, q in enumerate(Q):
        I = q.reshape(28, 28)
        out[i] = I[~binary_dilation(I > 0.3, iterations=2)].sum()
    return out


def nearest(Q, R, Rsq=None):
    """(distance, index) of the nearest row of R for each row of Q."""
    Q = np.atleast_2d(Q)
    Rsq = (R ** 2).sum(1) if Rsq is None else Rsq
    out_d, out_i = np.empty(len(Q)), np.empty(len(Q), int)
    for s in range(0, len(Q), 2000):
        q = Q[s:s + 2000]
        d2 = (q ** 2).sum(1)[:, None] + Rsq[None] - 2 * q @ R.T
        i = d2.argmin(1)
        out_i[s:s + 2000] = i
        out_d[s:s + 2000] = np.sqrt(np.maximum(d2[np.arange(len(q)), i], 0))
    return out_d, out_i


# ---------------------------------------------------------------------------
# Validity: is a new image a straight 1?
# ---------------------------------------------------------------------------

class Checker:
    """Seven checks, each calibrated on real data. A candidate must pass all of them.

    1. cube      every pixel in [0, 1]   (candidates are clipped, so this is by construction)
    2. digit     at least 9 of its 10 nearest images among all 60,000 training digits are 1s
    3. near      within THETA of a real straight 1
    4. straight  best-fit straight stroke residual < STRAIGHT_CUT
    5. pieces    exactly one connected piece of ink
    6. crisp     sharpness (TV/ink) and grey fraction inside the range real straight 1s span
                 (1st-99th percentile) -- rejects blur, which is what averaging produces
    7. clean     ghost ink (ink > 2 px from the stroke) no more than real straight 1s have
                 (99th percentile) -- rejects faint second strokes
    """

    def __init__(self, straight, X_all, y_all):
        self.S = straight; self.Ssq = (straight ** 2).sum(1)
        self.X = X_all; self.Xsq = (X_all ** 2).sum(1); self.y = y_all
        tv, gf = tv_per_ink(straight), grey_frac(straight)
        self.tv_lo, self.tv_hi = np.percentile(tv, [1, 99])
        self.gf_hi = np.percentile(gf, 99)
        self.ghost_hi = np.percentile(ghost_ink(straight), 99)

    def knn_ones(self, Q, k=10):
        Q = np.atleast_2d(Q); out = np.empty(len(Q), int)
        for s in range(0, len(Q), 500):
            q = Q[s:s + 500]
            d2 = (q ** 2).sum(1)[:, None] + self.Xsq[None] - 2 * q @ self.X.T
            nn = np.argpartition(d2, k, axis=1)[:, :k]
            out[s:s + 500] = (self.y[nn] == 1).sum(1)
        return out

    def check(self, Q, full=False):
        """Boolean table [n, 7] and the stroke residuals. Cheap checks run first and the
        slow ones only on survivors, unless full=True (every check on every row --
        for calibration tables)."""
        Q = np.atleast_2d(Q)
        ok = np.zeros((len(Q), 7), bool)
        ok[:, 0] = (Q >= 0).all(1) & (Q <= 1).all(1)
        ok[:, 2] = nearest(Q, self.S, self.Ssq)[0] <= THETA
        tv = tv_per_ink(Q)
        ok[:, 5] = (tv >= self.tv_lo) & (tv <= self.tv_hi) & (grey_frac(Q) <= self.gf_hi)
        ok[:, 4] = [n_pieces(q) == 1 for q in Q]
        ok[:, 6] = ghost_ink(Q) <= self.ghost_hi
        res = np.full(len(Q), np.nan)
        live = np.arange(len(Q)) if full else np.where(ok[:, [0, 2, 4, 5, 6]].all(1))[0]
        if len(live):
            ok[live, 1] = self.knn_ones(Q[live]) >= 9
            live = live if full else live[ok[live, 1]]
            for i in live:
                res[i] = stroke_fit(Q[i])[1]
            ok[live, 3] = res[live] < STRAIGHT_CUT
        return ok, res


CHECK_NAMES = ["in cube", "kNN says 1", "near real", "straight", "1 piece", "crisp", "clean"]


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def pca(Z):
    mu = Z.mean(0)
    w, V = np.linalg.eigh(np.cov((Z - mu).T))
    return mu, np.maximum(w[::-1], 0), V[:, ::-1]


def n_for(w, f):
    return int(np.searchsorted(np.cumsum(w) / w.sum(), f) + 1)


def pca_stats(Z):
    mu, w, V = pca(Z)
    return {f: n_for(w, f) for f in FRACS}, w


# ---------------------------------------------------------------------------
# Candidate new 1s
# ---------------------------------------------------------------------------

def affine(img, rot=0.0, dx=0.0, dy=0.0, sy=1.0, sx=1.0):
    """Rotate (degrees) / shift (px) / stretch a 28x28 image about its ink centroid.
    Bilinear resampling: it cannot overshoot, so unlike cubic it leaves no faint ripple
    of ink around the stroke (which real MNIST never has)."""
    I = img.reshape(28, 28); m = I.sum()
    cx, cy = (XX * I).sum() / m, (YY * I).sum() / m
    a = np.radians(rot); c, s = np.cos(a), np.sin(a)
    X0, Y0 = XX - cx - dx, YY - cy - dy                  # output -> source (inverse map)
    xs = (c * X0 + s * Y0) / sx + cx
    ys = (-s * X0 + c * Y0) / sy + cy
    return np.clip(map_coordinates(I, [ys, xs], order=1, mode="constant"), 0, 1).ravel()


WARPS = ([("rotate", dict(rot=r)) for r in (-4, -2, 2, 4)]
         + [("shift", dict(dx=a, dy=b)) for a, b in ((-1, 0), (1, 0), (0, -1), (0, 1))]
         + [("stretch", dict(sy=s)) for s in (0.92, 1.08)]
         + [("thicken", dict(sx=s)) for s in (0.85, 1.2)])


def warp_pool(src):
    """Hand-plausible variations of each source: small rotations, shifts, stretches."""
    Q, kind, frm = [], [], []
    for i, v in enumerate(src):
        for nm, kw in WARPS:
            Q.append(affine(v, **kw)); kind.append(nm); frm.append(i)
    return np.array(Q), np.array(kind), np.array(frm)


def pca_pool(src, mu, V, k, theta=THETA):
    """Two moves that use the current PCA directly, each capped at THETA of movement.

    trim  -- remove part of the point's small-component (tail) part: x - s * tail(x)
    push  -- move the point further out along its own big-component part
    """
    Z = src - mu
    top = (Z @ V[:, :k]) @ V[:, :k].T
    tail = Z - top
    Q, kind, frm = [], [], []
    tn = np.linalg.norm(tail, axis=1, keepdims=True)
    for s in (0.5, 1.0):
        step = np.minimum(s, theta / np.maximum(tn, 1e-9))
        Q.append(np.clip(src - step * tail, 0, 1)); kind += ["trim"] * len(src)
    un = top / np.maximum(np.linalg.norm(top, axis=1, keepdims=True), 1e-9)
    for s in (0.5, 1.0):
        Q.append(np.clip(src + s * theta * un, 0, 1)); kind += ["push"] * len(src)
    frm = np.tile(np.arange(len(src)), 4)
    return np.vstack(Q), np.array(kind), frm


# ---------------------------------------------------------------------------
# Greedy augmentation
# ---------------------------------------------------------------------------

def tail_share(w, k):
    return w[k:].sum() / w.sum()


def augment(real, pool, checker, min_sep, batch=100, max_rounds=60, patience=6,
            targeted=True, seed=0, n_check=1500, log=print):
    """Add valid straight 1s, `batch` per round, to shrink the 95% PC count.

    pool      dict with Q, kind, frm: a fixed set of candidates (warps), pre-checked valid
    targeted  True: pick the candidates that most lower the variance outside the top k-1
              PCs, k = current 95% count (first-order estimate, then recompute exactly).
              False: pick valid candidates at random -- the control.
    Also each round, trim/push candidates are regenerated from the CURRENT PCA and checked.
    A candidate must sit >= min_sep from every point already in the set, so the set can't
    be stuffed with near-copies.
    Stops when no valid candidate lowers the tail share, or the 95% count has not moved
    for `patience` rounds.
    """
    rng = np.random.default_rng(seed)
    added, akind = np.zeros((0, 784)), []
    cur = real.copy()
    counts, w = pca_stats(cur)
    hist = [dict(n_added=0, **{f"k{int(f*100)}": counts[f] for f in FRACS},
                 share=np.nan, kinds={})]
    used = np.zeros(len(pool["Q"]), bool)
    best_k, stale = counts[0.95], 0
    for rnd in range(max_rounds):
        mu, w, V = pca(cur)
        k = n_for(w, 0.95) - 1                   # the target: one fewer PC
        n = len(cur)
        # fresh PCA-guided candidates from the REAL points (never from added ones)
        share_now = tail_share(w, k)
        tot = w.sum() * (n - 1)

        def gain(Q):                                        # first-order effect of adding one
            Z = Q - mu
            z2 = (Z ** 2).sum(1); t2 = z2 - ((Z @ V[:, :k]) ** 2).sum(1)
            return (share_now * tot + t2) / (tot + z2)

        if targeted:
            # checking is the slow part, so only check the most promising n_check
            Qp, kp, _ = pca_pool(real, mu, V, k)
            gp = gain(Qp)
            shortlist = np.argsort(gp)[:n_check]
            shortlist = shortlist[gp[shortlist] < share_now]
            vp = shortlist[checker.check(Qp[shortlist])[0].all(1)]
        else:
            Qp, kp, vp = np.zeros((0, 784)), np.array([], str), np.array([], int)
        n_pca_valid = len(vp)
        CQ = np.vstack([pool["Q"][~used], Qp[vp]])
        CK = np.concatenate([pool["kind"][~used], kp[vp]])
        CI = np.concatenate([np.where(~used)[0], -np.ones(len(vp), int)])
        if not len(CQ):
            break
        new_share = gain(CQ)
        if targeted:
            order = np.argsort(new_share)
            order = order[new_share[order] < share_now]
        else:
            order = rng.permutation(len(CQ))
        # take the best `batch`, keeping min_sep from the set and from each other
        dset = nearest(CQ[order], cur)[0] if len(order) else np.array([])
        order = order[dset >= min_sep]
        pick = []
        for i in order:
            if len(pick) >= batch: break
            if pick and np.linalg.norm(CQ[pick] - CQ[i], axis=1).min() < min_sep:
                continue
            pick.append(i)
        if not pick:
            log(f"round {rnd}: no valid candidate lowers the tail share -- stop")
            break
        pick = np.array(pick)
        added = np.vstack([added, CQ[pick]]); akind += list(CK[pick])
        used[CI[pick][CI[pick] >= 0]] = True
        cur = np.vstack([real, added])
        counts, w = pca_stats(cur)
        kinds = {kk: int((CK[pick] == kk).sum()) for kk in np.unique(CK[pick])}
        hist.append(dict(n_added=len(added), **{f"k{int(f*100)}": counts[f] for f in FRACS},
                         share=tail_share(w, k), kinds=kinds, pca_valid=n_pca_valid))
        log(f"round {rnd:2d}: +{len(pick):3d} (total {len(added):5d})  "
            f"PCs 90/95/99 = {counts[0.9]}/{counts[0.95]}/{counts[0.99]}  {kinds}")
        if counts[0.95] < best_k:
            best_k, stale = counts[0.95], 0
        else:
            stale += 1
            if stale >= patience:
                log(f"95% count stuck at {best_k} for {patience} rounds -- stop")
                break
    return added, np.array(akind), hist
