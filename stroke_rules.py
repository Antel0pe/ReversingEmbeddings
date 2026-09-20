"""Rule 3 and Rule 4: the ink must BE a stroke, and the stroke must have a minimum width.

The first two rules (one connected piece, at least `solid_frac` of ink fully black)
are statistical -- they constrain how much of what kind of pixel there is. They do
not stop the ink from being a tangle. These two are structural:

    STROKE     the skeleton of the ink is a SIMPLE PATH: one piece, exactly two
               ends, no junctions, no loops. A 1 is a line that was thickened;
               a scribble is not.
    WIDTH      every point along that path carries at least `w_min` pixels of
               thickness, so the stroke cannot taper to nothing.

Skeletonisation is Zhang-Suen thinning, written out here because skimage is not
in this environment. Thickness at a skeleton pixel is twice the Euclidean
distance from that pixel to the nearest background pixel.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, label

NB8 = np.ones((3, 3))
# P2..P9 = N, NE, E, SE, S, SW, W, NW  (clockwise from north)
_OFF = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def _neighbours(I):
    """Stack of the 8 neighbours P2..P9 of every pixel, zero-padded at the border."""
    P = np.pad(I.astype(np.int8), 1)
    return np.stack([P[1 + dr:1 + dr + I.shape[0], 1 + dc:1 + dc + I.shape[1]]
                     for dr, dc in _OFF])


def skeletonize(I):
    """Zhang-Suen thinning. I is a boolean 2-D array; returns a boolean skeleton."""
    I = I.astype(bool).copy()
    while True:
        changed = False
        for step in (0, 1):
            N = _neighbours(I)
            B = N.sum(0)
            seq = np.concatenate([N, N[:1]], 0)
            A = ((seq[:-1] == 0) & (seq[1:] == 1)).sum(0)
            P2, P3, P4, P5, P6, P7, P8, P9 = N
            if step == 0:
                c = (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                c = (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
            kill = I & (B >= 2) & (B <= 6) & (A == 1) & c
            if kill.any():
                I[kill] = False; changed = True
        if not changed:
            return I


def skeleton_stats(I):
    """(n_pieces, n_endpoints, n_junctions, n_loops) of the skeleton of I."""
    S = skeletonize(I)
    if S.sum() == 0:
        return 0, 0, 0, 0
    deg = (_neighbours(S).sum(0)) * S
    ends = int(((deg == 1) & S).sum())
    junc = int(((deg >= 3) & S).sum())
    npieces = int(label(S, NB8)[1])
    V = int(S.sum())
    E = int((_neighbours(S).sum(0) * S).sum() // 2)      # each edge counted twice
    loops = E - V + npieces
    return npieces, ends, junc, loops


def is_stroke(I):
    """The skeleton is a simple path: one piece, two ends, no junctions, no loops."""
    n, ends, junc, loops = skeleton_stats(I)
    if int(I.sum()) == 1:
        return True                     # a single pixel is a degenerate but legal stroke
    return n == 1 and junc == 0 and loops <= 0 and ends == 2


def min_width(I):
    """Smallest thickness anywhere along the stroke, in pixels."""
    I = I.astype(bool)
    if not I.any():
        return 0.0
    S = skeletonize(I)
    if not S.any():
        return 0.0
    d = distance_transform_edt(np.pad(I, 1))[1:-1, 1:-1]
    return float(2 * d[S].min())


# ---------------------------------------------------------------------------
# generating things that obey the stroke rule by construction
# ---------------------------------------------------------------------------
N = 28
STEPS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
STEPS8 = STEPS4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def walk(n_steps, rng, margin=3, persistence=0.0, diag=True):
    """A self-avoiding walk whose pixel set is a simple 8-connected path.

    A new pixel is accepted only if the ONLY already-visited pixel in its
    8-neighbourhood is the current head -- that is what stops the path from
    touching itself and creating a junction or a loop.

    persistence > 0 biases the walk toward carrying straight on, which makes
    long smooth strokes instead of tight tangles.
    """
    I = np.zeros((N, N), bool)
    r, c = rng.integers(margin, N - margin, 2)
    I[r, c] = True
    steps = STEPS8 if diag else STEPS4
    last = None
    for _ in range(n_steps - 1):
        cand, wts = [], []
        for dr, dc in steps:
            a, b = r + dr, c + dc
            if not (margin <= a < N - margin and margin <= b < N - margin): continue
            if I[a, b]: continue
            nb = I[max(a-1,0):a+2, max(b-1,0):b+2].sum()
            if nb != 1: continue                       # touches only the head
            cand.append((a, b, dr, dc))
            w = 1.0
            if last is not None and persistence:
                align = (dr * last[0] + dc * last[1]) / (np.hypot(dr, dc) * np.hypot(*last))
                w = np.exp(persistence * align)
            wts.append(w)
        if not cand: break
        wts = np.array(wts); i = rng.choice(len(cand), p=wts / wts.sum())
        a, b, dr, dc = cand[i]
        I[a, b] = True; r, c, last = a, b, (dr, dc)
    return I


def thicken(I, radius):
    """Dilate the path with a Euclidean disk so it carries a minimum width."""
    if radius <= 0:
        return I
    d = distance_transform_edt(~I)
    return d <= radius


def shade(X_bool, solid_frac, rng):
    """Grey the outermost pixels, keeping >= solid_frac of the ink fully black."""
    X = X_bool.astype(np.float64)
    idx = np.flatnonzero(X_bool)
    n_grey = int(round(len(idx) * (1 - solid_frac)))
    if n_grey:
        d = distance_transform_edt(np.pad(X_bool, 1))[1:-1, 1:-1].ravel()[idx]
        pick = idx[np.argsort(d)[:n_grey]]             # shallowest pixels = the edge
        X.ravel()[pick] = rng.uniform(0.15, 0.94, n_grey)
    return X


def long_walk(min_len, rng, persistence=0.0, tries=200, margin=3):
    """A self-avoiding walk that actually reaches min_len pixels (it can get stuck)."""
    best = None
    for _ in range(tries):
        I = walk(min_len + 12, rng, margin=margin, persistence=persistence)
        if best is None or I.sum() > best.sum():
            best = I
        if I.sum() >= min_len:
            return I
    return best


def core_and_halo(path, r_core, r_out, rng):
    """Black core + grey halo, the way MNIST's own anti-aliasing works.

    The CORE (value 1.0) is the path thickened to r_core -- this is the visible
    ink, and it is what the stroke rules are checked on. The HALO is the shell
    out to r_out, shaded by depth so the outermost pixels are faintest. Nothing
    in the halo reaches 0.5, so it cannot change the structure of the visible ink.
    """
    core = thicken(path, r_core)
    outer = thicken(path, r_out)
    halo = outer & ~core
    X = core.astype(np.float64)
    if halo.any():
        d = distance_transform_edt(~path.astype(bool))[halo]
        t = (d - r_core) / max(r_out - r_core, 1e-9)            # 0 at the core, 1 at the rim
        X[halo] = np.clip(0.46 - 0.30 * t, 0.06, 0.49)
        X[halo] *= rng.uniform(0.75, 1.0, halo.sum())
    return X


def mean_width(I):
    """Average thickness of the stroke: inked area divided by skeleton length.

    More useful than the minimum on a 28x28 grid, where the minimum saturates
    at 2 for anything one pixel wide.
    """
    I = np.asarray(I).astype(bool)
    S = skeletonize(I)
    n = int(S.sum())
    return float(I.sum() / n) if n else 0.0
