"""Stroke-drawn 1s with a chosen number of knobs, for calibrating dimension counts.

A dimension estimator applied to MNIST 1s returns some number. To read that
number we need to know what the same estimator returns, with the same number of
images, on image families whose number of knobs is KNOWN. The generated 1s
(`grey_ones`) give one such family (five knobs). This module gives a ladder of
them: 5, 8, 12 or 16 knobs, all drawn the same way, all looking roughly like
handwritten 1s.

Drawing: a centre line runs from the bottom of the stroke to the top. Every
pixel's ink is darkness * sigmoid((half_width - distance_to_line) / softness),
taking the line point that gives the most ink (sigmoid is increasing, so the
maximum is taken inside it). Optional extras are a flag at
the top (the hook of a European 1) and a bar at the bottom (a base serif).

Knob order (the first K vary, the rest sit at their "off" value):

     1 cx         horizontal centre, px
     2 cy         vertical centre, px
     3 height     stroke length, px
     4 width      stroke width, px
     5 lean       degrees, positive tilts the top right
     6 bend       sideways bow at the middle, px
     7 wiggle     S-shaped sideways offset, px
     8 taper      top width / bottom width
     9 bulge      extra width at the middle, px
    10 hook_len   length of the top flag, px
    11 hook_ang   flag direction, degrees below horizontal-left
    12 bar_len    length of the base bar, px
    13 bar_shift  base bar centre offset from the stroke foot, px
    14 darkness   peak ink
    15 softness   edge blur, px
    16 hook_bend  curl of the top flag, px

`make_wobbly` adds many small knobs on top (sideways and width wobbles along
the stroke), to mimic a family with a few big knobs and many tiny ones.

Flags and bars are always present with a length well above zero whenever their
angle or offset knobs vary, so that no knob goes invisible somewhere inside its
range (that would lower the true dimension locally).
"""

import numpy as np

NAMES = ["cx", "cy", "height", "width", "lean", "bend", "wiggle", "taper",
         "bulge", "hook_len", "hook_ang", "bar_len", "bar_shift", "darkness",
         "softness", "hook_bend"]
RANGES = np.array([
    [12.5, 15.5], [12.5, 15.5], [16.0, 21.0], [1.8, 4.2], [-10.0, 30.0],
    [-2.0, 2.0], [-1.2, 1.2], [0.7, 1.4], [-0.6, 0.6],
    [2.5, 6.0], [10.0, 50.0], [3.0, 8.0], [-1.5, 1.5],
    [0.75, 1.0], [0.25, 0.5], [-1.5, 1.5]])
# value used when a knob is switched off (not among the first K)
OFF = np.array([14.0, 14.0, 18.5, 3.0, 10.0, 0.0, 0.0, 1.0, 0.0,
                0.0, 30.0, 0.0, 0.0, 1.0, 0.4, 0.0])
N_LINE = 64      # points along the main stroke
N_EXTRA = 24     # points along a flag or bar

_yy, _xx = np.mgrid[0:28, 0:28] + 0.5
PIX = np.stack([_xx.ravel(), _yy.ravel()], 1)          # (784, 2) x, y


def sample(n, k, rng):
    """n random settings with the first k knobs uniform in their ranges."""
    P = np.tile(OFF, (n, 1))
    lo, hi = RANGES[:k, 0], RANGES[:k, 1]
    P[:, :k] = rng.uniform(lo, hi, (n, k))
    return P


def _points(p):
    """Centre-line points (m, 2) and their half widths (m,) for one setting."""
    (cx, cy, h, w, lean, bend, wig, taper, bulge, hl, ha, bl, bs,
     _dark, _soft, hb) = p[:16]
    t = np.linspace(-1.0, 1.0, N_LINE)                 # -1 bottom, +1 top
    s = np.tan(np.radians(lean))
    y = cy - t * h / 2                                  # image y grows downward
    x = cx + s * (t * h / 2) + bend * (1 - t ** 2) + wig * (t ** 3 - t) * 2.6
    wb = w / (1 + taper) * 2                            # bottom width so mean stays w
    half = 0.5 * (wb + (wb * taper - wb) * (t + 1) / 2 + bulge * (1 - t ** 2))
    tiny = p[16:]
    if len(tiny):                                       # small wobbles, see make_wobbly
        nm = len(tiny) // 2
        modes = np.sin(np.outer(np.arange(2, 2 + nm), np.pi * (t + 1) / 2))   # (nm, m)
        x = x + tiny[:nm] @ modes
        half = half + 0.5 * tiny[nm:2 * nm] @ modes
    pts, halves = [np.c_[x, y]], [half]
    wt = wb * taper * 0.5
    if hl > 0:                                          # flag from the top, down-left
        u = np.linspace(0, 1, N_EXTRA)
        a = np.radians(ha)
        dx, dy = -np.cos(a), np.sin(a)                  # left and down
        fx = x[-1] + u * hl * dx + hb * (u - u ** 2) * 2 * (-dy)
        fy = y[-1] + u * hl * dy + hb * (u - u ** 2) * 2 * dx
        pts.append(np.c_[fx, fy]); halves.append(np.full(N_EXTRA, wt * 0.85))
    if bl > 0:                                          # base bar through the foot
        u = np.linspace(-0.5, 0.5, N_EXTRA)
        pts.append(np.c_[x[0] + bs + u * bl, np.full(N_EXTRA, y[0])])
        halves.append(np.full(N_EXTRA, wb * 0.4))
    return np.concatenate(pts), np.concatenate(halves)


def render(P, batch=64):
    """P (n, 16) -> images (n, 28, 28) float32 in [0, 1]."""
    P = np.atleast_2d(P)
    out = np.empty((len(P), 784), np.float32)
    pix = PIX.astype(np.float32)
    for b in range(0, len(P), batch):
        chunk = P[b:b + batch]
        pts, half = zip(*(_points(p) for p in chunk))
        pts = np.stack(pts).astype(np.float32)              # (B, m, 2)
        half = np.stack(half).astype(np.float32)            # (B, m)
        dx = pix[None, :, 0, None] - pts[:, None, :, 0]          # (B, 784, m)
        dy = pix[None, :, 1, None] - pts[:, None, :, 1]
        d2 = dx * dx + dy * dy
        # most ink comes from the largest half_width - distance; take it first
        gap = (half[:, None, :] - np.sqrt(d2)).max(2)        # (B, 784)
        soft = chunk[:, 14:15].astype(np.float32)
        dark = chunk[:, 13:14].astype(np.float32)
        out[b:b + batch] = dark / (1 + np.exp(-gap / soft))
    return out.reshape(-1, 28, 28)


def make(n, k, seed=0):
    rng = np.random.default_rng(seed)
    return render(sample(n, k, rng))


def make_wobbly(n, k, n_tiny, amp, seed=0):
    """k big knobs plus n_tiny SMALL knobs: sideways and width wobbles of the
    centre line (sine modes 2, 3, ... along the stroke), mode m uniform in
    [-amp, amp] * 2/m px. True dimension k + n_tiny, but the tiny knobs only show
    once images are packed closer together than about amp."""
    rng = np.random.default_rng(seed)
    P = sample(n, k, rng)
    m = np.arange(2, 2 + n_tiny // 2)
    scale = np.tile(2.0 / m, 2)                 # higher, finer modes are smaller
    T = rng.uniform(-amp, amp, (n, 2 * len(m))) * scale
    return render(np.c_[P, T])
