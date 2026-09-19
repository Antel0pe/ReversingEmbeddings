"""Explainable equations for a ladder of increasingly free straight 1s.

Each rung is DEFINED by a generator -- a function that draws a valid 1 from a few
knobs -- and then described by an equation E(x) = sum of small, named checks.
Every check returns a non-negative violation count, so E(x) = 0 exactly when
every check passes. The claim tested for each rung: E(x) = 0  <=>  the
generator can draw x.

    rung 1  MOVE   slides anywhere on the canvas          height 8, width 2
    rung 2  SIZE   ... and any height 5-10, width 1-3
    rung 3  TILT   ... and leans up to 23 degrees either way, still straight

Images are 10x10 and binary (0 = off, 1 = on). Tilt is drawn the way a pixel
grid forces it to be: each row is one run of `w` pixels, and the run's left
edge follows a straight line rounded down to whole pixels,

    left edge of row k  =  s0 + floor(a*k + b),     |a| <= tan(23 deg)

so a lean shows up as a staircase of one-pixel sideways steps.
"""

import itertools
import numpy as np
from scipy.optimize import linprog

H = W = 10
AMAX = float(np.tan(np.radians(23.0)))      # 0.4245 columns per row

RUNGS = {
    "move": dict(h=(8, 8), w=(2, 2), tilt=False),
    "size": dict(h=(5, 10), w=(1, 3), tilt=False),
    "tilt": dict(h=(5, 10), w=(1, 3), tilt=True),
}


# ---------------------------------------------------------------------------
# The generator: the DEFINITION of a valid 1 on each rung
# ---------------------------------------------------------------------------

def draw(top, height, width, lefts):
    x = np.zeros((H, W), np.int8)
    for k, s in enumerate(lefts):
        x[top + k, s:s + width] = 1
    return x


def is_straight(steps, amax=AMAX):
    """Ground truth: can floor(a*k + b) with |a| <= amax produce this staircase?

    `steps` are the row-to-row moves of the left edge. The staircase is
    achievable iff the linear inequalities  S_k <= a*k + b < S_k + 1  have a
    solution, which is a two-variable feasibility problem.
    """
    S = np.concatenate([[0], np.cumsum(steps)])
    k = np.arange(len(S))
    eps = 1e-4       # 'strictly below the next pixel'; must exceed the LP solver's own
                     # tolerance (1e-7), and is far below the nearest slope breakpoint
                     # to 23 degrees (3/7 = 0.4286, a gap of 0.004)
    A = np.vstack([np.column_stack([-k, -np.ones_like(k)]),        # a k + b >= S_k
                   np.column_stack([k, np.ones_like(k)])])         # a k + b <= S_k + 1 - eps
    ub = np.concatenate([-S, S + 1 - eps])
    r = linprog([0, 0], A_ub=A, b_ub=ub, bounds=[(-amax, amax), (0, 1 - eps)], method="highs")
    return r.status == 0


_WORDS = {}


def straight_words(n):
    """Every staircase of n steps that a line of lean <= 23 degrees can make."""
    if n not in _WORDS:
        _WORDS[n] = [w for w in itertools.product((-1, 0, 1), repeat=n) if is_straight(w)]
    return _WORDS[n]


def valid_set(rung):
    cfg = RUNGS[rung]
    out = set()
    for h in range(cfg["h"][0], cfg["h"][1] + 1):
        words = straight_words(h - 1) if cfg["tilt"] else [(0,) * (h - 1)]
        for w in range(cfg["w"][0], cfg["w"][1] + 1):
            for word in words:
                rel = np.concatenate([[0], np.cumsum(word)])
                for s0 in range(-rel.min(), W - w - rel.max() + 1):
                    for top in range(0, H - h + 1):
                        out.add(draw(top, h, w, s0 + rel).tobytes())
    return out


# ---------------------------------------------------------------------------
# Reading an image's structure (used by the checks)
# ---------------------------------------------------------------------------

def rows_of(x):
    inked = np.where(x.any(1))[0]
    runs = [int(((x[r] == 1) & (np.concatenate([[0], x[r][:-1]]) == 0)).sum()) for r in inked]
    widths = [int(x[r].sum()) for r in inked]
    lefts = [int(np.argmax(x[r])) for r in inked]
    return inked, runs, widths, lefts


def row_steps(x):
    _, _, _, lefts = rows_of(x)
    return np.diff(lefts)


# ---------------------------------------------------------------------------
# The checks. Each returns 0 when satisfied and a positive count otherwise.
# ---------------------------------------------------------------------------

def one_block_of_rows(x):
    """Inked rows form ONE unbroken block (also rejects a blank image)."""
    on = x.any(1).astype(int)
    blocks = int(((on == 1) & (np.concatenate([[0], on[:-1]]) == 0)).sum())
    return (blocks - 1) ** 2


def height_in(lo, hi):
    def check(x):
        h = int(x.any(1).sum())
        return max(0, lo - h) + max(0, h - hi)
    check.__name__ = f"height_{lo}_to_{hi}"
    return check


def one_run_per_row(x):
    """Every inked row is a single solid run of pixels."""
    _, runs, _, _ = rows_of(x)
    return sum((r - 1) ** 2 for r in runs)


def same_width_every_row(x):
    _, _, widths, _ = rows_of(x)
    return sum((a - b) ** 2 for a, b in zip(widths, widths[1:]))


def width_in(lo, hi):
    def check(x):
        _, _, widths, _ = rows_of(x)
        if not widths:
            return 0
        return max(0, lo - widths[0]) + max(0, widths[0] - hi)
    check.__name__ = f"width_{lo}_to_{hi}"
    return check


def no_sideways_steps(x):
    """Each row starts in the same column as the row above: perfectly upright."""
    return int((row_steps(x) ** 2).sum())


def leans_one_way(x):
    """Never steps left AND right: a straight line cannot change direction."""
    d = row_steps(x)
    return int(min((d > 0).sum(), (d < 0).sum()))


def leans_evenly(x):
    """Balance: any two stretches of the same length step sideways the same number
    of times, give or take one. This is what makes a staircase straight rather
    than bent -- a bend packs its steps into one end."""
    s = np.abs(row_steps(x))
    pen = 0
    for L in range(1, len(s)):
        c = np.convolve(s, np.ones(L, int), "valid")
        pen += max(0, int(c.max() - c.min()) - 1)
    return pen


def lean_at_most(amax=AMAX):
    def check(x):
        """No stretch of L rows steps sideways more than floor(0.42 L) + 1 times.
        The +1 is the rounding of a real line onto whole pixels."""
        s = np.abs(row_steps(x))
        pen = 0
        for L in range(1, len(s) + 1):
            c = np.convolve(s, np.ones(L, int), "valid")
            pen += max(0, int(c.max()) - (int(np.floor(amax * L)) + 1))
        return pen
    check.__name__ = "lean_at_most_23deg"
    return check


def checks(rung):
    cfg = RUNGS[rung]
    base = [one_block_of_rows, height_in(*cfg["h"]), one_run_per_row,
            same_width_every_row, width_in(*cfg["w"])]
    if not cfg["tilt"]:
        return base + [no_sideways_steps]
    return base + [leans_one_way, leans_evenly, lean_at_most()]


def E(x, rung):
    return sum(c(x) for c in checks(rung))


def breakdown(x, rung):
    return {c.__name__: c(x) for c in checks(rung)}
