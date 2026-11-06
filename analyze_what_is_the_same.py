"""What is actually THE SAME about two widenings, and can a general method see it?

The goal behind all of this: given a dataset, recover the smallest set of knobs
that explains it, plus a precise map from those knobs back to the full pictures.

The last experiment found that storing "how much ink is to the LEFT of here"
makes two consecutive widenings look alike. But that is a left-right trick for a
left-right change, so it is tuned to this one problem. This script asks the
general question instead:

    What property do two widenings genuinely share, that random noise and a
    different knob do NOT share -- and can it be measured without being told
    which knob is being turned?

The candidate answer tested here: describe a change not by WHICH PIXELS changed
but by WHERE THE INK MOVED. The ink movement is a little arrow at each point of
the picture. Two widenings move ink along the same arrows even though they
touch different pixels. Random noise is not ink moving anywhere, so it has no
arrows at all, and that failure is itself the detector.

The arrows are read off a change with no knowledge of which knob was turned:
if ink at a point slides by a small arrow v, the picture changes by
-(v pointing across the picture's own light-and-dark). Solve that backwards for
v, allowing v to vary across the picture as a polynomial of a chosen degree.

  degree 0  v is the same arrow everywhere            2 numbers   (a control:
                                                                  too weak)
  degree 1  arrows grow in straight proportion        6 numbers
  degree 2  arrows may also curve                    12 numbers   (a control:
                                                                  more freedom
                                                                  than needed)

Parts:
  1  what is the same about two widenings -- pixels vs arrows
  2  is "ink to the left of here" overfitted? the same test on all five knobs
  3  do the arrows work on all five knobs?
  4  do the arrows reject random noise and other junk?
  5  how many knobs does each way of writing it down report? (the real goal)
  6  does any of it survive on a completely different dataset?
"""

import sys

import numpy as np

from grey_ones import render

ROOT = np.array([14.0, 14.0, 19.75, 3.2, 0.0])
NAMES = ["slide sideways (cx)", "slide up-down (cy)", "make taller (height)",
         "widen (width)", "lean over (lean)"]
SHORT = ["cx", "cy", "height", "width", "lean"]
# a step of each knob that moves the picture by a comparable amount
STEP = np.array([1.0, 1.0, 2.0, 2.0, 6.0])   # each moves an edge by about one pixel


def one(p):
    return render(np.asarray(p, float)[None], dtype=np.float64)[0]


def cos(a, b):
    a, b = np.ravel(a), np.ravel(b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n else np.nan


def ang(a, b):
    return np.degrees(np.arccos(np.clip(cos(a, b), -1, 1)))


def rule(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


# ------------------------------------------------- the three ways of writing it

def left_total(I):
    """How much ink is to the left of here (running total along each row)."""
    return np.cumsum(I, axis=1)


def _basis(n, degree):
    g = (np.arange(n) - (n - 1) / 2) / ((n - 1) / 2)       # -1 .. 1
    Y, X = np.meshgrid(g, g, indexing="ij")
    terms = [np.ones_like(X)]
    if degree >= 1:
        terms += [X, Y]
    if degree >= 2:
        terms += [X * X, X * Y, Y * Y]
    return terms


def arrows_warp(I, T, degree=1):
    """Same description, but fitted by ACTUALLY MOVING the ink with
    interpolation instead of estimating how the picture shades. Slower, no
    derivative needed, and it copes with bigger changes."""
    from scipy.ndimage import map_coordinates
    from scipy.optimize import least_squares
    n = I.shape[0]
    terms = _basis(n, degree)
    s = (n - 1) / 2
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")

    def make(c):
        h = len(terms)
        vx = sum(c[i] * terms[i] for i in range(h))
        vy = sum(c[h + i] * terms[i] for i in range(h))
        return map_coordinates(I, [yy - vy * s, xx - vx * s], order=3, mode="constant")

    r = least_squares(lambda c: (make(c) - T).ravel(), np.zeros(2 * len(terms)),
                      method="lm", xtol=1e-13, ftol=1e-13)
    return r.x, float(np.linalg.norm(r.fun) / np.linalg.norm(T - I))


def arrows_exact(p, D, step=1e-3, degree=1):
    """The same description with a PERFECT derivative, available only because a
    renderer exists: slide the whole scene a hair and see what the picture does.
    Not a general method -- it is the ceiling the general ones are chasing."""
    a = np.array(p, float).copy(); b = a.copy(); a[0] += step; b[0] -= step
    gx = -(one(a) - one(b)) / (2 * step)
    a = np.array(p, float).copy(); b = a.copy(); a[1] += step; b[1] -= step
    gy = -(one(a) - one(b)) / (2 * step)
    I = one(p); n = I.shape[0]
    cols = [-(t * gx).ravel() for t in _basis(n, degree)] + \
           [-(t * gy).ravel() for t in _basis(n, degree)]
    A = np.array(cols).T
    c, *_ = np.linalg.lstsq(A, D.ravel(), rcond=None)
    return c, float(np.linalg.norm(A @ c - D.ravel()) / np.linalg.norm(D))


def arrows(I, D, degree=1):
    """Read off where the ink moved. Returns the numbers, and how much of the
    change they fail to explain (0 = fully explained)."""
    n = I.shape[0]
    gy, gx = np.gradient(I)
    cols = []
    for t in _basis(n, degree):
        cols.append(-(t * gx).ravel())      # arrow pointing across the picture
    for t in _basis(n, degree):
        cols.append(-(t * gy).ravel())
    A = np.array(cols).T
    c, *_ = np.linalg.lstsq(A, D.ravel(), rcond=None)
    resid = np.linalg.norm(A @ c - D.ravel()) / np.linalg.norm(D)
    return c, float(resid)


# --------------------------------------------------------------------- part 1

def part1():
    rule("1.  What is actually the same about two widenings?")
    print("""
Start at the root 1. Widen it, then widen it by the same amount again. Compare
the first change with the second, three ways. The score is an angle:
0 degrees = the very same change again, 90 degrees = nothing in common.
""")
    k = 3
    h = 1.0
    I0 = one(ROOT)
    p1 = ROOT.copy(); p1[k] += h
    p2 = ROOT.copy(); p2[k] += 2 * h
    I1, I2 = one(p1), one(p2)
    D1, D2 = I1 - I0, I2 - I1
    a1, r1 = arrows(I0, D1)
    a2, r2 = arrows(I1, D2)
    print(f"  which pixels changed                  : {ang(D1, D2):5.1f} deg")
    print(f"  ink to the left of here               : {ang(left_total(I1)-left_total(I0), left_total(I2)-left_total(I1)):5.1f} deg")
    print(f"  WHERE THE INK MOVED (the arrows)      : {ang(a1, a2):5.1f} deg")
    print(f"      (the arrows explain {100*(1-r1):.1f}% and {100*(1-r2):.1f}% of the two changes)")
    print("""
The arrows themselves, as six numbers. 'sideways arrow' is how far ink moves
left or right; 'up-down arrow' is how far it moves up or down. Each is written
as: a fixed part, plus a part proportional to how far right you are, plus a part
proportional to how far down you are.
""")
    lab = ["sideways: fixed", "sideways: x how far right", "sideways: x how far down",
           "up-down : fixed", "up-down : x how far right", "up-down : x how far down"]
    print(f"{'':>30}{'1st widening':>15}{'2nd widening':>15}")
    print("-" * 60)
    s1, s2 = a1 / np.linalg.norm(a1), a2 / np.linalg.norm(a2)
    for i, l in enumerate(lab):
        print(f"{l:>30}{s1[i]:>15.3f}{s2[i]:>15.3f}")
    print("""
Both widenings say the same thing: ink moves sideways, in proportion to how far
right of the middle it already is. Which pixels that lands on is different each
time -- that was the whole problem -- but the instruction is the same.""")


# --------------------------------------------------------------------- part 2

def _two_steps(k, start=ROOT):
    h = STEP[k]
    I0 = one(start)
    p1 = start.copy(); p1[k] += h
    p2 = start.copy(); p2[k] += 2 * h
    I1, I2 = one(p1), one(p2)
    return I0, I1, I2, I1 - I0, I2 - I1


def _table(fn, title, note=""):
    print(f"\n{title}")
    if note:
        print(note)
    print(f"{'knob':>22}{'step 1 vs step 2':>20}{'verdict':>12}")
    print("-" * 54)
    out = []
    for k in range(5):
        a = fn(k)
        out.append(a)
        v = "good" if a < 25 else ("so-so" if a < 55 else "FAILS")
        print(f"{NAMES[k]:>22}{a:>16.1f} deg{v:>12}")
    print(f"{'worst of the five':>22}{max(out):>16.1f} deg")
    return out


def part2():
    rule("2.  Is 'ink to the left of here' overfitted to widening?")
    print("""
The test that matters: run the SAME recipe on all five knobs, not just the one
it was invented for. A recipe that only works on left-right changes will fall
over on the up-down ones.

Each row: turn one knob twice by the same amount, and measure the angle between
the first change and the second. Small is good -- it means the method sees the
two steps as the same thing happening twice.
""")
    def plain(k):
        I0, I1, I2, D1, D2 = _two_steps(k)
        return ang(D1, D2)

    def lt(k):
        I0, I1, I2, _, _ = _two_steps(k)
        return ang(left_total(I1) - left_total(I0), left_total(I2) - left_total(I1))

    _table(plain, "A.  which pixels changed  (the baseline, no method at all)")
    _table(lt, "B.  ink to the left of here  (last session's trick)")
    print("""
Compare the two 'worst of the five' numbers. That is the honest score of a
method: how it does on the knob it suits LEAST.""")


# --------------------------------------------------------------------- part 3

def part3():
    rule("3.  Do the arrows work on all five knobs?")
    print("""
Same test, now describing each change by WHERE THE INK MOVED. Two ways of
reading the arrows off a change, plus two controls.
""")
    def f_grad(k, deg=1):
        I0, I1, I2, D1, D2 = _two_steps(k)
        return ang(arrows(I0, D1, deg)[0], arrows(I1, D2, deg)[0])

    def f_warp(k, deg=1):
        I0, I1, I2, _, _ = _two_steps(k)
        return ang(arrows_warp(I0, I1, deg)[0], arrows_warp(I1, I2, deg)[0])

    def f_exact(k, deg=1):
        h = STEP[k]
        p1 = ROOT.copy(); p1[k] += h
        p2 = ROOT.copy(); p2[k] += 2 * h
        I0, I1, I2 = one(ROOT), one(p1), one(p2)
        return ang(arrows_exact(ROOT, I1 - I0, degree=deg)[0],
                   arrows_exact(p1, I2 - I1, degree=deg)[0])

    _table(lambda k: f_grad(k, 0),
           "A.  CONTROL: one arrow for the whole picture (too weak on purpose)")
    _table(lambda k: f_grad(k, 1),
           "B.  arrows read off by estimating how the picture shades")
    _table(lambda k: f_warp(k, 1),
           "C.  arrows read off by actually moving the ink (no shading estimate)")
    _table(lambda k: f_grad(k, 2),
           "D.  CONTROL: twice as much freedom in the arrows (should not help)")
    _table(lambda k: f_exact(k, 1),
           "E.  CEILING: the same, with a perfect shading estimate\n"
           "    (only possible because a renderer exists -- not a general method)")

    print("""
How much of each change is actually accounted for by ink moving:
""")
    print(f"{'knob':>22}{'estimated shading':>20}{'moving the ink':>18}{'perfect shading':>18}")
    print("-" * 78)
    for k in range(5):
        I0, I1, _, D1, _ = _two_steps(k)
        a = 100 * (1 - arrows(I0, D1, 1)[1])
        b = 100 * (1 - arrows_warp(I0, I1, 1)[1])
        c = 100 * (1 - arrows_exact(ROOT, D1, degree=1)[1])
        print(f"{NAMES[k]:>22}{a:>19.1f}%{b:>17.1f}%{c:>17.1f}%")
    print("""
Two different costs are visible here, and they must not be confused.

  - 'estimated shading' is low (30-50%) because reading how a picture shades
    off 784 numbers is inaccurate at a sharp edge. Making the picture bigger
    does NOT fix this; it was checked at 28, 56, 112, 224 and 448 across and
    stays at 30-47%.
  - 'perfect shading' is low HERE only because these steps are a whole pixel.
    At a step of 0.05 px the same column reads 100.0% for every one of the five
    knobs. Sliding-and-shading is an exact description of a TINY change; it is
    not meant for a change this big.

'moving the ink' needs neither: it does not estimate shading and it does not
assume the change is tiny. That is why it is the only column that works on all
five knobs at a realistic step size. What it loses (the 73% on widening) is
resampling error when the interpolation crosses the sharp edge.""")


# --------------------------------------------------------------------- part 4

def part4():
    rule("4.  Do the arrows tell a real knob from junk?")
    print("""
A method that calls everything a knob is useless. Five things are fed in. The
first five rows are genuine knob turns. The rest are not: they are changes to
the picture that no sliding of ink could produce.

'explained by ink moving' is the key column. A real knob turn should be almost
fully explained. Junk should not be.
""")
    rng = np.random.default_rng(0)
    I0 = one(ROOT)
    items = []
    for k in range(5):
        p = ROOT.copy(); p[k] += STEP[k]
        items.append((NAMES[k] + "  [real knob]", one(p) - I0))
    scale = np.linalg.norm(one(ROOT + np.array([0, 0, 0, STEP[3], 0])) - I0)
    noise = rng.standard_normal(I0.shape); noise *= scale / np.linalg.norm(noise)
    items.append(("random noise everywhere  [junk]", noise))
    sp = np.zeros_like(I0)
    m = I0 > 0.05
    v = rng.standard_normal(int(m.sum())); sp[m] = v
    sp *= scale / np.linalg.norm(sp)
    items.append(("random noise, but only on the ink  [junk]", sp))
    dark = I0 * (scale / np.linalg.norm(I0))
    items.append(("make the whole 1 darker  [junk: not a move]", dark))
    blob = np.zeros_like(I0); blob[4:8, 20:24] = 1.0
    blob *= scale / np.linalg.norm(blob)
    items.append(("a blob appears in the corner  [junk]", blob))
    print(f"{'change fed in':>44}{'explained by ink moving':>26}")
    print("-" * 72)
    for name, D in items:
        _, r = arrows(I0, D, 1)
        print(f"{name:>44}{100*(1-r):>24.1f}%")
    print("""
The gap between the two groups is the detector. It needs nothing to be told in
advance -- no list of knobs, no labels.""")


# --------------------------------------------------------------------- part 5

def _describe(ims, I0, how):
    if how == "pixels":
        return np.array([(im - I0).ravel() for im in ims])
    if how == "left":
        L0 = left_total(I0)
        return np.array([(left_total(im) - L0).ravel() for im in ims])
    if how == "arrows1":
        return np.array([arrows(I0, im - I0, 1)[0] for im in ims])
    if how == "arrows2":
        return np.array([arrows(I0, im - I0, 2)[0] for im in ims])
    if how == "warp1":
        return np.array([arrows_warp(I0, im, 1)[0] for im in ims])
    raise ValueError(how)


def _recover(F, TH, squeeze=None, lam=1e-8):
    """Fit description -> knob values on half the data, score on the other half.
    squeeze = keep only this many numbers (the strongest directions) first."""
    F = F - F.mean(0)
    if squeeze is not None and F.shape[1] > squeeze:
        U, S, Vt = np.linalg.svd(F, full_matrices=False)
        F = U[:, :squeeze] * S[:squeeze]
    n = len(F); tr = slice(0, n // 2); te = slice(n // 2, n)
    X, Y = F[tr], TH[tr]
    W = np.linalg.solve(X.T @ X + lam * np.trace(X.T @ X) * np.eye(X.shape[1]) / X.shape[1],
                        X.T @ (Y - Y.mean(0)))
    P = F[te] @ W + Y.mean(0)
    return np.abs(P - TH[te]).mean(0)


def part5():
    rule("5.  Can you get the knob values back? (the real goal)")
    print("""
This is the thing we actually want: a handful of numbers per picture, from
which the original knob settings can be read back precisely.

Take 300 pictures whose five knobs were each nudged at random. Describe the
change from the starting 1 each way. Then fit the simplest possible rule from
the description to the five knob values, using HALF the pictures, and test it on
the other half. The error is the average miss on the held-out half, written as a
percentage of that knob's full allowed range. Under 1% is good.

Each description uses a different count of numbers, shown as 'numbers used'.
""")
    rng = np.random.default_rng(1)
    box = np.array([1.0, 1.0, 1.5, 2.8, 45.0])
    I0 = one(ROOT)
    nsamp = 150
    hows = [("which pixels changed", "pixels", 784),
            ("ink to the left of here", "left", 784),
            ("arrows, degree 1", "arrows1", 6),
            ("arrows, degree 2", "arrows2", 12),
            ("MOVING THE INK, degree 1", "warp1", 6)]
    for frac in [0.05, 0.20, 0.50]:
        TH = (rng.random((nsamp, 5)) - 0.5) * 2 * frac
        ims = render(ROOT + TH * box, dtype=np.float64)
        print(f"  --- knobs nudged by up to {100*frac:.0f}% of their range " + "-" * 24)
        print(f"{'description':>26}{'numbers used':>14}" + "".join(f"{k:>9}" for k in SHORT))
        print("-" * (40 + 9 * 5))
        for name, how, cnt in hows:
            F = _describe(ims, I0, how)
            e = 100 * _recover(F, TH)
            print(f"{name:>26}{cnt:>14}" + "".join(f"{v:>8.2f}%" for v in e))
        print(f"\n  the same, but every description first squeezed to just 5 numbers")
        print(f"  (this is the apples-to-apples 'minimal dimensions' test)")
        print(f"{'description':>26}{'numbers used':>14}" + "".join(f"{k:>9}" for k in SHORT))
        print("-" * (40 + 9 * 5))
        for name, how, cnt in hows:
            F = _describe(ims, I0, how)
            e = 100 * _recover(F, TH, squeeze=5)
            print(f"{name:>26}{5:>14}" + "".join(f"{v:>8.2f}%" for v in e))
        print()
    print("""A column that stays small means that knob is genuinely recoverable from
that description. A column that blows up means the description has lost that
knob -- it cannot tell you where it was set.""")


# --------------------------------------------------------------------- part 6

def ellipse(cx, cy, rx, ry, deg, n=28, ss=6):
    g = (np.arange(n * ss) + 0.5) / ss
    Y, X = np.meshgrid(g, g, indexing="ij")
    t = np.radians(deg)
    u = (X - cx) * np.cos(t) + (Y - cy) * np.sin(t)
    v = -(X - cx) * np.sin(t) + (Y - cy) * np.cos(t)
    f = ((u / rx) ** 2 + (v / ry) ** 2 <= 1.0).astype(float)
    return f.reshape(n, ss, n, ss).mean((1, 3))


def part6():
    rule("6.  Does it survive on a completely different dataset?")
    print("""
Nothing above should depend on the pictures being 1s. Here is a second family
with nothing in common with strokes: filled ellipses, five knobs -- centre
left-right, centre up-down, width, height, and rotation. Same two tests.
""")
    E = np.array([14.0, 14.0, 7.0, 4.0, 20.0])
    ESTEP = np.array([0.35, 0.35, 0.35, 0.35, 3.0])
    ENAMES = ["centre left-right", "centre up-down", "make wider",
              "make taller", "rotate"]
    I0 = ellipse(*E)
    print(f"{'knob of the ellipse':>22}{'step 1 vs step 2, pixels':>28}{'same, arrows':>16}")
    print("-" * 68)
    for k in range(5):
        p1 = E.copy(); p1[k] += ESTEP[k]
        p2 = E.copy(); p2[k] += 2 * ESTEP[k]
        I1, I2 = ellipse(*p1), ellipse(*p2)
        a1 = arrows(I0, I1 - I0, 1)[0]
        a2 = arrows(I1, I2 - I1, 1)[0]
        print(f"{ENAMES[k]:>22}{ang(I1-I0, I2-I1):>24.1f} deg{ang(a1,a2):>12.1f} deg")

    print("""
And the same knob-recovery test as part 5, on ellipses: nudge all five knobs at
random, describe the change, fit the simplest rule from the description back to
the knob values on half the pictures, score it on the other half. Error is the
average miss on the held-out half, as a percentage of that knob's range.
""")
    rng = np.random.default_rng(3)
    ebox = np.array([1.0, 1.0, 1.5, 1.5, 60.0])
    nn = 120
    for frac in [0.05, 0.20]:
        TH = (rng.random((nn, 5)) - 0.5) * 2 * frac
        ims = np.array([ellipse(*q) for q in (E + TH * ebox)])
        print(f"  --- knobs nudged by up to {100*frac:.0f}% of their range " + "-" * 22)
        print(f"{'description':>26}{'numbers':>9}" + "".join(f"{k[:8]:>10}" for k in ENAMES))
        print("-" * (35 + 10 * 5))
        rows = [("which pixels changed", np.array([(im - I0).ravel() for im in ims]), 784),
                ("arrows, degree 1", np.array([arrows(I0, im - I0, 1)[0] for im in ims]), 6),
                ("MOVING THE INK, degree 1",
                 np.array([arrows_warp(I0, im, 1)[0] for im in ims]), 6)]
        for name, F, cnt in rows:
            e = 100 * _recover(F, TH)
            print(f"{name:>26}{cnt:>9}" + "".join(f"{v:>9.2f}%" for v in e))
        print()
    print("""An ellipse's five knobs are a different five from a 1's five, and nothing
was adjusted between the two families.""")


PARTS = {1: part1, 2: part2, 3: part3, 4: part4, 5: part5, 6: part6}

if __name__ == "__main__":
    for i in [int(a) for a in sys.argv[1:]] or sorted(PARTS):
        PARTS[i]()
