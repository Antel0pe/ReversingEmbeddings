"""Is a picture better thought of as a matrix than as a long list of numbers?

Five questions, all asked by the user, tested on the generated 1s. No blurring
anywhere.

  Q1  Widen a 1 one pixel at a time. Does each step move the picture the same
      distance?
  Q2  Is each step at right angles to the step before it?
  Q3  Does writing the picture as a 28x28 grid of numbers, instead of a list of
      784 numbers, change any distance or any angle?
  Q4  The user's idea: describe a picture by the multiplication that turns the
      starting picture into it, and use that multiplication as the picture's
      description. Does that help?
  Q5  Claude's idea, offered separately: store "how much ink is to the LEFT of
      here" instead of "how much ink is here". Nothing is thrown away -- you can
      get the original back by subtracting neighbours. Does that make the
      widening steps point the same way?

Run one question with its number, or all with no argument.
"""

import sys

import numpy as np

from grey_ones import render

ROOT = np.array([14.0, 14.0, 19.75, 3.2, 0.0])
W0 = 3.2


def img(width, base=ROOT):
    p = base.copy(); p[3] = width
    return render(p, dtype=np.float64)[0]


def cosine(a, b):
    a, b = np.ravel(a), np.ravel(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na * nb else float("nan")


def rule(t):
    print()
    print("=" * 72)
    print(t)
    print("=" * 72)


# --------------------------------------------------------------------- Q1, Q2

def q12():
    rule("Q1 and Q2.  Widen a 1 one step at a time, in the ordinary picture")
    print("""
Start at the root 1 (width 3.2 px) and widen it by 0.5 px, again, again...
Each 'step' is the difference between one picture and the last. Both questions
below are about those steps.

  distance  = how far the picture moved on that step
  angle     = the angle between this step and the step before it
              0 deg = exactly the same change again
              90 deg = completely unrelated change (at right angles)
""")
    hs = 0.5
    ws = W0 + hs * np.arange(13)
    ims = [img(w) for w in ws]
    steps = [ims[i + 1] - ims[i] for i in range(len(ims) - 1)]
    print(f"{'from width':>11}{'to width':>10}{'distance':>11}{'angle vs previous':>20}"
          f"{'pixels changed':>16}{'shared with prev':>18}")
    print("-" * 86)
    prev = None
    for i, s in enumerate(steps):
        d = np.linalg.norm(s)
        ang = "" if prev is None else f"{np.degrees(np.arccos(np.clip(cosine(prev, s), -1, 1))):.1f} deg"
        nz = int((np.abs(s) > 1e-12).sum())
        sh = "" if prev is None else int(((np.abs(s) > 1e-12) & (np.abs(prev) > 1e-12)).sum())
        print(f"{ws[i]:>11.2f}{ws[i+1]:>10.2f}{d:>11.4f}{str(ang):>20}{nz:>16}{str(sh):>18}")
        prev = s
    print("""
What to look at: the distance column, and the angle column.""")

    rule("Q2 again, but with hard black-and-white ink instead of grey")
    print("""
The greys above come from the renderer sharing a pixel between 'ink' and 'no
ink'. With hard ink a pixel is either full or empty, which is the case the user
has in mind: each widening lights up a FRESH column and nothing else.
""")
    from coverage_grids import render_coverage
    def hard(w):
        p = ROOT.copy(); p[3] = w
        return (render_coverage(p[None], n_pix=28) >= 0.5).astype(float)[0]
    ws = W0 + 2.0 * np.arange(7)
    ims = [hard(w) for w in ws]
    steps = [ims[i + 1] - ims[i] for i in range(len(ims) - 1)]
    print(f"{'from width':>11}{'to width':>10}{'distance':>11}{'angle vs previous':>20}"
          f"{'pixels changed':>16}{'shared with prev':>18}")
    print("-" * 86)
    prev = None
    for i, s in enumerate(steps):
        ang = "" if prev is None else f"{np.degrees(np.arccos(np.clip(cosine(prev, s), -1, 1))):.1f} deg"
        sh = "" if prev is None else int(((np.abs(s) > 1e-12) & (np.abs(prev) > 1e-12)).sum())
        print(f"{ws[i]:>11.2f}{ws[i+1]:>10.2f}{np.linalg.norm(s):>11.4f}{str(ang):>20}"
              f"{int((np.abs(s) > 1e-12).sum()):>16}{str(sh):>18}")
        prev = s


# ------------------------------------------------------------------------- Q3

def q3():
    rule("Q3.  Does calling it a 28x28 grid instead of a list of 784 change anything?")
    print("""
Take the same two pictures. Measure the distance between them twice: once as
two lists of 784 numbers, once as two 28x28 grids (the usual size of a grid of
numbers, the Frobenius norm). Then the same for the angle between two steps.
""")
    a, b = img(W0), img(W0 + 1.0)
    c = img(W0 + 2.0)
    s1, s2 = b - a, c - b
    print(f"  distance, as a list of 784 numbers : {np.linalg.norm((b - a).ravel()):.12f}")
    print(f"  distance, as a 28x28 grid          : {np.linalg.norm(b - a, 'fro'):.12f}")
    print(f"  difference between the two         : {abs(np.linalg.norm((b-a).ravel()) - np.linalg.norm(b-a,'fro')):.3e}")
    print()
    print(f"  angle between two steps, as lists  : "
          f"{np.degrees(np.arccos(np.clip(cosine(s1, s2), -1, 1))):.6f} deg")
    print(f"  angle between two steps, as grids  : "
          f"{np.degrees(np.arccos(np.clip(float((s1*s2).sum()/np.linalg.norm(s1,'fro')/np.linalg.norm(s2,'fro')), -1, 1))):.6f} deg")
    print("""
Also check that the arrangement really is ignored: shuffle all 784 pixels of
both pictures with the SAME random shuffle (so the grid is destroyed) and
remeasure.""")
    rng = np.random.default_rng(0)
    perm = rng.permutation(784)
    pa, pb = a.ravel()[perm], b.ravel()[perm]
    print(f"  distance after shuffling the pixels: {np.linalg.norm(pb - pa):.12f}")


# ------------------------------------------------------------------------- Q4

def _batch(width, n=240, seed=0):
    """Many different 1s, all at the same width, to pin the multiplication down."""
    rng = np.random.default_rng(seed)
    P = np.tile(ROOT, (n, 1))
    P[:, 0] += rng.uniform(-1.5, 1.5, n)      # cx
    P[:, 1] += rng.uniform(-1.0, 1.0, n)      # cy
    P[:, 2] += rng.uniform(-3.0, 3.0, n)      # height
    P[:, 3] = width
    P[:, 4] += rng.uniform(-10.0, 30.0, n)    # lean
    return render(P, dtype=np.float64)


def _fit_right(M, Mp):
    """One 28x28 matrix A with M @ A = M' for every picture in the batch."""
    X = np.vstack(list(M)); Y = np.vstack(list(Mp))
    A = np.linalg.lstsq(X, Y, rcond=None)[0]
    err = np.linalg.norm(X @ A - Y) / np.linalg.norm(Y - X)
    return A, err


def _fit_left(M, Mp):
    """One 28x28 matrix B with B @ M = M' for every picture in the batch."""
    X = np.hstack(list(M)); Y = np.hstack(list(Mp))
    B = np.linalg.lstsq(X.T, Y.T, rcond=None)[0].T
    err = np.linalg.norm(B @ X - Y) / np.linalg.norm(Y - X)
    return B, err


def q4():
    rule("Q4.  Describe a picture by the multiplication that produces it")
    print("""
The idea: instead of storing the picture, store the matrix you multiply the
starting picture by to get it. First, two things have to be checked before the
idea can even be tried.

(a) Which side do you multiply on?
    A 28x28 grid of ink can be multiplied on the left or on the right, and they
    do different things. Multiplying on the LEFT mixes ROWS together. Multiplying
    on the RIGHT mixes COLUMNS together. Widening a 1 moves ink sideways, so it
    is a mixing of columns.
    Below: fit the single best matrix of each kind that widens 240 different 1s
    by 1 px, and report what fraction of the required change it fails to make.
    0% = it did the job perfectly, 100% = it did nothing useful.
""")
    M, Mp = _batch(W0), _batch(W0 + 1.0)
    A, ea = _fit_right(M, Mp)
    B, eb = _fit_left(M, Mp)
    print(f"      multiply on the RIGHT (mixes columns): {100*ea:6.2f}% of the change missed")
    print(f"      multiply on the LEFT  (mixes rows)   : {100*eb:6.2f}% of the change missed")

    print("""
(b) Is the matrix even pinned down by ONE picture?
    The root 1 is an upright bar: every row of ink is identical, so as a grid of
    numbers it has only one independent row. Its rank is printed below. A grid
    of that rank cannot pin down a 28x28 matrix -- there are endlessly many
    matrices that do the job, so "the multiplication that makes this picture"
    is not one well-defined thing unless you fit it on many pictures at once.
""")
    print(f"      rank of the root 1 as a grid of numbers  : {np.linalg.matrix_rank(img(W0), tol=1e-9)}  (out of 28)")
    print(f"      rank of a leaning 1                      : "
          f"{np.linalg.matrix_rank(render(ROOT + np.array([0,0,0,0,20.0]), dtype=np.float64)[0], tol=1e-9)}  (out of 28)")
    print(f"      rank of the stack of 240 varied 1s       : {np.linalg.matrix_rank(np.vstack(list(M)), tol=1e-9)}  (out of 28)")

    print("""
Now the real question. In the ordinary picture, widening step 2 is at right
angles to widening step 1. In the multiplication description, "another step"
means another matrix. If widening really is one consistent thing, the matrix
should be the SAME every step. Two ways of stepping are compared:

  equal pixel steps      width 3.2 -> 4.2 -> 5.2 -> 6.2   (+1 px each time)
  equal percentage steps width 3.2 -> 3.84 -> 4.61 -> 5.53 (x1.2 each time)

'change from doing nothing' is the matrix minus the do-nothing matrix: that is
the step itself. The angle between consecutive ones is the number to watch.
""")
    for label, widths in [("equal pixel steps (+1 px)", [W0, W0 + 1, W0 + 2, W0 + 3]),
                          ("equal percentage steps (x1.2)", [W0 * 1.2 ** k for k in range(4)])]:
        print(f"  --- {label} " + "-" * (46 - len(label)))
        mats = []
        for i in range(len(widths) - 1):
            Ai, e = _fit_right(_batch(widths[i]), _batch(widths[i + 1]))
            mats.append(Ai)
            print(f"      step {i+1}: width {widths[i]:.2f} -> {widths[i+1]:.2f},"
                  f" {100*e:5.2f}% of the change missed")
        I = np.eye(28)
        for i in range(len(mats) - 1):
            ang = np.degrees(np.arccos(np.clip(cosine(mats[i] - I, mats[i + 1] - I), -1, 1)))
            rel = np.linalg.norm(mats[i + 1] - mats[i]) / np.linalg.norm(mats[i] - I)
            print(f"      angle between step {i+1}'s matrix and step {i+2}'s : {ang:5.1f} deg"
                  f"   (they differ by {100*rel:.1f}% )")
        print(f"      for comparison, the angle between the same two steps")
        print(f"      measured on the ORDINARY picture:", end=" ")
        a0 = img(widths[0]); a1 = img(widths[1]); a2 = img(widths[2])
        print(f"{np.degrees(np.arccos(np.clip(cosine(a1 - a0, a2 - a1), -1, 1))):.1f} deg")
        print()

    print("""Last check for Q4: if one matrix really is 'widen by 20%', then doing
it twice should equal widening by 44% (1.2 x 1.2 = 1.44) in one go. Compare the
matrix for one 44% widening against the 20% matrix multiplied by itself.""")
    A20, _ = _fit_right(_batch(W0), _batch(W0 * 1.2))
    A44, _ = _fit_right(_batch(W0), _batch(W0 * 1.44))
    print(f"      difference between (widen 20%) applied twice and (widen 44%) once:"
          f" {100*np.linalg.norm(A20 @ A20 - A44)/np.linalg.norm(A44 - np.eye(28)):.1f}%")


# ------------------------------------------------------------------------- Q5

def q5():
    rule("Q5.  Store 'ink to the left of here' instead of 'ink here'")
    print("""
Claude's suggestion, kept separate from the user's. Nothing is blurred and
nothing is thrown away: running along each row, replace each pixel by the total
ink in that row up to and including it. Subtracting neighbours gets the original
picture back exactly, so this is just a different way of writing down the same
picture.

Why it might help: widening a 1 by a pixel changes only a thin sliver of ink,
and the next widening changes a DIFFERENT thin sliver, which is why the two look
unrelated. But 'ink to the left of here' changes over the whole right-hand half
of the row both times, so the two steps overlap heavily.
""")
    def left_total(m):
        return np.cumsum(m, axis=1)

    ws = [W0, W0 + 1.0, W0 + 2.0, W0 + 3.0]
    ims = [img(w) for w in ws]
    steps_px = [ims[i + 1] - ims[i] for i in range(3)]
    steps_ct = [left_total(ims[i + 1]) - left_total(ims[i]) for i in range(3)]
    print(f"{'':>26}{'ordinary picture':>20}{'ink-to-the-left':>20}")
    print("-" * 66)
    for i in range(2):
        ap = np.degrees(np.arccos(np.clip(cosine(steps_px[i], steps_px[i + 1]), -1, 1)))
        ac = np.degrees(np.arccos(np.clip(cosine(steps_ct[i], steps_ct[i + 1]), -1, 1)))
        print(f"{'angle, step %d vs step %d' % (i+1, i+2):>26}{ap:>17.1f} deg{ac:>17.1f} deg")
    print(f"{'pixels touched by a step':>26}{int((np.abs(steps_px[0])>1e-12).sum()):>20}"
          f"{int((np.abs(steps_ct[0])>1e-12).sum()):>20}")

    print("""
The catch to check: does it now call EVERYTHING related? Three changes that have
nothing to do with widening are scored against widening step 1. If these are
also near 0 degrees, the new way of writing the picture has simply made
everything look alike, which is no better than before.
""")
    others = {
        "the same widening, 10 px to the left": np.roll(steps_px[0], -10, axis=1),
        "leaning the 1 by 5 degrees": render(ROOT + np.array([0, 0, 0, 0, 5.0]), dtype=np.float64)[0] - ims[0],
        "making the 1 taller by 2 px": render(ROOT + np.array([0, 0, 2.0, 0, 0]), dtype=np.float64)[0] - ims[0],
        "sliding the 1 right by 1 px": render(ROOT + np.array([1.0, 0, 0, 0, 0]), dtype=np.float64)[0] - ims[0],
    }
    print(f"{'compared with widening step 1':>40}{'ordinary':>14}{'ink-to-the-left':>18}")
    print("-" * 72)
    ap = np.degrees(np.arccos(np.clip(cosine(steps_px[0], steps_px[1]), -1, 1)))
    ac = np.degrees(np.arccos(np.clip(cosine(steps_ct[0], steps_ct[1]), -1, 1)))
    print(f"{'widening step 2 (should be SMALL)':>40}{ap:>11.1f} deg{ac:>15.1f} deg")
    for name, d in others.items():
        ap = np.degrees(np.arccos(np.clip(cosine(steps_px[0], d), -1, 1)))
        ac = np.degrees(np.arccos(np.clip(cosine(left_total(ims[0] + d) - left_total(ims[0]),
                                                 steps_ct[0]), -1, 1)))
        print(f"{name + ' (should be BIG)':>40}{ap:>11.1f} deg{ac:>15.1f} deg")


def q6():
    rule("Q6.  Why the multiplication idea fails: a self-defeating requirement")
    print("""
Q4 showed the idea does not work. This asks why, by giving it the friendliest
possible conditions and watching where it breaks.

To widen a 1 by multiplying on the right, the matrix has to apply ONE sideways
recipe to every row of the picture. Below, four batches of 1s. Each batch holds
one thing fixed and varies another. For each batch two numbers:

  missed   what fraction of the widening a single best matrix fails to do
           (0% = one matrix does the whole job)
  rank     how many independent directions the batch of pictures spans, out of
           28. The matrix has 28 columns to pin down, so a rank below 28 means
           the leftover directions are pure guesswork and the matrix you get
           back is not a real description of anything.
""")
    rng = np.random.default_rng(1)
    n = 240
    def batch(vary, width):
        P = np.tile(ROOT, (n, 1))
        P[:, 3] = width
        if "cy" in vary:     P[:, 1] += rng.uniform(-1.0, 1.0, n)
        if "height" in vary: P[:, 2] += rng.uniform(-3.0, 3.0, n)
        if "cx" in vary:     P[:, 0] += rng.uniform(-1.5, 1.5, n)
        if "lean" in vary:   P[:, 4] += rng.uniform(-10.0, 30.0, n)
        return render(P, dtype=np.float64)

    cases = [
        ("upright, centred; only cy and height vary", ["cy", "height"]),
        ("...plus the 1 slides sideways (cx varies)", ["cy", "height", "cx"]),
        ("...plus the 1 leans (lean varies)", ["cy", "height", "lean"]),
        ("everything varies", ["cy", "height", "cx", "lean"]),
    ]
    print(f"{'batch':>45}{'missed':>10}{'rank':>8}")
    print("-" * 63)
    for name, vary in cases:
        rng = np.random.default_rng(1)
        M = batch(vary, W0)
        rng = np.random.default_rng(1)
        Mp = batch(vary, W0 + 1.0)
        A, e = _fit_right(M, Mp)
        r = np.linalg.matrix_rank(np.vstack(list(M)), tol=1e-9)
        print(f"{name:>45}{100*e:>9.2f}%{r:>8}")
    print("""
Read the two columns together. Where one matrix CAN do the job, the pictures do
not pin the matrix down. Where the pictures DO pin it down, one matrix can no
longer do the job. The requirement defeats itself.""")


PARTS = {1: q12, 2: q12, 3: q3, 4: q4, 5: q5, 6: q6}

if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:]] or [1, 3, 4, 6, 5]
    for i in want:
        PARTS[i]()
