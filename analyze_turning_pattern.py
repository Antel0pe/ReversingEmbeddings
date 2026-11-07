"""Walking a knob: does the direction turn by a steady amount?

Two consecutive widenings of a 1 came out 63 degrees apart. This asks whether
that number is part of a pattern -- does the direction of travel turn by a
consistent amount, at consistent moments, as you walk along a knob? And does it
behave the same way for lean?

Picture a knob sweep as a journey. Each image is a place; turning the knob a
little is a step; the step has a direction. The question is how that direction
rotates as the journey goes on.

  turn        the angle between one step's direction and the next step's.
              0 = still going the same way. 90 = swerved completely.
  distance    how far the picture moved on that step.
  corner      a step where the turn is large: the journey changed course.

Parts:
  1  walk each knob in tiny steps and look at the turn at every step
  2  are the corners where they should be? a closed-form prediction
  3  how big is each corner, and is it the same size every time
  4  does the total turning add up at a steady rate? (the clock question)
  5  lean, which is the awkward one, and why
  6  the same sweeps on the crease-free renderer, as a control
"""

import sys

import numpy as np

from grey_ones import render, RANGES
import smooth_ones

KNOBS = ["cx", "cy", "height", "width", "lean"]
ROOT = np.array([14.0, 14.0, 19.75, 3.2, 0.0])


def sweep(k, lo, hi, n, base=ROOT, renderer=None):
    """n images along knob k. Returns the knob values and the images."""
    t = np.linspace(lo, hi, n)
    P = np.tile(np.asarray(base, float), (n, 1))
    P[:, k] = t
    if renderer is None:
        return t, render(P, dtype=np.float64)
    return t, np.array([renderer(p) for p in P])


def turns(ims):
    """Direction of each step, how far it went, and the turn between steps."""
    D = np.diff(ims.reshape(len(ims), -1), axis=0)
    L = np.linalg.norm(D, axis=1)
    U = D / np.maximum(L[:, None], 1e-300)
    c = np.clip(np.sum(U[:-1] * U[1:], axis=1), -1, 1)
    return np.degrees(np.arccos(c)), L


def rule(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


# --------------------------------------------------------------------- part 1

def part1():
    rule("1.  Walk each knob in tiny steps. What does the turn look like?")
    print("""
Each knob is swept across its whole allowed range in 2000 tiny steps. For every
step the turn to the next step is measured. The question is whether the turning
is spread evenly along the journey or bunched up.

  'steps with turn under 1 deg'   how often the journey is going dead straight
  'steps with turn over 30 deg'   how often it swerves
  'biggest single turn'           the worst swerve
""")
    print(f"{'knob':>8}{'range swept':>18}{'under 1 deg':>14}{'over 30 deg':>14}"
          f"{'biggest turn':>14}{'total turning':>15}")
    print("-" * 83)
    for k in range(5):
        lo, hi = RANGES[k]
        t, ims = sweep(k, lo, hi, 2000)
        a, L = turns(ims)
        print(f"{KNOBS[k]:>8}{f'{lo:g} to {hi:g}':>18}"
              f"{100*np.mean(a < 1):>13.1f}%{100*np.mean(a > 30):>13.1f}%"
              f"{a.max():>13.1f}°{a.sum():>14.0f}°")
    print("""
Read the first two columns together. If the journey were a smooth curve the
turn would be small and similar at every step. If it is a POLYGON -- straight
runs joined by sharp corners -- then almost every step is under 1 degree and a
few are huge.""")


# --------------------------------------------------------------------- part 2

def part2():
    rule("2.  Are the corners where they should be?")
    print("""
A guess at what a corner IS. The renderer works out, for each pixel, what
fraction of it the stroke covers. While a stroke edge stays inside one column of
pixels, turning the knob just darkens that one column a little more -- the same
change again and again, so the journey runs straight. The moment the edge
crosses into the NEXT column, a different pixel starts filling, and the journey
turns.

So corners should happen exactly when a stroke edge crosses a whole-number
column boundary. The stroke's two edges sit at cx - width/2 and cx + width/2.

PREDICTION for the width knob, starting from cx = 14.0: both edges cross a
boundary together whenever width/2 is a whole number, i.e. at width = 2, 4, 6...
PREDICTION for the cx knob at width = 3.2: the edges are at cx - 1.6 and
cx + 1.6, so corners at cx = ...4, ...6 of any whole number.
""")
    for k, lo, hi, pred, desc in [
        (3, 1.0, 9.0, [2.0, 4.0, 6.0, 8.0], "width, every 2.0"),
        (0, 11.0, 17.0, [x + d for x in range(11, 17) for d in (0.4, 0.6)],
         "cx, at .4 and .6 of each whole number"),
    ]:
        t, ims = sweep(k, lo, hi, 4000)
        a, _ = turns(ims)
        mid = t[1:-1]
        thr = max(5.0, np.percentile(a, 99.0))
        found = []
        i = 0
        while i < len(a):
            if a[i] > thr:
                j = i
                while j + 1 < len(a) and a[j + 1] > thr:
                    j += 1
                found.append(float(mid[(i + j) // 2]))
                i = j + 1
            else:
                i += 1
        pred = [p for p in pred if lo < p < hi]
        print(f"  --- {KNOBS[k]} knob ({desc}) " + "-" * (34 - len(desc)))
        print(f"      predicted corners: {', '.join(f'{p:.2f}' for p in pred)}")
        print(f"      measured  corners: {', '.join(f'{p:.2f}' for p in found)}")
        if len(found) == len(pred):
            err = np.abs(np.array(found) - np.array(pred)).max()
            print(f"      worst disagreement: {err:.4f}  (one sweep step is "
                  f"{(hi-lo)/4000:.4f})")
        else:
            print(f"      counts differ: {len(found)} found vs {len(pred)} predicted")
        print()


# --------------------------------------------------------------------- part 3

def part3():
    rule("3.  Is every corner the same size?")
    print("""
If corners come at steady intervals AND each one turns by the same amount, then
the turning really is a steady clock. Below, each corner on a width sweep and a
cx sweep, with how far apart they are and how much each one turns.
""")
    for k, lo, hi in [(3, 1.0, 9.0), (0, 11.0, 17.0)]:
        t, ims = sweep(k, lo, hi, 8000)
        a, L = turns(ims)
        mid = t[1:-1]
        thr = max(5.0, np.percentile(a, 99.0))
        groups, i = [], 0
        while i < len(a):
            if a[i] > thr:
                j = i
                while j + 1 < len(a) and a[j + 1] > thr:
                    j += 1
                groups.append((float(mid[(i + j) // 2]), float(a[i:j + 1].sum())))
                i = j + 1
            else:
                i += 1
        print(f"  --- {KNOBS[k]} knob " + "-" * 44)
        print(f"{'corner at':>12}{'gap from last':>16}{'turn at it':>14}")
        prev = None
        for pos, tot in groups:
            g = "" if prev is None else f"{pos - prev:.3f}"
            print(f"{pos:>12.3f}{g:>16}{tot:>13.1f}°")
            prev = pos
        tv = [g[1] for g in groups]
        print(f"      turns: smallest {min(tv):.1f}°, largest {max(tv):.1f}°, "
              f"spread {100*(max(tv)-min(tv))/np.mean(tv):.0f}% of the average")
        print()


# --------------------------------------------------------------------- part 4

def part4():
    rule("4.  Does the turning add up at a steady rate?")
    print("""
The real question behind 'a consistent amount'. Walk each knob across its range
and keep a running total of how much the journey has turned. If that total
climbs at a steady rate, then 'how much have I turned' is a usable clock: it
tells you how far the knob has moved, read off the pictures alone with no labels.

The test: split each knob's range into five equal quarters-of-the-way chunks and
report the turning done in each. Equal numbers across a row = a steady rate.
""")
    print(f"{'knob':>8}" + "".join(f"{f'chunk {i+1}':>12}" for i in range(5))
          + f"{'spread':>10}{'corners':>9}")
    print("-" * 87)
    for k in range(5):
        lo, hi = RANGES[k]
        t, ims = sweep(k, lo, hi, 4000)
        a, _ = turns(ims)
        parts = np.array_split(a, 5)
        tot = [p.sum() for p in parts]
        thr = max(5.0, np.percentile(a, 99.0))
        ncor = int(np.sum((a[1:] > thr) & (a[:-1] <= thr))) + int(a[0] > thr)
        sp = 100 * (max(tot) - min(tot)) / np.mean(tot)
        print(f"{KNOBS[k]:>8}" + "".join(f"{v:>11.0f}°" for v in tot)
              + f"{sp:>9.0f}%{ncor:>9}")
    print("""
'spread' is how much the biggest chunk beats the smallest, as a percentage of
the average. Small spread = steady. 'corners' counts the sharp turns found in
the whole sweep.""")


# --------------------------------------------------------------------- part 5

def part5():
    rule("5.  Lean: the awkward one")
    print("""
Does any of this work for lean? Lean is different in kind. Widening moves both
stroke edges by the same amount in every row at once, so all the rows cross a
column boundary together and you get one big shared corner. Leaning moves the
top rows a lot and the middle rows not at all, so every row crosses its own
boundary at its own moment.

Below, a lean sweep, counting corners and measuring how much each turns.
""")
    t, ims = sweep(4, -10.0, 35.0, 6000)
    a, L = turns(ims)
    thr = max(2.0, np.percentile(a, 97.0))
    big = a > thr
    ncor = int(np.sum(big[1:] & ~big[:-1])) + int(big[0])
    print(f"      total turning over the whole lean range : {a.sum():.0f}°")
    print(f"      corners counted                          : {ncor}")
    print(f"      average turn at a corner                 : {a[big].sum()/max(ncor,1):.1f}°")
    print(f"      biggest single turn                      : {a.max():.1f}°")
    print(f"      share of all turning that happens at corners: {100*a[big].sum()/a.sum():.0f}%")
    print()
    for k, name in [(3, "width"), (0, "cx")]:
        lo, hi = RANGES[k]
        _, im2 = sweep(k, lo, hi, 6000)
        a2, _ = turns(im2)
        th2 = max(2.0, np.percentile(a2, 97.0))
        b2 = a2 > th2
        n2 = int(np.sum(b2[1:] & ~b2[:-1])) + int(b2[0])
        print(f"      for comparison, {name:>5}: {a2.sum():6.0f}° of turning, "
              f"{n2:3d} corners, {100*a2[b2].sum()/a2.sum():.0f}% of turning at corners")
    print("""
Then the steadiness test for lean on its own, in ten chunks across its range:
""")
    parts = np.array_split(a, 10)
    tot = [p.sum() for p in parts]
    print("      " + "  ".join(f"{v:.0f}°" for v in tot))
    print(f"      biggest chunk beats smallest by "
          f"{100*(max(tot)-min(tot))/np.mean(tot):.0f}% of the average")


# --------------------------------------------------------------------- part 6

def part6():
    rule("6.  Control: the same sweeps on a renderer with no corners at all")
    print("""
`smooth_ones` draws the same five-knob family with soft edges and no
whichever-is-smaller decisions anywhere, so it has no pixel-boundary crossings
to make corners out of. If the pattern above is really made by the pixel grid,
it should vanish here and be replaced by gentle, even turning.
""")
    def sm(p):
        return smooth_ones.render(np.asarray(p, float)[None])[0]

    print(f"{'knob':>8}{'under 1 deg':>14}{'over 30 deg':>14}{'biggest turn':>14}"
          f"{'total turning':>15}{'spread over 5 chunks':>22}")
    print("-" * 87)
    for k in range(5):
        lo, hi = RANGES[k]
        t, ims = sweep(k, lo, hi, 1200, renderer=sm)
        a, _ = turns(ims)
        tot = [p.sum() for p in np.array_split(a, 5)]
        print(f"{KNOBS[k]:>8}{100*np.mean(a < 1):>13.1f}%{100*np.mean(a > 30):>13.1f}%"
              f"{a.max():>13.2f}°{a.sum():>14.1f}°"
              f"{100*(max(tot)-min(tot))/np.mean(tot):>21.0f}%")
    print("""
Compare with part 1's table. Same five knobs, same ranges, soft edges.""")


def part7():
    rule("7.  Where 90 and 60 come from, in closed form")
    print("""
The corner sizes were not round by accident. A guess at the rule:

A knob step changes a handful of pixel columns. At a corner, some of those
columns hand over to their neighbours and the rest carry on. If the step spreads
itself evenly over N columns and M of them hand over, then the part of the
change that survives the corner is (N - M) out of N, so

    turn at the corner = the angle whose cosine is (N - M) / N

  widening   both stroke edges move outward together, so the step touches the
             left-edge column and the right-edge column: N = 2. At a corner
             BOTH cross at once, M = 2, giving cosine 0, a turn of 90 degrees.
  sliding    the step touches the same two columns, N = 2, but the left edge and
             the right edge reach their boundaries at different moments, so only
             M = 1 crosses, giving cosine 1/2, a turn of 60 degrees.

Below the columns are actually counted on either side of each corner, so the
rule is checked rather than asserted.
""")
    print(f"{'knob':>8}{'corner at':>12}{'columns touched':>18}{'that hand over':>17}"
          f"{'predicted':>12}{'measured':>11}")
    print("-" * 78)
    for k, lo, hi in [(3, 1.0, 9.0), (0, 11.0, 17.0)]:
        t, ims = sweep(k, lo, hi, 8000)
        a, _ = turns(ims)
        mid = t[1:-1]
        thr = max(5.0, np.percentile(a, 99.0))
        i = 0
        while i < len(a):
            if a[i] > thr:
                j = i
                while j + 1 < len(a) and a[j + 1] > thr:
                    j += 1
                D = np.diff(ims.reshape(len(ims), -1), axis=0)
                before = D[max(i - 4, 0)].reshape(28, 28)
                after = D[min(j + 4, len(D) - 1)].reshape(28, 28)
                cb = set(np.flatnonzero(np.abs(before).sum(0) > 1e-9))
                ca = set(np.flatnonzero(np.abs(after).sum(0) > 1e-9))
                N = len(cb)
                M = len(cb - ca)
                pred = np.degrees(np.arccos(np.clip((N - M) / N, -1, 1)))
                meas = float(a[i:j + 1].sum())
                print(f"{KNOBS[k]:>8}{mid[(i+j)//2]:>12.3f}{N:>18}{M:>17}"
                      f"{pred:>11.1f}\u00b0{meas:>10.1f}\u00b0")
                i = j + 1
            else:
                i += 1
    print("""
If predicted and measured agree, the corner size is not a measurement at all --
it is arithmetic on how many columns are in play.""")


def part8():
    rule("8.  Is lean the same mechanism, with the corners packed too close to see?")
    print("""
Widening moves every row's edges by the same amount, so all the rows cross a
boundary at the same moment and you get ONE big corner. Leaning moves the top
rows a long way and the middle rows not at all, so each row crosses its own
boundary at its own moment. If that is right, lean should have the same corners,
just very many of them, each tiny -- which is why the sweep looked like a smooth
curve. Three checks.

(a) ZOOM IN. If lean is really made of corners, a very fine sweep over a tiny
    slice of lean should look spiky, not smooth.
""")
    for rng_deg, n in [((-10.0, 35.0), 6000), ((10.0, 11.0), 6000), ((10.0, 10.05), 6000)]:
        t, ims = sweep(4, rng_deg[0], rng_deg[1], n)
        a, _ = turns(ims)
        peak = a.max() / max(np.median(a), 1e-12)
        print(f"      lean {rng_deg[0]:>6.2f} to {rng_deg[1]:<6.2f}: "
              f"biggest turn {a.max():7.3f} deg, typical turn {np.median(a):7.4f} deg, "
              f"biggest is {peak:6.1f}x typical")
    print("""
    For comparison, the same ratio on the width knob, which is definitely made
    of corners, and on the crease-free renderer, which definitely is not:""")
    t, ims = sweep(3, 1.8, 4.6, 6000)
    a, _ = turns(ims)
    print(f"      width (corners)        : biggest is {a.max()/max(np.median(a),1e-12):.0f}x typical")
    t, ims = sweep(4, 10.0, 11.0, 2000,
                   renderer=lambda p: smooth_ones.render(np.asarray(p, float)[None])[0])
    a, _ = turns(ims)
    print(f"      lean, soft edges       : biggest is {a.max()/max(np.median(a),1e-12):.1f}x typical")

    print("""
(b) MAKE THE STROKE TALLER. A row crosses a boundary once per pixel it travels,
    and leaning moves a row by its distance from the middle, so a taller stroke
    has rows further out, crossing more boundaries, and should turn MORE.

    My guess at how much was the SQUARE of the height, counting the crossings.
    That guess is WRONG: the measurement below follows height to the power 1.42,
    fitted to within 3.2%, not height squared. Counting crossings alone
    overpredicts, because the size of each corner changes with the height too,
    and at this sweep resolution the two effects cannot be separated cleanly.
    The direction of the prediction holds; the exponent does not.
""")
    print(f"{'stroke height':>15}{'total turning':>16}{'vs the shortest':>18}"
          f"{'height^1.42, vs the shortest':>34}")
    print("-" * 83)
    base = None
    for h in [8.0, 12.0, 16.0, 20.0, 24.0]:
        st = ROOT.copy(); st[2] = h
        t, ims = sweep(4, 0.0, 20.0, 3000, base=st)
        a, _ = turns(ims)
        if base is None:
            base, bh = a.sum(), h
        print(f"{h:>15.1f}{a.sum():>15.0f}\u00b0{a.sum()/base:>17.2f}x"
              f"{(h/bh)**1.42:>33.2f}x")

    print("""
(c) IS THERE A CLOCK FOR LEAN? Lean's turning was uneven when measured in
    degrees. A row's sideways travel goes with the TANGENT of the lean, not the
    lean itself, so tangent is the natural clock. Below, the turning split into
    ten equal chunks, measured both ways. Lower spread = steadier.
""")
    for label, vals in [("equal chunks of lean in degrees", np.linspace(-10, 35, 4001)),
                        ("equal chunks of tan(lean)",
                         np.degrees(np.arctan(np.linspace(np.tan(np.radians(-10)),
                                                          np.tan(np.radians(35)), 4001))))]:
        P = np.tile(ROOT, (len(vals), 1)); P[:, 4] = vals
        ims = render(P, dtype=np.float64)
        a, _ = turns(ims)
        tot = [p.sum() for p in np.array_split(a, 10)]
        print(f"      {label}")
        print("        " + "  ".join(f"{v:.0f}" for v in tot)
              + f"   -> spread {100*(max(tot)-min(tot))/np.mean(tot):.0f}%")


PARTS = {1: part1, 2: part2, 3: part3, 4: part4, 5: part5, 6: part6, 7: part7, 8: part8}

if __name__ == "__main__":
    for i in [int(a) for a in sys.argv[1:]] or sorted(PARTS):
        PARTS[i]()
