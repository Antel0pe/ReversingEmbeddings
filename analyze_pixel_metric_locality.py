"""Does pixel space know that two nearby pixels are nearby?

The complaint this script tests, in the user's words: treating each pixel as its
own dimension makes every pixel exactly as far from every other pixel.  So two
ink patterns that do not share a single pixel score as perpendicular -- cosine
exactly zero -- whether they are neighbouring slivers along the SAME edge, which
are obviously the same gesture continued, or an unrelated blob on the far side
of the canvas.  Plain pixel distance cannot tell those two cases apart, and that
is also where the infinite derivative comes from (Wakin et al. 2005, and
`literature/multiscale-structure-of-non-differentiable-image-manifolds.md`).

Every candidate fix tested here has the same shape:

    re-encode the picture, then use ORDINARY Euclidean distance on the encoding.

and every LINEAR re-encoding that is also translation-covariant and isotropic is
a reweighting of spatial frequencies.  So the whole family of linear answers is
one function w(mu), mu = squared spatial frequency:

    pixel      w = 1                     every frequency equal -> no geometry
    blur s     w = exp(-s^2 mu / 2)      kill detail finer than s
    H^{-1/2}   w = (kappa + mu)^{-1/2}   kill detail at every scale, gently
    sharpen    w = (kappa + mu)^{+1/2}   control: the wrong direction

Implemented in the DCT-II basis, where the Neumann Laplacian is exactly
diagonal, so the same code is exact at any grid fineness.  One nonlinear
candidate is also tested: optimal transport (Sinkhorn), where distance is "how
far does the ink have to walk", which encodes proximity by construction instead
of by filtering.

Parts (run one with its number as an argument, or all with none):
  1  the step function: one blob vs the same blob d pixels away
  2  same gesture, consecutive disjoint increments, vs three wrong-answer controls
  3  speed: ||R(t+h) - R(t)|| / h as h shrinks, the 1/sqrt(h) blow-up
  4  tangent-plane twist: principal angles between the 5-knob plane at h and h/2
  5  optimal transport: does ink-walking distance scale linearly in the step
  6  refining the grid: the sub-pixel blow-up is the sampling, not the edges
"""

import sys

import numpy as np
from scipy.fft import dctn, idctn

from coverage_grids import render_coverage

ROOT = np.array([14.0, 14.0, 19.75, 3.2, 0.0])
KNOBS = ["cx", "cy", "height", "width", "lean"]
CANVAS = 28.0


# ---------------------------------------------------------------- renderers

def cov(P, n_pix=28):
    """Coverage render: the repo's default, already partly anti-aliased."""
    return render_coverage(np.atleast_2d(P), n_pix=n_pix)


def hard(P, n_pix=224):
    """Binary ink: a true cliff, which is the paper's object. Needs fine pitch."""
    return (render_coverage(np.atleast_2d(P), n_pix=n_pix) >= 0.5).astype(np.float64)


# ------------------------------------------------- metrics as frequency weights

def _mu(n_pix):
    """Eigenvalues of the Neumann (-Laplacian) in the DCT-II basis, per canvas^2."""
    pitch = CANVAS / n_pix
    k = np.arange(n_pix)
    d = 2.0 * (1.0 - np.cos(np.pi * k / n_pix)) / pitch ** 2
    return d[:, None] + d[None, :]


_MU_CACHE = {}


def reweight(X, weight, n_pix):
    """Apply a frequency weight to each image in X: the re-encoding Phi."""
    if n_pix not in _MU_CACHE:
        _MU_CACHE[n_pix] = _mu(n_pix)
    w = weight(_MU_CACHE[n_pix])
    X = np.atleast_3d(np.asarray(X, np.float64))
    C = dctn(X, axes=(-2, -1), norm="ortho")
    return idctn(C * w, axes=(-2, -1), norm="ortho")


KAPPA = (2 * np.pi / CANVAS) ** 2          # softest frequency the canvas holds

METRICS = {
    "pixel":      lambda mu: np.ones_like(mu),
    "blur 0.5px": lambda mu: np.exp(-0.5 * 0.5 ** 2 * mu),
    "blur 1px":   lambda mu: np.exp(-0.5 * 1.0 ** 2 * mu),
    "blur 2px":   lambda mu: np.exp(-0.5 * 2.0 ** 2 * mu),
    "blur 4px":   lambda mu: np.exp(-0.5 * 4.0 ** 2 * mu),
    "H^-1/2":     lambda mu: (KAPPA + mu) ** -0.5,
    "H^-1":       lambda mu: (KAPPA + mu) ** -1.0,
    "sharpen":    lambda mu: (KAPPA + mu) ** 0.5,
}


def cosines(A, B, n_pix):
    """cos(A, B) under every metric. A, B: (28,28)-shaped arrays."""
    out = {}
    for name, w in METRICS.items():
        a, b = reweight(A, w, n_pix)[0], reweight(B, w, n_pix)[0]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        out[name] = float(a.ravel() @ b.ravel() / (na * nb)) if na * nb > 0 else np.nan
    return out


def table(rows, header, widths=None):
    widths = widths or [max(11, len(h)) for h in header]
    print("  ".join(h.rjust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(c).rjust(w) for c, w in zip(r, widths)))


# ---------------------------------------------------------------------- part 1

def part1():
    print(__doc__.split("Parts")[0].strip()[:0] or "", end="")
    print("=" * 78)
    print("PART 1  The step function in space")
    print("=" * 78)
    print("""
One pixel of ink switched on at (row 14, col 8), against the SAME single pixel
of ink switched on d columns to the right.  Nothing else in the picture.  As
pictures these are near-identical gestures when d = 1 and unrelated when d = 14.
Numbers are cos between the two one-pixel pictures, 1.00 = identical direction,
0.00 = perpendicular.
""")
    n = 28
    ds = [0, 1, 2, 3, 4, 6, 8, 12, 19]
    base = np.zeros((n, n)); base[14, 8] = 1.0
    names = list(METRICS)
    rows = []
    for d in ds:
        other = np.zeros((n, n)); other[14, 8 + d] = 1.0
        c = cosines(base[None], other[None], n)
        rows.append([f"d={d}"] + [f"{c[k]:.3f}" for k in names])
    table(rows, ["offset"] + names, [7] + [10] * len(names))
    print("""
Read the 'pixel' column: 1.000 then 0.000 forever.  One pixel over and one
quarter of the canvas over score IDENTICALLY.  That is the complaint, as a
measurement: in the pixel metric spatial distance does not exist at all, there
is only same-pixel or different-pixel.

Every other column is a decaying curve instead of a step, so each of them does
give pixel space a notion of proximity.  'Connection range' below is the offset
at which cos first falls under 0.1 -- how far apart two patterns must be before
that metric calls them unrelated.""")
    print()
    rows = []
    for name, w in METRICS.items():
        rng = "never <0.1"
        for d in range(1, 27):
            other = np.zeros((n, n)); other[14, 8 + d] = 1.0
            a = reweight(base[None], w, n)[0].ravel()
            b = reweight(other[None], w, n)[0].ravel()
            if a @ b / (np.linalg.norm(a) * np.linalg.norm(b)) < 0.1:
                rng = f"{d} px"
                break
        rows.append([name, rng])
    table(rows, ["metric", "connection range"], [11, 17])


# ---------------------------------------------------------------------- part 2

def part2():
    print("=" * 78)
    print("PART 2  Same gesture continued, against three wrong answers")
    print("=" * 78)
    print(f"""
Binary-ink render on a {224}x{224} grid (a true cliff; the canvas still spans
0..28, so the pitch is {28/224:.4f} px).  Start at root 1.  Take the increment of
one knob over [0, h] and compare it with four other directions:

  same-knob-next   the SAME knob's increment over [h, 2h] -- the gesture
                   continued outward, touching the first sliver but sharing no
                   pixel with it.  This is the pair that SHOULD score high.
  other-knob       a different knob's increment over [0, h] at the same place.
  far-away         the first increment itself, translated 10 px sideways:
                   right gesture, wrong place.
  scramble         the first increment's own pixels with random signs, forced
                   to exactly zero mean: right place, no gesture.  Controls for
                   a metric that only measures support overlap.  (The signs MUST
                   be balanced: a free coin flip over ~1300 pixels leaves a ~3%
                   mean offset, and after heavy blurring almost nothing but that
                   offset survives, which fakes a score of +0.90.)
  white noise      standard normal over the whole canvas: no gesture, no place.

A representation that captures what the user wants scores same-knob-next well
above the other three.  A representation that cannot tell them apart scores all
four the same.
""")
    n = 224
    rng = np.random.default_rng(0)
    score = {k: [] for k in METRICS}
    for ki, knob in enumerate(["width", "lean", "cx"]):
        k = KNOBS.index(knob)
        print(f"--- knob {knob}, step h = 1.0 {'deg' if knob=='lean' else 'px'} " + "-" * 30)
        h = 1.0
        p0 = ROOT.copy()
        p1 = ROOT.copy(); p1[k] += h
        p2 = ROOT.copy(); p2[k] += 2 * h
        I0, I1, I2 = hard(p0, n)[0], hard(p1, n)[0], hard(p2, n)[0]
        d1 = I1 - I0
        partners = {"same-knob-next": I2 - I1}
        ok = KNOBS.index("lean" if knob != "lean" else "width")
        q = ROOT.copy(); q[ok] += 1.0
        partners["other-knob"] = hard(q, n)[0] - I0
        partners["far-away"] = np.roll(d1, int(10 * n / 28), axis=1)
        s = np.zeros_like(d1)
        m = d1 != 0
        nm = int(m.sum())
        v = np.r_[np.ones(nm // 2), -np.ones(nm - nm // 2)]
        rng.shuffle(v)
        s[m] = v * np.abs(d1[m])
        partners["scramble"] = s
        partners["white noise"] = rng.standard_normal(d1.shape)
        names = list(METRICS)
        rows, got = [], {}
        for pname, P in partners.items():
            c = cosines(d1[None], P[None], n)
            got[pname] = c
            rows.append([pname] + [f"{c[kk]:+.3f}" for kk in names])
        table(rows, ["partner"] + names, [15] + [10] * len(names))
        for m in names:
            wrong = max(abs(got[q][m]) for q in partners if q != "same-knob-next")
            score[m].append(got["same-knob-next"][m] - wrong)
        print(f"    overlapping pixels, increment 1 vs same-knob-next: "
              f"{int(((d1 != 0) & (partners['same-knob-next'] != 0)).sum())}")
        print()
    print("""SEPARATION SCORE: the same-knob-next cosine minus the largest
|cosine| among the four wrong answers, averaged over the three knobs.  1.00
would be a representation that scores the continued gesture perfectly and
everything else at zero; 0.00 cannot tell them apart at all.
""")
    rows = sorted(([m, f"{np.mean(v):+.3f}"] for m, v in score.items()),
                  key=lambda r: -float(r[1]))
    table(rows, ["metric", "separation"], [11, 11])


# ---------------------------------------------------------------------- part 3

def part3():
    print("=" * 78)
    print("PART 3  Speed: does the derivative exist")
    print("=" * 78)
    print("""
||R(root + h e_k) - R(root)|| / h, with the norm taken in each metric and then
divided by that metric's norm of the image itself, so the numbers are comparable
across metrics (units: fractions of an image, per pixel of knob).  A real
derivative means these settle to a constant as h shrinks.  The paper's result is
that on a sharp edge the pixel metric instead grows like 1/sqrt(h).

'slope' is the fitted log-log slope over the whole h range:
  -0.5  the paper's blow-up, no derivative
   0.0  a genuine finite derivative
""")
    n = 224
    hs = np.array([4.0, 2.0, 1.0, 0.5, 0.25])
    for knob in ["width", "lean", "cx"]:
        k = KNOBS.index(knob)
        I0 = hard(ROOT, n)[0]
        print(f"--- knob {knob}, binary ink, {n}x{n} " + "-" * 34)
        rows = []
        for name, w in METRICS.items():
            e0 = reweight(I0[None], w, n)[0]
            scale = np.linalg.norm(e0)
            sp = []
            for h in hs:
                p = ROOT.copy(); p[k] += h
                e1 = reweight(hard(p, n)[0][None], w, n)[0]
                sp.append(np.linalg.norm(e1 - e0) / scale / h)
            sp = np.array(sp)
            slope = np.polyfit(np.log(hs), np.log(sp), 1)[0]
            rows.append([name] + [f"{v:.4f}" for v in sp] + [f"{slope:+.2f}"])
        table(rows, ["metric"] + [f"h={h}" for h in hs] + ["slope"],
              [11] + [9] * len(hs) + [7])
        print()
    print("--- the same, on the repo's 28x28 COVERAGE render " + "-" * 23)
    print("(anti-aliasing is already a regulariser, so this is not the cliff case)")
    n = 28
    k = KNOBS.index("width")
    I0 = cov(ROOT, n)[0]
    rows = []
    for name, w in METRICS.items():
        e0 = reweight(I0[None], w, n)[0]
        scale = np.linalg.norm(e0)
        sp = []
        for h in hs:
            p = ROOT.copy(); p[k] += h
            e1 = reweight(cov(p, n)[0][None], w, n)[0]
            sp.append(np.linalg.norm(e1 - e0) / scale / h)
        sp = np.array(sp)
        rows.append([name] + [f"{v:.4f}" for v in sp]
                    + [f"{np.polyfit(np.log(hs), np.log(sp), 1)[0]:+.2f}"])
    table(rows, ["metric"] + [f"h={h}" for h in hs] + ["slope"],
          [11] + [9] * len(hs) + [7])


# ---------------------------------------------------------------------- part 4

def part4():
    print("=" * 78)
    print("PART 4  Tangent-plane twist: does the 5-knob plane settle down")
    print("=" * 78)
    print("""
At root 1, build the five one-sided knob increments at step h (h is in
NORMALISED knob units: h times each knob's allowed range), span them as a
5-dimensional plane, and compare that plane with the plane built at h/2.
Reported is the LARGEST principal angle between the two planes, in degrees.

  angle shrinking toward 0 as h shrinks -> a tangent plane exists there
  angle stuck at a constant             -> the paper's twisting; no tangent

A binary render has a floor: once a knob step moves the edge by less than one
pixel pitch, NO pixel flips, the increment is exactly zero, and the angle is a
meaningless 90.  The live-direction count below guards against reading that as
a result.
""")
    rng_box = np.array([1.0, 1.0, 1.5, 2.8, 45.0])

    def planes(renderer, n, hs):
        I0 = renderer(ROOT, n)[0].ravel()
        out, alive = {}, {}
        for h in hs:
            D = []
            for k in range(5):
                p = ROOT.copy(); p[k] += h * rng_box[k]
                D.append(renderer(p, n)[0].ravel() - I0)
            D = np.array(D)
            out[h] = D
            alive[h] = int((np.abs(D).max(1) > 0).sum())
        return out, alive

    def report(P, alive, hs, n):
        print("    live directions (of 5) at each h: "
              + ", ".join(f"h={h:g}: {alive[h]}" for h in hs))
        usable = [h for h in hs if alive[h] == 5]
        if len(usable) < 2:
            print("    too few usable steps"); return
        rows = []
        for name, w in METRICS.items():
            enc = {}
            for h in usable:
                E = reweight(P[h].reshape(-1, n, n), w, n).reshape(5, -1)
                enc[h] = np.linalg.qr(E.T)[0]
            angs = []
            for a, b in zip(usable[:-1], usable[1:]):
                sv = np.linalg.svd(enc[a].T @ enc[b], compute_uv=False)
                angs.append(np.degrees(np.arccos(np.clip(sv.min(), -1, 1))))
            rows.append([name] + [f"{v:.1f}" for v in angs])
        table(rows, ["metric"] + [f"{a:g}->{b:g}" for a, b in zip(usable[:-1], usable[1:])],
              [11] + [12] * (len(usable) - 1))

    n = 448
    hs = [0.5, 0.25, 0.125, 0.0625]
    print(f"--- binary ink, {n}x{n}, pitch {CANVAS/n:.4f} px " + "-" * 26)
    P, alive = planes(hard, n, hs)
    report(P, alive, hs, n)
    print()

    n = 28
    hs = [0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
    print(f"--- the repo's own object: {n}x{n} coverage render " + "-" * 22)
    print("    (continuous in h, so no grid floor; already partly anti-aliased)")
    P, alive = planes(cov, n, hs)
    report(P, alive, hs, n)


# ---------------------------------------------------------------------- part 5

def part5():
    print("=" * 78)
    print("PART 5  Optimal transport: distance as how far the ink walks")
    print("=" * 78)
    print("""
The non-filtering answer.  Instead of re-encoding the picture, change what
distance MEANS: the cost of carrying the first picture's ink onto the second's,
paying ink times the distance it travels.  Proximity is then built in, because
carrying ink one pixel over is cheap and carrying it across the canvas is dear.
Nothing is blurred away, so unlike the metrics above this costs no detail.

Computed with Sinkhorn (entropic OT, regulariser eps) on the 28x28 coverage
render, with both images normalised to equal total ink.  The entropic term adds
a constant floor of order sqrt(eps), which swamps small steps, so the DEBIASED
Sinkhorn divergence S(a,b) - (S(a,a) + S(b,b))/2 is reported alongside the raw
value.  Distances are in pixels of travel.

A pure translation by h must cost exactly h: slope +1.00 and a flat speed of
1.00, with no blur and nothing destroyed.  Compare the pixel metric's +0.50.
""")
    n = 28
    eps = 0.25
    x = np.arange(float(n))
    C1 = (x[:, None] - x[None, :]) ** 2
    K = np.exp(-C1 / eps)
    Kc = K * C1

    def ot(a, b, iters=3000):
        a = np.maximum(a, 1e-11); a = a / a.sum()
        b = np.maximum(b, 1e-11); b = b / b.sum()
        u = np.ones((n, n)); v = np.ones((n, n))
        for _ in range(iters):
            u = a / np.maximum(K @ (K @ v.T).T, 1e-300)
            v = b / np.maximum(K @ (K @ u.T).T, 1e-300)
        return float((u * (Kc @ v @ K.T + K @ v @ Kc.T)).sum())

    def divergence(a, b):
        return ot(a, b) - 0.5 * (ot(a, a) + ot(b, b))

    hs = [4.0, 2.0, 1.0, 0.5, 0.25]
    rows = []
    for knob in ["cx", "width", "lean"]:
        k = KNOBS.index(knob)
        I0 = cov(ROOT, n)[0]
        raw, deb = [], []
        for h in hs:
            p = ROOT.copy(); p[k] += h
            I1 = cov(p, n)[0]
            raw.append(np.sqrt(max(ot(I0, I1), 0.0)))
            deb.append(np.sqrt(max(divergence(I0, I1), 0.0)))
        sl_r = np.polyfit(np.log(hs), np.log(np.maximum(raw, 1e-12)), 1)[0]
        sl_d = np.polyfit(np.log(hs), np.log(np.maximum(deb, 1e-12)), 1)[0]
        rows.append([f"{knob} raw", *[f"{v:.4f}" for v in raw], f"{sl_r:+.2f}"])
        rows.append([f"{knob} debiased", *[f"{v:.4f}" for v in deb], f"{sl_d:+.2f}"])
        rows.append([f"{knob} speed", *[f"{v/h:.4f}" for v, h in zip(deb, hs)], ""])
    table(rows, ["quantity"] + [f"h={h}" for h in hs] + ["slope"],
          [15] + [9] * len(hs) + [7])
    print("""
'speed' is the debiased distance divided by the knob step, the same quantity
part 3 tabulates.  Flat means the derivative exists.""")


def part6():
    print("=" * 78)
    print("PART 6  Where the breakdown actually comes from: the pixel pitch")
    print("=" * 78)
    print("""
Part 5 found that optimal transport gives EXACTLY the right answer for a
translation -- cost 1.0000 per pixel of travel -- but only down to some step,
below which the 1/sqrt(h) blow-up comes back.  This part asks what sets that
breakpoint.  If it were the sharp edges, refining the grid would not move it.

Same measurement as part 5 (debiased Sinkhorn, cx knob, coverage render), at
three grid finenesses.  eps is held at 0.02 canvas-px^2 throughout, i.e. an
entropic blur of 0.14 px, finer than every pitch tested.
""")
    hs = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    rows = []
    for n in [28, 56, 112]:
        pitch = CANVAS / n
        x = np.arange(float(n)) * pitch
        C1 = (x[:, None] - x[None, :]) ** 2
        K = np.exp(-C1 / 0.02); Kc = K * C1

        def ot(a, b, iters=4000):
            a = np.maximum(a, 1e-11); a = a / a.sum()
            b = np.maximum(b, 1e-11); b = b / b.sum()
            u = np.ones((n, n)); v = np.ones((n, n))
            for _ in range(iters):
                u = a / np.maximum(K @ (K @ v.T).T, 1e-300)
                v = b / np.maximum(K @ (K @ u.T).T, 1e-300)
            return float((u * (Kc @ v @ K.T + K @ v @ Kc.T)).sum())

        I0 = cov(ROOT, n)[0]; s0 = ot(I0, I0)
        vals = []
        for h in hs:
            p = ROOT.copy(); p[0] += h
            I1 = cov(p, n)[0]
            vals.append(np.sqrt(max(ot(I0, I1) - 0.5 * (s0 + ot(I1, I1)), 0.0)) / h)
        rows.append([f"{n}^2 (pitch {pitch:g})"] + [f"{v:.4f}" for v in vals])
    rows.append(["1/sqrt(h)"] + [f"{1/np.sqrt(h):.4f}" for h in hs])
    table(rows, ["grid"] + [f"h={h:g}" for h in hs] + [],
          [18] + [9] * len(hs))
    print("""
The flat run of 1.0000 ends at h = the pitch, at every fineness: 1.0 for the
28x28 grid, 0.5 for 56x56, 0.25 for 112x112.  So below one pixel the blow-up is
the SAMPLING GRID, not the sharp edges -- shifting a sampled picture by less
than a pixel is not a translation of it at all, it is a resampling, and no
choice of metric can make it one.  Above the pitch, optimal transport is exact
and costs nothing in lost detail.""")


PARTS = {1: part1, 2: part2, 3: part3, 4: part4, 5: part5, 6: part6}

if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:]] or sorted(PARTS)
    for i in want:
        PARTS[i]()
        print()
