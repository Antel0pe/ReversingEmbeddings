"""Five lines from the root 1: do they add up, and how many lines cover the family?

Question (user's words): start at the root 1, move each knob by +1 on its own
dial with the others held still -> five lines in pixel space.  If I combine two
lines, do I land on the image that really has both knobs changed?  How far off?
How many lines do I need to reach any point of the family?

Everything is measured in PIXEL space: an image is 784 coverage numbers, a move
is the difference of two images, and every error below is a length divided by
another length, in percent.  Nothing here is fitted unless the part says so.

Run:  .venv/bin/python analyze_generated_one_line_interpolation.py [part ...]
"""
import itertools
import json
import sys

import numpy as np

from grey_ones import KNOBS, RANGES, render

ROOT = np.array([14.0, 14.0, 19.75, 3.2, 0.0])
UNIT = np.eye(5)                      # +1 on each knob's own dial (px, px, px, px, deg)
SPAN = RANGES[:, 1] - RANGES[:, 0]    # 1, 1, 1.5, 2.8, 45
OUT = {}


def img(p):
    return render(np.asarray(p, float)).ravel().astype(np.float64)


def imgs(P):
    return render(np.asarray(P, float)).reshape(len(P), -1).astype(np.float64)


def pct(a, b):
    return 100.0 * np.linalg.norm(a - b) / np.linalg.norm(b)


I0 = img(ROOT)
LINES = np.array([img(ROOT + UNIT[k]) - I0 for k in range(5)])      # 5 x 784


# ---------------------------------------------------------------- part 1
def part1():
    """The five lines themselves: how long, and how independent."""
    print("PART 1  the five lines from the root 1 (+1 on each dial)")
    print(f"  root image length ||I0|| = {np.linalg.norm(I0):.4f}")
    print(f"  {'knob':8s} {'step':>8s} {'% of range':>11s} {'||line||':>9s} {'% of ||I0||':>12s}")
    rows = []
    for k in range(5):
        L = np.linalg.norm(LINES[k])
        print(f"  {KNOBS[k]:8s} {'+1':>8s} {100 / SPAN[k]:10.1f}% {L:9.4f} {100 * L / np.linalg.norm(I0):11.2f}%")
        rows.append(dict(knob=KNOBS[k], line_norm=L, pct_of_root=100 * L / np.linalg.norm(I0),
                         pct_of_range=100 / SPAN[k]))
    C = (LINES / np.linalg.norm(LINES, axis=1, keepdims=True))
    G = C @ C.T
    print("\n  cosine between lines (1 = same direction, 0 = perpendicular):")
    print("            " + " ".join(f"{n[:6]:>7s}" for n in KNOBS))
    for k in range(5):
        print(f"  {KNOBS[k]:9s} " + " ".join(f"{G[k, j]:7.3f}" for j in range(5)))
    s = np.linalg.svd(LINES, compute_uv=False)
    print(f"\n  singular values of the 5 lines: {np.array2string(s, precision=4)}")
    print(f"  -> 5 genuinely different directions (smallest/largest = {s[-1] / s[0]:.3f})")
    OUT["part1"] = dict(lines=rows, cosines=G.tolist(), singular_values=s.tolist())


# ---------------------------------------------------------------- part 2
def part2():
    """Add two lines -> the corner.  Average two images -> the midpoint."""
    print("PART 2  combining two lines, both ways, at +1 on each dial")
    print("  CORNER  : predict render(root + e_i + e_j) by I0 + line_i + line_j")
    print("  MIDPOINT: predict render(root + .5 e_i + .5 e_j) by (img_i + img_j)/2")
    print("  error = |predicted - true| / |true - I0|, i.e. percent of the move you made\n")
    print(f"  {'pair':22s} {'move len':>9s} {'corner err':>11s} {'mid err':>9s}")
    rows = []
    for i, j in itertools.combinations(range(5), 2):
        tc = img(ROOT + UNIT[i] + UNIT[j])
        pc = I0 + LINES[i] + LINES[j]
        tm = img(ROOT + 0.5 * UNIT[i] + 0.5 * UNIT[j])
        pm = 0.5 * (img(ROOT + UNIT[i]) + img(ROOT + UNIT[j]))
        mv = np.linalg.norm(tc - I0)
        ec = 100 * np.linalg.norm(pc - tc) / mv
        em = 100 * np.linalg.norm(pm - tm) / np.linalg.norm(tm - I0)
        print(f"  {KNOBS[i] + ' + ' + KNOBS[j]:22s} {mv:9.4f} {ec:10.2f}% {em:8.2f}%")
        rows.append(dict(pair=[KNOBS[i], KNOBS[j]], move_len=mv, corner_err=ec, mid_err=em))
    allp = ROOT + UNIT.sum(0)
    t5, p5 = img(allp), I0 + LINES.sum(0)
    e5 = 100 * np.linalg.norm(p5 - t5) / np.linalg.norm(t5 - I0)
    print(f"\n  all five at once (+1,+1,+1,+1,+1): error {e5:.2f}% of the move")
    print(f"  worst pair {max(rows, key=lambda r: r['corner_err'])['pair']}"
          f" at {max(r['corner_err'] for r in rows):.2f}%,"
          f" best {min(r['corner_err'] for r in rows):.2f}%")
    OUT["part2"] = dict(pairs=rows, all_five_err=e5)


# ---------------------------------------------------------------- part 3
def part3():
    """Error vs step size: does the miss shrink, and how fast."""
    print("PART 3  how the miss depends on how far you step")
    print("  step t means knob k moves by t * 1 unit; all five moved together")
    print("  err_abs = |pred - true| / |I0| ;  err_rel = |pred - true| / |true - I0|\n")
    ts = np.array([2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002])
    print(f"  {'t':>7s} {'move len':>10s} {'err_abs':>10s} {'err_rel':>10s}")
    rows = []
    for t in ts:
        lines = np.array([img(ROOT + t * UNIT[k]) - I0 for k in range(5)])
        tt = img(ROOT + t * UNIT.sum(0))
        pp = I0 + lines.sum(0)
        mv = np.linalg.norm(tt - I0)
        ea = 100 * np.linalg.norm(pp - tt) / np.linalg.norm(I0)
        er = 100 * np.linalg.norm(pp - tt) / mv
        print(f"  {t:7.3f} {mv:10.5f} {ea:9.4f}% {er:9.3f}%")
        rows.append(dict(t=float(t), move_len=mv, err_abs=ea, err_rel=er))
    for lo, hi in ((0.002, 0.05), (0.002, 0.5)):
        m = (ts >= lo) & (ts <= hi)
        sl = np.polyfit(np.log(ts[m]), np.log([r["err_abs"] for r, k in zip(rows, m) if k]), 1)[0]
        print(f"\n  log-log slope of err_abs vs t over t in [{lo}, {hi}]: {sl:.3f}")
    print("  slope 2 = the miss is purely second order: halve the step, quarter the miss")
    print("  the big steps sit above the power law, so they are past the local regime")
    OUT["part3"] = dict(rows=rows, slope=float(sl))


# ---------------------------------------------------------------- part 4
def part4():
    """Do the five root lines SPAN the family, or only point along it locally?"""
    print("PART 4  is the whole family inside the 5-line flat from the root?")
    rng = np.random.default_rng(0)
    P = RANGES[:, 0] + rng.random((3000, 5)) * SPAN
    D = imgs(P) - I0
    Q = np.linalg.qr(LINES.T)[0]                       # orthonormal basis of the 5 lines
    res = D - (D @ Q) @ Q.T
    r = 100 * np.linalg.norm(res, axis=1) / np.linalg.norm(D, axis=1)
    print("  3000 settings drawn uniformly from the default range box.")
    print(f"  distance from the 5-line flat, as % of the move from the root:")
    print(f"    median {np.median(r):.2f}%   p95 {np.percentile(r, 95):.2f}%   max {r.max():.2f}%")
    # best possible 5 directions through the root, for comparison
    s = np.linalg.svd(D, compute_uv=False)
    tot = (s ** 2).sum()
    best5 = 100 * np.sqrt(1 - (s[:5] ** 2).sum() / tot)
    got5 = 100 * np.linalg.norm(res) / np.linalg.norm(D)
    print(f"\n  pooled residual with the 5 KNOB lines        : {got5:.2f}%")
    print(f"  pooled residual with the best 5 directions   : {best5:.2f}%  (uncentred SVD of D)")
    ks = [5, 10, 20, 40, 70, 100]
    print("  best-possible residual vs number of directions through the root:")
    for k in ks:
        print(f"    {k:4d} directions -> {100 * np.sqrt(1 - (s[:k] ** 2).sum() / tot):6.2f}%")
    OUT["part4"] = dict(median=float(np.median(r)), p95=float(np.percentile(r, 95)),
                        max=float(r.max()), knob5=got5, best5=best5,
                        svd_curve={int(k): float(100 * np.sqrt(1 - (s[:k] ** 2).sum() / tot))
                                   for k in ks})


# ---------------------------------------------------------------- part 5
def part5():
    """Lines + products of lines: a polynomial in the five knobs, fitted."""
    print("PART 5  how many lines, if you may also use products of knobs (FITTED)")
    rng = np.random.default_rng(1)
    Ptr = RANGES[:, 0] + rng.random((4000, 5)) * SPAN
    Pte = RANGES[:, 0] + rng.random((2000, 5)) * SPAN
    Itr, Ite = imgs(Ptr), imgs(Pte)
    u_tr = 2 * (Ptr - RANGES[:, 0]) / SPAN - 1
    u_te = 2 * (Pte - RANGES[:, 0]) / SPAN - 1

    def design(u, deg):
        cols, names = [], []
        for e in itertools.product(*[range(deg + 1)] * 5):
            if sum(e) <= deg:
                cols.append(np.prod(u ** np.array(e), axis=1))
                names.append(e)
        return np.column_stack(cols), names

    rows = []
    print(f"  {'model':34s} {'lines':>6s} {'train':>8s} {'held-out':>9s}")
    for deg in (1, 2, 3, 4):
        A, nm = design(u_tr, deg)
        B, _ = design(u_te, deg)
        W = np.linalg.lstsq(A, Itr, rcond=None)[0]
        etr = np.median(100 * np.linalg.norm(A @ W - Itr, axis=1) / np.linalg.norm(Itr, axis=1))
        ete = np.median(100 * np.linalg.norm(B @ W - Ite, axis=1) / np.linalg.norm(Ite, axis=1))
        lbl = {1: "straight lines (affine)", 2: "+ pair products", 3: "+ triple products",
               4: "+ quadruple products"}[deg]
        print(f"  degree {deg}  {lbl:25s} {len(nm):6d} {etr:7.2f}% {ete:8.2f}%")
        rows.append(dict(degree=deg, n_terms=len(nm), train=etr, test=ete))

    # PCA ceiling: the best any set of k fixed images can do, coefficients measured
    mu = Itr.mean(0)
    V = np.linalg.svd(Itr - mu, full_matrices=False)[2]
    print("\n  for comparison, the best possible k fixed images (PCA of the family),")
    print("  with the k numbers read off the target image rather than from the knobs:")
    pca = {}
    for k in (1, 2, 5, 10, 20, 40, 70, 100):
        R = (Ite - mu) - ((Ite - mu) @ V[:k].T) @ V[:k]
        e = np.median(100 * np.linalg.norm(R, axis=1) / np.linalg.norm(Ite, axis=1))
        pca[k] = e
        print(f"    {k:4d} images -> {e:7.3f}%")
    R = (Ite - mu)
    need = {}
    for tol in (5.0, 1.0, 0.1):
        for k in range(1, 400):
            r = R - (R @ V[:k].T) @ V[:k]
            if np.median(100 * np.linalg.norm(r, axis=1) / np.linalg.norm(Ite, axis=1)) < tol:
                need[tol] = k
                break
    print("  fixed images needed, best case: " +
          ", ".join(f"{t:g}% -> {k}" for t, k in need.items()))
    print("  (a tensor-product guess from part 7, 3*2*2*3*13, would be 468)")
    OUT["part5"] = dict(poly=rows, pca={int(k): float(v) for k, v in pca.items()},
                        need={str(t): int(k) for t, k in need.items()})


# ---------------------------------------------------------------- part 6
def part6():
    """If 5 lines only work locally, how many patches of 5 lines cover the box?"""
    print("PART 6  patches: 5 fresh lines per patch, how many patches for 1%?")
    print("  a patch is a centre plus its 5 lines, measured with step h (fraction of")
    print("  each knob's range).  Points are drawn uniformly in the +-h box around the")
    print("  centre.  error = |pred - true| / |true| , percent of the image.\n")
    rng = np.random.default_rng(2)
    centres = [ROOT] + list(RANGES[:, 0] + rng.random((4, 5)) * SPAN)
    hs = 0.5 ** np.arange(1, 11)
    print(f"  {'h (frac of range)':>18s} " + " ".join(f"{'c' + str(i):>8s}" for i in range(5)) + "   p95 all")
    rows = []
    for h in hs:
        p95s, pool = [], []
        for c in centres:
            lines = np.array([img(c + h * SPAN * UNIT[k]) - img(c) for k in range(5)])
            ic = img(c)
            du = rng.uniform(-h, h, (400, 5)) * SPAN
            T = imgs(c + du)
            pred = ic + (du / (h * SPAN)) @ lines
            e = 100 * np.linalg.norm(pred - T, axis=1) / np.linalg.norm(T, axis=1)
            p95s.append(np.percentile(e, 95))
            pool.append(e)
        allp = np.percentile(np.concatenate(pool), 95)
        print(f"  {h:18.5f} " + " ".join(f"{v:7.3f}%" for v in p95s) + f"  {allp:7.3f}%")
        rows.append(dict(h=float(h), per_centre=[float(v) for v in p95s], p95=float(allp)))
    fit = [r for r in rows if r["p95"] < 5.0]          # asymptotic regime only
    a, b = np.polyfit(np.log([r["h"] for r in fit]), np.log([r["p95"] for r in fit]), 1)
    h1 = float(np.exp((np.log(1.0) - b) / a))
    n1 = (1 / (2 * h1)) ** 5
    print(f"\n  fit: p95 error = {np.exp(b):.3f}% * h^{a:.2f}   (slope 2 = second order)")
    print(f"  h for 1% p95 error: {h1:.4f} of each range  ->  {1 / (2 * h1):.1f} patches per knob")
    print(f"  patches to tile the whole 5-knob box at 1%: {n1:,.0f}  ({5 * n1:,.0f} lines)")
    for tol in (0.1, 1.0, 5.0):
        ht = float(np.exp((np.log(tol) - b) / a))
        print(f"    tolerance {tol:4.1f}%  ->  h {ht:.4f}, {(1 / (2 * ht)) ** 5:,.0f} patches,"
              f" {5 * (1 / (2 * ht)) ** 5:,.0f} lines")
    OUT["part6"] = dict(rows=rows, slope=float(a), h_1pct=h1, patches_1pct=float(n1))


# ---------------------------------------------------------------- part 7
def part7():
    """One knob at a time: how many lines does a single knob's own sweep need?"""
    print("PART 7  one knob, swept across its whole default range, others at root")
    print("  single line  = error of root + (one line, scaled) against the true sweep")
    print("  lines for 1% = how many fixed images it takes to hold that one curve")
    print("  (uncentred SVD of the sweep's moves away from the root)\n")
    print(f"  {'knob':8s} {'range':>14s} {'sweep len':>10s} {'single line':>12s} "
          f"{'2 lines':>8s} {'3 lines':>8s} {'lines for 1%':>13s}")
    rows = []
    for k in range(5):
        g = np.linspace(RANGES[k, 0], RANGES[k, 1], 201)
        P = np.tile(ROOT, (201, 1))
        P[:, k] = g
        D = imgs(P) - I0
        q = LINES[k] / np.linalg.norm(LINES[k])
        pred = np.outer(D @ q, q)
        e1 = 100 * np.linalg.norm(D - pred) / np.linalg.norm(imgs(P))
        sv = np.linalg.svd(D, compute_uv=False)
        tot = (sv ** 2).sum()
        nrm = np.linalg.norm(imgs(P))
        def res(n):
            return 100 * np.sqrt(max(tot - (sv[:n] ** 2).sum(), 0)) / nrm
        n1 = int(np.argmax([res(n) < 1.0 for n in range(1, 31)]) + 1)
        print(f"  {KNOBS[k]:8s} {f'{RANGES[k,0]:g} to {RANGES[k,1]:g}':>14s} "
              f"{np.linalg.norm(D[-1]):10.4f} {e1:11.2f}% {res(2):7.2f}% {res(3):7.2f}% {n1:13d}")
        rows.append(dict(knob=KNOBS[k], single_line=e1, two=res(2), three=res(3), lines_1pct=n1))
    print("\n  a straight knob would read 0.00% on a single line; these are the")
    print("  curvatures that break the 5-line picture once the step is big.")
    OUT["part7"] = rows


# ---------------------------------------------------------------- part 8
def part8():
    """The same five lines, moved out of pixel space and into warp space."""
    from scipy.ndimage import map_coordinates
    print("PART 8  five lines that DO add up: warp coefficients, not pixels")
    print("  every setting is the root stroke pushed through one affine map of the")
    print("  canvas:   x = bx + sx*X + a12*Y ,  y = by + sy*Y   (no vertical shear)")
    print("  so each image is described by five warp numbers (sx, a12, sy, bx, by).\n")
    cx0, cy0, h0, w0 = ROOT[0], ROOT[1], ROOT[2], ROOT[3]

    def coeffs(P):
        cx, cy, h, w, lean = P.T
        sx, sy = w / w0, h / h0
        a12 = -np.tan(np.radians(lean)) * sy
        bx = cx - cx0 * sx - cy0 * a12
        by = cy - cy0 * sy
        return np.column_stack([sx, a12, sy, bx, by])

    rng = np.random.default_rng(3)
    P = RANGES[:, 0] + rng.random((2000, 5)) * SPAN
    C = coeffs(P)
    for name, U in (("the dials as given  (cx, cy, height, width, lean)", P),
                    ("lean swapped for height*tan(lean)", np.column_stack(
                        [P[:, 0], P[:, 1], P[:, 2], P[:, 3],
                         P[:, 2] * np.tan(np.radians(P[:, 4]))]))):
        A = np.column_stack([U, np.ones(len(U))])
        R = C - A @ np.linalg.lstsq(A, C, rcond=None)[0]
        print(f"  5 lines in warp space, {name}:")
        print(f"    residual {np.abs(R).max():.3e} absolute"
              f"  ({100 * np.linalg.norm(R) / np.linalg.norm(C - C.mean(0)):.2e}% relative)")
    print("  -> with the right fifth dial the five warp lines add EXACTLY: the family")
    print("     is a flat 5-dimensional plane in warp space, not a curved sheet.\n")

    gy, gx = np.mgrid[0:28, 0:28]
    xq, yq = gx + 0.5, gy + 0.5
    errs = []
    for c, p in zip(C[:300], P[:300]):
        sx, a12, sy, bx, by = c
        Y = (yq - by) / sy
        X = (xq - bx - a12 * Y) / sx
        warped = map_coordinates(I0.reshape(28, 28), [Y - 0.5, X - 0.5], order=1, cval=0.0)
        errs.append(pct(warped.ravel(), img(p)))
    print(f"  cost of turning those five warp numbers back into 28x28 pixels, by")
    print(f"  resampling the root image: median {np.median(errs):.2f}%,"
          f" p95 {np.percentile(errs, 95):.2f}% of the image")
    print("\n  control: is that 15% the five lines, or the 28x28 template?  Redo the")
    print("  identical warp on a finer root template, then area-average back to 28x28.")
    from coverage_grids import render_coverage
    curve = {28: float(np.median(errs))}
    for n in (56, 112, 224, 448):
        f = n // 28
        root_f = render_coverage(ROOT[None], n_pix=n)[0]
        pit = 28.0 / n
        gy2, gx2 = np.mgrid[0:n, 0:n]
        xq2, yq2 = (gx2 + 0.5) * pit, (gy2 + 0.5) * pit
        e = []
        for c, q in zip(C[:100], P[:100]):
            sx, a12, sy, bx, by = c
            Y = (yq2 - by) / sy
            X = (xq2 - bx - a12 * Y) / sx
            wf = map_coordinates(root_f, [Y / pit - 0.5, X / pit - 0.5], order=1, cval=0.0)
            e.append(pct(wf.reshape(28, f, 28, f).mean((1, 3)).ravel(), img(q)))
        curve[n] = float(np.median(e))
        print(f"    root template {n}x{n} (pitch {pit:g} px) -> median {curve[n]:6.2f}%")
    print("  the five warp lines are unchanged throughout; only the template changes.")
    OUT["part8"] = dict(warp_decode_median=float(np.median(errs)),
                        warp_decode_p95=float(np.percentile(errs, 95)),
                        template_curve=curve)


PARTS = dict(part1=part1, part2=part2, part3=part3, part4=part4, part5=part5, part6=part6, part7=part7, part8=part8)

if __name__ == "__main__":
    want = sys.argv[1:] or list(PARTS)
    for name in want:
        PARTS[name]()
        print()
    with open("figures/generated_one_line_interpolation/results.json", "w") as f:
        json.dump(OUT, f, indent=1)
