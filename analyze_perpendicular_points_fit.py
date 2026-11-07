"""Fitting a line through mutually perpendicular points in 784-D pixel space.

Question: a set of perpendicular images looks like stepwise jumps to ordinary
fits. Does high dimension help, or is there a kind of fit that treats these
points differently? Small decisive tests on the generated 1 family.

Run from the repo root: .venv/bin/python analyze_perpendicular_points_fit.py
"""
from pathlib import Path

import matplotlib
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as nd_shift
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

from grey_ones import KNOBS, RANGES, render

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("figures/perpendicular_points_fit")
CY, H, W = 14.0, 19.75, 3.2
TRAIN = np.array([4.0, 10.0, 16.0, 22.0])
TEST = np.array([c for c in np.arange(4, 22.01, 0.25) if c not in TRAIN])
MID = np.array([7.0, 13.0, 19.0])


def side(cx):
    return render([[c, CY, H, W, 0.0] for c in np.atleast_1d(cx)], dtype=np.float64)


def lean_imgs(a):
    return render([[14.0, 14.0, H, W, x] for x in np.atleast_1d(a)], dtype=np.float64)


def rel(pred, true):
    """Miss as a percent of the true image's length, one number per image."""
    p, t = pred.reshape(len(true), -1), true.reshape(len(true), -1)
    return 100 * np.linalg.norm(p - t, axis=1) / np.linalg.norm(t, axis=1)


def best_blend(train, true):
    """Closest any weighted mix of the training images can get to each image."""
    A = train.reshape(len(train), -1).T
    t = true.reshape(len(true), -1).T
    w = np.linalg.lstsq(A, t, rcond=None)[0]
    return (A @ w).T.reshape(true.shape)


def fourier_shift(img, dx, dy=0.0):
    fy = np.fft.fftfreq(img.shape[0])[:, None]
    fx = np.fft.rfftfreq(img.shape[1])[None, :]
    F = np.fft.rfft2(img) * np.exp(-2j * np.pi * (fx * dx + fy * dy))
    return np.fft.irfft2(F, s=img.shape)


def fit_one_shape(train, t):
    """Fit `one shape, slid at a steady speed` from the images and their t only.

    The speed is the slope of the lowest-frequency Fourier angle against t. It
    is a straight-line fit, and it is the only thing fitted besides the shape.
    """
    F = np.fft.rfft2(train)
    vx = np.polyfit(t, np.unwrap(np.angle(F[:, 0, 1])), 1)[0] * -28 / (2 * np.pi)
    vy = np.polyfit(t, np.unwrap(np.angle(F[:, 1, 0])), 1)[0] * -28 / (2 * np.pi)
    t0 = t.mean()
    shape = np.mean(
        [fourier_shift(im, -vx * (ti - t0), -vy * (ti - t0)) for im, ti in zip(train, t)],
        axis=0,
    )
    return shape, t0, vx, vy


def slide(shape, t0, vx, vy, t, order="fourier"):
    out = []
    for ti in t:
        if order == "fourier":
            out.append(fourier_shift(shape, vx * (ti - t0), vy * (ti - t0)))
        else:
            out.append(nd_shift(shape, (vy * (ti - t0), vx * (ti - t0)), order=1))
    return np.clip(np.array(out), 0, 1)


def shear(img, s):
    rows = np.arange(28) + 0.5
    ink = img.sum(1)
    pivot = (rows * ink).sum() / ink.sum()
    return np.array([np.interp(np.arange(28), np.arange(28) - s * (pivot - r), img[i], left=0, right=0)
                     for i, r in enumerate(rows)])


EDGES = np.arange(29.0)
U = np.linspace(0, 1, 65)


def ink_positions(imgs):
    """Per row: how much ink, and where the 0%, 1/64, ... 100% marks of that ink sit.

    Returns (n, 28, 66). A straight move of ink is a straight line in these
    numbers, whereas in pixel values it is a run of one-pixel humps.
    """
    imgs = np.asarray(imgs)
    out = np.zeros((len(imgs), 28, 66))
    for n, im in enumerate(imgs):
        filled = []
        for r, row in enumerate(im):
            m = row.sum()
            out[n, r, 0] = m
            if m <= 1e-12:
                continue
            cdf = np.r_[0.0, np.cumsum(row)] / m
            idx = np.where(np.diff(cdf) > 0)[0]
            xp = np.r_[cdf[idx], cdf[idx[-1] + 1]]
            fp = np.r_[EDGES[idx], EDGES[idx[-1] + 1]]
            out[n, r, 1:] = np.interp(U, xp, fp)
            filled.append(r)
        for r in range(28):  # an empty row borrows the nearest inked row's positions
            if r not in filled:
                out[n, r, 1:] = out[n, filled[int(np.argmin(np.abs(np.array(filled) - r)))], 1:]
    return out


def from_ink_positions(code):
    imgs = np.zeros((len(code), 28, 28))
    for n, c in enumerate(code):
        for r in range(28):
            q = np.maximum.accumulate(c[r, 1:])
            imgs[n, r] = max(c[r, 0], 0) * np.diff(np.interp(EDGES, q, U, left=0, right=1))
    return np.clip(imgs, 0, 1)


def quad_features(p):
    z = (p - LO) / (HI - LO) * 2 - 1
    cols = [np.ones(len(z))] + [z[:, i] for i in range(5)]
    cols += [z[:, i] * z[:, j] for i in range(5) for j in range(i, 5)]
    return np.stack(cols, 1)


def cubic_features(p):
    z = (p - LO) / (HI - LO) * 2 - 1
    cols = [c for c in quad_features(p).T]
    cols += [z[:, i] * z[:, j] * z[:, k] for i in range(5) for j in range(i, 5) for k in range(j, 5)]
    return np.stack(cols, 1)


LO = np.asarray(RANGES, dtype=float)[:, 0]
HI = np.asarray(RANGES, dtype=float)[:, 1]


def mlp_fit(train, t, test_t, seed=0, hidden=128, steps=6000):
    try:
        import torch
    except ImportError:
        return None
    torch.manual_seed(seed)
    m = (t.mean(), t.std())
    X = torch.tensor((t - m[0]) / m[1], dtype=torch.float32)[:, None]
    Y = torch.tensor(train.reshape(len(train), -1), dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(1, hidden), torch.nn.Tanh(),
                              torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
                              torch.nn.Linear(hidden, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((net(X) - Y) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        q = torch.tensor((test_t - m[0]) / m[1], dtype=torch.float32)[:, None]
        return net(q).numpy().reshape(len(test_t), 28, 28).astype(np.float64), float(loss)


def report(name, pred, true, midx):
    e = rel(pred, true)
    print(f"  {name:<44s} all {e.mean():6.1f}%   midpoints {e[midx].mean():6.1f}%")
    return e


# ---------------------------------------------------------------- the points
train = side(TRAIN)
true = side(TEST)
midx = np.isin(TEST, MID)
v = train.reshape(4, -1)
cos = (v @ v.T) / np.outer(np.linalg.norm(v, axis=1), np.linalg.norm(v, axis=1))
D = np.linalg.norm(v[:, None] - v[None], axis=2)
print("PART 0: the four training images, cx =", TRAIN)
print("  largest cosine between any two:", cos[~np.eye(4, dtype=bool)].max())
print("  pairwise distances: min %.6f max %.6f" % (D[D > 0].min(), D.max()))
print("  midpoint image vs all training images, largest cosine: %.4f" % (
    (side(MID).reshape(3, -1) @ v.T / np.outer(np.linalg.norm(side(MID).reshape(3, -1), axis=1),
                                               np.linalg.norm(v, axis=1))).max()))

# ------------------------------------------------------------- ordinary fits
print("\nPART 1: ordinary fits, knob -> image, miss as % of image (blank canvas = 100%)")
near = train[np.abs(TEST[:, None] - TRAIN[None]).argmin(1)]
line = np.stack([(1 - (c - TRAIN[i]) / 6) * train[i] + (c - TRAIN[i]) / 6 * train[i + 1]
                 for c in TEST for i in [min(int((c - 4) // 6), 2)]])
spl = CubicSpline(TRAIN, train, axis=0)(TEST)
sig = 6.0
Kt = np.exp(-((TRAIN[:, None] - TRAIN[None]) ** 2) / (2 * sig**2))
Kq = np.exp(-((TEST[:, None] - TRAIN[None]) ** 2) / (2 * sig**2))
kern = (Kq @ np.linalg.solve(Kt, v)).reshape(true.shape)
bb = best_blend(train, true)
ordinary = {}
ordinary["stairs"] = report("stairs (nearest training image)", near, true, midx)
ordinary["line"] = report("straight lines between points", line, true, midx)
ordinary["spline"] = report("smooth curve, each pixel separately", spl, true, midx)
ordinary["kernel"] = report("Gaussian kernel fit", kern, true, midx)
res = mlp_fit(train, TRAIN, TEST)
if res is not None:
    mlp, loss = res
    ordinary["mlp"] = report(f"neural net (train loss {loss:.1e})", mlp, true, midx)
    r2 = np.linalg.norm(mlp.reshape(len(TEST), -1) - best_blend(train, mlp).reshape(len(TEST), -1), axis=1)
    print("    share of net's output outside the training images' mixes: %.1f%%" % (
        100 * (r2 / np.linalg.norm(mlp.reshape(len(TEST), -1), axis=1)).mean()))
else:
    print("  neural net: torch not installed, skipped")
inkfit = from_ink_positions(CubicSpline(TRAIN, ink_positions(train), axis=0)(TEST))
ordinary["ink"] = report("same smooth curve, on ink positions", inkfit, true, midx)
print(f"    at whole-pixel positions: {ordinary['ink'][np.isclose(TEST, np.round(TEST))].mean():.1f}%; between pixels: {ordinary['ink'][~np.isclose(TEST, np.round(TEST))].mean():.1f}%")
ordinary["bb"] = report("BEST POSSIBLE mix of the 4 images (cheating)", bb, true, midx)

# ----------------------------------------------------------- one shape, moved
print("\nPART 2: fit one shape plus a sliding speed, from the same 4 images and labels")
shape, t0, vx, vy = fit_one_shape(train, TRAIN)
print(f"  fitted speed: {vx:.4f} px per unit cx sideways (truth 1), {vy:.4f} vertically (truth 0)")
pred = slide(shape, t0, vx, vy, TEST)
e_shape = report("one shape slid (Fourier resampling)", pred, true, midx)
e_ord1 = report("one shape slid (linear resampling)", slide(shape, t0, vx, vy, TEST, "linear"), true, midx)
ints = np.isclose(TEST, np.round(TEST))
print(f"    at whole-pixel positions only: {e_shape[ints].mean():.1f}%; between pixels: {e_shape[~ints].mean():.1f}%")

# ------------------------------------------------------------ scramble control
print("\nPART 3: same experiment with pixel order scrambled (same permutation for every image)")
perm = np.random.default_rng(0).permutation(784)
scr = lambda a: a.reshape(len(a), -1)[:, perm].reshape(a.shape)
inv = np.argsort(perm)
s_train, s_true = scr(train), scr(true)
sv = s_train.reshape(4, -1)
print("  largest cosine: %.1e; pairwise distances unchanged: %s" % (
    (sv @ sv.T)[~np.eye(4, dtype=bool)].max(), np.allclose(np.linalg.norm(sv[:, None] - sv[None], axis=2), D)))
s_spl = CubicSpline(TRAIN, s_train, axis=0)(TEST)
report("smooth curve, each pixel separately", s_spl, s_true, midx)
s_ink = from_ink_positions(CubicSpline(TRAIN, ink_positions(s_train), axis=0)(TEST))
report("same smooth curve, on ink positions", s_ink, s_true, midx)
s = fit_one_shape(s_train, TRAIN)
report("one shape slid", slide(*s, TEST), s_true, midx)

# --------------------------------------------------------- image -> number
print("\nPART 4: the other direction, image -> cx, both by a straight-line rule")
A = np.c_[v, np.ones(4)]
coef = np.linalg.lstsq(A, TRAIN, rcond=None)[0]
memo = lambda im: im.reshape(len(im), -1) @ coef[:-1] + coef[-1]
print("  fitted on the 4 images: training miss %.1e px, held-out miss %.2f px" % (
    np.abs(memo(train) - TRAIN).max(), np.abs(memo(true) - TEST).mean()))
print("    held-out midpoints read as", np.round(memo(side(MID)), 2), "truth", MID)
xs = np.arange(28) + 0.5
com = lambda im: (im.sum(1) * xs).sum(1) / im.sum((1, 2))
print("  centre of ink, nothing fitted: held-out miss %.2e px" % np.abs(com(true) - TEST).mean())
print("    scrambled pixels, same rule: held-out miss %.2f px" % np.abs(com(s_true) - TEST).mean())
print("    scrambled pixels, fitted on the 4: held-out miss %.2f px (identical to unscrambled)" % (
    np.abs(s_true.reshape(len(TEST), -1) @ np.linalg.lstsq(np.c_[sv, np.ones(4)], TRAIN, rcond=None)[0][:-1]
           + np.linalg.lstsq(np.c_[sv, np.ones(4)], TRAIN, rcond=None)[0][-1] - TEST).mean()))

# ------------------------------------------------------------- Pythagoras
print("\nPART 5: do squared distances add up along the sweep?")
fine = np.arange(4, 22.01, 1.0)
im = side(fine)
f = im.reshape(len(fine), -1)
d2 = ((f[:, None] - f[None]) ** 2).sum(2)
for sep in [1, 2, 3, 4, 5, 6, 8, 12]:
    print(f"  {sep:2d} px apart: squared distance {np.diag(d2, sep).mean():7.3f}  (per px {np.diag(d2, sep).mean()/sep:6.3f})")
rng = np.random.default_rng(1)
uneven = np.sort(rng.uniform(4, 22, 40))
g = side(uneven).reshape(40, -1)
d2u = ((g[:, None] - g[None]) ** 2).sum(2)
uneven2 = np.sort(rng.uniform(4, 22, 160))
g = side(uneven2).reshape(160, -1)
d2v = ((g[:, None] - g[None]) ** 2).sum(2)
for name, tt, g2 in [("19 images 1 px apart", fine, d2), ("40 unevenly spaced images", uneven, d2u),
                     ("160 unevenly spaced images", uneven2, d2v),
                     ("the 4 perpendicular images", TRAIN, D**2)]:
    sat = g2.max() * 0.999
    links = np.where(g2 < sat, g2, 0)
    path = shortest_path(links, method="D", directed=False)
    connected = np.isfinite(path).all()
    if connected:
        end = np.unravel_index(np.argmax(path), path.shape)[0]
        print(f"  order from distances, {name}: rank correlation {abs(spearmanr(path[end], tt)[0]):.4f}")
    else:
        print(f"  order from distances, {name}: no overlapping pairs, so nothing to chain; all {len(tt)}! orders fit equally")

print("\nPART 6: how close before plain blending works? (straight line, midpoints across the sweep)")
cs = np.arange(8, 20.001, 0.05)
for h in [6, 4, 3, 2, 1, 0.5, 0.25, 0.1]:
    e = rel(0.5 * (side(cs - h / 2) + side(cs + h / 2)), side(cs))
    print(f"  points {h:5.2f} px apart: miss at midpoint mean {e.mean():6.1f}% worst {e.max():6.1f}%")

print("\nPART 6b: blur first, then fit. Miss is measured against the blurred truth; detail is gone either way")
for sg in [0, 1, 2, 3, 4]:
    b = lambda a: gaussian_filter(a, (0, sg, sg)) if sg else a
    bt = b(train).reshape(4, -1)
    cb = (bt @ bt.T / np.outer(np.linalg.norm(bt, axis=1), np.linalg.norm(bt, axis=1)))[0, 1]
    li = rel(CubicSpline(TRAIN, b(train), axis=0)(TEST), b(true))
    print(f"  blur {sg} px: neighbouring points' cosine {cb:.3f}, smooth curve miss {li.mean():5.1f}% (mid {li[midx].mean():5.1f}%), "
          f"blurred truth vs sharp truth {rel(b(true), true).mean():5.1f}%")

# ------------------------------------------------------------- lean
print("\nPART 7: the lean knob, where whole images overlap but moves turn")
LT = np.array([-10.0, 5.0, 20.0, 35.0])
LQ = np.array([a for a in np.arange(-10, 35.01, 1.0) if a not in LT])
lt, lq = lean_imgs(LT), lean_imgs(LQ)
lv = lt.reshape(4, -1)
mv = np.diff(lean_imgs(np.arange(-10, 36, 5.0)).reshape(10, -1), axis=0)
mc = (mv[:-1] * mv[1:]).sum(1) / (np.linalg.norm(mv[:-1], axis=1) * np.linalg.norm(mv[1:], axis=1))
print("  cosine between consecutive 5-degree moves, mean %.3f" % mc.mean())
print("  cosine between training images, min %.3f" % (
    (lv @ lv.T / np.outer(np.linalg.norm(lv, axis=1), np.linalg.norm(lv, axis=1))).min()))
lmid = np.isin(LQ, [-2.0, 13.0, 27.0])
lline = np.stack([(1 - (a - LT[i]) / 15) * lt[i] + (a - LT[i]) / 15 * lt[i + 1]
                  for a in LQ for i in [min(int((a + 10) // 15), 2)]])
report("straight lines between points", lline, lq, lmid)
report("smooth curve, each pixel separately", CubicSpline(LT, lt, axis=0)(LQ), lq, lmid)
report("BEST POSSIBLE mix of the 4 images (cheating)", best_blend(lt, lq), lq, lmid)
T = lt[1]
grid = np.linspace(-1.5, 1.5, 1201)
sfit = [grid[np.argmin([np.linalg.norm(shear(T, g) - x) for g in grid])] for x in lt]
print("  shear that best carries the 5-degree image to each training image:", np.round(sfit, 3))
print("    true tan(lean) - tan(5):", np.round(np.tan(np.radians(LT)) - np.tan(np.radians(5)), 3))
poly = np.polyfit(LT, sfit, 3)
slanted = np.stack([np.clip(shear(T, np.polyval(poly, a)), 0, 1) for a in LQ])
report("one shape, slanted; fit the slant by lean", slanted, lq, lmid)

li = from_ink_positions(CubicSpline(LT, ink_positions(lt), axis=0)(LQ))
report("same smooth curve, on ink positions", li, lq, lmid)

print("\nPART 8: all five knobs at once, default ranges, random settings, held-out test of 400")
rng = np.random.default_rng(2)
Pq = LO + (HI - LO) * rng.random((400, 5))
Iq = render(Pq, dtype=np.float64)
Cq = ink_positions(Iq)
for N, feats, nm in [(40, quad_features, "2nd-degree"), (200, quad_features, "2nd-degree"),
                     (200, cubic_features, "3rd-degree"), (1000, cubic_features, "3rd-degree")]:
    Pt = LO + (HI - LO) * rng.random((N, 5))
    It = render(Pt, dtype=np.float64)
    Ct = ink_positions(It)
    Xt, Xq = feats(Pt), feats(Pq)
    Bp = np.linalg.lstsq(Xt, It.reshape(N, -1), rcond=None)[0]
    Bi = np.linalg.lstsq(Xt, Ct.reshape(N, -1), rcond=None)[0]
    ep = rel(Xq @ Bp, Iq)
    ei = rel(from_ink_positions((Xq @ Bi).reshape(400, 28, 66)), Iq)
    print(f"  {N:5d} examples, {nm} formula: fit pixels {ep.mean():5.1f}%   fit ink positions {ei.mean():5.1f}%   (worst {ei.max():.1f}%)")
print("  round trip, encode then decode with no fit: %.2f%%" % rel(from_ink_positions(Cq), Iq).mean())
print("  by knob, ink-position fit (1000 examples, 3rd-degree), error vs knob quartile:")
for k, kn in enumerate(KNOBS):
    q = np.digitize(Pq[:, k], np.quantile(Pq[:, k], [0.25, 0.5, 0.75]))
    print(f"    {kn:7s}", " ".join(f"{ei[q == j].mean():5.1f}%" for j in range(4)))

# ------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 4, figsize=(15, 5.3), constrained_layout=True)
c = 7.25
i = int(np.argmin(np.abs(TEST - c)))
show = [(train.sum(0), "The 4 training 1s, drawn on one canvas\nNo pixel shared, so all perpendicular\nStroke centres: 4, 10, 16, 22"),
        (true[i], "The true 1 centred at 7.25\nHeld out: no fit ever saw it\n "),
        (spl[i], f"Smooth curve through pixel values\nRead at 7.25\nMiss: {ordinary['spline'][i]:.0f}% of the image"),
        (inkfit[i], f"Same smooth curve through ink positions\nRead at 7.25\nMiss: {ordinary['ink'][i]:.0f}% of the image")]
for ax, (img, title) in zip(axes, show):
    h = ax.imshow(img, cmap="RdGy", vmin=-1, vmax=1)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([]), ax.set_yticks([])
    if ax is not axes[0]:
        ax.axvline(c - 0.5, color="tab:blue", lw=1.2, ls=":")
cb = fig.colorbar(h, ax=axes, shrink=0.8, pad=0.01)
cb.set_label("Ink: 0 white, 1 black. Red is negative ink, which is impossible")
fig.suptitle("Filling in a 1 between four perpendicular examples (generated 1s, moved sideways only). "
             "Blue dots mark centre 7.25. A blank canvas would miss by 100%", fontsize=13)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "in_between_points.png", dpi=130)
print("\nsaved", OUT / "in_between_points.png")
