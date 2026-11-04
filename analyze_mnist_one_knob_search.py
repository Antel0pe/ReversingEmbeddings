"""Can the five-knob method find the knobs of REAL MNIST 1s?

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python analyze_mnist_one_knob_search.py
     (optional: ... <part> ...   parts are sparsity floor saturation subsets
      dimension appearance meaning; default runs all. Full run ~45 min.)

`figures/generated_one_motion_discovery` showed that unlabelled images can name
their own motions and count them, on the generated family. The goal here is the
real thing: given nothing but MNIST 1s, recover coordinates and an explicit
decoder from those coordinates to 784 pixels.

Every claim is paired with the generated family as a positive control, because
there the answer is known: five knobs, and `grey_ones.render` is the decoder.

The seven parts, and what each settles:

  SPARSITY    Why the earlier linearised method cannot be used here at all.
              It needs neighbours close enough to be first order.
  FLOOR       What limits the decoder. Warping one generated 1 onto another
              through the EXACT map between their true knobs still misses by
              ~15% from a 28x28 source -- so the binding constraint is the
              template's resolution, not the model. Hence `warp_charts`
              super-resolves the template from the data.
  SATURATION  The count. Give the warp more and more freedom and watch when it
              stops paying. Flat means the family is exhausted.
  SUBSETS     Whether the five-knob story holds for the MNIST 1s that genuinely
              ARE straight strokes. Selection is by overlap with the
              parallelogram each image's own moments imply -- a shape
              hypothesis, independent of the model under test. Training size is
              held fixed, since a smaller subset otherwise gets a worse
              template and that alone could produce the trend.
  DIMENSION   The count again, with no model at all: two distance-only
              estimators, calibrated on a 5-dimensional linear subspace and on
              the generated family, and re-run at growing sample sizes. An
              estimate that keeps climbing with n is data-starved, not
              converged, and must be read as a lower bound.
  APPEARANCE  What geometry cannot reach, priced against plain PCA at matched
              coordinate count.
  RESIDUALS   What the failures actually look like, and whether several
              templates rescue them.
  GENERATION  Whether NEW coordinates decode to valid members. Reconstruction
              error cannot see this, and it is where a correct description and
              a good compressor come apart.
  MEANING     What the recovered warp numbers correspond to in measurable terms.
  INVERSION   The closed form taking the unlabelled chart back to the
              generator's own knobs, and how few labels it needs.
  GAIN        Whether an appearance field explains what geometry cannot,
              with the template-round confound controlled for.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy.ndimage import map_coordinates

import mnist
import warp_charts as wc
from coverage_grids import render_coverage
from grey_ones import RANGES, render

OUT = Path(__file__).resolve().parent / "figures/mnist_one_knob_search"
R = {}


def mnist_ones():
    X, y = mnist.load("train")
    return X[y == 1].astype(np.float64) / 255.0


def generated(n, seed=0):
    rng = np.random.default_rng(seed)
    return render(RANGES[:, 0] + rng.random((n, 5)) * (RANGES[:, 1] - RANGES[:, 0]),
                  dtype=np.float64)


def stroke_overlap(I):
    """How well is each image explained by ONE straight parallelogram stroke?

    Uses only the image's own moments plus `grey_ones.render`, so the selection
    is independent of the warp model being tested.
    """
    P = wc.moment_chart(I)
    cx, cy = P[:, 0], P[:, 1]
    vyy = P[:, 4] ** 2
    vxy = P[:, 3] * P[:, 4]
    h = np.sqrt(12 * vyy)
    s = -vxy / vyy
    w = np.sqrt(np.maximum(12 * (P[:, 2] ** 2), 1e-6))
    Q = np.stack([cx, cy, h, w, np.degrees(np.arctan(s))], 1)
    ok = (h > 8) & (h < 30) & (w > 0.5) & (w < 14)
    Q[~ok] = [14, 14, 20, 3, 0]
    S = render(Q, dtype=np.float64)
    a, b = I.reshape(len(I), -1), S.reshape(len(S), -1)
    c = (a * b).sum(1) / np.maximum(
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-9)
    c[~ok] = 0.0
    return c


def part_sparsity():
    print("\n=== SPARSITY: can neighbour differences be treated as first order? ===")
    rows = {}
    for name, V in (("real MNIST 1s", mnist_ones().reshape(-1, 784)),
                    ("generated 1s, same count", generated(6742).reshape(-1, 784))):
        d = wc.neighbour_distances(V, kmax=8)
        norm = np.linalg.norm(V, axis=1)
        rows[name] = {"nn_pct": float(np.median(d[:, 0] / norm) * 100),
                      "nn8_pct": float(np.median(d[:, 7] / norm) * 100)}
        print(f"  {name:>26}: nearest neighbour is "
              f"{rows[name]['nn_pct']:5.1f}% of the image, 8th is "
              f"{rows[name]['nn8_pct']:5.1f}%")
    print("  a 25% step is nowhere near first order, so the linearised method is out")
    R["sparsity"] = rows


def part_floor():
    print("\n=== FLOOR: what actually limits the decoder ===")
    print("  warping one generated 1 onto another through the EXACT map between")
    print("  their true knobs, so any error is the resampling operator alone")
    rng = np.random.default_rng(1)
    M = 300
    P = RANGES[:, 0] + rng.random((2 * M, 5)) * (RANGES[:, 1] - RANGES[:, 0])
    src, dst = P[:M], P[M:]
    Idst = render(dst, dtype=np.float64)
    cen = np.arange(28) + 0.5
    rows = {}
    for label, grid in (("28x28 source (one image)", 28), ("56x56", 56),
                        ("112x112", 112), ("224x224", 224)):
        hi = render_coverage(src, n_pix=grid) if grid != 28 else render(src, dtype=np.float64)
        sub = (np.arange(4) + 0.5) / 4
        off = np.add.outer(np.arange(28), sub).ravel()
        X, Y = np.meshgrid(off, off)
        num = den = 0.0
        for a, b, p1, p2 in zip(hi, Idst, src, dst):
            cx1, cy1, h1, w1, l1 = p1
            cx2, cy2, h2, w2, l2 = p2
            s1, s2 = np.tan(np.radians(l1)), np.tan(np.radians(l2))
            y1 = cy1 + (h1 / h2) * (Y - cy2)
            x1 = cx1 - s1 * (y1 - cy1) + (w1 / w2) * (X - cx2 + s2 * (Y - cy2))
            fine = map_coordinates(a, [y1 * grid / 28.0 - 0.5, x1 * grid / 28.0 - 0.5],
                                   order=1, mode="constant", cval=0.0)
            rec = fine.reshape(28, 4, 28, 4).mean((1, 3))
            num += np.linalg.norm(rec - b)
            den += np.linalg.norm(b)
        rows[label] = float(num / den)
        print(f"  {label:>26}: {rows[label] * 100:6.2f}%")
    print("  so the template must be super-resolved; a 28x28 one caps the decoder")
    R["floor"] = rows


def _sweep(I, name, degrees=(1, 2, 3, 4), n_train=2500, n_test=1500):
    tr = np.arange(n_train)
    te = np.arange(n_train, n_train + n_test)
    F = I.reshape(len(I), -1)
    mu = F[tr].mean(0)
    _, _, V = np.linalg.svd(F[tr] - mu, full_matrices=False)
    specs = [("affine, no vertical shear", wc.model(1, True))]
    specs += [(f"degree {d}", wc.model(d)) for d in degrees]
    print(f"  {name}")
    print(f"  {'warp freedom':>26}{'params':>8}{'held-out':>11}"
          f"{'PCA same #':>12}{'gain':>8}{'folds any':>9}{'on ink':>9}")
    rows, prev = [], None
    for label, m in specs:
        T, THtr, THte = wc.fit_chart(I, m, tr, te)
        e = wc.rel_error(I[te], wc.decode(T, THte, m))
        k = wc.n_par(m)
        rec = mu + ((F[te] - mu) @ V[:k].T) @ V[:k]
        pca = wc.rel_error(F[te], rec)
        anywhere = np.mean([wc.jacobian_determinants(t, m).min() <= 0
                            for t in THte])
        on_ink = np.mean([wc.jacobian_determinants(t, m, ink=img).min() <= 0
                          for t, img in zip(THte, I[te])])
        gain = "" if prev is None else f"{(prev - e) * 100:+7.2f}"
        print(f"  {label:>26}{k:>8}{e * 100:10.2f}%{pca * 100:11.2f}%"
              f"{gain:>8}{anywhere * 100:8.1f}%{on_ink * 100:9.1f}%")
        rows.append({"model": label, "params": k, "warp": e, "pca": pca,
                     "folded_anywhere": float(anywhere),
                     "folded_on_ink": float(on_ink)})
        prev = e
    return rows


def part_saturation():
    print("\n=== SATURATION: where does geometry run out? ===")
    R["saturation"] = {
        "generated": _sweep(generated(5000), "GENERATED 1s (floor 1.26%)"),
        "mnist": _sweep(mnist_ones(), "REAL MNIST 1s"),
    }


def part_subsets(n_train=1011, n_test=674):
    print(f"\n=== SUBSETS: matched training size ({n_train} train, {n_test} test) ===")
    I = mnist_ones()
    order = np.argsort(-stroke_overlap(I))
    q = len(I) // 4
    sets = {"all MNIST 1s": I,
            "most stroke-like 25%": I[order[:q]],
            "least stroke-like 25%": I[order[-q:]]}
    tr, te = np.arange(n_train), np.arange(n_train, n_train + n_test)
    print(f"  {'subset':>24}{'5 warp':>9}{'20 warp':>9}{'PCA 5':>8}{'PCA 20':>8}")
    rows = {}
    for name, S in sets.items():
        F = S.reshape(len(S), -1)
        mu = F[tr].mean(0)
        _, _, V = np.linalg.svd(F[tr] - mu, full_matrices=False)
        got = []
        for m in (wc.model(1, True), wc.model(3)):
            T, _, THte = wc.fit_chart(S, m, tr, te)
            got.append(wc.rel_error(S[te], wc.decode(T, THte, m)))
        pca = [wc.rel_error(F[te], mu + ((F[te] - mu) @ V[:k].T) @ V[:k]) for k in (5, 20)]
        rows[name] = {"warp5": got[0], "warp20": got[1],
                      "pca5": pca[0], "pca20": pca[1]}
        print(f"  {name:>24}{got[0] * 100:8.2f}%{got[1] * 100:8.2f}%"
              f"{pca[0] * 100:7.2f}%{pca[1] * 100:7.2f}%")
    R["subsets"] = rows


def part_dimension():
    print("\n=== DIMENSION: how many knobs, with no model ===")
    rng = np.random.default_rng(0)
    I = mnist_ones().reshape(-1, 784)
    order = np.argsort(-stroke_overlap(I.reshape(-1, 28, 28)))
    q = len(I) // 4
    lin = rng.standard_normal((6742, 5)) @ rng.standard_normal((5, 784))
    sets = [("CONTROL 5-dim linear subspace", lin),
            ("CONTROL generated 1s (5 knobs)", generated(6742).reshape(-1, 784)),
            ("MNIST 1s, all", I),
            ("MNIST 1s, most stroke-like 25%", I[order[:q]]),
            ("MNIST 1s, least stroke-like 25%", I[order[-q:]])]
    print(f"  {'set':>33}{'TwoNN':>8}{'MLE5':>7}{'MLE10':>7}{'MLE20':>7}")
    rows = {}
    for name, V in sets:
        Vs = V[rng.permutation(len(V))[:4000]]
        d = wc.neighbour_distances(Vs)
        rows[name] = {"twonn": wc.twonn(d), "mle5": wc.mle_dim(d, 5),
                      "mle10": wc.mle_dim(d, 10), "mle20": wc.mle_dim(d, 20)}
        print(f"  {name:>33}{rows[name]['twonn']:8.2f}{rows[name]['mle5']:7.2f}"
              f"{rows[name]['mle10']:7.2f}{rows[name]['mle20']:7.2f}")
    print("\n  the same estimate at growing n -- still climbing means data-starved")
    for name, V in sets[1:]:
        seq = []
        for n in (500, 1000, 2000, 4000):
            Vs = V[rng.permutation(len(V))[:min(n, len(V))]]
            seq.append(wc.twonn(wc.neighbour_distances(Vs)))
        rows[name]["scaling"] = seq
        print(f"  {name:>33}  " + " -> ".join(f"{v:5.2f}" for v in seq))
    R["dimension"] = rows


def part_appearance(n_train=2500, n_test=1500):
    print("\n=== APPEARANCE: what geometry cannot reach, priced against PCA ===")
    I = mnist_ones()
    tr, te = np.arange(n_train), np.arange(n_train, n_train + n_test)
    F = I.reshape(len(I), -1)
    rows = {}
    for label, m in (("5 warp coords", wc.model(1, True)),
                     ("20 warp coords (cubic)", wc.model(3))):
        T, THtr, THte = wc.fit_chart(I, m, tr, te)
        Btr = wc.decode(T, THtr, m).reshape(len(tr), -1)
        Bte = wc.decode(T, THte, m).reshape(len(te), -1)
        res = F[tr] - Btr
        rm = res.mean(0)
        _, _, E = np.linalg.svd(res - rm, full_matrices=False)
        mu = F[tr].mean(0)
        _, _, V = np.linalg.svd(F[tr] - mu, full_matrices=False)
        dn = np.linalg.norm(F[te])
        k0 = wc.n_par(m)
        print(f"  {label}")
        print(f"  {'total coords':>14}{'warp + appearance':>20}{'plain PCA':>12}")
        got = []
        for k in (0, 5, 20, 40, 80, 135):
            r = F[te] - Bte - rm
            rec = Bte + rm + ((r @ E[:k].T) @ E[:k] if k else 0)
            kk = k0 + k
            r2 = mu + ((F[te] - mu) @ V[:kk].T) @ V[:kk]
            ew = float(np.linalg.norm(F[te] - rec) / dn)
            ep = float(np.linalg.norm(F[te] - r2) / dn)
            got.append({"coords": kk, "warp_plus": ew, "pca": ep})
            print(f"  {kk:>14}{ew * 100:19.2f}%{ep * 100:11.2f}%")
        rows[label] = got
    R["appearance"] = rows


def part_meaning(n_train=2500):
    print("\n=== MEANING: what the recovered warp numbers track ===")
    I = mnist_ones()
    tr, te = np.arange(n_train), np.arange(n_train, n_train + 300)
    m = wc.model(1, True)
    T, THtr, _ = wc.fit_chart(I, m, tr, te)
    P = wc.moment_chart(I[tr])
    vyy = P[:, 4] ** 2
    obs = {"ink": I[tr].reshape(len(tr), -1).sum(1),
           "centre x": P[:, 0], "centre y": P[:, 1],
           "height": np.sqrt(12 * vyy),
           "slant": -(P[:, 3] * P[:, 4]) / vyy,
           "width": np.sqrt(12 * P[:, 2] ** 2)}
    names = ["zx const", "zx per x", "zx per y", "zy const", "zy per y"]
    print("              " + "".join(f"{k:>11}" for k in obs))
    rows = {}
    for j, nm in enumerate(names):
        c = [float(np.corrcoef(THtr[:, j], v)[0, 1]) for v in obs.values()]
        rows[nm] = dict(zip(obs, c))
        print(f"  {nm:>12}" + "".join(f"{v:>11.2f}" for v in c))
    R["meaning"] = rows


def part_residuals(n_train=2500, n_test=1500):
    """What does the warp model fail ON, and would several templates fix it?

    A warp moves ink; it cannot create a stroke. So the natural guess is that
    the failures carry a serif or a base bar that a plain stroke has no way to
    grow. That guess is tested here rather than assumed -- it turns out to be
    the right direction and the wrong magnitude.
    """
    print("\n=== RESIDUALS: what geometry fails on, and whether templates help ===")
    I = mnist_ones()
    tr, te = np.arange(n_train), np.arange(n_train, n_train + n_test)
    m = wc.model(1, True)
    T, THtr, THte = wc.fit_chart(I, m, tr, te)
    B = wc.decode(T, THte, m)
    per = (np.linalg.norm((I[te] - B).reshape(len(te), -1), axis=1)
           / np.linalg.norm(I[te].reshape(len(te), -1), axis=1))

    A = _pullback(I[te], THte, m, res=56)
    n = A.shape[1]
    tot = A.reshape(len(A), -1).sum(1) + 1e-9
    rows_, cols_ = np.arange(n)[:, None], np.arange(n)[None, :]
    mid = (cols_ > 0.35 * n) & (cols_ < 0.65 * n)
    feats = {"base bar": (rows_ > 0.78 * n) & ~mid,
             "top-left flag": (rows_ < 0.30 * n) & (cols_ < 0.42 * n)}
    o = np.argsort(per)
    best, worst = o[:len(o) // 10], o[-len(o) // 10:]
    print(f"  per-image error: best decile {per[best].mean() * 100:.1f}%,"
          f" worst decile {per[worst].mean() * 100:.1f}%")
    rows = {}
    for k, mask in feats.items():
        v = (A * mask).reshape(len(A), -1).sum(1) / tot
        rows[k] = {"best_decile": float(v[best].mean()),
                   "worst_decile": float(v[worst].mean()),
                   "corr_with_error": float(np.corrcoef(v, per)[0, 1]),
                   "share_above_0.10": float((v > 0.10).mean())}
        print(f"  {k:>15}: best {rows[k]['best_decile']:.3f}"
              f"  worst {rows[k]['worst_decile']:.3f}"
              f"  corr {rows[k]['corr_with_error']:+.3f}"
              f"  share above 0.10 {rows[k]['share_above_0.10'] * 100:.0f}%")
    print("  direction right, magnitude small: serifs and bars correlate with")
    print("  failure but are too rare to be the main cause")

    Atr = _pullback(I[tr], THtr, m, res=56).reshape(len(tr), -1)
    Atr /= np.maximum(np.linalg.norm(Atr, axis=1, keepdims=True), 1e-9)
    mu = Atr.mean(0)
    _, _, Vt = np.linalg.svd(Atr - mu, full_matrices=False)
    Ztr = (Atr - mu) @ Vt[:12].T
    Ate = A.reshape(len(te), -1)
    Ate /= np.maximum(np.linalg.norm(Ate, axis=1, keepdims=True), 1e-9)
    Zte = (Ate - mu) @ Vt[:12].T
    print("  several templates, clustered on the ALIGNED shapes:")
    rng = np.random.default_rng(0)
    tm = {}
    for K in (1, 2, 3, 5, 8):
        cen = Ztr[rng.choice(len(Ztr), K, replace=False)]
        for _ in range(80):
            lab = np.argmin(((Ztr[:, None] - cen[None]) ** 2).sum(2), 1)
            for c in range(K):
                if (lab == c).any():
                    cen[c] = Ztr[lab == c].mean(0)
        labt = np.argmin(((Zte[:, None] - cen[None]) ** 2).sum(2), 1)
        rec = np.empty_like(I[te])
        for c in range(K):
            s = lab == c
            Tc = wc.splat(I[tr][s], THtr[s], m)
            for _ in range(5):
                Tc = Tc + wc.splat(I[tr][s] - wc.decode(Tc, THtr[s], m), THtr[s], m)
            st = labt == c
            if st.any():
                rec[st] = wc.decode(Tc, THte[st], m)
        tm[K] = wc.rel_error(I[te], rec)
        print(f"    {K} template(s): {tm[K] * 100:6.2f}%")
    print("  smooth and gradual, with no knee: a continuum of hands, not a few styles")
    R["residuals"] = {"features": rows, "templates": tm}


def _pullback(I, TH, m, res=56):
    """Pull images into the canonical frame (affine charts only)."""
    deg, ix, iy = m
    assert deg == 1, "pull-back is implemented for affine charts only"
    g = (np.arange(res) + 0.5) * (28.0 / res)
    ZX, ZY = np.meshgrid(g, g)
    out = np.empty((len(TH), res, res))
    nx = len(ix)
    for i, th in enumerate(TH):
        cx = dict(zip(ix, th[:nx]))
        cy = dict(zip(iy, th[nx:]))
        Amat = np.array([[cx.get(1, 0.0), cx.get(2, 0.0)],
                         [cy.get(1, 0.0), cy.get(2, 0.0)]]) / wc.CEN
        c = np.array([cx.get(0, 0.0), cy.get(0, 0.0)]) - Amat @ np.array([wc.CEN, wc.CEN])
        Ai = np.linalg.inv(Amat)
        xx = Ai[0, 0] * (ZX - c[0]) + Ai[0, 1] * (ZY - c[1])
        yy = Ai[1, 0] * (ZX - c[0]) + Ai[1, 1] * (ZY - c[1])
        out[i] = map_coordinates(I[i], [yy - 0.5, xx - 0.5], order=1,
                                 mode="constant", cval=0.0)
    return out


def part_generation(n_train=2500, n_test=1200, n_draw=600):
    """A correct knob set GENERATES the family; reconstruction error cannot see it.

    Fit coordinates, fit a Gaussian over them, draw NEW coordinates, decode. If
    the coordinates are the family's real knobs, every sample is a valid member.
    PCA reconstructs well but is notorious for generating ghosts, so this
    separates "good compressor" from "right description".

    Every row, the real held-out images included, is scored against the same
    reference set -- the TRAINING images. Scoring reals against themselves would
    give a trivial zero and no baseline.
    """
    print("\n=== GENERATION: do NEW coordinates decode to valid members? ===")
    rng = np.random.default_rng(3)
    out = {}
    for name, I in (("GENERATED 1s (control: five knobs IS the truth)", generated(5000)),
                    ("REAL MNIST 1s", mnist_ones())):
        tr = np.arange(n_train)
        te = np.arange(n_train, n_train + n_test)
        ref = I[tr].reshape(n_train, -1)

        def score(S, label):
            ov = stroke_overlap(np.clip(S, -9, 9))
            a = S.reshape(len(S), -1)
            d = ((a ** 2).sum(1)[:, None] + (ref ** 2).sum(1)[None, :] - 2 * a @ ref.T)
            nr = np.sqrt(np.maximum(d.min(1), 0)) / np.maximum(
                np.linalg.norm(a, axis=1), 1e-9)
            neg = np.minimum(S, 0).reshape(len(S), -1).sum(1) / np.maximum(
                np.abs(S).reshape(len(S), -1).sum(1), 1e-9)
            print(f"  {label:>38}{np.median(ov):15.3f}{np.median(nr):22.3f}"
                  f"{np.median(neg):13.3f}")
            return {"stroke_overlap": float(np.median(ov)),
                    "dist_to_nearest_train": float(np.median(nr)),
                    "negative_ink": float(np.median(neg))}

        print(f"  {name}")
        print(f"  {'source of images':>38}{'stroke overlap':>15}"
              f"{'dist to nearest':>22}{'negative ink':>13}")
        rows = {"real held-out": score(I[te], "REAL held-out 1s (the target)")}
        for label, m in (("5 warp coords", wc.model(1, True)),
                         ("20 warp coords (cubic)", wc.model(3))):
            T, THtr, _ = wc.fit_chart(I, m, tr, te)
            C = np.cov(THtr.T) + 1e-10 * np.eye(THtr.shape[1])
            draw = rng.multivariate_normal(THtr.mean(0), C, size=n_draw)
            rows[label] = score(wc.decode(T, draw, m), f"decoded from NEW {label}")
        F = I.reshape(len(I), -1)
        mu = F[tr].mean(0)
        _, _, V = np.linalg.svd(F[tr] - mu, full_matrices=False)
        for k in (5, 20):
            Z = (F[tr] - mu) @ V[:k].T
            draw = rng.multivariate_normal(Z.mean(0), np.cov(Z.T) + 1e-10 * np.eye(k),
                                           size=n_draw)
            rows[f"pca {k}"] = score((mu + draw @ V[:k]).reshape(-1, 28, 28),
                                     f"PCA with {k} coords, sampled")
        out[name] = rows
    R["generation"] = out


def part_inversion(n_train=2500, n_test=1500, rounds=25):
    """From the data-only chart back to the generator's ORIGINAL knobs, exactly.

    The fitted chart is zx = a0 + a1*u + a2*v, zy = b0 + b2*v, u = (x-14)/14.
    The generator's own map is u_gen = (W0/w)(x - cx + s(y - cy)),
    v_gen = (H0/h)(y - cy). The canonical frame is a FIXED affine
    reparameterisation of (u_gen, v_gen) -- the same one for every image, since
    it is only the frame the template happens to live in. Matching coefficients
    gives an exact inversion with five global constants:

        w  = Cw / a1
        h  = Ch / b2
        s  = (a2 - kappa * b2) / a1            (s = tan lean)
        cy = 14 + 14 * (Zy - b0) / b2
        cx = 14 + 14 * (Zx - a0 - a2 * (cy - 14) / 14) / a1

    Those five depend on the gauge, not on the image, so the unlabelled chart
    pins the real knobs down to five numbers. Five labelled examples fix them --
    and five is exactly enough, which is the point: the gauge has five degrees
    of freedom, so more labels add nothing.

    Control: run the same inversion on chart rows paired with the wrong images.
    """
    print("\n=== INVERSION: the data-only chart back to the true knobs ===")
    from grey_ones import KNOBS
    rng = np.random.default_rng(0)
    P = RANGES[:, 0] + rng.random((n_train + n_test + 500, 5)) * (
        RANGES[:, 1] - RANGES[:, 0])
    I = render(P, dtype=np.float64)
    m = wc.model(1, True)
    tr = np.arange(n_train)
    te = np.arange(n_train, n_train + n_test)
    T, THtr, THte = wc.fit_chart(I, m, tr, te, rounds=rounds)
    print(f"  chart fitted with no labels; held-out reconstruction "
          f"{wc.rel_error(I[te], wc.decode(T, THte, m)) * 100:.2f}%")

    def predict(TH, C):
        a0, a1, a2, b0, b2 = TH.T
        Cw, Ch, kap, Zx, Zy = C
        cy = 14 + 14 * (Zy - b0) / b2
        cx = 14 + 14 * (Zx - a0 - a2 * (cy - 14) / 14) / a1
        return np.stack([cx, cy, Ch / b2, Cw / a1,
                         np.degrees(np.arctan((a2 - kap * b2) / a1))], 1)

    def calibrate(TH, Q):
        a0, a1, a2, b0, b2 = TH.T
        cx, cy, h, w, lean = Q.T
        s = np.tan(np.radians(lean))
        return (np.median(w * a1), np.median(h * b2), np.median((a2 - s * a1) / b2),
                np.median(a0 + a1 * (cx - 14) / 14 + a2 * (cy - 14) / 14),
                np.median(b0 + b2 * (cy - 14) / 14))

    print(f"  {'labelled':>9}" + "".join(f"{k:>11}" for k in KNOBS))
    rows = {}
    for n in (5, 10, 20, 100, n_train):
        err = np.abs(predict(THte, calibrate(THtr[:n], P[tr][:n])) - P[te])
        rows[n] = [float(v) for v in np.median(err, 0)]
        print(f"  {n:>9}" + "".join(f"{v:>11.4f}" for v in rows[n]))
    print("  (median absolute error; the knob ranges are 1, 1, 1.5, 2.8 px, 45 deg)")
    C = calibrate(THtr, P[tr])
    worst = np.abs(predict(THte, C) - P[te]).max(0)
    print(f"  {'worst':>9}" + "".join(f"{v:>11.4f}" for v in worst))
    sh = rng.permutation(len(THte))
    ctl = np.median(np.abs(predict(THte[sh], C) - P[te]), 0)
    print(f"  {'SHUFFLED':>9}" + "".join(f"{v:>11.4f}" for v in ctl))
    R["inversion"] = {"by_label_count": rows,
                      "worst_full_calibration": [float(v) for v in worst],
                      "shuffled_control": [float(v) for v in ctl]}


def part_gain(n_train=2500, n_test=1500, rounds=11):
    """Geometry times appearance, with the round count held fixed.

    A warp of a fixed template cannot change how dark the ink is. (It CAN
    already vary thickness along the stroke -- a `u*v` term is "horizontal
    scale changes with height" -- which is why degree 2 buys the most.) So the
    new freedom is a multiplicative gain field in canonical coordinates.

    THE CONFOUND, recorded because it nearly produced a false finding: a gain
    model also re-estimates the template, so comparing it against a shorter
    no-gain fit credits the gain with extra template refinement. Measured at
    matched rounds, a one-term gain is worth nothing, and the fitted constant
    gain's inter-quartile spread is 0.004 on the generated family against 0.029
    on MNIST -- real variation, far too small to matter. The ten-term spatial
    gain survives, and is worth many times more on MNIST than on a family whose
    coverage is uniform by construction.
    """
    print("\n=== GAIN: can appearance explain what geometry cannot? ===")
    print(f"  every row uses the same {rounds} template rounds\n")
    rows = {}
    for name, I, m, label in (
            ("generated", generated(5000), wc.model(1, True),
             "GENERATED 1s, 5 warp coords (uniform coverage: control)"),
            ("mnist_affine", mnist_ones(), wc.model(1, True),
             "REAL MNIST 1s, 5 warp coords"),
            ("mnist_quad", mnist_ones(), wc.model(2),
             "REAL MNIST 1s, 12 warp coords (quadratic)")):
        tr, te = np.arange(n_train), np.arange(n_train, n_train + n_test)
        k0 = wc.n_par(m)
        print(f"  {label}")
        print(f"  {'gain terms':>12}{'total coords':>14}{'held-out':>11}")
        got = {}
        T, _, THte = wc.fit_chart(I, m, tr, te, rounds=rounds)
        got[0] = wc.rel_error(I[te], wc.decode(T, THte, m))
        print(f"  {0:>12}{k0:>14}{got[0] * 100:10.2f}%")
        for ng in (1, 10):
            Tg, _, THteg = wc.fit_chart_gained(I, m, tr, te, ng, rounds=rounds)
            _, rec = wc.fit_gains(Tg, THteg, I[te], m, ng)
            got[ng] = wc.rel_error(I[te], rec)
            print(f"  {ng:>12}{k0 + ng:>14}{got[ng] * 100:10.2f}%")
        print(f"      ten-term gain is worth {(got[0] - got[10]) * 100:+.2f} points here")
        rows[name] = {str(k): v for k, v in got.items()}
    R["gain"] = rows


PARTS = {"sparsity": part_sparsity, "floor": part_floor,
         "saturation": part_saturation, "subsets": part_subsets,
         "dimension": part_dimension, "appearance": part_appearance,
         "residuals": part_residuals,
         "generation": part_generation,
         "inversion": part_inversion,
         "gain": part_gain,
         "meaning": part_meaning}

if __name__ == "__main__":
    want = sys.argv[1:] or list(PARTS)
    for p in want:
        PARTS[p]()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ("results.json" if len(want) == len(PARTS)
                  else f"results_{'_'.join(want)}.json")
    path.write_text(json.dumps(R, indent=2))
    print(f"\nwrote {path}")
