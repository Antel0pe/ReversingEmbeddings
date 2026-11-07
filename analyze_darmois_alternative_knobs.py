"""Build an explicit alternative set of five knobs for the generated 1s.

Hyvarinen & Pajunen (1999) prove the generating map is not identifiable from
the data alone. The proof is CONSTRUCTIVE (Darmois 1952 / Rosenblatt 1952):
given the data's joint distribution, the conditional CDFs

    z_1 = F(x_1),  z_2 = F(x_2 | x_1),  z_3 = F(x_3 | x_1, x_2),  ...

are exactly independent and the map is invertible, so it is a complete
alternative generator.  Here `grey_ones.sample` draws the five knobs
INDEPENDENT UNIFORM on a box, so the recipe collapses to something we can
write in closed form: any volume-preserving bijection of the unit cube sends
independent uniforms to independent uniforms.

Two such maps are built, both mixing normalized `cx` (index 0) and `lean`
(index 4):

  WRAP   two cyclic shears on the torus.  Exactly volume preserving, exact
         inverse in closed form, but discontinuous at the wrap.
  FLOW   time-1 map of the divergence-free field of H = (eps/pi) sin(pi p)
         sin(pi q) on the unit square.  Smooth, fixes the boundary, volume
         preserving up to RK4 error.

For each: does the alternative decoder produce the SAME images, are the new
coordinates really independent, and are they the generator's own knobs?
Finally a tie-breaker that does NOT use the data distribution: how well do
one-coordinate image changes add, in each coordinate system.

Reference point for the additivity test is the BOX CENTRE
(cx, cy, height, width, lean) = (14.5, 14.5, 19.75, 3.2, 12.5), i.e. u = 0.5
in every normalized coordinate -- not AGENTS.md's root 1, which sits on the
lower bound of cx and cy where a cube map has no room to move.
"""
import numpy as np
from grey_ones import render, RANGES, KNOBS

LO, HI = RANGES[:, 0], RANGES[:, 1]
SPAN = HI - LO
MIX = (0, 4)                      # the pair that gets mixed: cx and lean


def to_u(P):     return (np.asarray(P, float) - LO) / SPAN
def to_knobs(U): return LO + np.asarray(U, float) * SPAN


# ---------------------------------------------------------------- WRAP map
def _f(u, A=0.37): return A * np.sin(2 * np.pi * u)
def _g(u, B=0.29): return B * np.sin(2 * np.pi * u + 1.1)


def wrap_fwd(U):
    U = np.array(U, float); a, b = MIX
    t = (U[:, b] + _f(U[:, a])) % 1.0
    Z = U.copy()
    Z[:, a] = (U[:, a] + _g(t)) % 1.0
    Z[:, b] = t
    return Z


def wrap_inv(Z):
    Z = np.array(Z, float); a, b = MIX
    t = Z[:, b]
    U = Z.copy()
    U[:, a] = (Z[:, a] - _g(t)) % 1.0
    U[:, b] = (t - _f(U[:, a])) % 1.0
    return U


# ---------------------------------------------------------------- FLOW map
def _vfield(p, q, eps):
    return (eps * np.sin(np.pi * p) * np.cos(np.pi * q),
            -eps * np.cos(np.pi * p) * np.sin(np.pi * q))


def _flow(P, Q, eps, steps=400):
    h = 1.0 / steps
    for _ in range(steps):
        k1p, k1q = _vfield(P, Q, eps)
        k2p, k2q = _vfield(P + h / 2 * k1p, Q + h / 2 * k1q, eps)
        k3p, k3q = _vfield(P + h / 2 * k2p, Q + h / 2 * k2q, eps)
        k4p, k4q = _vfield(P + h * k3p, Q + h * k3q, eps)
        P = P + h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        Q = Q + h / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
    return np.clip(P, 0, 1), np.clip(Q, 0, 1)


def flow_fwd(U, eps=1.6):
    U = np.array(U, float); a, b = MIX
    U[:, a], U[:, b] = _flow(U[:, a], U[:, b], eps)
    return U


def flow_inv(Z, eps=1.6):
    Z = np.array(Z, float); a, b = MIX
    Z[:, a], Z[:, b] = _flow(Z[:, a], Z[:, b], -eps)
    return Z


# ------------------------------------------------------------- statistics
def dcor(x, y):
    """Distance correlation: 0 iff independent, catches nonlinear coupling."""
    x = np.asarray(x, float)[:, None]; y = np.asarray(y, float)[:, None]
    A = np.abs(x - x.T); B = np.abs(y - y.T)
    A = A - A.mean(0) - A.mean(1)[:, None] + A.mean()
    B = B - B.mean(0) - B.mean(1)[:, None] + B.mean()
    dcov2 = (A * B).mean(); dvx = (A * A).mean(); dvy = (B * B).mean()
    return float(np.sqrt(max(dcov2, 0) / np.sqrt(dvx * dvy))) if dvx * dvy > 0 else 0.0


def ks_uniform(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    e = (np.arange(1, n + 1)) / n
    return float(max(np.abs(e - x).max(), np.abs(x - (e - 1 / n)).max()))


def max_pair_dcor(Z, n=1500, rng=None):
    idx = rng.choice(len(Z), size=min(n, len(Z)), replace=False)
    S = Z[idx]
    return max(dcor(S[:, i], S[:, j]) for i in range(5) for j in range(i + 1, 5))


def binned_r2(x, y, bins=24):
    """How much of y a smooth function of x alone explains (0 = none, 1 = all)."""
    e = np.linspace(0, 1, bins + 1); k = np.clip(np.digitize(x, e) - 1, 0, bins - 1)
    pred = np.array([y[k == b].mean() if (k == b).sum() > 4 else y.mean()
                     for b in range(bins)])[k]
    return float(1 - ((y - pred) ** 2).mean() / y.var())


# ------------------------------------------------------------ the checks
def report(name, fwd, inv, rng, n=6000):
    U = rng.random((n, 5))
    Z = fwd(U)
    Uback = inv(Z)
    print(f"\n=== {name} ===")
    print(f"  round trip  max |inv(fwd(u)) - u|          {np.abs(Uback - U).max():.3e}")

    imgs_true = render(to_knobs(U))
    imgs_alt = render(to_knobs(inv(Z)))
    d = np.abs(imgs_true - imgs_alt).max()
    rel = np.linalg.norm((imgs_true - imgs_alt).ravel()) / np.linalg.norm(imgs_true.ravel())
    print(f"  SAME IMAGES max pixel difference          {d:.3e}  (rel RMS {rel:.3e})")

    print(f"  marginals   worst KS vs uniform   new {max(ks_uniform(Z[:, k]) for k in range(5)):.4f}"
          f"   true knobs {max(ks_uniform(U[:, k]) for k in range(5)):.4f}")
    print(f"  INDEPENDENT worst pair dist-corr  new {max_pair_dcor(Z, rng=rng):.4f}"
          f"   true knobs {max_pair_dcor(U, rng=rng):.4f}")

    a, b = MIX
    print(f"  but is it the knobs?  new coord {KNOBS[a]!r}-slot explained by")
    print(f"      true {KNOBS[a]:<6} alone  R2 = {binned_r2(U[:, a], Z[:, a]):.3f}")
    print(f"      true {KNOBS[b]:<6} alone  R2 = {binned_r2(U[:, b], Z[:, a]):.3f}")
    print(f"      Pearson r with true {KNOBS[a]:<6} = {np.corrcoef(U[:, a], Z[:, a])[0,1]:+.3f}")
    return U, Z


def additivity(decode, root, delta):
    """Residual of adding five one-coordinate image changes, as a fraction."""
    base = decode(root[None, :])[0]
    lines = np.stack([decode((root + delta * np.eye(5)[k])[None, :])[0] - base
                      for k in range(5)])
    joint = decode((root + delta * np.ones(5))[None, :])[0] - base
    return float(np.linalg.norm(joint - lines.sum(0)) / np.linalg.norm(joint))


def poly_features(C, deg):
    """All monomials in the five coordinates up to total degree `deg`."""
    from itertools import combinations_with_replacement as cwr
    cols = [np.ones(len(C))]
    for d in range(1, deg + 1):
        for idx in cwr(range(5), d):
            v = np.ones(len(C))
            for i in idx:
                v = v * C[:, i]
            cols.append(v)
    return np.stack(cols, 1)


def decoder_simplicity(C, imgs, deg, n_train=3000):
    """Held-out error of the best polynomial decoder of this degree."""
    X = poly_features(C - 0.5, deg)
    Xtr, Xte = X[:n_train], X[n_train:]
    Ytr, Yte = imgs[:n_train], imgs[n_train:]
    W, *_ = np.linalg.lstsq(Xtr, Ytr, rcond=1e-10)
    r = Xte @ W - Yte
    return float(np.linalg.norm(r) / np.linalg.norm(Yte)) * 100, X.shape[1]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    np.set_printoptions(suppress=True)

    report("WRAP  (exact, discontinuous)", wrap_fwd, wrap_inv, rng)
    report("FLOW  (smooth, eps = 1.6)", flow_fwd, flow_inv, rng)

    print("\n=== infinitely many: sweep the flow strength ===")
    print("  eps    max pixel diff   worst dist-corr   r(z_cx, true cx)")
    for eps in [0.0, 0.4, 0.8, 1.2, 1.6, 2.4, 3.2]:
        U = rng.random((3000, 5))
        Z = flow_fwd(U, eps)
        same = np.abs(render(to_knobs(U)) - render(to_knobs(flow_inv(Z, eps)))).max()
        print(f"  {eps:4.1f}   {same:.3e}        {max_pair_dcor(Z, rng=rng):.4f}"
              f"            {np.corrcoef(U[:, 0], Z[:, 0])[0,1]:+.3f}")

    print("\n=== tie-breaker that does NOT look at the distribution ===")
    print("  how badly five one-coordinate image changes fail to add up")
    root = np.full(5, 0.5)
    print("  step    true knobs      flow knobs (eps=1.6)     ratio")
    for delta in [0.2, 0.1, 0.05, 0.025, 0.0125]:
        rt = additivity(lambda U: render(to_knobs(U)), root, delta)
        rz = additivity(lambda Z: render(to_knobs(flow_inv(Z, 1.6))),
                        flow_fwd(root[None, :], 1.6)[0], delta)
        print(f"  {delta:6.4f}  {rt:9.5f}       {rz:9.5f}          {rz/rt:6.2f}x")
    print("  -> VERDICT: additivity does NOT separate them.")

    print("\n=== tie-breaker 2: how simple is the decoder in each coordinate system ===")
    print("  held-out relative error (%) of the best polynomial decoder")
    rng2 = np.random.default_rng(7)
    U = rng2.random((5000, 5))
    imgs = render(to_knobs(U)).reshape(len(U), -1).astype(np.float64)
    Zf = flow_fwd(U, 1.6)
    Zw = wrap_fwd(U)
    print("  degree  terms   true knobs    flow knobs    wrap knobs")
    for deg in [1, 2, 3, 4, 5]:
        eu, nt = decoder_simplicity(U, imgs, deg)
        ef, _ = decoder_simplicity(Zf, imgs, deg)
        ew, _ = decoder_simplicity(Zw, imgs, deg)
        print(f"  {deg:^6d}  {nt:5d}   {eu:9.3f}    {ef:9.3f}    {ew:9.3f}")
