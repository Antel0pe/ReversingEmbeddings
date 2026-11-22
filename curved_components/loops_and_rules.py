"""The first experiment of this folder, kept as a record, now run on lean as well.

Question (user, 2026-11-22): a curve needs many straight (PCA) components; can
we instead find a few CURVED components, e.g. by combining the straight ones?

Three checks per sweep:
  1. How many straight PCA components does one knob need?
  2. Plot those components in pairs (1&2, 3&4, ...). Does each pair trace a
     loop, and does the knob drive every loop at a steady rate?
  3. "One repeated rule": learn, from the MIDDLE THIRD of the sweep only, a
     small turning rule per frequency (per row, for lean, because each row of a
     leaning stroke slides by its own amount). Then rebuild every position from
     a single picture by applying the rule again and again, including positions
     the rule never saw. Compared with 6 PCA components fitted to the same
     middle third.
Control: the EXACT slide/lean done by the same rule with the true speeds, on the
same 28x28 grid. Whatever that misses is the pixel grid, not the rule.

Lean is run twice: steps even in degrees, and steps even in tan(lean), which is
lean's natural clock (each row's shift is (row - cy) * tan(lean)).

Run:  .venv/bin/python curved_components/loops_and_rules.py
"""

import numpy as np

from data import ROOT_ONE, first_below, render, save

N = 28
FREQ = 2 * np.pi * np.arange(N // 2 + 1) / N       # turning speed of each frequency, per pixel shifted


def sweep(knob, values):
    P = np.repeat(ROOT_ONE[None], len(values), 0)
    P[:, knob] = values
    return render(P, dtype=np.float64)


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def pca_count(X):
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    tot = np.linalg.norm(X)
    left = [100 * np.sqrt(max((S[k:] ** 2).sum(), 0)) / tot for k in range(len(S) + 1)]
    return U * S, left


def loops(Z, t):
    out = []
    for a in [0, 2, 4]:
        z = Z[:, a] + 1j * Z[:, a + 1]
        ang = np.unwrap(np.angle(z))
        fit = np.polyfit(t, ang, 1)
        res = ang - np.polyval(fit, t)
        out.append((100 * np.abs(z).std() / np.abs(z).mean(), abs(ang[-1] - ang[0]) / (2 * np.pi),
                    1 - res.var() / ang.var()))
    return out


def rule_test(img, t, mid, per_row):
    """Learn one turning multiplier per frequency (and per row if per_row) from consecutive middle frames."""
    F = np.fft.rfft(img, axis=2)
    idx = np.where(mid)[0]
    a, b = F[idx[:-1]], F[idx[1:]]
    axes = (0,) if per_row else (0, 1)
    m = (np.conj(a) * b).sum(axes) / np.maximum((np.abs(a) ** 2).sum(axes), 1e-300)
    m = np.where(np.abs(m) > 1e-12, m / np.maximum(np.abs(m), 1e-300), 1.0)   # pure turning (circle); empty rows stay put
    start = idx[0]
    out = np.array([rel(np.fft.irfft(F[start] * m ** (i - start), n=N, axis=-1), img[i]) for i in range(len(t))])
    return 100 * out


def pca_seen(X, mid, k=6):
    mu = X[mid].mean(0)
    V = np.linalg.svd(X[mid] - mu, full_matrices=False)[2][:k]
    return 100 * np.array([rel(mu + ((x - mu) @ V.T) @ V, x) for x in X])


def exact_shift(img, t, start, per_row, knob_to_shift):
    """True speeds: each row shifted by the amount the knob really moves it."""
    F = np.fft.rfft(img, axis=2)
    rows = np.arange(N) + 0.5 - ROOT_ONE[1]
    out = []
    for i in range(len(t)):
        sh = knob_to_shift(t[i]) - knob_to_shift(t[start])
        sh = sh * rows if per_row else sh * np.ones(N)
        out.append(rel(np.fft.irfft(F[start] * np.exp(-1j * FREQ[None] * sh[:, None]), n=N, axis=-1), img[i]))
    return 100 * np.array(out)


def main():
    cases = [
        ("slide", 0, np.arange(5.0, 23.0 + 1e-9, 0.25), False, lambda c: c),
        ("lean, even degrees", 4, np.arange(-10.0, 35.0 + 1e-9, 0.5), True,
         lambda L: -np.tan(np.radians(L))),
        ("lean, even tan(lean)", 4, np.degrees(np.arctan(np.linspace(np.tan(np.radians(-10)),
                                                                     np.tan(np.radians(35)), 91))), True,
         lambda L: -np.tan(np.radians(L))),
    ]
    res = {}
    for name, knob, t, per_row, k2s in cases:
        img = sweep(knob, t)
        X = img.reshape(len(t), -1)
        Z, left = pca_count(X)
        lo, hi = t.min() + (t.max() - t.min()) / 3, t.min() + 2 * (t.max() - t.min()) / 3
        mid = (t >= lo) & (t <= hi)
        r = rule_test(img, t, mid, per_row)
        p = pca_seen(X, mid)
        ex = exact_shift(img, t, np.where(mid)[0][0], per_row, k2s)
        print(f"== {name}: {len(t)} pictures, PCA needs {first_below(left, 10)} straight components "
              f"for <10%, {first_below(left, 1)} for <1%")
        clock = np.tan(np.radians(t)) if knob == 4 else t
        for j, (spread, turns, r2) in enumerate(loops(Z, clock)):
            print(f"   components {2 * j + 1}&{2 * j + 2}: loop radius varies {spread:5.1f}%, goes round {turns:4.2f} "
                  f"times, steady with the knob r2 = {r2:.3f}")
        print(f"   rebuilt from ONE picture (median miss) - seen middle third: rule {np.median(r[mid]):5.1f}%, "
              f"6 PCA {np.median(p[mid]):5.1f}% | never seen: rule {np.median(r[~mid]):5.1f}%, "
              f"6 PCA {np.median(p[~mid]):6.1f}% | exact move, same grid: {np.median(ex[~mid]):5.1f}%")
        key = name.split(",")[0] + ("_tan" if "tan" in name else "")
        res.update({f"{key}_t": t, f"{key}_rule": r, f"{key}_pca6": p, f"{key}_exact": ex,
                    f"{key}_mid": mid, f"{key}_Z": Z[:, :6], f"{key}_left": np.array(left[:40])})
    save("loops_and_rules", **res)
    return res


if __name__ == "__main__":
    main()
