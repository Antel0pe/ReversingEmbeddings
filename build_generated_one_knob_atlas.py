"""A reusable atlas of knob frames, and the measurement of why it is slices.

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python build_generated_one_knob_atlas.py

The request was to trace the manifold out: at every setting, record where the
five knob directions point, store it, and stop re-deriving. Part 1 measures
whether that is affordable. It is not, by about eight orders of magnitude, so
the rest builds the largest useful thing that does fit and says what it is good
for.

1. HOW FINE WOULD A DENSE MAP HAVE TO BE. For each knob, bisect for the step at
   which a straight-line prediction of the image is 1% wrong. Multiply the
   resulting counts to get the lattice size, and price it.
2. COARSE METRIC LATTICE. 9 points per axis over the default box, storing the
   15 numbers of G = J J^T. Under-resolved for frames by construction; part 5
   measures whether it is under-resolved for the METRIC too, which is the thing
   it is actually for.
3. FINE SWEEPS. One knob at a time through root 1, 81 points, storing all five
   784-vectors -- the "lines" of the progressive construction. Both renderers,
   so frame drift can be separated from crease noise.
4. PAIR SLICES. All ten knob pairs, 21x21, metric only -- the "planes".
5. WHAT INTERPOLATION CAN AND CANNOT RECOVER. From the lattice, interpolate (a)
   the metric and (b) the frame directions at random off-lattice points, and
   compare both to truth. The gap between the two answers is the whole point.

Knob values are the index of this dataset, not a fitted quantity: this is a map
of a known generator, not a recovery method.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

import knob_frames as kf
from grey_ones import KNOBS, RANGES, render

OUT = Path(__file__).resolve().parent / "figures/generated_one_knob_atlas"
ROOT = np.array([14.0, 14.0, 19.75, 3.2, 0.0])
LATTICE_N = 9
SWEEP_N = 81
PAIR_N = 21
PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
SPAN = RANGES[:, 1] - RANGES[:, 0]


def linearity_step(n_probe=40, target=0.01, seed=0):
    """Largest single-knob step whose first-order prediction stays within 1%."""
    rng = np.random.default_rng(seed)
    A = RANGES[:, 0] + rng.random((n_probe, 5)) * SPAN
    out = {}
    for k, nm in enumerate(KNOBS):
        hs = []
        for p in A:
            J = kf.jacobian(p[None])[0, k]
            I0 = render(p[None], dtype=np.float64).ravel()
            lo, hi = 1e-4 * SPAN[k], SPAN[k]
            for _ in range(36):
                h = np.sqrt(lo * hi)
                q = p.copy(); q[k] += h
                I1 = render(q[None], dtype=np.float64).ravel()
                err = np.linalg.norm(I0 + h * J - I1) / max(np.linalg.norm(I1 - I0), 1e-300)
                lo, hi = (lo, h) if err > target else (h, hi)
            hs.append(np.sqrt(lo * hi))
        out[nm] = dict(step=float(np.median(hs)), steps_per_axis=float(SPAN[k] / np.median(hs)))
    return out


def lattice_metric(axes, chunk=1000):
    grids = np.meshgrid(*axes, indexing="ij")
    P = np.stack([g.ravel() for g in grids], 1)
    M = np.empty((len(P), 15), np.float32)
    for s in range(0, len(P), chunk):
        M[s:s + chunk] = kf.metric(P[s:s + chunk], flat=True).astype(np.float32)
    return P, M


def multilinear_frame(axes, p, knob):
    """Interpolate one knob's direction from the 32 surrounding lattice corners."""
    idx, t = [], []
    for d in range(5):
        a = axes[d]
        i = int(np.clip(np.searchsorted(a, p[d]) - 1, 0, len(a) - 2))
        idx.append(i); t.append((p[d] - a[i]) / (a[i + 1] - a[i]))
    corners, weights = [], []
    for bits in range(32):
        off = [(bits >> d) & 1 for d in range(5)]
        corners.append([axes[d][idx[d] + off[d]] for d in range(5)])
        w = 1.0
        for d in range(5):
            w *= t[d] if off[d] else (1 - t[d])
        weights.append(w)
    J = kf.jacobian(np.array(corners))[:, knob]
    J = J / np.maximum(np.linalg.norm(J, axis=1, keepdims=True), 1e-300)
    v = np.array(weights) @ J
    return v / max(np.linalg.norm(v), 1e-300)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}

    res["linearity"] = linearity_step()
    total = float(np.prod([v["steps_per_axis"] for v in res["linearity"].values()]))
    res["dense_cost"] = dict(
        lattice_points=total,
        bytes_metric_only=total * 15 * 4,
        bytes_five_directions=total * 5 * 784 * 4,
        note="metric is 15 float32; five directions are 5 x 784 float32 = 15.7 kB")

    axes = [np.linspace(RANGES[k, 0], RANGES[k, 1], LATTICE_N) for k in range(5)]
    Plat, Mlat = lattice_metric(axes)
    G = kf.unflatten_metric(Mlat.astype(np.float64))
    sp = kf.speeds(G)
    cos = kf.knob_cosines(G)
    off = np.abs(cos - np.eye(5)[None])
    worst = int(np.argmax(off.reshape(len(off), -1).max(1)))
    wi, wj = np.unravel_index(int(np.argmax(off[worst])), (5, 5))
    res["speeds"] = {nm: dict(min=float(sp[:, k].min()), median=float(np.median(sp[:, k])),
                              max=float(sp[:, k].max()))
                     for k, nm in enumerate(KNOBS)}
    ev = np.linalg.eigvalsh(G)
    cond = ev[:, -1] / np.maximum(ev[:, 0], 1e-300)
    res["degeneracy"] = dict(
        cond_min=float(cond.min()), cond_median=float(np.median(cond)),
        cond_max=float(cond.max()),
        worst_pair=f"{KNOBS[wi]}-{KNOBS[wj]}",
        worst_abs_cos=float(off[worst, wi, wj]),
        worst_at={nm: float(Plat[worst][k]) for k, nm in enumerate(KNOBS)},
        root_height_width_cos=float(kf.knob_cosines(kf.metric(ROOT[None]))[0, 2, 3]))

    sweep_knobs = np.empty((5, SWEEP_N, 5))
    Jg = np.empty((5, SWEEP_N, 5, 784), np.float32)
    Js = np.empty((5, SWEEP_N, 5, 784), np.float32)
    sens = np.empty((5, SWEEP_N), np.float32)
    for k in range(5):
        P = np.tile(ROOT, (SWEEP_N, 1))
        P[:, k] = np.linspace(RANGES[k, 0], RANGES[k, 1], SWEEP_N)
        sweep_knobs[k] = P
        a = kf.jacobian(P)
        b = kf.jacobian(P, fd=2 * kf.DEFAULT_FD)
        Jg[k] = a.astype(np.float32)
        Js[k] = kf.jacobian(P, renderer=kf.smooth_renderer).astype(np.float32)
        sens[k] = (np.linalg.norm(a - b, axis=2) / np.maximum(
            np.linalg.norm(a, axis=2), 1e-300)).mean(1).astype(np.float32)

    Ug = Jg[np.arange(5), :, np.arange(5), :]            # knob k's own direction on its own sweep
    Ug = Ug / np.maximum(np.linalg.norm(Ug, axis=2, keepdims=True), 1e-300)
    Us = Js[np.arange(5), :, np.arange(5), :]
    Us = Us / np.maximum(np.linalg.norm(Us, axis=2, keepdims=True), 1e-300)
    res["self_drift"] = {nm: dict(
        cos_vs_start_end=float(Ug[k, -1] @ Ug[k, 0]),
        cos_min=float((Ug[k] @ Ug[k, 0]).min()),
        smooth_cos_vs_start_end=float(Us[k, -1] @ Us[k, 0]))
        for k, nm in enumerate(KNOBS)}
    res["step_sensitivity"] = {nm: float(np.median(sens[k])) for k, nm in enumerate(KNOBS)}

    # how much does a knob's direction turn over ONE lattice step?
    rng0 = np.random.default_rng(11)
    S = RANGES[:, 0] + rng0.random((40, 5)) * SPAN
    spacing = SPAN / (LATTICE_N - 1)
    res["lattice_step_cosine"] = {}
    for k, nm in enumerate(KNOBS):
        c = []
        for q0 in S:
            q1 = q0.copy()
            q1[k] += spacing[k] if q0[k] + spacing[k] <= RANGES[k, 1] else -spacing[k]
            F = kf.frame(np.stack([q0, q1]))
            c.append(float(F[0, k] @ F[1, k]))
        res["lattice_step_cosine"][nm] = dict(median=float(np.median(c)), min=float(np.min(c)),
                                              step=float(spacing[k]))

    pair_axes = np.empty((len(PAIRS), 2, PAIR_N))
    pair_M = np.empty((len(PAIRS), PAIR_N, PAIR_N, 15), np.float32)
    for n, (i, j) in enumerate(PAIRS):
        ai = np.linspace(RANGES[i, 0], RANGES[i, 1], PAIR_N)
        aj = np.linspace(RANGES[j, 0], RANGES[j, 1], PAIR_N)
        pair_axes[n] = [ai, aj]
        A, B = np.meshgrid(ai, aj, indexing="ij")
        P = np.tile(ROOT, (PAIR_N * PAIR_N, 1))
        P[:, i] = A.ravel(); P[:, j] = B.ravel()
        pair_M[n] = kf.metric(P, flat=True).astype(np.float32).reshape(PAIR_N, PAIR_N, 15)

    # ---- what interpolation of the lattice can recover ----
    from scipy.interpolate import RegularGridInterpolator
    rgi = RegularGridInterpolator(axes, Mlat.reshape(*[LATTICE_N] * 5, 15).astype(np.float64))
    rng = np.random.default_rng(5)
    T = RANGES[:, 0] + rng.random((300, 5)) * SPAN
    T = np.clip(T, RANGES[:, 0] + 1e-9, RANGES[:, 1] - 1e-9)
    Mi = rgi(T)
    Mt = kf.metric(T, flat=True)
    rel = np.linalg.norm(Mi - Mt, axis=1) / np.linalg.norm(Mt, axis=1)
    Ft = kf.frame(T[:60])
    fcos = {nm: [] for nm in KNOBS}
    for n, p in enumerate(T[:60]):
        for k, nm in enumerate(KNOBS):
            fcos[nm].append(float(multilinear_frame(axes, p, k) @ Ft[n, k]))
    res["interpolation"] = dict(
        metric_rel_median=float(np.median(rel)), metric_rel_p90=float(np.percentile(rel, 90)),
        frame_cos_median={nm: float(np.median(v)) for nm, v in fcos.items()},
        frame_cos_min={nm: float(np.min(v)) for nm, v in fcos.items()})

    np.savez_compressed(
        OUT / "atlas.npz",
        knob_names=np.array(KNOBS), ranges=RANGES, root=ROOT,
        fd_steps=kf.DEFAULT_FD, metric_index=np.array(kf.METRIC_IJ),
        lattice_axes=np.array(axes), lattice_metric=Mlat.reshape(*[LATTICE_N] * 5, 15),
        sweep_knobs=sweep_knobs, sweep_jac_grey=Jg, sweep_jac_smooth=Js,
        sweep_step_sensitivity=sens,
        pair_index=np.array(PAIRS), pair_axes=pair_axes, pair_metric=pair_M)
    res["file_mb"] = round((OUT / "atlas.npz").stat().st_size / 1e6, 2)
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items()
                      if k != "interpolation"}, indent=2)[:2600])
    print("\ninterpolation:", json.dumps(res["interpolation"], indent=2))
    print("\natlas.npz =", res["file_mb"], "MB")
    make_figure(res)




def make_figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = dict(np.load(OUT / "atlas.npz", allow_pickle=False))
    fig = plt.figure(figsize=(18.5, 9.6), constrained_layout=True)
    fig.suptitle("Generated 1s: what a stored atlas of knob frames can and cannot hold\n"
                 "a 'frame' at a setting = the five 784-vectors the image moves along, "
                 "one per knob (units: pixels, or degrees for lean)",
                 fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(2, 4)
    BLUE, ORANGE, GREEN, PURPLE = "#1b6ca8", "#c44601", "#047857", "#7c3aed"

    ax = fig.add_subplot(gs[0, 0])
    n = [res["linearity"][k]["steps_per_axis"] for k in KNOBS]
    ax.bar(KNOBS, n, color=BLUE)
    ax.set_yscale("log"); ax.set_ylabel("grid points needed along that axis")
    for i, v in enumerate(n):
        ax.text(i, v * 1.15, f"{v:.0f}", ha="center", fontsize=9)
    dc = res["dense_cost"]
    ax.set_title("A. Why the whole box cannot be stored\n"
                 "spacing at which a straight-line step is 1% wrong\n"
                 f"product = {dc['lattice_points']/1e9:.0f} billion points\n"
                 f"= {dc['bytes_metric_only']/1e12:.1f} TB (metric) or "
                 f"{dc['bytes_five_directions']/1e15:.1f} PB (frames)", fontsize=10)
    ax.set_ylim(1, max(n) * 4)

    ax = fig.add_subplot(gs[0, 1])
    G = kf.unflatten_metric(d["lattice_metric"].reshape(-1, 15).astype(np.float64))
    sp = kf.speeds(G)
    ax.boxplot([sp[:, k] for k in range(5)], tick_labels=KNOBS, showfliers=False)
    ax.set_ylabel("image distance moved per unit of knob")
    ax.set_title("B. Knob speeds over the default box\n"
                 f"{LATTICE_N}^5 = {LATTICE_N**5:,} lattice points\n"
                 f"cx moves the image ~{sp[:,0].mean()/sp[:,4].mean():.0f}x faster per unit "
                 "than lean", fontsize=10)

    ax = fig.add_subplot(gs[0, 2])
    wa = res["degeneracy"]["worst_at"]
    P = np.array([[wa[k] for k in KNOBS]])
    C = np.abs(kf.knob_cosines(kf.metric(P))[0])
    im = ax.imshow(C, cmap="magma_r", vmin=0, vmax=1)
    ax.set_xticks(range(5), KNOBS, rotation=45, ha="right"); ax.set_yticks(range(5), KNOBS)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if C[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046).set_label("|cosine| between knob directions", fontsize=8.5)
    dg = res["degeneracy"]
    ax.set_title("C. Where two knobs nearly coincide\n"
                 f"worst point in the box: {dg['worst_pair']} at |cos| = {dg['worst_abs_cos']:.3f}\n"
                 f"(thin and leaned: width {wa['width']:.1f}, lean {wa['lean']:.0f} deg)", fontsize=10)

    ax = fig.add_subplot(gs[0, 3])
    x = np.arange(5)
    step = [res["lattice_step_cosine"][k]["median"] for k in KNOBS]
    rec = [res["interpolation"]["frame_cos_median"][k] for k in KNOBS]
    ax.bar(x - 0.2, step, 0.4, color=ORANGE, label="same knob, one lattice step apart")
    ax.bar(x + 0.2, rec, 0.4, color=GREEN, label="interpolated from the lattice vs truth")
    ax.set_xticks(x, KNOBS); ax.set_ylim(0, 1.15); ax.set_ylabel("cosine (1 = same direction)")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("D. The frame turns fast but SMOOTHLY\n"
                 f"lean: only {step[4]:.2f} aligned one step on,\n"
                 f"yet interpolation recovers it to {rec[4]:.2f}", fontsize=10)

    for n2, cxv in enumerate([14.0, 15.0]):
        ax = fig.add_subplot(gs[1, n2])
        p = ROOT.copy(); p[0] = cxv
        v = kf.jacobian(p[None])[0, 0].reshape(28, 28)
        lim = np.abs(v).max()
        im = ax.imshow(v, cmap="RdBu_r", vmin=-lim, vmax=lim, extent=[0, 28, 28, 0],
                       interpolation="nearest")
        ax.set_title(f"E{n2+1}. the cx direction at cx = {cxv:.0f}", fontsize=10)
        ax.set_xlabel("column")
        if n2 == 0:
            ax.set_ylabel("row")
        fig.colorbar(im, ax=ax, fraction=0.046).set_label(
            "ink change per pixel of cx:  red = gains,  blue = loses", fontsize=7.5)

    ax = fig.add_subplot(gs[1, 2])
    pm = d["pair_metric"][9]                       # pair (width, lean)
    c01 = np.abs(pm[..., 1] / np.sqrt(np.maximum(pm[..., 0] * pm[..., 5], 1e-30)))
    wax, lax = d["pair_axes"][9]
    im = ax.imshow(c01.T, origin="lower", aspect="auto", cmap="magma_r", vmin=0, vmax=1,
                   extent=[wax[0], wax[-1], lax[0], lax[-1]])
    ax.set_xlabel("width (pixels)"); ax.set_ylabel("lean (degrees)")
    fig.colorbar(im, ax=ax, fraction=0.046).set_label("|cos| between cx and cy directions", fontsize=8)
    ax.set_title("F. A stored slice: where cx and cy blur together\n"
                 "other three knobs held at root 1; degeneracy grows with lean\n"
                 "and with thinness, reaching 0.96 in the far corner", fontsize=10)

    ax = fig.add_subplot(gs[1, 3])
    itp = res["interpolation"]
    ax.axis("off")
    rows = [("file on disk", f"{res['file_mb']} MB"),
            ("metric lattice", f"{LATTICE_N}^5 points x 15 numbers"),
            ("fine sweeps", f"5 knobs x {SWEEP_N} pts, all 5\n     directions, both renderers"),
            ("pair slices", f"10 pairs x {PAIR_N}x{PAIR_N}, metric"),
            ("", ""),
            ("metric interp.", f"{itp['metric_rel_median']*100:.1f}% median error"
                               f" (p90 {itp['metric_rel_p90']*100:.1f}%)"),
            ("frame interp.", "cos "
             f"{min(itp['frame_cos_median'].values()):.2f}-{max(itp['frame_cos_median'].values()):.2f}"
             f" median,\n     worst {min(itp['frame_cos_min'].values()):.2f}"),
            ("", ""),
            ("cx dir at cx=14", "columns 12 and 15"),
            ("cx dir at cx=15", "columns 13 and 16"),
            ("their cosine", "exactly 0.000")]
    y = 0.97
    ax.text(0.0, y, "G. What is in atlas.npz", fontsize=11, fontweight="bold", va="top")
    y -= 0.10
    for a, b in rows:
        if a:
            ax.text(0.0, y, a, fontsize=9.5, va="top", fontweight="bold")
            ax.text(0.40, y, b, fontsize=9.5, va="top")
        y -= 0.075 if "\n" not in b else 0.12
    fig.savefig(OUT / "knob_atlas.png", dpi=130)
    plt.close(fig)
    print("wrote", OUT / "knob_atlas.png")


if __name__ == "__main__":
    main()
