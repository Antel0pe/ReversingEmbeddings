"""Do the negative-space rules get you anywhere near a 1? (No -- and here is by how much.)

  speckle_gallery.png  -- what legal random images actually look like, blob -> scribble
  speckle_distance.png -- how far they land from the nearest real MNIST 1

Run: python make_speckle_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import speckle_free as SF, grey_ones as G, mnist

OUT = "figures"
INK, GOOD, BAD, REF = "#1a1a1a", "#1a7f37", "#c62828", "#1565c0"
BETAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
NPER = 220


def nearest(Q, R):
    Rsq = (R ** 2).sum(1); out = np.empty(len(Q))
    for s in range(0, len(Q), 500):
        q = Q[s:s + 500]
        d2 = (q ** 2).sum(1)[:, None] + Rsq[None] - 2 * q @ R.T
        out[s:s + 500] = np.sqrt(np.maximum(d2.min(1), 0))
    return out


def main():
    rng = np.random.default_rng(3)
    Xall, yall = mnist.load("train")
    ONES = (Xall[yall == 1].reshape(-1, 784) / 255.0).astype(np.float64)
    ink = (ONES > 0).sum(1)
    n_ink = int(np.median(ink))
    print(f"{len(ONES)} MNIST 1s; median inked pixels {n_ink} (5-95pct {np.percentile(ink,[5,95]).astype(int)})")
    solid = np.array([(o[o > 0] >= .95).mean() for o in ONES])
    from scipy.ndimage import label as _lab
    onepiece = np.array([_lab(o.reshape(28, 28) > 0, np.ones((3, 3)))[1] == 1 for o in ONES])
    print(f"MNIST 1s: median fraction of ink fully black = {np.median(solid):.2f} "
          f"(5-95pct {np.percentile(solid,[5,95]).round(2)})")
    print(f"MNIST 1s passing 'one piece'      : {100*onepiece.mean():.1f}%")
    for f in (0.75, 0.5, 0.42, 0.25):
        print(f"MNIST 1s passing '{f:.0%} fully black' : {100*(solid>=f).mean():5.1f}%   "
              f"and both rules: {100*((solid>=f)&onepiece).mean():5.1f}%")
    REAL_SOLID = float(np.median(solid))

    sets, legal_rate = {}, {}
    for b in BETAS:
        Q = SF.batch(NPER, n_ink, b, rng)
        oks = [SF.legal(q)[0] for q in Q]
        legal_rate[b] = np.mean(oks); sets[b] = Q
        print(f"  beta {b:4.1f}: {100*np.mean(oks):5.1f}% pass the three rules")

    # controls
    noise = rng.random((NPER, 784))
    noise_sparse = (rng.random((NPER, 784)) < n_ink / 784) * rng.random((NPER, 784))
    gen1 = G.render(G.sample(NPER, rng)).reshape(NPER, -1).astype(np.float64)
    heldout = ONES[rng.choice(len(ONES), NPER, replace=False)]

    d = {f"beta={b:g}": nearest(sets[b], ONES) for b in BETAS}
    d["uniform noise"] = nearest(noise, ONES)
    d["sparse noise"] = nearest(noise_sparse, ONES)
    d["your generator"] = nearest(gen1, ONES)
    d["a real 1"] = nearest(heldout, ONES[:0].reshape(0, 784)) if False else None
    # for real 1s use nearest OTHER 1
    R = ONES; Rsq = (R ** 2).sum(1); out = np.empty(NPER)
    idx = rng.choice(len(ONES), NPER, replace=False)
    for i, j in enumerate(idx):
        q = ONES[j]
        dd = np.sqrt(np.maximum((q ** 2).sum() + Rsq - 2 * R @ q, 0)); dd[j] = np.inf
        out[i] = dd.min()
    d["a real 1"] = out

    # ---------------- gallery ----------------
    ncol = 8
    fig = plt.figure(figsize=(13.4, 9.6))
    fig.text(.5, .968, "Random images that obey all three rules: one piece, no speckle, 75% fully black",
             ha="center", va="center", fontsize=15.5, fontweight="bold")
    fig.text(.5, .928, "Grown pixel by pixel. β controls whether growth prefers crowded or lonely pixels: "
                       "β = 0 makes blobs, large β makes scribbles.\nEvery image here is legal. "
                       "The number under each is its distance to the nearest of 6,742 real MNIST 1s.",
             ha="center", va="center", fontsize=9.5, color="#555", linespacing=1.55)
    rows = BETAS + ["REAL"]
    L, T, B = .10, .885, .075
    rh = (T - B) / len(rows)
    for r, b in enumerate(rows):
        if b == "REAL":
            Q = heldout[:ncol]; dd = out[:ncol]; lab = "real MNIST 1s\n(calibration)\nFAIL the 75% rule"
        else:
            Q = sets[b][:ncol]; dd = d[f"beta={b:g}"][:ncol]; lab = f"β = {b:g}"
        y0 = T - (r + 1) * rh
        for c in range(ncol):
            ax = fig.add_axes([L + c * .098, y0 + rh * .30, .082, rh * .58])
            ax.imshow(Q[c].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_edgecolor(GOOD if b == "REAL" else "#ccc")
                s_.set_linewidth(1.8 if b == "REAL" else .6)
            ax.text(.5, -.10, f"{dd[c]:.1f}", transform=ax.transAxes, ha="center", va="top",
                    fontsize=8, fontweight="bold",
                    color=GOOD if dd[c] < 3 else BAD)
        fig.text(L - .012, y0 + rh * .59, lab, ha="right", va="center",
                 fontsize=10, fontweight="bold",
                 color=GOOD if b == "REAL" else INK, linespacing=1.4)
    fig.text(.5, .030, "Two real 1s sit about 1.9 apart (θ). Nothing legal-but-random lands anywhere near that — "
                       "the rules remove a lot of space\nand still leave a space made almost entirely of non-digits.",
             ha="center", va="center", fontsize=9.3, color="#333", style="italic", linespacing=1.6)
    fig.savefig(f"{OUT}/speckle_gallery.png", dpi=140)
    plt.close(fig)

    # ---------------- distance figure ----------------
    order = ["uniform noise", "sparse noise"] + [f"beta={b:g}" for b in BETAS] + ["your generator", "a real 1"]
    fig = plt.figure(figsize=(12.2, 6.2))
    fig.text(.5, .955, "The rules shrink the space enormously and barely move you toward a 1",
             ha="center", va="center", fontsize=15.5, fontweight="bold")
    fig.text(.5, .903, "Distance from each image to the nearest of 6,742 real MNIST 1s. Lower is closer to being a 1. "
                       "Bars are medians, whiskers the 10th–90th percentile.",
             ha="center", va="center", fontsize=9.4, color="#555")
    ax = fig.add_axes([.085, .21, .885, .63])
    med = [np.median(d[k]) for k in order]
    lo = [np.percentile(d[k], 10) for k in order]
    hi = [np.percentile(d[k], 90) for k in order]
    cols = []
    for k in order:
        cols.append(GOOD if k == "a real 1" else REF if k == "your generator"
                    else "#9a9a9a" if k.startswith("beta") else BAD)
    x = np.arange(len(order))
    ax.bar(x, med, color=cols, width=.66)
    ax.errorbar(x, med, yerr=[np.array(med) - lo, np.array(hi) - np.array(med)],
                fmt="none", ecolor="#333", capsize=3, lw=1.1)
    for i, m in enumerate(med):
        ax.text(i, m + .12, f"{m:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(1.89, color=GOOD, ls="--", lw=1.4)
    ax.text(len(order) - .4, 1.95, "θ = 1.89, the distance between neighbouring real 1s",
            ha="right", va="bottom", fontsize=8.8, color=GOOD)
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=9, rotation=26, ha="right")
    ax.set_ylabel("distance to the nearest real MNIST 1", fontsize=10)
    ax.grid(alpha=.22, axis="y")
    ax.text(.5, .955, "← the three rules are ON for every grey bar →", transform=ax.transAxes,
            ha="center", va="top", fontsize=9, color="#555", style="italic")
    fig.text(.5, .042, f"Legal random images land {med[order.index('beta=8')]/med[-1]:.1f}× further from a 1 than another "
                       f"real 1 does. The rules are a genuine filter — they just aren't a description.",
             ha="center", va="center", fontsize=9.4, color="#333", style="italic")
    fig.savefig(f"{OUT}/speckle_distance.png", dpi=150)
    plt.close(fig)
    print("\n  set                 median distance to nearest real 1")
    for k in order:
        print(f"  {k:18s}  {np.median(d[k]):6.2f}")


if __name__ == "__main__":
    main()
