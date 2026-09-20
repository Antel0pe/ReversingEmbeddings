"""What the 214 directions actually ARE: seven of them, drawn as images.

Each principal direction is a unit vector in 784-d PIXEL space -- a weighted
combination of the 784 pixels, i.e. an image. It is NOT a combination of the
five knobs; the knobs are only 5 and they enter the pixels non-linearly.

Run: python make_pc_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import grey_ones as G

OUT = "figures"
INK, BAD, REF = "#1a1a1a", "#c62828", "#1565c0"
KN = ["cx", "cy", "height", "width", "lean"]
SHOW = [1, 2, 3, 5, 20, 100, 213]


def main():
    rng = np.random.default_rng(0)
    N = 20000
    P = G.sample(N, rng)
    X = G.render(P).reshape(N, -1).astype(np.float64)
    mu = X.mean(0); Z = X - mu
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    var = S ** 2; share = var / var.sum(); cum = np.cumsum(share)
    C = U * S

    # how much of each PC's coefficient is explained by the five knobs, linearly
    Pz = (P - P.mean(0)) / P.std(0)
    fig = plt.figure(figsize=(15.0, 6.4))
    fig.text(.5, .955, "Seven of the 214 directions — each one is an image, not a knob",
             ha="center", va="center", fontsize=16, fontweight="bold")
    fig.text(.5, .898, "A principal direction is a weighted sum of the 784 PIXELS (red = add ink, blue = remove). "
                       "It is a direction in image space, not in knob space.\nThe bar under each shows how strongly that "
                       "direction's coefficient tracks each knob across 20,000 samples.",
             ha="center", va="center", fontsize=9.5, color="#555", linespacing=1.55)

    n = len(SHOW); L, R = .035, .988
    w = (R - L) / n
    for i, k in enumerate(SHOW):
        v = Vt[k - 1]; m = np.abs(v).max()
        ax = fig.add_axes([L + i * w + w * .085, .455, w * .70, .34])
        ax.imshow(v.reshape(28, 28), cmap="RdBu_r", vmin=-m, vmax=m, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values(): s_.set_edgecolor("#bbb"); s_.set_linewidth(.7)
        ax.text(.5, 1.10, f"PC {k}", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=INK)
        ax.text(.5, -.075, f"{100*share[k-1]:.2f}% of variance\n{100*cum[k-1]:.1f}% cumulative",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.3,
                color="#555", linespacing=1.5)
        # knob correlations
        c = np.array([abs(np.corrcoef(C[:, k - 1], Pz[:, j])[0, 1]) for j in range(5)])
        ax2 = fig.add_axes([L + i * w + w * .085, .135, w * .70, .155])
        ax2.bar(np.arange(5), c, color=[BAD if x == c.max() and x > .3 else "#9a9a9a" for x in c],
                width=.68)
        ax2.set_ylim(0, 1); ax2.set_xticks(np.arange(5))
        ax2.set_xticklabels(KN, fontsize=6.6, rotation=62, ha="right")
        ax2.tick_params(axis="y", labelsize=6.5)
        if i: ax2.set_yticklabels([])
        else: ax2.set_ylabel("|corr|\nwith knob", fontsize=7.4, linespacing=1.4)
        ax2.grid(alpha=.2, axis="y")
        best = int(np.argmax(c))
        lab = f"{KN[best]}  {c[best]:.2f}" if c[best] > .3 else "no single knob"
        ax2.text(.5, 1.06, lab, transform=ax2.transAxes, ha="center", va="bottom",
                 fontsize=8, fontweight="bold",
                 color=BAD if c[best] > .3 else "#777")

    fig.text(.5, .038, "The first few directions each track one knob — PC 1 is essentially “lean” written as an image. "
                       "By PC 100 no knob owns the direction any more:\nthose are the fine edge corrections that make the "
                       "difference between 95% of the variance and all of it.",
             ha="center", va="center", fontsize=9.4, color="#333", style="italic", linespacing=1.6)
    fig.savefig(f"{OUT}/statespace_pcs.png", dpi=150)
    plt.close(fig)

    # the literal readout for PC1
    print("PC 1 as a literal weighted sum of pixel axes (top 8 of 784 weights):")
    v = Vt[0]; idx = np.argsort(-np.abs(v))[:8]
    terms = "  ".join(f"{v[j]:+.3f}*p[{j//28},{j%28}]" for j in idx)
    print("   PC1 = " + terms + "  ... (776 more, most near zero)")
    print(f"   weights above 0.01 in magnitude: {(np.abs(v)>.01).sum()} of 784")
    for k in SHOW:
        v = Vt[k-1]
        c = np.array([abs(np.corrcoef(C[:, k-1], Pz[:, j])[0,1]) for j in range(5)])
        print(f"   PC{k:4d}  var {100*share[k-1]:6.3f}%  cum {100*cum[k-1]:6.2f}%  "
              f"nonzero px {(np.abs(v)>.01).sum():3d}  best knob {KN[int(np.argmax(c))]} {c.max():.2f}")


if __name__ == "__main__":
    main()
