"""One summary figure: run the other scripts first (they write results/*.npz).

Run:  .venv/bin/python curved_components/make_figure.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import curve_dictionary as cd  # noqa: E402
from data import OUT, ROOT_ONE, render  # noqa: E402

BLUE, ORANGE, AQUA, MAGENTA, GREY, INK, MUTED, GRID = (
    "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#8a8980", "#1a1a19", "#5f5e58", "#e1e0d9")


def outside_shape_list(knob, t, mid, n_shapes=10):
    P = np.repeat(ROOT_ONE[None], len(t), 0)
    P[:, knob] = t
    X = render(P, dtype=np.float64).reshape(len(t), -1)
    s = cd.scale(t, t.min(), t.max())
    chosen, _ = cd.greedy(s[mid], X[mid], n_shapes)
    F = np.c_[np.ones(len(t)), cd.library(s)[0]][:, chosen]
    W = np.linalg.lstsq(F[mid], X[mid], rcond=None)[0]
    rebuilt = F @ W
    return 100 * np.linalg.norm(rebuilt - X, axis=1) / np.linalg.norm(X, axis=1)


def main():
    pp = np.load(os.path.join(OUT, "principal_polynomial.npz"))
    dc = np.load(os.path.join(OUT, "curve_dictionary.npz"))
    lr = np.load(os.path.join(OUT, "loops_and_rules.npz"))
    pc = {n: np.load(os.path.join(OUT, f"principal_curve_{n}.npz")) for n in ["slide", "lean"]}
    pc_miss = {"slide": 5.30, "lean": 0.89}   # printed by principal_curves.py (held-out, 1 number)

    fig = plt.figure(figsize=(16, 6.6))
    gs = fig.add_gridspec(1, 3, wspace=0.28, left=0.05, right=0.985, top=0.80, bottom=0.12)
    fig.suptitle("Curved-component methods on generated 1s with one knob changing: "
                 "how much of the picture do they miss on pictures they were not shown?",
                 fontsize=13.5, color=INK)
    fig.text(0.05, 0.885, "Miss = size of the error as a % of the picture's own size (lower is better). "
             "Panels A-B test positions BETWEEN the ones a method saw; panel C tests positions BEYOND them.",
             fontsize=10.5, color=MUTED)
    for j, (name, title) in enumerate([("slide", "A. The 1 sliding sideways (cx 5 to 23 px)"),
                                       ("lean", "B. The 1 leaning (-10 to 35 degrees)")]):
        ax = fig.add_subplot(gs[0, j])
        k = np.arange(26)
        ax.plot(k, pp[f"{name}_pca"][:26], color=BLUE, lw=2.2, label="PCA: straight components")
        ax.plot(k, pp[f"{name}_deg3"][:26], color=ORANGE, lw=2, label="bent components (principal polynomial)")
        ax.plot(k, dc[f"{name}_between"][:26], color=AQUA, lw=2, ls="--",
                label="curve shapes from a list (told the knob)")
        ax.plot([1], [pc_miss[name]], "o", color=MAGENTA, ms=11, mec="white", mew=2,
                label="principal curve: ONE number")
        ax.annotate(f"principal curve\n1 number, {pc_miss[name]:.1f}%", xy=(1, pc_miss[name]),
                    xytext=(4.5, 45), fontsize=10, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
        ax.axhline(10, color=GREY, lw=0.8, ls=":")
        ax.text(25, 11.5, "10% line", ha="right", fontsize=9, color=MUTED)
        ax.set_xlim(0, 25); ax.set_ylim(0, 85)
        ax.set_xlabel("how many components / shapes kept", color=MUTED)
        ax.set_ylabel("miss on in-between positions (%)", color=MUTED)
        ax.set_title(title, fontsize=12, color=INK, loc="left")
        ax.grid(color=GRID, lw=0.6)
        if j == 0:
            ax.legend(fontsize=9.5, frameon=False, loc="upper right")

    ax = fig.add_subplot(gs[0, 2])
    bars = []
    for name, knob, key in [("slide", 0, "slide"), ("lean", 4, "lean")]:
        t, mid = lr[f"{key}_t"], lr[f"{key}_mid"]
        shapes = outside_shape_list(knob, t, mid)
        bars.append([np.median(lr[f"{key}_pca6"][~mid]), np.median(shapes[~mid]),
                     np.median(lr[f"{key}_rule"][~mid]), np.median(lr[f"{key}_exact"][~mid])])
    bars = np.array(bars)
    labels = ["6 PCA\ncomponents", "10 curve shapes\nfrom the list", "one repeated\nrule (turning)"]
    cols = [BLUE, AQUA, ORANGE]
    x = np.arange(2)
    for i in range(3):
        b = ax.bar(x + (i - 1) * 0.26, np.minimum(bars[:, i], 130), 0.24, color=cols[i], label=labels[i].replace("\n", " "))
        for xi, v in zip(x + (i - 1) * 0.26, bars[:, i]):
            ax.text(xi, min(v, 130) + 2, f"{v:.0f}%" + (" (cut off)" if v > 130 else ""), ha="center",
                    fontsize=10, color=INK)
    for xi, v in zip(x, bars[:, 3]):
        ax.plot([xi - 0.4, xi + 0.4], [v, v], color=INK, lw=1.2, ls="--")
    ax.plot([], [], color=INK, lw=1.2, ls="--", label="the exact move on the same\n28x28 grid (the grid's floor)")
    ax.set_xticks(x, ["slide", "lean"])
    ax.set_ylim(0, 140)
    ax.set_ylabel("median miss on positions never seen (%)", color=MUTED)
    ax.set_title("C. Trained on the middle third only:\nwhich ones carry on past what they saw?",
                 fontsize=12, color=INK, loc="left")
    ax.legend(fontsize=9.5, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.84))
    ax.grid(axis="y", color=GRID, lw=0.6)
    path = os.path.join(OUT, "curved_components_summary.png")
    fig.savefig(path, dpi=110)
    print(path, bars.round(1))


if __name__ == "__main__":
    main()
