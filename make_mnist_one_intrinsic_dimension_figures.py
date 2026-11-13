"""Figures for figures/mnist_one_intrinsic_dimension (run the analysis first).

    .venv/bin/python make_mnist_one_intrinsic_dimension_figures.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_mnist_one_intrinsic_dimension as A

OUT = A.OUT
R = {p: json.loads((OUT / f"{p}.json").read_text())
     for p in ("size", "blursize", "corr", "local")}

MNIST_C = "#eb6834"                       # orange: the real data
WOB_C = "#1baf7a"                         # aqua: big + tiny knobs
FAM = {"strokes_5": ("5 knobs", "#b9cde6"), "strokes_8": ("8 knobs", "#7fa6d6"),
       "strokes_12": ("12 knobs", "#3f76c0"), "strokes_16": ("16 knobs", "#173f7a")}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 13.5, "axes.titleweight": "bold"})


def end_label(ax, x, y, text, color, dy=0.0, bold=False):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center",
                fontsize=11, color="#222" if not bold else color,
                fontweight="bold" if bold else "normal")


# ------------------------------------------------------------------ 1 samples
def fig_samples():
    data = A.datasets()
    rows = [("MNIST 1s (real)", data["mnist"])] + \
           [(FAM[k][0] + " (drawn)", data[k]) for k in FAM] + \
           [("8 big + 40 tiny knobs", data["wobbly_8_40"])]
    fig, ax = plt.subplots(len(rows), 10, figsize=(11, 7.4))
    for r, (name, X) in enumerate(rows):
        for j in range(10):
            ax[r, j].imshow(X[j].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
            ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
        ax[r, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=12)
    fig.suptitle("The real 1s, and the drawn families whose number of knobs is known",
                 fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.2, right=0.99, top=0.93, bottom=0.01, wspace=0.05, hspace=0.08)
    fig.savefig(OUT / "fig1_samples.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ 2 count vs n
def fig_size():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    panels = [("size", "A. Sharp images"), ("blursize", "B. Same images, blurred by 2 pixels")]
    for ax, (part, title) in zip(axes, panels):
        res = R[part]
        for key, (label, col) in FAM.items():
            if key not in res:
                continue
            n = np.array([int(k) for k in res[key]]); y = [v["twonn"] for v in res[key].values()]
            ax.plot(n, y, "-o", color=col, lw=2, ms=5)
            end_label(ax, n[-1], y[-1], label, col)
        if "wobbly_8_40" in res:
            n = np.array([int(k) for k in res["wobbly_8_40"]])
            y = [v["twonn"] for v in res["wobbly_8_40"].values()]
            ax.plot(n, y, "--o", color=WOB_C, lw=2, ms=5)
            end_label(ax, n[-1], y[-1], "8 big +\n40 tiny", WOB_C, bold=True)
        n = np.array([int(k) for k in res["mnist"]]); y = [v["twonn"] for v in res["mnist"].values()]
        ax.plot(n, y, "-o", color=MNIST_C, lw=3.2, ms=7)
        end_label(ax, n[-1], y[-1], f"MNIST 1s\n{y[-1]:.1f}", MNIST_C, bold=True)
        ax.set_xscale("log"); ax.set_xlim(400, 90000)
        ax.set_xticks([500, 1000, 2000, 4000, 8000, 13533, 32000])
        ax.set_xticklabels(["500", "1k", "2k", "4k", "8k", "13.5k", "32k"])
        ax.set_xlabel("number of images the count is given")
        ax.set_title(title, loc="left")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("dimension count (nearest-neighbour ratio method)")
    axes[0].text(0.02, 0.97, "real MNIST 1s run out at 13,533;\ndrawn families go to 32,000",
                 transform=axes[0].transAxes, va="top", fontsize=10.5, color="#555")
    fig.suptitle("Drawn families level off as images are added; MNIST 1s keep climbing",
                 fontsize=14.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig2_count_vs_images.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ 3 count vs scale
def corr_slope(d):
    r = np.array(d["r_rel"]); C = np.array(d["C"])
    ok = C > 0
    lr, lc = np.log(r[ok]), np.log(C[ok])
    xs, ys = [], []
    for i in range(5, len(lc) - 5):
        if np.exp(lc[i]) < 1e-6 or np.exp(lc[i]) > 0.2:
            continue
        xs.append(np.exp(lc[i])); ys.append(np.polyfit(lr[i - 5:i + 6], lc[i - 5:i + 6], 1)[0])
    return np.array(xs), np.array(ys)


def fig_scale():
    fig, ax = plt.subplots(figsize=(11, 6))
    for key, (label, col) in FAM.items():
        x, y = corr_slope(R["corr"][key]); ax.plot(x, y, color=col, lw=2.2)
        end_label(ax, x[-1], y[-1], label, col)
    x, y = corr_slope(R["corr"]["wobbly_8_40"]); ax.plot(x, y, "--", color=WOB_C, lw=2.2)
    end_label(ax, x[-1], y[-1], "8 big + 40 tiny", WOB_C, dy=-12, bold=True)
    x, y = corr_slope(R["corr"]["mnist"]); ax.plot(x, y, color=MNIST_C, lw=3.4)
    end_label(ax, x[-1], y[-1], "MNIST 1s", MNIST_C, dy=6, bold=True)
    ax.set_xscale("log"); ax.set_xlim(1e-6, 1.2)
    ax.set_xticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_xticklabels(["closest\n1 in a million", "1 in\n100,000", "1 in\n10,000",
                        "1 in\n1,000", "1 in\n100", "1 in\n10"])
    ax.set_xlabel("which pairs of images are being compared  (fine detail  ←   →  overall shape)")
    ax.set_ylabel("dimension count at that scale")
    ax.grid(alpha=0.25)
    ax.set_title("At the scale of overall shape MNIST 1s count like 8 knobs; "
                 "at fine scale, like 16 or more", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_count_vs_scale.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ 4 local planes
def fig_local():
    fig, ax = plt.subplots(figsize=(11, 6))
    j = np.arange(31)
    for key, (label, col) in FAM.items():
        y = 100 * np.array(R["local"][key]["median"]); ax.plot(j, y, color=col, lw=2.2)
        end_label(ax, j[-1], y[-1], label, col)
    y = 100 * np.array(R["local"]["wobbly_8_40"]["median"]); ax.plot(j, y, "--", color=WOB_C, lw=2.2)
    end_label(ax, j[-1], y[-1], "8 big + 40 tiny", WOB_C, bold=True)
    y = 100 * np.array(R["local"]["mnist"]["median"]); ax.plot(j, y, color=MNIST_C, lw=3.4)
    end_label(ax, j[-1], y[-1], f"MNIST 1s  {y[-1]:.0f}%", MNIST_C, bold=True)
    ax.set_xlim(0, 36); ax.set_ylim(0, 102)
    ax.set_xlabel("number of flat directions through the 50 nearest neighbours")
    ax.set_ylabel("% of a held-out 1's offset still unexplained\n(median over 600 held-out 1s)")
    ax.grid(alpha=0.25)
    ax.set_title("Pointing at a held-out 1 from its neighbours: drawn families get there, "
                 "MNIST keeps a long tail", loc="left", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_local_planes.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ 5 neighbours
def fig_pairs():
    X = A.mnist_ones()
    D, idx = A.knn(X, 1)
    scale = np.median(np.linalg.norm(X - X.mean(0), axis=1))
    rel = D[:, 0] / scale
    order = np.argsort(np.abs(rel - np.median(rel)))[:400]        # typical pairs
    picks = order[[0, 60, 120, 200, 300, 399]]
    fig, ax = plt.subplots(3, len(picks), figsize=(11, 6.4))
    for c, i in enumerate(picks):
        a, b = X[i].reshape(28, 28), X[idx[i, 0]].reshape(28, 28)
        ax[0, c].imshow(a, cmap="gray_r", vmin=0, vmax=1)
        ax[1, c].imshow(b, cmap="gray_r", vmin=0, vmax=1)
        ax[2, c].imshow(a - b, cmap="RdBu_r", vmin=-1, vmax=1)
        ax[0, c].set_title(f"apart: {100 * rel[i]:.0f}%", fontsize=11)
        for r in range(3):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    for r, t in enumerate(["a real 1", "its closest\nother 1\n(of 13,533)",
                           "difference\nred: more ink\nin the top one\nblue: less"]):
        ax[r, 0].set_ylabel(t, rotation=0, ha="right", va="center", fontsize=11.5)
    fig.suptitle("Typical nearest neighbours: same overall shape, many small differences along the edges",
                 fontsize=13.5, fontweight="bold")
    fig.text(0.99, 0.01, "'apart' = distance between the two images, as % of a typical 1's distance "
             "from the average 1", ha="right", fontsize=9.5, color="#555")
    fig.subplots_adjust(left=0.17, right=0.99, top=0.88, bottom=0.05, wspace=0.06, hspace=0.06)
    fig.savefig(OUT / "fig5_nearest_pairs.png", dpi=110)
    plt.close(fig)
    return float(np.median(rel))


if __name__ == "__main__":
    fig_samples(); fig_size(); fig_scale(); fig_local()
    print("median nearest-neighbour separation", fig_pairs())
