"""Figures for figures/drawn_one_warp_scorecard (no MNIST images are drawn).

  examples.png   what the drawn families look like (drawn 1s only)
  leftover.png   leftover vs how many numbers each image gets, every family
  knobs.png      how well the numbers can be read as each TRUE knob (12-knob 1s)
  shape.png      where the leftover sits: drawn families vs MNIST
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_drawn_one_warp_scorecard import OUT, load

S = json.loads((OUT / "scorecard.json").read_text())
SHAPE = json.loads((OUT / "leftover_shape.json").read_text())
CROSS = json.loads((OUT / "cross_section.json").read_text())
B90 = json.loads((OUT / "best_ninety.json").read_text())

LABEL = {"drawn5": "drawn 1s, 5 knobs", "drawn8": "drawn 1s, 8 knobs (+ bend, wiggle, taper)",
         "drawn12": "drawn 1s, 12 knobs (+ bulge, hook, bar)",
         "wobbly_0.8": "12 knobs + smooth wobbles (0.8 px)",
         "sharp5": "drawn 1s, 5 knobs (sharp edges like MNIST)",
         "sharp12": "drawn 1s, 12 knobs (sharp edges like MNIST)",
         "mnist": "real MNIST 1s"}
COL = {"drawn5": "#9aa5b1", "drawn8": "#5b8fd1", "drawn12": "#1f4e9c",
       "wobbly_0.8": "#b07cc6", "sharp5": "#e8b27a", "sharp12": "#d9661f", "mnist": "#111111"}
SHORT = {"drawn5": "5 knobs", "drawn8": "8 knobs", "drawn12": "12 knobs\n(always a\nhook & bar)",
         "wobbly_0.8": "12 knobs\n+ wobbles", "sharp5": "5 knobs,\nsharp\nedges",
         "sharp12": "12 knobs,\nsharp\nedges", "mnist": "real\nMNIST 1s"}
KNOB_LABEL = ["horizontal position", "vertical position", "height", "width", "lean",
              "bend", "wiggle", "taper", "bulge", "hook length", "hook angle", "bar length"]


def pieces(name):
    p = OUT / f"pieces_{name}_20.json"
    return json.loads(p.read_text()) if p.exists() else None


def examples():
    rows = [n for n in ("drawn5", "drawn8", "drawn12", "sharp12") if n in S]
    fig = plt.figure(figsize=(10, 1.9 * len(rows) + 0.9))
    fig.text(0.5, 0.975, "What the drawn 1s look like (6 random members of each family)",
             ha="center", va="top", fontsize=13, weight="bold")
    h = 0.86 / len(rows)
    for r, name in enumerate(rows):
        I = load(name)[0][:6]
        y0 = 0.90 - (r + 1) * h
        fig.text(0.01, y0 + h * 0.5, LABEL[name].replace(" (", "\n("), fontsize=9.5,
                 va="center", color=COL[name], weight="bold")
        for c in range(6):
            ax = fig.add_axes([0.30 + c * 0.115, y0 + 0.08 * h, 0.105, 0.84 * h])
            ax.imshow(I[c], cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.5, 0.012, "Each image is 28 x 28 pixels; black = full ink. "
             "No real MNIST 1s are shown in any of these figures.",
             ha="center", fontsize=9, color="#555")
    fig.savefig(OUT / "examples.png", dpi=130)
    plt.close(fig)


def leftover():
    fig = plt.figure(figsize=(11, 6.4))
    ax = fig.add_axes([0.08, 0.13, 0.56, 0.72])
    fig.text(0.5, 0.97, "How much of each image is left over, as each image gets more numbers",
             ha="center", va="top", fontsize=13, weight="bold")
    fig.text(0.5, 0.925, "Solid: warp numbers only (where the ink goes).   "
             "Dashed: 20 warp numbers plus 1-16 'ink pieces' (extra ink the template can add).",
             ha="center", va="top", fontsize=9.5, color="#444")
    order = [n for n in ("sharp5", "drawn8", "drawn12", "sharp12", "mnist") if n in S]
    ends = []
    for name in order:
        lo = S[name]["leftover"]
        x = [5, 12, 20]
        y = [lo["5"] * 100, lo["12"] * 100, lo["20"] * 100]
        lw = 3.2 if name == "mnist" else 2
        ax.plot(x, y, "-o", color=COL[name], lw=lw, ms=5, label=LABEL[name])
        p = pieces(name)
        if p:
            ks_ = sorted(int(k) for k in p["rows"])
            ax.plot([20 + k for k in ks_], [p["rows"][str(k)]["leftover"] * 100 for k in ks_],
                    "--", color=COL[name], lw=lw)
            end = p["rows"][str(ks_[-1])]["leftover"] * 100
            ends.append((end, name))
    placed = []
    for end, name in sorted(ends):
        y = end if not placed else max(end, placed[-1] + 1.3)
        placed.append(y)
        ax.text(36.6, y, f"{end:.1f}%", color=COL[name], fontsize=8.5, va="center")
    pca = S["mnist"]["pca_leftover"]
    ax.plot([5, 12, 20], [pca["5"] * 100, pca["12"] * 100, pca["20"] * 100], ":",
            color="#111", lw=1.5, label="real MNIST 1s, plain flat compression (PCA)")
    ax.axvline(20, color="#ccc", lw=1, zorder=0)
    ax.set_xlim(3, 39); ax.set_ylim(0, 42)
    ax.set_xticks([5, 12, 20, 24, 28, 36])
    ax.set_xlabel("numbers per image")
    ax.set_ylabel("leftover: size of (real image - rebuilt image),\nas % of the real image's size")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.03, 1.0), fontsize=9, frameon=False)
    ax2 = fig.add_axes([0.76, 0.13, 0.21, 0.30])
    names = [n for n in order if pieces(n)]
    gone = [(1 - pieces(n)["rows"]["16"]["leftover"] / pieces(n)["rows"]["0"]["leftover"]) * 100
            for n in names]
    ax2.barh(range(len(names)), gone, color=[COL[n] for n in names])
    for i, g in enumerate(gone):
        ax2.text(g + 1, i, f"{g:.0f}%", va="center", fontsize=8.5)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels([SHORT[n].replace("\n", " ").replace(" (always a hook & bar)", "")
                         .replace("sharp edges", "sharp") for n in names], fontsize=8)
    ax2.invert_yaxis(); ax2.set_xlim(0, 100)
    ax2.set_title("share of the 20-number leftover\nremoved by 16 ink pieces", fontsize=9)
    fig.text(0.08, 0.025, "Held-out images: the template and pieces are learned on 3,000 images "
             "and scored on 1,500 others. Grey line at 20 marks where the pieces start.",
             fontsize=8.5, color="#555")
    fig.savefig(OUT / "leftover.png", dpi=130)
    plt.close(fig)


def knobs():
    names = [n for n in ("drawn12", "sharp12") if n in S and "knob_r2" in S[n]]
    fig = plt.figure(figsize=(max(11.5, 6.4 * len(names) + 1.6), 7.2))
    fig.text(0.5, 0.975, "Can the numbers be read back as the TRUE knobs?\n(1 = perfectly, 0 = no better than guessing)",
             ha="center", va="top", fontsize=12.5, weight="bold")
    fig.text(0.5, 0.895, "Each cell: share of a knob's variety, across 1,500 held-out images, that a smooth "
             "(quadratic) formula of the numbers predicts.", ha="center", va="top", fontsize=9.5, color="#444")
    cols = [("5 moment sums", "5 ink sums\n(no fitting)"), ("5 warp numbers", "5 warp"),
            ("12 warp numbers", "12 warp"), ("20 warp numbers", "20 warp"),
            ("PIECES", "20 warp\n+ 16 pieces"), ("20 PCA numbers", "20 PCA\n(flat)"),
            ("20 warp numbers, SHUFFLED", "20 warp,\nshuffled")]
    w = 0.84 / len(names)
    for j, name in enumerate(names):
        card = dict(S[name]["knob_r2"])
        p = pieces(name)
        if p:
            card["PIECES"] = p["rows"]["16"]["knob_r2"]
        cols_ = [c for c in cols if c[0] in card]
        M = np.array([[card[c][k] for c, _ in cols_] for k in card[cols_[0][0]]])
        ax = fig.add_axes([0.14 + j * w, 0.11, w - 0.04, 0.64])
        ax.imshow(np.clip(M, 0, 1), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for (r, c), v in np.ndenumerate(M):
            ax.text(c, r, "<0" if v < 0 else f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 else "black")
        ax.set_xticks(range(len(cols_)))
        ax.set_xticklabels([lab for _, lab in cols_], fontsize=8)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(M)))
        ax.set_yticklabels(KNOB_LABEL[:len(M)] if j == 0 else [], fontsize=9)
        ax.axhline(7.5, color="#d9661f", lw=1.5)
        ax.set_title(LABEL[name], fontsize=10, color=COL[name], weight="bold", pad=38)
    fig.text(0.14, 0.055, "Rows above the orange line bend or move the stroke; rows below add or "
             "reshape ink (bulge, hook, bar).", fontsize=8.5, color="#555")
    fig.text(0.14, 0.025, "'Shuffled' pairs each image's numbers with a different image: "
             "the no-information control.", fontsize=8.5, color="#555")
    fig.savefig(OUT / "knobs.png", dpi=130)
    plt.close(fig)


def shape():
    order = [n for n in ("drawn12", "wobbly_0.8", "sharp5", "sharp12", "mnist") if n in SHAPE]
    meas = [("share of the leftover lying on the\nstroke's centre line",
             lambda n: SHAPE[n]["across_share"][0] * 100, "%"),
            None,
            ("share of the ink in part-inked\n(grey) edge pixels (typical image)",
             lambda n: CROSS[n]["grey share"]["median"] * 100, "%")]
    fig = plt.figure(figsize=(13, 5.6))
    fig.text(0.5, 0.97, "Is the leftover the same KIND of thing in the drawn 1s and in MNIST?",
             ha="center", va="top", fontsize=13, weight="bold")
    fig.text(0.5, 0.925, "All measured after 20 warp numbers, on 1,500 held-out images, from the "
             "images alone. The black bar is real MNIST; the others are families whose knobs are known.",
             ha="center", va="top", fontsize=9.5, color="#444")
    for j, m in enumerate(meas):
        ax = fig.add_axes([0.06 + j * 0.325, 0.14, 0.27, 0.66])
        if m is None:
            b = [B90[n]["20"]["best90"] * 100 for n in order]
            w = [B90[n]["20"]["worst10"] * 100 for n in order]
            x = np.arange(len(order))
            ax.bar(x - 0.2, b, 0.4, color=[COL[n] for n in order], label="best-fitted 90%")
            ax.bar(x + 0.2, w, 0.4, color=[COL[n] for n in order], alpha=0.45, hatch="//",
                   edgecolor="white", label="worst-fitted 10%")
            for i in range(len(order)):
                ax.text(i - 0.2, b[i] + 0.6, f"{b[i]:.0f}", ha="center", fontsize=8)
                ax.text(i + 0.2, w[i] + 0.6, f"{w[i]:.0f}", ha="center", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels([SHORT[n] for n in order], fontsize=8.5)
            ax.set_ylim(0, max(w) * 1.25)
            ax.set_title("leftover (%) of the best-fitted 90% (solid)\nand worst-fitted 10% (hatched) of images",
                         fontsize=9.5)
            ax.spines[["top", "right"]].set_visible(False)
            continue
        title, f, unit = m
        v = [f(n) for n in order]
        ax.bar(range(len(order)), v, color=[COL[n] for n in order])
        for i, x in enumerate(v):
            ax.text(i, x + 1, f"{x:.0f}{unit}", ha="center", fontsize=9)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([SHORT[n] for n in order], fontsize=8.5)
        ax.set_ylim(0, max(v) * 1.25)
        ax.set_title(title, fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "shape.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    examples(); leftover(); knobs(); shape()
    print("wrote", [p.name for p in sorted(OUT.glob("*.png"))])
