"""One figure for analyze_mnist_one_autoencoder.py: three questions, three panels.

  A  How many numbers does the decoder need?  held-out error vs k, MNIST and
     generated 1s, with the best flat-plane (PCA) decoder for comparison.
  B  What does each of the five numbers do?  decode the typical code, then slide
     along each of the five main directions of the code cloud (MNIST, k=5).
  C  What sits halfway between two real 1s?  pixel average vs decoded midpoint.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_mnist_one_autoencoder as A

OUT = A.OUT
R = {**json.loads((OUT / "results_mnist.json").read_text())}
G = json.loads((OUT / "results_generated.json").read_text())
D = A.datasets()


def curve(res, name):
    ks = sorted(int(k) for k in res["sweep"][name]["ae"])
    ae = [np.mean([r["heldout"] for r in res["sweep"][name]["ae"][str(k)]]) * 100 for k in ks]
    pk = list(range(1, 31))
    pca = [res["sweep"][name]["pca"][str(k)] * 100 for k in pk]
    return ks, ae, pk, pca


def tile(images, gap=2):
    """Lay out a 2-D list of 28x28 images on a white sheet."""
    r, c = len(images), len(images[0])
    sheet = np.ones((r * (28 + gap) - gap, c * (28 + gap) - gap))
    for i, row in enumerate(images):
        for j, im in enumerate(row):
            sheet[i * 30:i * 30 + 28, j * 30:j * 30 + 28] = 1 - im.reshape(28, 28)
    return sheet


NAMES = {"slant": "slant", "width": "width", "ink": "ink amount", "centre x": "left-right",
         "centre y": "up-down", "height": "height", "bend": "bend",
         "slant2": "slant²"}

fig = plt.figure(figsize=(19.5, 6.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 0.95], wspace=0.42)

# ---- A: error curve
ax = fig.add_subplot(gs[0])
for res, name, col, lab in ((R, "mnist", "#c0392b", "real MNIST 1s"),
                            (G, "generated", "#2471a3", "generated 1s (truly 5 knobs)")):
    ks, ae, pk, pca = curve(res, name)
    ax.plot(ks, ae, "o-", color=col, lw=2.5, ms=6, label=f"{lab}: learned decoder")
    ax.plot(pk, pca, "--", color=col, lw=1.5, alpha=0.7, label=f"{lab}: best flat plane")
ax.axvline(5, color="grey", lw=1, ls=":")
ax.text(5.3, 2, "5 numbers", color="grey", fontsize=10)
ax.set_xscale("log"); ax.set_xticks([1, 2, 3, 5, 8, 12, 20, 30]); ax.set_xticklabels([1, 2, 3, 5, 8, 12, 20, 30])
ax.set_ylim(0, 75)
ax.set_xlabel("how many numbers the decoder gets", fontsize=12)
ax.set_ylabel("what's lost on unseen 1s (% of the image)", fontsize=12)
ax.set_title("A. How many numbers does it need?", fontsize=14, loc="left")
ax.legend(fontsize=9.5, loc="upper right")
ax.grid(alpha=0.3)

# ---- B: five directions, MNIST k=5
Xtr, Xte, _, _ = D["mnist"]
m = A.get_model("mnist", Xtr, 5, 0)
Z = A.encode(m, Xtr)
mu, V, S = A.code_directions(Z)
qs = [5, 25, 50, 75, 95]
rows, labels = [], []
MEAS = A.measurements(Xtr)
MEAS["slant2"] = (MEAS["slant"] - MEAS["slant"].mean()) ** 2
for d in range(5):
    vals = np.percentile(S[:, d], qs)
    codes = mu + np.outer(vals, V[d])
    rows.append(list(A.decode(m, codes)))
    corr = {q: float(np.corrcoef(S[:, d], v)[0, 1]) for q, v in MEAS.items()}
    top = sorted(corr, key=lambda q: -abs(corr[q]))[:2]
    labels.append(f"#{d + 1}  " + "\n".join(f"{NAMES[q]} {abs(corr[q]):.2f}" for q in top))
ax = fig.add_subplot(gs[1])
ax.imshow(tile(rows), cmap="gray", vmin=0, vmax=1)
ax.set_xticks([14 + 30 * j for j in range(5)])
ax.set_xticklabels(["low\n(5%)", "", "typical", "", "high\n(95%)"], fontsize=10)
ax.set_yticks([14 + 30 * i for i in range(5)])
ax.yaxis.tick_right()
ax.set_yticklabels(labels, fontsize=10)
ax.text(1.02, 1.01, "tracks (|correlation|):", transform=ax.transAxes, fontsize=10, color="dimgrey")
ax.set_title("B. Real 1s, 5 numbers: slide one at a time", fontsize=14, loc="left")
ax.set_xlabel("other four held at typical values", fontsize=10)

# ---- C: midpoints
rng = np.random.default_rng(11)
mz = A.get_model("mnist", Xtr, 12, 0)
pairs = []
Zte = A.encode(mz, Xte)
slant = A.measurements(Xte)["slant"]
lo, hi = np.argsort(slant)[:200], np.argsort(slant)[-200:]
for _ in range(3):
    a, b = rng.choice(lo), rng.choice(hi)
    pairs.append([Xte[a], 0.5 * (Xte[a] + Xte[b]), A.decode(mz, 0.5 * (Zte[a] + Zte[b])[None])[0], Xte[b]])
ax = fig.add_subplot(gs[2])
ax.imshow(tile(pairs), cmap="gray", vmin=0, vmax=1)
ax.set_xticks([14 + 30 * j for j in range(4)])
ax.set_xticklabels(["real 1\n(A)", "average\nof pixels", "decoder's\nhalfway", "real 1\n(B)"], fontsize=10)
ax.set_yticks([])
gm = R["gaps"]["mnist"]["rows"]
ax.set_title("C. What's halfway between two real 1s?", fontsize=14, loc="left")
ax.set_xlabel(f"grey-ink share: real 1s {gm['real held-out 1s']['grey_share']:.2f}, "
              f"pixel average {gm['pixel average of two real 1s']['grey_share']:.2f}, "
              f"decoder halfway {gm['k=12 midpoint of two codes']['grey_share']:.2f}\n"
              "(decoder: 12 numbers; pairs: one upright, one strongly slanted)", fontsize=10)

fig.suptitle("A small neural network that turns a few numbers into an image of a 1 "
             "(trained on 5,742 MNIST 1s; scored on 1,000 it never saw)", fontsize=14, y=0.995)
fig.savefig(OUT / "autoencoder_summary.png", dpi=120, bbox_inches="tight")
print("saved", OUT / "autoencoder_summary.png")
