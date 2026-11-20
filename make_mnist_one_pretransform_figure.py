"""Figure for analyze_mnist_one_pretransform.py: three panels.

  A  what each reshaping does to three real 1s
  B  real 1s: error on unseen 1s vs TOTAL numbers (learned + set aside), per reshaping
  C  generated 1s (truly five knobs): no reshaping vs straighten-and-centre
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

import analyze_mnist_one_autoencoder as A
import pre_transforms as T

PT = A.OUT / "pretransform"
D = A.datasets()


def raw_curve(name):
    r = json.loads((A.OUT / f"results_{name}.json").read_text())["sweep"][name]["ae"]
    ks = sorted(int(k) for k in r)
    return ks, [100 * np.mean([x["heldout"] for x in r[str(k)]]) for k in ks]


def curve(tname, dname):
    r = json.loads((PT / f"{tname}_{dname}.json").read_text())
    ks = sorted(int(k) for k in r["k"])
    return ([k + r["extra_numbers"] for k in ks],
            [100 * r["k"][str(k)]["vs_original"] for k in ks], 100 * r["floor"])


STYLE = {"none": ("black", "no reshaping"),
         "pose": ("#c0392b", "straighten + centre + resize (5 set aside)"),
         "deslant": ("#e67e22", "straighten + centre left-right only (2 set aside)"),
         "sdf": ("#2471a3", "distance-to-edge map"),
         "blur": ("#7d3c98", "blur (1 px)"),
         "upsample": ("grey", "check: no reshaping, but the same finer 56x56 grid")}

fig = plt.figure(figsize=(19, 6.8))
gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.15, 0.85], wspace=0.3)

# ---- A: what the reshapings look like
Xte = D["mnist"][1]
slant = T.moments(Xte)[:, 2]
pick = [int(np.argsort(slant)[len(slant) // 2]), int(np.argsort(slant)[-40]), int(np.argsort(slant)[30])]
X3 = Xte[pick]
pose, _ = T.pose_forward(X3)
des, _ = T.deslant_forward(X3)
sdf, _ = T.sdf_forward(X3)
blur, _ = T.blur_forward(X3)
small = lambda Y: np.stack([ndimage.zoom(y.reshape(56, 56), 0.5, order=1).ravel() for y in Y])
cols = [X3, small(des), small(pose), sdf, blur]
sheet = np.ones((3 * 30 - 2, 5 * 30 - 2))
for i in range(3):
    for j, C in enumerate(cols):
        sheet[i * 30:i * 30 + 28, j * 30:j * 30 + 28] = 1 - C[i].reshape(28, 28)
ax = fig.add_subplot(gs[0])
ax.imshow(sheet, cmap="gray", vmin=0, vmax=1)
ax.set_xticks([14 + 30 * j for j in range(5)])
ax.set_xticklabels(["original", "straight-\nened,\ncentred", "straight-\nened,\ncentred,\nresized",
                    "distance\nto edge\n(darker =\nfurther in)", "blurred"], fontsize=9.5)
ax.set_yticks([])
ax.set_title("A. Three real 1s, reshaped four ways", fontsize=14, loc="left")

# ---- B: real 1s
ax = fig.add_subplot(gs[1])
k, e = raw_curve("mnist")
ax.plot(k, e, "o-", color="black", lw=3, ms=6, label=STYLE["none"][1], zorder=5)
for t in ("pose", "deslant", "upsample", "sdf", "blur"):
    try:
        kk, ee, fl = curve(t, "mnist")
    except FileNotFoundError:
        continue
    col, lab = STYLE[t]
    ax.plot(kk, ee, "s--" if t == "upsample" else "o-", color=col, lw=2, ms=5, label=lab)
    if t in ("sdf", "blur"):
        ax.axhline(fl, color=col, lw=1, ls=":")
        ax.text(1.03, fl + 0.5, f"{STYLE[t][1]}: reshaping alone\nloses {fl:.0f}%",
                color=col, fontsize=9, va="bottom", ha="left", zorder=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
ax.set_xscale("log"); ax.set_xticks([1, 2, 3, 5, 8, 12, 20, 30]); ax.set_xticklabels([1, 2, 3, 5, 8, 12, 20, 30])
ax.set_xlim(0.9, 30); ax.set_ylim(0, 52)
ax.set_xlabel("TOTAL numbers per image (learned + set aside)", fontsize=12)
ax.set_ylabel("what's lost on unseen real 1s\n(% of the ORIGINAL image)", fontsize=12)
ax.set_title("B. Real 1s: does reshaping first help?", fontsize=14, loc="left")
ax.legend(fontsize=9.5, loc="lower left")
ax.grid(alpha=0.3)

# ---- C: generated control
ax = fig.add_subplot(gs[2])
k, e = raw_curve("generated")
ax.plot(k, e, "o-", color="black", lw=3, ms=6, label=STYLE["none"][1])
try:
    kk, ee, fl = curve("pose", "generated")
    ax.plot(kk, ee, "o-", color="#c0392b", lw=2, ms=5, label="straighten + centre + resize\n(5 set aside)")
except FileNotFoundError:
    pass
ax.axvline(5, color="grey", lw=1, ls=":")
ax.set_xscale("log"); ax.set_xticks([1, 2, 3, 5, 8, 12, 20, 30]); ax.set_xticklabels([1, 2, 3, 5, 8, 12, 20, 30])
ax.set_xlim(0.9, 30); ax.set_ylim(0, 40)
ax.set_xlabel("TOTAL numbers per image", fontsize=12)
ax.set_title("C. Check: generated 1s (truly 5 knobs)", fontsize=14, loc="left")
ax.legend(fontsize=9.5, loc="upper right")
ax.grid(alpha=0.3)

fig.suptitle("Reshape the 1s first, learn the rest, undo the reshaping, score against the original image "
             "(same small network; 1,000 unseen 1s)", fontsize=14, y=1.0)
fig.savefig(PT / "pretransform_summary.png", dpi=120, bbox_inches="tight")
print("saved")
