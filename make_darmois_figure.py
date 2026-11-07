"""Figure for analyze_darmois_alternative_knobs.py."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from grey_ones import render
from analyze_darmois_alternative_knobs import (
    to_knobs, flow_fwd, flow_inv, wrap_fwd, dcor, decoder_simplicity, MIX)

EPS = 1.6
rng = np.random.default_rng(3)
a, b = MIX

# ---- data -------------------------------------------------------------
U3 = np.array([[0.25, 0.40, 0.55, 0.30, 0.20],
               [0.60, 0.55, 0.35, 0.70, 0.65],
               [0.85, 0.30, 0.70, 0.45, 0.90]])
Z3 = flow_fwd(U3, EPS)
img_true = render(to_knobs(U3))
img_alt = render(to_knobs(flow_inv(Z3, EPS)))
diff = img_alt.astype(np.float64) - img_true.astype(np.float64)

Ubig = rng.random((6000, 5))
worst = np.abs(render(to_knobs(Ubig)).astype(np.float64)
               - render(to_knobs(flow_inv(flow_fwd(Ubig, EPS), EPS))).astype(np.float64)).max()
DLIM = max(worst, 1e-12)

U = rng.random((4000, 5))
Z = flow_fwd(U, EPS)
s = slice(0, 1200)
dc_true, dc_new = dcor(U[s, a], U[s, b]), dcor(Z[s, a], Z[s, b])

imgs = render(to_knobs(U)).reshape(len(U), -1).astype(np.float64)
Zw = wrap_fwd(U)
degs = [1, 2, 3, 4, 5]
curves = {n: [decoder_simplicity(C, imgs, d, n_train=3000)[0] for d in degs]
          for n, C in [("the true knobs", U), ("flow knobs", Z), ("wrap knobs", Zw)]}
nterms = [decoder_simplicity(U, imgs, d, n_train=3000)[1] for d in degs]

# ---- figure -----------------------------------------------------------
fig = plt.figure(figsize=(16.5, 13.0))
fig.suptitle("A second set of five knobs that makes exactly the same 1s\n"
             "Hyvarinen & Pajunen's impossibility proof is a recipe -- here it is, run on grey_ones",
             fontsize=18, fontweight="bold", y=0.988)
fig.text(0.045, 0.906,
         "A.  One picture, two unrelated sets of five numbers.  The two generators are the same generator.",
         fontsize=13.5, fontweight="bold", va="bottom")

gs = GridSpec(2, 3, figure=fig, height_ratios=[1.30, 1.0],
              hspace=0.46, wspace=0.42,
              left=0.045, right=0.975, top=0.868, bottom=0.105)

gsa = gs[0, :].subgridspec(3, 5, width_ratios=[0.90, 1, 1, 1, 0.16],
                           height_ratios=[1, 1, 0.80], hspace=0.62, wspace=0.08)
rows = [("rendered from\nthe TRUE knobs\n(the baseline)", img_true, "gray_r", 1.0),
        ("rendered from\nthe NEW knobs,\nthrough their\nown decoder", img_alt, "gray_r", 1.0),
        ("difference\nbetween the two\nrows above", diff, "RdBu_r", DLIM)]
for r, (lab, arr, cmap, lim) in enumerate(rows):
    axl = fig.add_subplot(gsa[r, 0]); axl.axis("off")
    axl.text(0.97, 0.52, lab, ha="right", va="center", fontsize=11,
             fontweight="bold" if r < 2 else "normal", transform=axl.transAxes)
    for c in range(3):
        ax = fig.add_subplot(gsa[r, c + 1])
        kw = dict(cmap=cmap, vmin=0, vmax=1) if r < 2 else dict(cmap=cmap, vmin=-lim, vmax=lim)
        im = ax.imshow(arr[c], **kw)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("cx %.2f   cy %.2f   height %.2f\nwidth %.2f   lean %.1f deg"
                         % tuple(to_knobs(U3)[c]), fontsize=9.5, pad=5)
        elif r == 1:
            ax.set_xlabel("its five new numbers\n%.3f  %.3f  %.3f  %.3f  %.3f"
                          % tuple(Z3[c]), fontsize=9, labelpad=4, color="#b03a00")
        else:
            ax.set_xlabel("largest pixel gap  %.1e" % np.abs(diff[c]).max(),
                          fontsize=9, labelpad=4)
axcb = fig.add_subplot(gsa[2, 4])
cb = fig.colorbar(im, cax=axcb)
cb.set_label("ink difference\n(red: new render darker)", fontsize=8)
cb.ax.tick_params(labelsize=7)
fig.text(0.045, 0.597,
         "Colour scale of the bottom row is +/- %.0e, the worst single-pixel gap over 6000 random 1s. "
         "Flat white means the two renders are bit-for-bit equal in float32." % DLIM,
         fontsize=10, color="#333", va="bottom", style="italic")

# --- B: independence
axb = fig.add_subplot(gs[1, 0])
axb.scatter(Z[:1500, a], Z[:1500, b], s=4, alpha=0.38, color="#b03a00", linewidths=0)
axb.set_xlim(0, 1); axb.set_ylim(0, 1); axb.set_aspect("equal")
axb.set_xlabel("new coordinate 1   (0 = low end of its range, 1 = high)", fontsize=10)
axb.set_ylabel("new coordinate 5", fontsize=10)
axb.set_title("B.  No statistical test prefers the true knobs\n"
              "both pairs fill the square evenly",
              fontsize=12, fontweight="bold", pad=10)
axi = axb.inset_axes([0.635, 0.055, 0.34, 0.34])
axi.scatter(U[:1500, a], U[:1500, b], s=2, alpha=0.3, color="#444", linewidths=0)
axi.set_xticks([]); axi.set_yticks([]); axi.set_xlim(0, 1); axi.set_ylim(0, 1)
axi.set_xlabel("baseline:\ntrue cx vs lean", fontsize=8, labelpad=2)
axb.text(0.035, 0.965,
         "dependence within the pair\n(distance correlation, 0 = independent)\n"
         "    true knobs   %.3f\n    new knobs    %.3f\n"
         "both sit at the n=1200 sampling floor" % (dc_true, dc_new),
         fontsize=9, va="top", transform=axb.transAxes,
         bbox=dict(fc="white", ec="#999", alpha=0.94, pad=4))

# --- C: but it is not cx
axc = fig.add_subplot(gs[1, 1])
sc = axc.scatter(U[:2000, a], Z[:2000, a], c=to_knobs(U[:2000])[:, b],
                 s=5, cmap="viridis", linewidths=0)
cb2 = fig.colorbar(sc, ax=axc, fraction=0.046, pad=0.02)
cb2.set_label("true lean (degrees)", fontsize=9)
cb2.ax.tick_params(labelsize=8)
axc.set_xlabel("true cx, as a fraction of its range", fontsize=10)
axc.set_ylabel("new coordinate 1", fontsize=10)
axc.set_title("C.  ...but it is not cx in disguise\n"
              "a renamed cx would lie on the dashed line",
              fontsize=12, fontweight="bold", pad=10)
axc.plot([0, 1], [0, 1], "k--", lw=1.2)
axc.set_xlim(-0.03, 1.03); axc.set_ylim(-0.08, 1.12)
axc.text(0.03, 0.965, "straight-line fit  r = %+.2f\nit mixes cx with lean"
         % np.corrcoef(U[:, a], Z[:, a])[0, 1], fontsize=9, ha="left", va="top",
         transform=axc.transAxes, bbox=dict(fc="white", ec="#999", alpha=0.94, pad=4))

# --- D: what does separate them
axd = fig.add_subplot(gs[1, 2])
styles = {"the true knobs": ("#1a5fb4", "o", 2.6), "flow knobs": ("#b03a00", "s", 2.0),
          "wrap knobs": ("#777", "^", 2.0)}
for name, y in curves.items():
    col, mk, lw = styles[name]
    axd.plot(degs, y, marker=mk, color=col, lw=lw, ms=7, label=name)
axd.set_xticks(degs)
axd.set_xticklabels(["%d\n(%d terms)" % (d, n) for d, n in zip(degs, nterms)], fontsize=9)
axd.set_xlabel("degree of the polynomial decoder tried", fontsize=10)
axd.set_ylabel("held-out error, % of image size", fontsize=10)
axd.set_title("D.  What DOES separate them: simplicity\n"
              "true knobs need the fewest terms",
              fontsize=12, fontweight="bold", pad=10)
axd.legend(fontsize=9.5, loc="lower left", framealpha=0.95)
axd.grid(alpha=0.3)
axd.set_ylim(-2, 66)
axd.text(0.97, 0.30, "at degree 5 the new knobs are\n2.3x and 3.7x worse.\n"
         "Independence cannot choose;\nsmoothness can.",
         fontsize=9.5, ha="right", va="bottom", transform=axd.transAxes,
         bbox=dict(fc="#fff8e1", ec="#c8a000", pad=5))

fig.text(0.045, 0.012,
         "Generated 1s only (grey_ones, 28x28 coverage, values 0-1); knobs drawn uniform and independent on the default range box.  "
         "'Flow knobs' = the time-1 map of a divergence-free flow (strength 1.6)\n"
         "mixing normalized cx and lean; it preserves volume, so it carries independent uniforms to independent uniforms.  'Wrap knobs' = an exactly\n"
         "volume-preserving pair of cyclic shears (discontinuous at the wrap).  Panels B-D use 4000 fresh 1s; panel D holds out 1000 of them.",
         fontsize=9, color="#444", va="bottom")

out = "figures/darmois_alternative_knobs/darmois_alternative_knobs.png"
fig.savefig(out, dpi=112)
print("wrote", out, " worst pixel gap over 6000:", worst)
