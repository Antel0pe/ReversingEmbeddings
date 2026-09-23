"""Lean an image with an operator that never recovers the generator's knobs.

Input: a 28x28 image and a lean change. Output: a 28x28 image. Nothing else passes
between them -- no cx, cy, height, width or lean. See lean_operator.py.

Bottom two rows are real MNIST 1s, which the generator cannot draw at all. The
operator leans them anyway, which is the point: it works off the image's own edges.

Run: python make_lean_operator_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import grey_ones as G, mnist
from lean_operator import lean_step, pivot_row

OUT = "figures"
INK, GOOD, BAD, REF = "#1a1a1a", "#1a7f37", "#c62828", "#1565c0"
DEG = [-20, -10, 0, 10, 20, 30]
EPS8 = 0.11


def main():
    Xa, ya = mnist.load("train")
    ONES = (Xa[ya == 1].reshape(-1, 784) / 255.0).astype(np.float64)
    rng = np.random.default_rng(5)
    import stroke_rules as SR
    idx = rng.choice(len(ONES), 400, replace=False)
    plain = [j for j in idx if SR.is_stroke(ONES[j].reshape(28, 28) >= .5)][0]
    fancy = [j for j in idx if not SR.is_stroke(ONES[j].reshape(28, 28) >= .5)][3]

    seeds = [
        ("generator\nplain upright", np.array([14.5, 14.5, 19.75, 3.2, 0.0]), None),
        ("generator\nnarrow, off-centre", np.array([12.8, 15.2, 20.5, 1.9, 0.0]), None),
        ("generator\nwide, short", np.array([16.0, 13.8, 15.0, 4.6, 0.0]), None),
        ("generator\nalready leaning 20°", np.array([14.5, 14.5, 19.75, 3.2, 20.0]), None),
        ("real MNIST 1\n(plain)", None, ONES[plain].reshape(28, 28)),
        ("real MNIST 1\n(with a foot)", None, ONES[fancy].reshape(28, 28)),
    ]

    fig = plt.figure(figsize=(12.6, 12.0))
    fig.text(.5, .972, "Leaning an image with an operator that never sees the knobs",
             ha="center", va="center", fontsize=15, fontweight="bold")
    fig.text(.5, .938, "Input a 28×28 image and a lean change; get a 28×28 image back. The operator slides row i sideways by "
                       "−(i − pivot)·Δu, finding each row's edge\nfrom that row's own pixels. Numbers are the error against what the "
                       "generator draws (green under 0.11 = invisible). The last two rows are real MNIST 1s,\nwhich the generator cannot draw — "
                       "so there is nothing to compare against, and the operator leans them anyway.",
             ha="center", va="center", fontsize=9.2, color="#555", linespacing=1.55)
    L, T, B = .165, .900, .055
    rh = (T - B) / len(seeds)
    for r, (lab, knobs, img) in enumerate(seeds):
        base = G.render(knobs)[0].astype(np.float64) if img is None else img
        lean0 = knobs[4] if knobs is not None else None
        y0 = T - (r + 1) * rh
        for c, dg in enumerate(DEG):
            if lean0 is not None:
                du = np.tan(np.radians(lean0 + dg)) - np.tan(np.radians(lean0))
            else:
                du = np.tan(np.radians(dg))
            out = lean_step(base, du, "interval")
            ax = fig.add_axes([L + c * .132, y0 + rh * .26, .105, rh * .62])
            ax.imshow(out, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            em = dg == 0
            for sp in ax.spines.values():
                sp.set_edgecolor(REF if em else "#bbb"); sp.set_linewidth(2 if em else .7)
            if r == 0:
                ax.text(.5, 1.06, f"{dg:+d}°" if dg else "start", transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=10.5, fontweight="bold")
            if knobs is not None:
                q = knobs.copy(); q[4] = lean0 + dg
                e = np.linalg.norm(out - G.render(q)[0].astype(np.float64))
                ax.text(.5, -.07, f"{e:.3f}", transform=ax.transAxes, ha="center", va="top",
                        fontsize=8.6, fontweight="bold", color=GOOD if e < EPS8 else BAD)
            else:
                ax.text(.5, -.07, "—", transform=ax.transAxes, ha="center", va="top",
                        fontsize=8.6, color="#999")
        fig.text(L - .014, y0 + rh * .57, lab, ha="right", va="center", fontsize=9.6,
                 fontweight="bold", linespacing=1.4,
                 color=INK if knobs is not None else GOOD)
    fig.text(.5, .030, "Row 4 starts already leaning, and its errors are the largest: once a stroke leans, the generator draws its edges slightly soft, "
                       "and this operator\ntreats every edge as sharp. It slides the edge to the right place and redraws it crisp. That is the whole of the remaining error.",
             ha="center", va="center", fontsize=9.1, color="#333", style="italic", linespacing=1.6)
    fig.savefig(f"{OUT}/lean_operator.png", dpi=125)
    plt.close(fig)


if __name__ == "__main__":
    main()
