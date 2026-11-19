"""Where across and along the stroke does each family's leftover sit?

Uses the cached 20-number fits of analyze_drawn_one_warp_scorecard.py. Measured
in each image's own stroke frame (from its moments): distance from the centre
line in units of half the stroke width, and position along the stroke. Also:
how much of the total leftover the worst tenth of images carry, and how often
the leftover flips sign as you walk down the stroke (a wobble flips often, a
wrong overall length or darkness does not).
"""
import json
import numpy as np
import warp_charts as wc
from analyze_drawn_one_warp_scorecard import NAMES, OUT, N_TRAIN, N_TEST, load, stroke_frame

res = {}
te = np.arange(N_TRAIN, N_TRAIN + N_TEST)
for name in [n for n in NAMES if (OUT / f"fit_{n}.npz").exists()]:
    Z = np.load(OUT / f"fit_{name}.npz")
    I, _ = load(name)
    I = I[te]
    R = I - Z["dec20"]
    cx, cy, half, slant, width = stroke_frame(I)
    X, Y = wc.PIX_X[None], wc.PIX_Y[None]
    t = (Y - cy[:, None, None]) / half[:, None, None]
    lat = ((X - cx[:, None, None]) - slant[:, None, None] * (Y - cy[:, None, None])) \
        / (width[:, None, None] / 2)
    e = R ** 2
    bins = [0, 0.5, 1.0, 1.5, 2.0, 99]
    a = np.abs(lat)
    across = [float(e[(a >= lo) & (a < hi)].sum() / e.sum()) for lo, hi in zip(bins[:-1], bins[1:])]
    per = e.reshape(len(e), -1).sum(1)
    worst = float(np.sort(per)[::-1][:len(per) // 10].sum() / per.sum())
    # signed leftover summed across each pixel row, inside the stroke span only
    flips = []
    for r, tt in zip(R, t):
        rows = np.where((tt[:, 14] > -0.9) & (tt[:, 14] < 0.9))[0]
        s = r[rows].sum(1)
        big = s[np.abs(s) > 0.05 * np.abs(s).max()]
        flips.append(np.mean(np.diff(np.sign(big)) != 0) if len(big) > 2 else 0.0)
    # left-minus-right: sideways shift (opposite signs) vs thickness (same sign)
    L = (R * ((lat < -0.5) & (lat > -2))).reshape(len(R), 28, 28).sum(2)
    Rr = (R * ((lat > 0.5) & (lat < 2))).reshape(len(R), 28, 28).sum(2)
    side = float(np.abs(L - Rr).sum() / (np.abs(L - Rr).sum() + np.abs(L + Rr).sum()))
    res[name] = {"across_bins_halfwidths": bins, "across_share": across,
                 "worst_tenth_share": worst, "row_sign_flip_rate": float(np.mean(flips)),
                 "sideways_share_of_edge_leftover": side}
    print(f"{name:>8}: across |offset| in half-widths 0-.5/.5-1/1-1.5/1.5-2/2+ = "
          + " ".join(f"{v * 100:4.0f}%" for v in across)
          + f" | worst tenth carry {worst * 100:.0f}% | row sign flips {np.mean(flips):.2f}"
          + f" | edge leftover that is sideways shift {side * 100:.0f}%")
(OUT / "leftover_shape.json").write_text(json.dumps(res, indent=2))
