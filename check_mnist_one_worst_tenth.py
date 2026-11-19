"""What sets MNIST's worst-fitted tenth apart, in measurements only (no pictures).

Uses the cached 20-number MNIST fit from analyze_drawn_one_warp_scorecard.py.
For each held-out 1: its leftover share, plus measurements taken from the image
alone -- total ink, moment height / width / slant, how much of its ink lies off
a straight stroke (outside +-1.5 half-widths of the centre line, or beyond the
ends), and its overlap with the straight stroke its own moments imply.
"""
import json
import numpy as np
import warp_charts as wc
from analyze_drawn_one_warp_scorecard import OUT, N_TRAIN, N_TEST, load, stroke_frame
from analyze_mnist_one_knob_search import stroke_overlap

te = np.arange(N_TRAIN, N_TRAIN + N_TEST)
I = load("mnist")[0][te]
D = np.load(OUT / "fit_mnist.npz")["dec20"]
per = np.linalg.norm((I - D).reshape(len(I), -1), axis=1) / np.linalg.norm(I.reshape(len(I), -1), axis=1)
cx, cy, half, slant, width = stroke_frame(I)
X, Y = wc.PIX_X[None], wc.PIX_Y[None]
t = (Y - cy[:, None, None]) / half[:, None, None]
lat = ((X - cx[:, None, None]) - slant[:, None, None] * (Y - cy[:, None, None])) / (width[:, None, None] / 2)
ink = I.reshape(len(I), -1).sum(1)
off = (I * ((np.abs(lat) > 1.5) | (np.abs(t) > 1))).reshape(len(I), -1).sum(1) / ink
meas = {"leftover": per, "total ink": ink, "height (px)": 2 * half, "width (px)": width,
        "slant (deg)": np.degrees(np.arctan(np.abs(slant))),
        "share of ink off a straight stroke": off,
        "overlap with own straight stroke": stroke_overlap(I)}
o = np.argsort(per)
worst, rest = o[-len(o) // 10:], o[:-len(o) // 10]
out = {}
for k, v in meas.items():
    out[k] = {"worst tenth median": float(np.median(v[worst])), "rest median": float(np.median(v[rest])),
              "corr with leftover": float(np.corrcoef(v, per)[0, 1])}
    print(f"{k:>36}: worst tenth {out[k]['worst tenth median']:8.3f}   rest {out[k]['rest median']:8.3f}"
          f"   corr with leftover {out[k]['corr with leftover']:+.2f}")
(OUT / "mnist_worst_tenth.json").write_text(json.dumps(out, indent=2))

# Row by row: a thick stroke is wide in EVERY row; a hook or serif is wide in a
# few rows; a curved stroke has row centres that leave a straight line.
def rows(img):
    w, c = [], []
    for r in img:
        if r.sum() > 0.5:
            x = np.arange(28) + 0.5
            w.append(r.sum())                          # ink width of that row, px
            c.append((r * x).sum() / r.sum())
    w, c = np.array(w), np.array(c)
    y = np.arange(len(c))
    resid = c - np.polyval(np.polyfit(y, c, 1), y)
    return np.median(w), np.mean(w > 1.8 * np.median(w)), np.sqrt(np.mean(resid ** 2))

R = np.array([rows(i) for i in I])
row = {"median row ink width (px)": R[:, 0], "share of rows over 1.8x the median width": R[:, 1],
       "row centres off a straight line (px rms)": R[:, 2]}
for k, v in row.items():
    out[k] = {"worst tenth median": float(np.median(v[worst])), "rest median": float(np.median(v[rest])),
              "corr with leftover": float(np.corrcoef(v, per)[0, 1])}
    print(f"{k:>42}: worst tenth {out[k]['worst tenth median']:7.3f}   rest {out[k]['rest median']:7.3f}"
          f"   corr {out[k]['corr with leftover']:+.2f}")
# how many of the worst tenth are mainly thick, mainly kinked/hooked, mainly curved
thick = R[:, 0] > np.percentile(R[rest, 0], 90)
wide_rows = R[:, 1] > np.percentile(R[rest, 1], 90)
curved = R[:, 2] > np.percentile(R[rest, 2], 90)
for k, v in (("thick stroke", thick), ("a few much wider rows", wide_rows), ("curved centre line", curved)):
    out[f"worst tenth flagged: {k}"] = float(v[worst].mean())
    print(f"  worst tenth beyond the rest's 90th percentile for {k}: {v[worst].mean() * 100:.0f}%  (10% by chance)")
none = ~(thick | wide_rows | curved)
out["worst tenth flagged by none"] = float(none[worst].mean())
print(f"  worst tenth flagged by none of the three: {none[worst].mean() * 100:.0f}%")
(OUT / "mnist_worst_tenth.json").write_text(json.dumps(out, indent=2))
