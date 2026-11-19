"""Row-to-row roughness of a stroke's edges, measured from the images alone.

For each pixel row inside the stroke (skipping two rows at each end): the ink
centre (ink-weighted x) and the ink width (row sum, px). Fit each along the
stroke with a cubic -- the smooth part a cubic warp can follow -- and keep what
is left. Report its typical size (robust: 1.4826 x median absolute deviation)
and how much neighbouring rows agree (lag-1 correlation).
"""
import json
import numpy as np
from analyze_drawn_one_warp_scorecard import NAMES, OUT, N_TRAIN, N_TEST, load


def roughness(I):
    dc, dw = [], []
    for img in I:
        s = img.sum(1)
        rows = np.where(s > 0.5)[0]
        if len(rows) < 10:
            continue
        rows = rows[2:-2]
        x = np.arange(28) + 0.5
        c = (img[rows] * x).sum(1) / s[rows]
        w = s[rows]
        for v, acc in ((c, dc), (w, dw)):
            r = v - np.polyval(np.polyfit(rows, v, 3), rows)
            acc.append(r)
    out = {}
    for key, acc in (("centre", dc), ("width", dw)):
        allr = np.concatenate(acc)
        mad = 1.4826 * np.median(np.abs(allr - np.median(allr)))
        lag = np.concatenate([a[:-1] for a in acc]), np.concatenate([a[1:] for a in acc])
        out[key] = {"size_px": float(mad), "lag1": float(np.corrcoef(*lag)[0, 1])}
    return out


if __name__ == "__main__":
    res = {}
    for name in NAMES:
        I = load(name)[0][:N_TRAIN]
        res[name] = r = roughness(I)
        print(f"{name:>11}: centre roughness {r['centre']['size_px']:.3f} px (neighbour rows agree "
              f"{r['centre']['lag1']:+.2f}), width roughness {r['width']['size_px']:.3f} px "
              f"(agree {r['width']['lag1']:+.2f})")
    (OUT / "edge_roughness.json").write_text(json.dumps(res, indent=2))
