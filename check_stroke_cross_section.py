"""How each image's ink is spread ACROSS its stroke, measured from the image alone.

grey share  -- share of the ink sitting in part-inked pixels (0.05-0.95), i.e.
               how blurred the edges are
peak        -- the 95th percentile of the inked pixels' values, i.e. how dark
               the stroke's core is
For each family: the spread across images (10th-90th percentile), and the
correlation with each image's leftover after the 20-number warp fit.
"""
import json
import numpy as np
from analyze_drawn_one_warp_scorecard import NAMES, OUT, N_TRAIN, N_TEST, load

te = np.arange(N_TRAIN, N_TRAIN + N_TEST)
res = {}
for name in NAMES:
    I = load(name)[0][te]
    D = np.load(OUT / f"fit_{name}.npz")["dec20"]
    F = I.reshape(len(I), -1)
    per = np.linalg.norm(F - D.reshape(len(I), -1), axis=1) / np.linalg.norm(F, axis=1)
    grey = (F * ((F > 0.05) & (F < 0.95))).sum(1) / F.sum(1)
    peak = np.array([np.percentile(f[f > 0.05], 95) for f in F])
    res[name] = {}
    line = f"{name:>11}:"
    for k, v in (("grey share", grey), ("peak", peak)):
        p10, p50, p90 = np.percentile(v, [10, 50, 90])
        c = float(np.corrcoef(v, per)[0, 1])
        res[name][k] = {"p10": float(p10), "median": float(p50), "p90": float(p90), "corr_with_leftover": c}
        line += f"  {k} {p10:.2f}/{p50:.2f}/{p90:.2f} (corr {c:+.2f})"
    print(line)
(OUT / "cross_section.json").write_text(json.dumps(res, indent=2))
