"""Leftover of each family's best-fitted 90% vs its worst-fitted 10%.

Same cut for every family, so the comparison is fair (cutting the worst tenth
always helps; the question is whether MNIST's best 90% then looks like a drawn
family). Pooled leftover = size of all leftovers together / size of all images.
"""
import json
import numpy as np
from analyze_drawn_one_warp_scorecard import NAMES, OUT, N_TRAIN, N_TEST, load

te = np.arange(N_TRAIN, N_TRAIN + N_TEST)
res = {}
for name in NAMES:
    F = load(name)[0][te].reshape(len(te), -1)
    Z = np.load(OUT / f"fit_{name}.npz")
    res[name] = {}
    line = f"{name:>11}:"
    for k in (5, 12, 20):
        R = F - Z[f"dec{k}"].reshape(len(te), -1)
        e, s = (R ** 2).sum(1), (F ** 2).sum(1)
        o = np.argsort(e / s)
        best, worst = o[: int(0.9 * len(o))], o[int(0.9 * len(o)):]
        b, w = np.sqrt(e[best].sum() / s[best].sum()), np.sqrt(e[worst].sum() / s[worst].sum())
        res[name][k] = {"best90": float(b), "worst10": float(w)}
        line += f"   {k:>2}: best 90% {b * 100:5.1f}%  worst 10% {w * 100:5.1f}%"
    print(line)
(OUT / "best_ninety.json").write_text(json.dumps(res, indent=2))
