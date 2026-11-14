"""Is the pulled flattening of the 1s one-to-one? The suite's double-layer check, no truth used.

For each saved layout of analyze_pull_ones.py (no pulling / walking / pulled), keep its first m
coordinates and ask: does the flat result put any image much closer to another than they really
are in pixel space (unfolding_suite/verify.py)? Also: how well do short distances survive
(flat length / pixel length of each image's 10 closest pairs, after one overall resize)?

Run: .venv/bin/python check_pull_ones_one_to_one.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import analyze_pull_ones as A
from unfolding_suite.verify import verify

OUT = Path(__file__).resolve().parent / "figures" / "pull_ones"


def short_lengths(P, F, k=10):
    _, near = cKDTree(P).query(P, k=k + 1)
    a, b = np.repeat(np.arange(len(P)), k), near[:, 1:].ravel()
    r = np.linalg.norm(F[a] - F[b], axis=1) / np.linalg.norm(P[a] - P[b], axis=1)
    r /= np.median(r)
    return float(np.percentile(r, 10)), float(np.percentile(r, 90))


results = {}
for name, images in (("generated", A.generated()[0]), ("mnist", A.mnist_ones())):
    P = images.reshape(len(images), -1)
    d = np.load(OUT / f"{name}_push0.001.npz")
    for method in ("raw", "walking", "pulled"):
        for m in (2, 3, 5, 10):
            v = verify(P, d[method][:, :m])
            lo, hi = short_lengths(P, d[method][:, :m])
            row = {k: v[k] for k in ("reading", "flagged_share", "stacked_share", "biggest_patch_share",
                                     "typical_ratio", "flagged_ratio")}
            row["short_lengths_10th_90th"] = [lo, hi]
            results[f"{name} / {method} / {m}"] = row
            fr = row["flagged_ratio"]
            print(f"{name:9s} {method:8s} {m:2d} numbers: {row['reading']:8s} flagged {100 * row['flagged_share']:5.1f}%  "
                  f"biggest patch {100 * row['biggest_patch_share']:5.1f}%  typical {row['typical_ratio']:.2f}x  "
                  f"flagged typically {fr if fr is None else round(fr, 1)}x | short lengths kept {lo:.2f}-{hi:.2f}",
                  flush=True)
(OUT / "one_to_one.json").write_text(json.dumps(results, indent=1))
