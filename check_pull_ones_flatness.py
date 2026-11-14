"""How flat is the pulled cloud, and in how many directions? Uses the saved whole pulled clouds.

For d directions: fit the best flat d-direction surface through the cloud (its d most-spread
directions) and measure how far EVERY image sits off it, in units of the typical distance between
an image and its 10 closest images (median string length). Flat in d directions = even the worst
image is a small fraction of a string length off. Same for the cloud before pulling.

Run: .venv/bin/python check_pull_ones_flatness.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import analyze_pull_ones as A

OUT = Path(__file__).resolve().parent / "figures" / "pull_ones"
DS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50, 100]
results = {}
for name, images in (("generated", A.generated()[0]), ("mnist", A.mnist_ones())):
    P = images.reshape(len(images), -1)
    X = np.load(OUT / f"{name}_pulled_full.npy")
    _, near = cKDTree(P).query(P, k=11)
    string = np.median(np.linalg.norm(P[near[:, 1:]] - P[:, None], axis=2))
    for label, C in (("before", P), ("pulled", X)):
        c = C - C.mean(0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        rows = {}
        for d in DS:
            off = np.linalg.norm(c - (c @ vt[:d].T) @ vt[:d], axis=1) / string
            rows[d] = {"median": float(np.median(off)), "p99": float(np.percentile(off, 99)), "worst": float(off.max())}
        results[f"{name} / {label}"] = rows
        print(f"{name} {label}: distance off the best flat surface, in string lengths (median / worst)")
        print("   " + "  ".join(f"d={d}: {r['median']:.2f}/{r['worst']:.2f}" for d, r in rows.items()))
(OUT / "flatness.json").write_text(json.dumps(results, indent=1))
