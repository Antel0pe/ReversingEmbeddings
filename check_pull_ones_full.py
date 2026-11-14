"""Is the pull itself one-to-one? Compare the original images with the WHOLE pulled cloud (no squashing).

Reruns the pull exactly as analyze_pull_ones.py did (1000 images, strings to 10 closest, push 1e-3,
10 rounds of 200 steps), keeps every coordinate of the result, and runs the suite's double-layer
check (unfolding_suite/verify.py) on original -> pulled: for each image, its 10 closest images
AFTER pulling, and how many times closer they are after than before. Also how much the pull
stretched pairs that were close before.

Run: .venv/bin/python check_pull_ones_full.py generated|mnist
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import analyze_pull_ones as A
from unfolding_suite.pull import pull
from unfolding_suite.verify import verify

OUT = Path(__file__).resolve().parent / "figures" / "pull_ones"
name = sys.argv[1]
images = A.generated()[0] if name == "generated" else A.mnist_ones()
P = images.reshape(len(images), -1)
links = A.closest_links(P)
X, history, strain = pull(P, links, push=1e-3, rounds=10, iters=200, verbose=True)
np.save(OUT / f"{name}_pulled_full.npy", X)

v = verify(P, X)
ratio = v.pop("per_dot")
# the other direction: pairs that were close before, how far apart are they after?
_, near = cKDTree(P).query(P, k=11)
a, b = np.repeat(np.arange(len(P)), 10), near[:, 1:].ravel()
grow = np.linalg.norm(X[a] - X[b], axis=1) / np.linalg.norm(P[a] - P[b], axis=1)
grow /= np.median(grow)
row = {**v, "squeeze_worst": float(ratio.max()), "squeeze_99th": float(np.percentile(ratio, 99)),
       "squeeze_median": float(np.median(ratio)),
       "close_pairs_after_over_before_10th_50th_90th_max": [float(np.percentile(grow, q)) for q in (10, 50, 90, 100)],
       "string_stretch_median": strain}
path = OUT / "pull_full_one_to_one.json"
old = json.loads(path.read_text()) if path.exists() else {}
old[name] = row
path.write_text(json.dumps(old, indent=1))
print(name, json.dumps(row, indent=1))
