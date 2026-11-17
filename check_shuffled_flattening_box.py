"""Follow-up to analyze_shuffled_pixel_flattening.py: is cy/height missed because
they are SMALL next to lean's long, bent sweep (not because of the ruler)?

Prediction (recorded before running): shrinking lean's range -> cy and height
appear with the plain pixel ruler; widening cy's range -> cy appears.
"""
import json
import numpy as np
from sklearn.manifold import Isomap
from grey_ones import KNOBS, RANGES, render
from analyze_shuffled_pixel_flattening import r2_cv, N

boxes = {
    "default box": RANGES,
    "lean narrowed to 10-13 deg": np.array([*RANGES[:4], [10.0, 13.0]]),
    "cy widened to 10-18 px": np.array([RANGES[0], [10.0, 18.0], *RANGES[2:]]),
}
out = {}
for name, R in boxes.items():
    rng = np.random.default_rng(1)
    P = R[:, 0] + rng.random((N, 5)) * (R[:, 1] - R[:, 0])
    X = render(P, dtype=np.float64).reshape(N, -1)[:, rng.permutation(784)]
    Pn = (P - R[:, 0]) / (R[:, 1] - R[:, 0])
    Y = Isomap(n_neighbors=10, n_components=5).fit_transform(X)
    kfl = dict(zip(KNOBS, r2_cv(Y, Pn)))
    lfk = r2_cv(Pn, (Y - Y.mean(0)) / Y.std(0))
    out[name] = {"knobs_from_layout": kfl, "layout_from_knobs": lfk}
    print(name, json.dumps(out[name]), flush=True)
json.dump(out, open("figures/generated_one_shuffled_flattening/box_check.json", "w"), indent=1)
