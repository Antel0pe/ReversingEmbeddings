"""Does the height split get fooled when the folded-over layer dominates a dot's closest dots?

For every dot with another layer among its 40 closest dots, record what share of those 40
are on another layer (truth), and whether the split went wrong: kept some other-layer dots,
or dropped some of its own layer. Bucket by that share.

Run: .venv/bin/python -m unfolding_suite.experiments.height_split_concern
"""
import json

import numpy as np
from scipy.spatial import cKDTree

from unfolding_suite.diagnose import FAR
from unfolding_suite.experiments.height_split import OUT, heights, my_layer
from unfolding_suite.sheets import SHEETS, make_instance

NAMES = ["s_curl", "h_roll5", "tight_roll", "crumple", "hard_crumple", "h_crumple17", "h_crumple29"]
EDGES = [0, 0.25, 0.5, 0.75, 1.0001]


def main(K=40, gap=0.5):
    out = {}
    for name in NAMES:
        inst = make_instance(SHEETS[name], 65, 100)
        near, h = heights(inst.points, K, quadratic=True)
        spacing = np.median(cKDTree(inst.points).query(inst.points, k=2)[0][:, 1])
        keep = my_layer(h, gap * spacing)
        g = inst.order
        other = np.linalg.norm(inst.uv[g[near]] - inst.uv[g[:, None]], axis=2) > FAR
        touched = other.any(1)
        share = other[:, 1:].mean(1)
        kept_wrong = (keep & other).any(1)
        dropped_own = (~keep & ~other)[:, 1:].any(1)
        rows = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            sel = touched & (share >= lo) & (share < hi)
            if sel.sum():
                rows.append({"share": f"{int(100 * lo)}-{min(100, int(100 * hi))}%", "dots": int(sel.sum()),
                             "kept_wrong": float(kept_wrong[sel].mean()), "dropped_own": float(dropped_own[sel].mean()),
                             "clean": float((~kept_wrong & ~dropped_own)[sel].mean())})
        out[name] = rows
        print(name)
        for r in rows:
            print(f"   other layer is {r['share']:>8s} of the 40 closest: {r['dots']:5d} dots | split clean "
                  f"{100 * r['clean']:5.1f}% | kept other-layer dots {100 * r['kept_wrong']:5.1f}% | "
                  f"dropped own-layer dots {100 * r['dropped_own']:5.1f}%")
    (OUT / "concern.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
