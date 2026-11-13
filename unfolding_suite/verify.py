"""Double-layer check: did the flattening stack part of the sheet on top of another part?

Uses ONLY what a real user would have: the 3-D dots and the flat result. No truth.

The idea. Paper that is bent (not stretched) can never come out CLOSER when flattened:
walking along the paper is never shorter than the straight line through space. So for each
dot, look at its nearest dots in the FLAT result and ask how much closer they are there than
they really are in 3-D:

    ratio = (3-D distance) / (flat distance)       worst of the dot's k nearest flat neighbours

The flat result is first resized so that ordinary 3-D nearest neighbours keep their length
(median 1). An honest flattening gives ratio ~1 everywhere. A dot is FLAGGED when its ratio
is above `flag` (default 2: "the flat result puts this dot at least twice as close to a
neighbour as it really is"). Flagged dots that touch each other (nearest neighbours in 3-D or
in the flat result) form PATCHES; one big patch is the signature of a stacked area.

    from unfolding_suite.verify import verify
    v = verify(points, flat)          # points (n, 3), flat (n, 2), same order
    v["reading"], v["flagged_share"], v["biggest_patch_share"]

Readings (calibrated on the reference run; see VERIFY.md):
    clean     no flags: nothing stacked that this check can see
    specks    flags, but only in tiny scattered patches: wobble, not a fold
    mild      a patch, but small (under 5% of dots) and only slightly too close (under 2.5x):
              mild squashing or stretched paper, not a fold
    stacked   a connected patch of at least 1% of the dots that is not mild: part of the
              sheet lies on top of another part (or, rarely, is squashed hard)

Run on a saved suite run (compares the reading with the suite's truth-based fold check):
    .venv/bin/python -m unfolding_suite.verify --run current
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

K = 10                  # nearest neighbours looked at, in the flat result and in 3-D
FLAG = 2.0              # a dot is flagged when a flat neighbour is this many times too close
STACKED_PATCH = 0.01    # a patch holding at least this share of the dots means "stacked"
MILD_SHARE = 0.05       # ...unless under this share of dots is in patches AND
MILD_RATIO = 2.5        # ...the flagged dots are typically less than this many times too close: "mild"
SPECK_PATCH = 5         # patches smaller than this many dots are not counted at all


def closer_than_real(points, flat, k=K):
    """Per dot: the worst (3-D distance / flat distance) over its k nearest dots in the flat result.

    Returns (ratio per dot, flat result rescaled so 3-D nearest neighbours keep their length).
    """
    points = np.asarray(points, float)
    flat = np.asarray(flat, float)
    n = len(points)
    _, near3 = cKDTree(points).query(points, k=k + 1)
    a, b = np.repeat(np.arange(n), k), near3[:, 1:].ravel()
    scale = np.median(np.linalg.norm(flat[a] - flat[b], axis=1) /
                      np.maximum(np.linalg.norm(points[a] - points[b], axis=1), 1e-300))
    F = flat / max(scale, 1e-300)
    dist_flat, nearF = cKDTree(F).query(F, k=k + 1)
    d3 = np.linalg.norm(points[nearF[:, 1:]] - points[:, None, :], axis=2)
    return (d3 / np.maximum(dist_flat[:, 1:], 1e-12)).max(1), F


def patches(points, F, flagged, k=K):
    """Sizes of the connected groups of flagged dots (touching in 3-D or in the flat result)."""
    n = len(points)
    idx = np.where(flagged)[0]
    if len(idx) == 0:
        return np.zeros(0, int), np.full(n, -1)
    rows = []
    for P in (points, F):
        _, near = cKDTree(P).query(P[idx], k=k + 1)
        rows.append(np.c_[np.repeat(idx, k + 1), near.ravel()])
    links = np.concatenate(rows)
    links = links[flagged[links[:, 1]]]
    labels = connected_components(csr_matrix((np.ones(len(links)), (links[:, 0], links[:, 1])), shape=(n, n)),
                                  directed=False)[1]
    labels = np.where(flagged, labels, -1)
    sizes = np.bincount(labels[flagged])
    return np.sort(sizes[sizes > 0])[::-1], labels


def verify(points, flat, k=K, flag=FLAG):
    """The double-layer check. Returns a dict; `per_dot` holds each dot's ratio, in input order."""
    ratio, F = closer_than_real(points, flat, k)
    flagged = ratio > flag
    sizes, labels = patches(points, F, flagged, k)
    n = len(ratio)
    counted = sizes[sizes >= SPECK_PATCH]
    biggest = float(sizes[0] / n) if len(sizes) else 0.0
    in_patches = np.isin(labels, np.where(np.bincount(labels[flagged]) >= SPECK_PATCH)[0]) if flagged.any() else flagged
    flagged_ratio = float(np.median(ratio[flagged])) if flagged.any() else None
    if biggest >= STACKED_PATCH and in_patches.mean() < MILD_SHARE and flagged_ratio < MILD_RATIO:
        reading = "mild"
    elif biggest >= STACKED_PATCH:
        reading = "stacked"
    elif flagged.any():
        reading = "specks"
    else:
        reading = "clean"
    return {"reading": reading,
            "flagged_share": float(flagged.mean()),             # dots above the flag line
            "stacked_share": float(in_patches.mean()),         # dots in patches of 5 or more
            "biggest_patch_share": biggest,                    # the largest connected patch
            "patches": int(len(counted)),                      # patches of 5 or more dots
            "typical_ratio": float(np.median(ratio)),          # ~1 for an honest flattening
            "flagged_ratio": flagged_ratio,                    # how many times too close the flagged dots typically are
            "per_dot": ratio}


def summary_words(v):
    """Short text for terminal lines and scorecards."""
    if v["reading"] == "clean":
        return "clean"
    if v["reading"] == "specks":
        return f"specks {100 * v['flagged_share']:.1f}%"
    if v["reading"] == "mild":
        return f"mild {100 * v['stacked_share']:.1f}% ({v['flagged_ratio']:.1f}x)"
    return f"stacked {100 * v['stacked_share']:.0f}% ({v['flagged_ratio']:.0f}x)"


# ---------------------------------------------------------------- calibration on a saved run
def check_run(run_dir):
    """Verify every saved layout of a suite run and compare with the suite's truth-based checks."""
    from .run import ROOT
    from .scoring import score_layout, to_grid_order
    from .sheets import SHEETS, make_instance

    run_dir = Path(run_dir) if Path(run_dir).exists() else ROOT / run_dir
    saved = json.loads((run_dir / "results.json").read_text())
    settings = saved["settings"]
    layouts = np.load(run_dir / "layouts.npz")
    rows = []
    for key in layouts.files:
        test, seed = key.rsplit("|", 1)
        sheet_name = test.split(" / ")[0]
        if sheet_name not in SHEETS:
            continue
        inst = make_instance(SHEETS[sheet_name], settings["grid"], int(seed), settings["noise"])
        flat_grid = layouts[key].astype(float)               # saved in paper (grid) order
        v = verify(_grid_points(inst), flat_grid)
        r = score_layout(flat_grid, inst.uv, inst.xyz, inst.grid, inst.sheet.family)
        truth_stacked = max(r["fold_share"], r["overlap_share"])
        rows.append({"test": test, "family": SHEETS[sheet_name].family, "seed": int(seed), "reading": v["reading"],
                     "stacked_share": v["stacked_share"], "flagged_share": v["flagged_share"],
                     "biggest_patch_share": v["biggest_patch_share"], "flagged_ratio": v["flagged_ratio"],
                     "typical_ratio": v["typical_ratio"],
                     "truth_folded": truth_stacked, "truth_verdict": r["verdict"], "passed": r["passed"]})
    return rows


def _grid_points(inst):
    """The noisy 3-D points the method saw, put back in paper order."""
    P = np.empty_like(inst.points)
    P[inst.order] = inst.points
    return P


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="current", help="a saved run folder name under unfolding_suite/runs, or a path")
    ap.add_argument("--save", help="write the rows to this JSON file")
    a = ap.parse_args(argv)
    rows = check_run(a.run)
    print(f"{'test':40s} {'reading':8s} {'stacked':>8s} {'biggest':>8s} {'ratio':>6s}   truth: folded  verdict")
    for r in sorted(rows, key=lambda r: (r["reading"], -r["truth_folded"])):
        ratio = "-" if r["flagged_ratio"] is None else f"{r['flagged_ratio']:.1f}"
        print(f"{r['test']:40s} {r['reading']:8s} {100 * r['stacked_share']:7.1f}% {100 * r['biggest_patch_share']:7.1f}% "
              f"{ratio:>6s}   {100 * r['truth_folded']:6.1f}%  {r['truth_verdict']}")
    folded = [r for r in rows if r["truth_verdict"] == "folded"]
    unfolded = [r for r in rows if r["truth_verdict"] != "folded"]
    print(f"\nTruth says folded: {len(folded)}. Read as stacked: {sum(r['reading'] == 'stacked' for r in folded)}"
          + "".join(f"\n    folded but read {r['reading']}: {r['test']} (truth {100 * r['truth_folded']:.1f}%)"
                    for r in folded if r["reading"] != "stacked"))
    print(f"Truth says not folded: {len(unfolded)}. Read as clean {sum(r['reading'] == 'clean' for r in unfolded)}, "
          f"specks {sum(r['reading'] == 'specks' for r in unfolded)}, mild {sum(r['reading'] == 'mild' for r in unfolded)}, stacked {sum(r['reading'] == 'stacked' for r in unfolded)}"
          + "".join(f"\n    stacked but not folded: {r['test']} ({r['family']}, {100 * r['stacked_share']:.1f}% at "
                    f"{r['flagged_ratio']:.1f}x)" for r in unfolded if r["reading"] == "stacked"))
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        Path(a.save).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
