"""Height split: keep only the nearby dots that sit on MY layer.

Paper has no thickness. For each dot: take its K nearest dots, fit a plane to them, and
measure every dot's height above that plane. One layer gives heights near 0; a second layer
nearby gives a second clump of heights. Split the sorted heights wherever two neighbouring
values differ by more than `gap` (in units of the typical dot spacing), keep the clump that
contains the dot itself, and link the dot to its 10 nearest dots in that clump.

The whole other layer is dropped in one decision, so dropping one dot cannot expose another.

Run:
  .venv/bin/python -m unfolding_suite.experiments.height_split diag     # can heights separate layers at all? (truth)
  OMP_NUM_THREADS=1 .venv/bin/python -m unfolding_suite.experiments.height_split score [methods...]
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from unfolding_suite.diagnose import FAR
from unfolding_suite.link_check import reattach_stranded
from unfolding_suite.methods import METHODS, layout_from_links, method
from unfolding_suite.run import failure_words
from unfolding_suite.scoring import score_layout, to_grid_order
from unfolding_suite.sheets import SHEETS, make_instance

OUT = Path(__file__).resolve().parent.parent / "runs" / "height_split"
SEEDS = (100, 101, 102)
ONLY = [s for s in __import__("os").environ.get("SHEETS", "").split(",") if s]   # optional subset of sheets


def heights(points, K, quadratic=False):
    """Nearest K dots (self first) and their heights above the plane fitted to all of them."""
    _, near = cKDTree(points).query(points, k=K + 1)
    patch = points[near] - points[:, None, :]                       # self at the origin
    centred = patch - patch.mean(1, keepdims=True)
    axes = np.linalg.svd(centred, full_matrices=False)[2]           # (n, 3, 3): rows are axes
    h = np.einsum("nkd,nd->nk", patch, axes[:, 2])
    if quadratic:                                                    # remove the bowl of a curved layer
        a = np.einsum("nkd,nd->nk", patch, axes[:, 0])
        b = np.einsum("nkd,nd->nk", patch, axes[:, 1])
        X = np.stack([a * a, a * b, b * b, a, b, np.ones_like(a)], -1)
        m = max(9, (K + 1) // 2)                                     # fit the bowl to the m flattest dots only
        w = np.argsort(np.abs(h), 1)[:, :m]
        for i in range(len(points)):
            coef = np.linalg.lstsq(X[i][w[i]], h[i][w[i]], rcond=None)[0]
            h[i] = h[i] - X[i] @ coef + coef[5]
    return near, h


def my_layer(h, gap):
    """Mask of the dots in the same height clump as the dot itself (column 0)."""
    order = np.argsort(h, 1)
    sorted_h = np.take_along_axis(h, order, 1)
    label_sorted = np.c_[np.zeros(len(h), int), np.cumsum(np.diff(sorted_h, axis=1) > gap, 1)]
    label = np.empty_like(label_sorted)
    np.put_along_axis(label, order, label_sorted, 1)
    return label == label[:, :1]


def height_links(points, K=40, gap=0.5, k=10, quadratic=False):
    near, h = heights(points, K, quadratic)
    spacing = np.median(cKDTree(points).query(points, k=2)[0][:, 1])
    keep = my_layer(h, gap * spacing)
    keep[:, 0] = False
    src, dst, short = [], [], 0
    for i in range(len(points)):
        mine = near[i][keep[i]][:k]
        short += len(mine) < 3
        src.append(np.full(len(mine), i))
        dst.append(mine)
    links = np.c_[np.concatenate(src), np.concatenate(dst)]
    return reattach_stranded(points, links), short


def make_method(K, gap, quadratic=False):
    def fn(points, dim, rng):
        links, _ = height_links(points, K, gap, quadratic=quadratic)
        flat, captured = layout_from_links(points, links, dim)
        return flat, {"edges": links, "captured": captured}
    return fn


VARIANTS = {f"height_K{K}_g{g}_bowl": (K, g, True) for K in (12, 20, 40) for g in (0.25, 0.5)}
VARIANTS["height_K20_g0.5_plane"] = (20, 0.5, False)
for name, (K, g, q) in VARIANTS.items():
    method(name)(make_method(K, g, q))


def jumps(inst, links):
    g = inst.order[links]
    return int((np.linalg.norm(inst.uv[g[:, 0]] - inst.uv[g[:, 1]], axis=1) > FAR).sum())


# --------------------------------------------------------------- diagnosis (uses the truth)
def diag(name, K=40, quadratic=False):
    """Per dot: can ANY height cut separate its own layer from the other layers nearby?"""
    inst = make_instance(SHEETS[name], 65, 100)
    near, h = heights(inst.points, K, quadratic)
    g = inst.order
    far = np.linalg.norm(inst.uv[g[near]] - inst.uv[g[:, None]], axis=2) > FAR
    spacing = np.median(cKDTree(inst.points).query(inst.points, k=2)[0][:, 1])
    touched = far.any(1)
    if not touched.any():
        return name, 0, None
    ok, margin = [], []
    for i in np.where(touched)[0]:
        same, other = h[i][~far[i]], h[i][far[i]]
        # separable if the other layer's heights all lie strictly beyond my layer's range, on either side
        above, below = other[other > 0], other[other <= 0]
        lo, hi = same.min(), same.max()
        sep = (above.min() if len(above) else np.inf) > hi and (below.max() if len(below) else -np.inf) < lo
        ok.append(sep)
        m = min((above.min() - hi) if len(above) else np.inf, (lo - below.max()) if len(below) else np.inf)
        margin.append(m / spacing)
    margin = np.array(margin)
    return name, int(touched.sum()), (np.mean(ok), np.median(margin[np.isfinite(margin)]))


def one(job):
    sheet_name, method_name, seed = job
    inst = make_instance(SHEETS[sheet_name], 65, seed)
    start = time.time()
    try:
        flat, info = METHODS[method_name](inst.points.copy(), 2, np.random.default_rng(seed + 1000))
        r = score_layout(to_grid_order(flat, inst), inst.uv, inst.xyz, inst.grid, inst.sheet.family)
        r["jumps"] = jumps(inst, info["edges"]) if "edges" in info else None
    except Exception as e:                                           # noqa: BLE001
        r = {"passed": False, "verdict": "crashed", "error": f"{type(e).__name__}: {e}"}
    keep = {k: r.get(k) for k in ("passed", "verdict", "fold_share", "overlap_share", "square_error_median",
                                  "roughness", "tear_share", "crush_share", "jumps", "error")}
    keep["words"] = failure_words(r)
    keep["seconds"] = time.time() - start
    return job, keep


# ------------------------------------------------- my addition: height split AND the link check
def tilt_ok(points, links, plane_k=12, limit=0.5):
    """The link check from link_check.py, applied to a given set of links."""
    _, near = cKDTree(points).query(points, k=plane_k + 1)
    patch = points[near]
    normals = np.linalg.svd(patch - patch.mean(1, keepdims=True), full_matrices=False)[2][:, 2]
    step = points[links[:, 1]] - points[links[:, 0]]
    step /= np.linalg.norm(step, axis=1, keepdims=True)
    tilt = np.maximum(np.abs((step * normals[links[:, 0]]).sum(1)), np.abs((step * normals[links[:, 1]]).sum(1)))
    return tilt <= limit


def make_both(K, gap):
    def fn(points, dim, rng):
        links, _ = height_links(points, K, gap, quadratic=True)
        links = reattach_stranded(points, links[tilt_ok(points, links)])
        flat, captured = layout_from_links(points, links, dim)
        return flat, {"edges": links, "captured": captured}
    return fn


for K, g in ((12, 0.25), (40, 0.5)):
    method(f"both_K{K}_g{g}")(make_both(K, g))


if __name__ == "__main__":
    part = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    if part == "diag":
        names = ["s_curl", "roll", "tight_roll", "h_roll5", "h_scurl65", "accordion", "crumple", "hard_crumple",
                 "h_crumple17", "h_crumple29"]
        rows = {}
        with Pool(4) as pool:
            for q in (False, True):
                for name, touched, res in pool.starmap(diag, [(n, 40, q) for n in names]):
                    rows[f"{name}{' quadratic' if q else ''}"] = [touched, res]
                    print(f"{name:13s} {'bowl removed' if q else 'plane       '} dots with another layer in reach: "
                          f"{touched:5d}" + ("" if res is None else
                                             f" | separable by some height cut: {100 * res[0]:5.1f}%"
                                             f" | median clear gap {res[1]:.2f} dot spacings"), flush=True)
        (OUT / "diag.json").write_text(json.dumps(rows, indent=1))
    else:
        methods = sys.argv[2:] or list(VARIANTS) + ["isomap_k10", "check_k10"]
        sheets = [s for s in SHEETS if not ONLY or s in ONLY]
        jobs = [(s, m, seed) for s in sheets for m in methods for seed in SEEDS]
        results = {}
        with Pool(4) as pool:
            for (s, m, seed), r in pool.imap_unordered(one, jobs):
                results.setdefault(f"{s} / {m}", {})[str(seed)] = r
                print(f"{'PASS' if r['passed'] else 'FAIL'} {s:15s} {m:22s} {seed} jumps {r['jumps']} {r['words']}",
                      flush=True)
        path = OUT / "scores.json"
        old = json.loads(path.read_text()) if path.exists() else {}
        old.update(results)
        path.write_text(json.dumps(old, indent=1))
