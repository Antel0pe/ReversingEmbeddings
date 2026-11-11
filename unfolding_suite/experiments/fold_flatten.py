"""Where does flattening end up on a sheet that is creased on one side and folded on the other?

Two new bent (never stretched) test sheets, both bent along one direction only, so a side
view shows everything:

  flap_gapXX   Left part: three mild creases, nothing overhangs. Then flat. Then a hairpin
               fold, and the rest of the paper lies flat back over the base at a constant
               gap XX/100. The whole flap is close to the base, everywhere, equally.
  loop_gapXX   The same creased part and flat base. Then the paper rolls up into a loop
               (270 degrees), comes down, swings past the base at a closest gap XX/100, and
               curves back up and away. Only a narrow strip is close to the base.

The gap is the closest 3-D distance between parts of the paper that are far apart on it.
The dot spacing is 2/64 = 0.031, so gap 0.02 means the other layer is closer than the
dot's own neighbours.

The hypothesis under test (the user's): the creased part flattens fine; the folded part
gets its fold squashed, so that area is not one-to-one.

Then two checks that use only the dots and the flat result (no truth):
  closer_than_real   for each dot, its nearest dots in the FLAT result; how much nearer
                     the flat result puts them than they really are in 3-D. Paper that was
                     only bent can never come out nearer when flattened (walking along the
                     paper is never shorter than the straight line), so a ratio well above 1
                     means two dots were pushed together that are really apart.
  pulled_apart       for each dot, its nearest dots in 3-D; how much further apart the flat
                     result puts them. Above 1 means a 3-D shortcut was undone (or a tear).

Run:
  OMP_NUM_THREADS=1 .venv/bin/python -m unfolding_suite.experiments.fold_flatten          # everything
  OMP_NUM_THREADS=1 .venv/bin/python -m unfolding_suite.experiments.fold_flatten score    # only pass/fail
  OMP_NUM_THREADS=1 .venv/bin/python -m unfolding_suite.experiments.fold_flatten small    # gaps 0.005-0.02
Writes unfolding_suite/runs/fold_flatten/{scores,regions,detect}.json and layouts.npz.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

from unfolding_suite.diagnose import FAR
from unfolding_suite.methods import METHODS, method
from unfolding_suite.run import failure_words
from unfolding_suite.scoring import fold_check, score_layout, to_grid_order, triangles, signed_area
from unfolding_suite.sheets import _curve, make_instance, sheet, SHEETS

OUT = Path(__file__).resolve().parent.parent / "runs" / "fold_flatten"
SEEDS = (100, 101, 102)
GAPS = (0.02, 0.05, 0.10)
METHOD_NAMES = ("isomap_k10", "check_k10", "convex_edge", "lle_k10")
CONTROLS = ("flat", "gentle_arc", "wave", "accordion")
CREASE_END = 0.5          # arc length where the creased part ends (s = u + 1, from 0 to 2)
FLAP_FOLD = 1.3           # arc length of the hairpin fold
LOOP_START, R_LOOP, R_TURN = 0.9, 0.16, 0.05


# ------------------------------------------------------------------------------- sheets
def creases(s):
    """Three mild creases (+35, -60, +25 degrees): the paper's direction stays within 35 degrees."""
    k = np.zeros_like(s)
    for centre, degrees in [(0.12, 35), (0.27, -60), (0.42, 25)]:
        width = 0.025
        k += np.radians(degrees) / (width * np.sqrt(2 * np.pi)) * np.exp(-(s - centre) ** 2 / (2 * width ** 2))
    return k


def flap_curvature(gap):
    r = gap / 2
    return lambda s: creases(s) + np.where((s >= FLAP_FOLD) & (s < FLAP_FOLD + np.pi * r), 1 / r, 0.0)


def loop_curvature(gap):
    loop_end = LOOP_START + 1.5 * np.pi * R_LOOP
    down_end = loop_end + (R_LOOP - R_TURN - gap)
    turn_end = down_end + 0.75 * np.pi * R_TURN
    return lambda s: creases(s) + np.where((s >= LOOP_START) & (s < loop_end), 1 / R_LOOP, 0.0) \
        + np.where((s >= down_end) & (s < turn_end), -1 / R_TURN, 0.0)


def bend_with(curvature):
    def build(uv):
        s, x, z = _curve(curvature, 2.0, n=200001)
        arc = uv[:, 0] + 1
        return np.c_[np.interp(arc, s, x), uv[:, 1], np.interp(arc, s, z)]
    return build


for g in GAPS:
    tag = f"{round(100 * g):02d}"
    sheet(f"flap_gap{tag}", "bent", f"Creased, then folded flat over itself at gap {g}.")(bend_with(flap_curvature(g)))
    sheet(f"loop_gap{tag}", "bent", f"Creased, then a loop that swings past the base at gap {g}.")(bend_with(loop_curvature(g)))
NEW_SHEETS = [f"{kind}_gap{round(100 * g):02d}" for kind in ("flap", "loop") for g in GAPS]


def side_view(name, n=4001):
    """(s, x, z) of the sheet's profile: every bend is along u, so this is the whole shape."""
    s = np.linspace(-1, 1, n)
    xyz = SHEETS[name].build(np.c_[s, np.zeros(n)])
    return s + 1, xyz[:, 0], xyz[:, 2]


def measured_gap(name):
    """Closest 3-D approach between two parts of the profile more than FAR apart on the paper."""
    s, x, z = side_view(name)
    P = np.c_[x, z]
    pairs = cKDTree(P).query_pairs(0.3, output_type="ndarray")
    far = np.abs(s[pairs[:, 0]] - s[pairs[:, 1]]) > FAR
    return float(np.linalg.norm(P[pairs[far, 0]] - P[pairs[far, 1]], axis=1).min()) if far.any() else None


def regions(name, s):
    """Truth labels (for judging only): 0 creased, 1 flat base, 2 the fold itself, 3 the far part."""
    lab = np.full(len(s), 1)
    lab[s < CREASE_END] = 0
    if name.startswith("flap"):
        g = int(name[-2:]) / 100
        lab[(s >= FLAP_FOLD) & (s < FLAP_FOLD + np.pi * g / 2)] = 2
        lab[s >= FLAP_FOLD + np.pi * g / 2] = 3
    elif name.startswith("loop"):
        g = int(name[-2:]) / 100
        loop_end = LOOP_START + 1.5 * np.pi * R_LOOP + (R_LOOP - R_TURN - g)
        lab[(s >= LOOP_START) & (s < loop_end)] = 2
        lab[s >= loop_end] = 3
    return lab


REGION_NAMES = {"flap": ["creased part", "flat base", "hairpin fold", "flap lying on the base"],
                "loop": ["creased part", "flat base", "loop", "swing-past and end"]}


# ------------------------------------------------------------------- one extra method
@method("lle_k10")
def lle(points, dim, rng, k=10, reg=1e-3):
    """Locally linear embedding: each dot as a weighted mix of its 10 nearest; keep the mixes, lay out flat."""
    n = len(points)
    _, near = cKDTree(points).query(points, k=k + 1)
    near = near[:, 1:]
    Z = points[near] - points[:, None, :]
    C = np.einsum("nkd,njd->nkj", Z, Z)
    C += reg * np.trace(C, axis1=1, axis2=2)[:, None, None] * np.eye(k)
    w = np.linalg.solve(C, np.ones((n, k, 1)))[..., 0]
    w /= w.sum(1, keepdims=True)
    W = csr_matrix((w.ravel(), (np.repeat(np.arange(n), k), near.ravel())), shape=(n, n))
    A = identity(n, format="csr") - W
    M = (A.T @ A).tocsc()
    values, vectors = eigsh(M, k=dim + 1, sigma=-1e-6, which="LM", v0=np.random.default_rng(0).standard_normal(n))
    order = np.argsort(values)
    links = np.c_[np.repeat(np.arange(n), k), near.ravel()]
    return vectors[:, order[1:dim + 1]], {"edges": links}


# ------------------------------------------------------------- the checks without truth
def closer_than_real(points, flat, k=10):
    """Per dot: the largest (3-D distance / flat distance) over its k nearest dots IN THE FLAT RESULT.

    The flat result is first rescaled so that 3-D nearest neighbours keep their length on
    average (median ratio 1). Uses only the dots and the flat result.
    """
    _, near3 = cKDTree(points).query(points, k=k + 1)
    a, b = np.repeat(np.arange(len(points)), k), near3[:, 1:].ravel()
    scale = np.median(np.linalg.norm(flat[a] - flat[b], axis=1) / np.linalg.norm(points[a] - points[b], axis=1))
    F = flat / scale
    dist_flat, nearF = cKDTree(F).query(F, k=k + 1)
    d3 = np.linalg.norm(points[nearF[:, 1:]] - points[:, None, :], axis=2)
    ratio = d3 / np.maximum(dist_flat[:, 1:], 1e-12)
    worst = ratio.argmax(1)
    return ratio.max(1), nearF[:, 1:][np.arange(len(points)), worst], F


def pulled_apart(points, flat_scaled, k=10):
    """Per dot: the largest (flat distance / 3-D distance) over its k nearest dots IN 3-D."""
    d3, near3 = cKDTree(points).query(points, k=k + 1)
    dF = np.linalg.norm(flat_scaled[near3[:, 1:]] - flat_scaled[:, None, :], axis=2)
    return (dF / d3[:, 1:]).max(1)


def flagged_groups(points, flat_scaled, flagged, k=10):
    """Share of all dots in the biggest connected group of flagged dots, and how many groups.

    Two flagged dots are connected if either is among the other's k nearest in 3-D or in
    the flat result, so a patch and its overlaid twin count as one group.
    """
    n = len(points)
    idx = np.where(flagged)[0]
    if len(idx) == 0:
        return 0.0, 0
    rows = []
    for P in (points, flat_scaled):
        _, near = cKDTree(P).query(P[idx], k=k + 1)
        rows.append(np.c_[np.repeat(idx, k + 1), near.ravel()])
    links = np.concatenate(rows)
    links = links[flagged[links[:, 1]]]
    labels = connected_components(csr_matrix((np.ones(len(links)), (links[:, 0], links[:, 1])), shape=(n, n)),
                                  directed=False)[1][idx]
    sizes = np.bincount(labels)
    sizes = sizes[sizes > 0]
    return float(sizes.max() / n), int((sizes >= 5).sum())


# ----------------------------------------------------------------------------- one run
def procrustes_scale_error(F, U):
    """Best turn/mirror/shift/resize of F onto U: (scale factor, median miss in % of the side)."""
    Fc, Uc = F - F.mean(0), U - U.mean(0)
    a, sv, b = np.linalg.svd(Fc.T @ Uc)
    scale = sv.sum() / max((Fc ** 2).sum(), 1e-300)
    miss = np.linalg.norm(scale * Fc @ (a @ b) - Uc, axis=1) / 2 * 100
    return scale, float(np.median(miss))


def one(job):
    name, method_name, seed = job
    inst = make_instance(SHEETS[name], seed=seed)
    answer = METHODS[method_name](inst.points.copy(), 2, np.random.default_rng(seed + 1000))
    flat = answer[0] if isinstance(answer, tuple) else answer
    flat_grid = to_grid_order(flat, inst)                     # paper order, for scoring and truth only
    result = score_layout(flat_grid, inst.uv, inst.xyz, inst.grid, inst.sheet.family)
    out = {"sheet": name, "method": method_name, "seed": seed, "passed": result["passed"],
           "words": failure_words(result), "fold_share": result["fold_share"],
           "overlap_share": result["overlap_share"], "square_error_median": result["square_error_median"]}
    if seed != SEEDS[0]:
        return out, None

    # --- the checks that use only the dots and the flat result
    pts = inst.points
    closer, partner, F = closer_than_real(pts, flat)
    apart = pulled_apart(pts, F)
    s = inst.uv[inst.order][:, 0] + 1                         # truth, for judging the checks only
    paper_far = np.abs(s - s[partner]) > FAR                  # is the flat neighbour really far on the paper?
    detect = {"closer": closer, "apart": apart, "partner_far_on_paper": paper_far}
    for threshold in (1.5, 2.0, 3.0):
        flagged = closer > threshold
        big, groups = flagged_groups(pts, F, flagged)
        detect[f"closer>{threshold}"] = {"share": float(flagged.mean()), "biggest_group": big, "groups": groups}
    detect["apart>3"] = {"share": float((apart > 3).mean())}
    detect = {k: v for k, v in detect.items() if not isinstance(v, np.ndarray)}

    # --- truth: how each region came out
    reg = {}
    if name in NEW_SHEETS:
        U = inst.uv[inst.order]
        lab = regions(name, s)
        T = triangles(inst.grid, 2)
        tri_lab = regions(name, inst.uv[T].mean(1)[:, 0] + 1)
        area = signed_area(flat_grid, T)
        paper = signed_area(inst.uv, T)
        major = 1 if paper[area > 0].sum() >= paper[area < 0].sum() else -1
        flipped = np.sign(area) != major
        # does a region land on top of another region? distance in flat to the nearest dot of the base
        base = cKDTree(F[lab == 1])
        spacing = 2 / (inst.grid - 1)
        for r in range(4):
            m = lab == r
            if m.sum() < 3:
                continue
            scale, miss = procrustes_scale_error(F[m], U[m])
            d_base = base.query(F[m])[0]
            reg[REGION_NAMES[name[:4]][r]] = {
                "dots": int(m.sum()),
                "size_in_flat": float(1 / scale),             # 1 = the right size; 0.1 = squashed to a tenth
                "shape_miss_pct": miss,                       # after resizing this region on its own
                "flipped_pct": float(100 * paper[(tri_lab == r) & flipped].sum() / paper[tri_lab == r].sum()),
                "on_top_of_base_pct": float(100 * (d_base < 0.5 * spacing).mean()) if r != 1 else None,
                "flagged_closer>2_pct": float(100 * (closer[m] > 2).mean()),
                "flagged_partner_far_pct": float(100 * paper_far[m & (closer > 2)].mean()) if (m & (closer > 2)).any() else None,
            }
    keep = {"flat": flat.astype(np.float32), "closer": closer.astype(np.float32), "apart": apart.astype(np.float32)}
    return out, {"detect": detect, "regions": reg, "arrays": keep}


def main(only_score=False):
    OUT.mkdir(parents=True, exist_ok=True)
    print("sheet         measured gap")
    for name in NEW_SHEETS:
        print(f"  {name:12s} {measured_gap(name):.4f}")
    jobs = [(n, m, s) for n in NEW_SHEETS for m in METHOD_NAMES for s in SEEDS]
    if not only_score:
        jobs += [(n, m, SEEDS[0]) for n in CONTROLS for m in METHOD_NAMES]
    with Pool(8) as pool:
        results = pool.map(one, jobs)
    scores, regions_out, detect_out, arrays = {}, {}, {}, {}
    for out, extra in results:
        key = f"{out['sheet']} / {out['method']}"
        scores.setdefault(key, []).append(out)
        if extra:
            detect_out[key] = extra["detect"]
            if extra["regions"]:
                regions_out[key] = extra["regions"]
            for k, v in extra["arrays"].items():
                arrays[f"{key}|{k}"] = v
    summary = {k: {"passed": all(o["passed"] for o in v), "seeds_passed": sum(o["passed"] for o in v),
                   "worst": max(v, key=lambda o: (not o["passed"], max(o["fold_share"], o["overlap_share"])))["words"],
                   "runs": v} for k, v in scores.items()}
    (OUT / "scores.json").write_text(json.dumps(summary, indent=1))
    if not only_score:
        (OUT / "regions.json").write_text(json.dumps(regions_out, indent=1))
        (OUT / "detect.json").write_text(json.dumps(detect_out, indent=1))
        np.savez_compressed(OUT / "layouts.npz", **arrays)
    print("\nPASS/FAIL (seeds 100-102, worst seed)")
    for k, v in summary.items():
        print(f"  {k:30s} {'PASS' if v['passed'] else 'FAIL'} {v['seeds_passed']}/{len(v['runs'])}  {v['worst']}")
    if only_score:
        return
    print("\nREGIONS (seed 100): size in flat (1 = right), shape miss %, flipped %, on top of base %, flagged %")
    for k, v in regions_out.items():
        print(f"  {k}")
        for r, d in v.items():
            otb = "-" if d["on_top_of_base_pct"] is None else f"{d['on_top_of_base_pct']:5.1f}"
            print(f"     {r:24s} size {d['size_in_flat']:5.2f}  miss {d['shape_miss_pct']:5.1f}  "
                  f"flipped {d['flipped_pct']:5.1f}  on-base {otb}  flagged {d['flagged_closer>2_pct']:5.1f}")
    print("\nCHECK WITHOUT TRUTH (seed 100): flagged share / biggest group share / groups of 5+  at ratio > 2")
    for k, v in detect_out.items():
        d = v["closer>2.0"]
        print(f"  {k:30s} {100 * d['share']:5.1f}%  biggest {100 * d['biggest_group']:5.1f}%  groups {d['groups']:3d}"
              f"   | pulled apart >3x: {100 * v['apart>3']['share']:5.1f}%")


def small_gaps():
    """How small can the gap get before the check goes blind? check_k10, seed 100."""
    rows = {}
    for g in (0.005, 0.01, 0.02):
        for kind, curvature in (("flap", flap_curvature), ("loop", loop_curvature)):
            name = f"{kind}_tiny{g}"
            sheet(name, "bent", "small-gap check")(bend_with(curvature(g)))
            inst = make_instance(SHEETS[name], seed=SEEDS[0])
            flat = METHODS["check_k10"](inst.points.copy(), 2, np.random.default_rng(0))[0]
            result = score_layout(to_grid_order(flat, inst), inst.uv, inst.xyz, inst.grid, "bent")
            closer = closer_than_real(inst.points, flat)[0]
            s = inst.uv[inst.order][:, 0] + 1
            rows[name] = {"verdict": result["verdict"], "overlap_share": result["overlap_share"],
                          "flagged_over_2": float((closer > 2).mean()),
                          "median_ratio_far_part": float(np.median(closer[s > 1.3])),
                          "median_ratio_creased_part": float(np.median(closer[s < CREASE_END]))}
            print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[name].items()})
    (OUT / "small_gaps.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "small":
        small_gaps()
    else:
        main(only_score=len(sys.argv) > 1 and sys.argv[1] == "score")
