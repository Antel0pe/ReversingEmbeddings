"""Press flat, then find the folded-over areas and flip them back across their creases.

The user's idea (2026-11-14): pressing (press_flat.py) often flattens the paper still folded.
A folded area should be detectable as a broad patch that breaks the one-to-one rule, its edge
is the crease, and near the crease "the distance between previous and current space slowly
decreases until it's close to 0". Rotate the folded area around that crease to unfold it.

Truth-free detection, per dot (uses only the 3-D dots, the strings and the flat result):
  1. The dot's little patch in 3-D: best-fit plane through the dot and its string partners.
  2. Point every patch's "up" the same way, by walking along the strings and flipping any
     neighbour whose up disagrees (most-similar neighbours first, i.e. a spanning tree).
  3. Fit the 2x2 map that takes the patch (in its own plane, with that up) to the flat result.
     Its determinant = how the patch's area changed: about +1 = laid flat as-is; about -1 =
     laid flat MIRROR-IMAGE (folded over); near 0 = squashed to a line (a crease).
  4. The majority sign is "right way up"; connected groups of the other sign are folded areas.
     Groups under SPECK dots are absorbed into their surroundings.

Unfolding: the largest right-way-up group is the base. Walking outward group by group, any group
that is (currently) mirror-image is reflected, together with everything beyond it, across the
straight line fitted to its border with the group it hangs from (the crease). Optionally the
strings then settle in the flat plane (patch_stitch.settle_piece) to smooth the seams.

Run: OMP_NUM_THREADS=1 .venv/bin/python -m unfolding_suite.experiments.press_unfold
"""
import json
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components, minimum_spanning_tree
from scipy.spatial import cKDTree

from unfolding_suite.experiments.patch_stitch import settle_piece
from unfolding_suite.experiments.press_flat import knn, true_links
from unfolding_suite.link_check import checked_links
import unfolding_suite.holdout  # noqa: F401
from unfolding_suite.run import failure_words
from unfolding_suite.scoring import score_layout, signed_area, to_grid_order, triangles
from unfolding_suite.sheets import SHEETS, make_instance
from unfolding_suite.verify import verify

RUNS = Path(__file__).resolve().parent.parent / "runs"
OUT = RUNS / "press_unfold"
SPECK = 0.005           # groups smaller than this share of the dots are absorbed
SHEETS_TRIED = ["cone", "s_curl", "roll", "h_roll5", "tight_roll", "crumple", "hard_crumple",
                "h_crumple17", "h_crumple29", "wave", "accordion"]


def adjacency(n, links):
    links = np.unique(np.sort(links, 1), axis=0)
    return csr_matrix((np.ones(2 * len(links)), (np.r_[links[:, 0], links[:, 1]], np.r_[links[:, 1], links[:, 0]])),
                      shape=(n, n)).tocsr(), links


def area_change(points, flat, A):
    """Per dot: signed area change of its patch from 3-D (consistently oriented) to the flat result."""
    n = len(points)
    nbrs = [A.indices[A.indptr[i]:A.indptr[i + 1]] for i in range(n)]
    few = [i for i in range(n) if len(nbrs[i]) < 3]           # dots left with under 3 strings: use nearest dots
    if few:
        _, near = cKDTree(points).query(points[few], k=7)
        for i, row in zip(few, near):
            nbrs[i] = row[1:]
    frames = np.zeros((n, 3, 3))
    for i in range(n):
        P = points[np.r_[i, nbrs[i]]]
        frames[i] = np.linalg.svd(P - P.mean(0), full_matrices=False)[2]
    normal = frames[:, 2].copy()
    # orient the normals consistently: spanning tree that prefers the most parallel neighbours
    i, j = A.nonzero()
    w = 1.0 - np.abs((normal[i] * normal[j]).sum(1)) + 1e-6
    tree = minimum_spanning_tree(csr_matrix((w, (i, j)), shape=(n, n)))
    tree = tree + tree.T
    sign = np.zeros(n)
    for root in range(n):
        if sign[root]:
            continue
        order, pred = breadth_first_order(tree, root, directed=False)
        sign[root] = 1
        for v in order[1:]:
            sign[v] = sign[pred[v]] * np.sign((normal[v] * normal[pred[v]]).sum() or 1)
    normal *= sign[:, None]
    det = np.zeros(n)
    for i in range(n):
        e1 = frames[i, 0]
        e2 = np.cross(normal[i], e1)
        d3 = points[nbrs[i]] - points[i]
        T = np.c_[d3 @ e1, d3 @ e2]
        F = flat[nbrs[i]] - flat[i]
        M = np.linalg.lstsq(T, F, rcond=None)[0]
        det[i] = np.linalg.det(M)
    if np.median(det) < 0:
        det = -det                                    # the majority is "right way up"
    return det


def groups(A, positive, speck=SPECK):
    """Connected same-sign groups along the strings; tiny groups absorbed into their surroundings."""
    n = len(positive)
    lab = positive.copy()
    for _ in range(4):
        i, j = A.nonzero()
        keep = lab[i] == lab[j]
        G = csr_matrix((np.ones(keep.sum()), (i[keep], j[keep])), shape=(n, n))
        k, comp = connected_components(G, directed=False)
        sizes = np.bincount(comp)
        small = sizes[comp] < speck * n
        if not small.any():
            break
        lab = lab.copy()
        lab[small] = ~lab[small]
    return lab, comp, k


def fit_line(P):
    c = P.mean(0)
    d = np.linalg.svd(P - c, full_matrices=False)[2][0]
    return c, d


def reflect(X, c, d):
    v = X - c
    return c + 2 * np.outer(v @ d, d) - v


def unfold(flat, A, positive, comp, k):
    """Reflect mirror-image groups (with everything beyond them) across their crease lines."""
    flat = flat.copy()
    i, j = A.nonzero()
    cross = comp[i] != comp[j]
    pairs = {}
    for a, b in zip(i[cross], j[cross]):
        pairs.setdefault((comp[a], comp[b]), set()).update((a, b))
    nbr = {g: set() for g in range(k)}
    for ga, gb in pairs:
        nbr[ga].add(gb)
    group_sign = np.array([positive[comp == g].mean() > 0.5 for g in range(k)])
    base = max((g for g in range(k) if group_sign[g]), key=lambda g: (comp == g).sum())
    parent = {base: None}
    queue = deque([base])
    order = []
    while queue:
        g = queue.popleft()
        order.append(g)
        for h in sorted(nbr[g]):
            if h not in parent:
                parent[h] = g
                queue.append(h)
    children = {g: [h for h in order if parent[h] == g] for g in order}

    def subtree(g):
        out, stack = [], [g]
        while stack:
            x = stack.pop()
            out.append(x)
            stack += children[x]
        return out

    current = group_sign.copy()
    creases = []
    for g in order[1:]:
        if current[g]:
            continue
        border = np.array(sorted(pairs[(parent[g], g)]))
        c, d = fit_line(flat[border])
        move = np.isin(comp, subtree(g))
        flat[move] = reflect(flat[move], c, d)
        for h in subtree(g):
            current[h] = ~current[h]
        creases.append((c, d, len(border)))
    return flat, creases


def truth_flipped(inst, flat):
    """Per dot (in points order): is most of its paper triangle fan mirror-image in the flat result?"""
    F = to_grid_order(flat, inst)
    T = triangles(inst.grid, 1)
    a = signed_area(F, T)
    flip = (np.sign(a) != np.sign(np.median(a))).astype(float)
    n = len(F)
    votes, count = np.zeros(n), np.zeros(n)
    for c in range(3):
        np.add.at(votes, T[:, c], flip)
        np.add.at(count, T[:, c], 1)
    by_grid = votes / np.maximum(count, 1) > 0.5
    return by_grid[inst.order]


def score(inst, flat):
    r = score_layout(to_grid_order(flat, inst), inst.uv, inst.xyz, inst.grid, inst.sheet.family)
    return {"passed": bool(r["passed"]), "words": failure_words(r),
            "fold": float(max(r["fold_share"], r["overlap_share"])), "layers": verify(inst.points, flat)["reading"]}


def run_one(job):
    name, which, flat = job
    inst = make_instance(SHEETS[name], 65, 100)
    links = {"nearest": lambda: knn(inst.points),
             "checked": lambda: checked_links(inst.points, 10, 12, 0.5),
             "true": lambda: true_links(inst)}[which]()
    flat = np.asarray(flat)
    A, _ = adjacency(len(flat), links)
    det = area_change(inst.points, flat, A)
    positive, comp, k = groups(A, det > 0)
    truth = truth_flipped(inst, flat)
    found = ~positive
    unfolded, creases = unfold(flat, A, positive, comp, k)
    settled = settle_piece(inst.points, np.unique(np.sort(links, 1), axis=0), np.arange(len(flat)), unfolded, 1500)
    # the user's crease intuition: how squashed are the dots on a detected group border vs elsewhere?
    i, j = A.nonzero()
    on_border = np.zeros(len(flat), bool)
    on_border[i[comp[i] != comp[j]]] = True
    return {"sheet": name, "links": which,
            "pressed": score(inst, flat),
            "truly_flipped_share": float(truth.mean()),
            "found_share": float(found.mean()),
            "agreement": float((found == truth).mean()),
            "missed": float((truth & ~found).mean()), "false_alarm": float((found & ~truth).mean()),
            "groups": int(k), "creases": len(creases),
            "abs_det_border_median": float(np.median(np.abs(det[on_border]))) if on_border.any() else None,
            "abs_det_elsewhere_median": float(np.median(np.abs(det[~on_border]))),
            "unfolded": score(inst, unfolded), "unfolded_settled": score(inst, settled),
            "det": det.tolist(), "found": found.tolist(), "truth": truth.tolist(),
            "flat_unfolded": unfolded.tolist(), "flat_settled": settled.tolist()}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pressed = json.loads((RUNS / "press_flat" / "main_seeds100.json").read_text())
    jobs = [(r["sheet"], r["links"], r["flat"]) for r in pressed
            if r["push"] == 0 and r["sheet"] in SHEETS_TRIED and r["flat"] is not None]
    rows = []
    with Pool(4) as pool:
        for r in pool.imap_unordered(run_one, jobs):
            rows.append(r)
            print(f"{r['sheet']:13s} {r['links']:8s} pressed {r['pressed']['words']:14s} | truly flipped "
                  f"{r['truly_flipped_share']:.2f} found {r['found_share']:.2f} agree {r['agreement']:.2f} "
                  f"(missed {r['missed']:.2f}, false {r['false_alarm']:.2f}) groups {r['groups']} creases {r['creases']} | "
                  f"|area| border {r['abs_det_border_median'] or 0:.2f} vs {r['abs_det_elsewhere_median']:.2f} | "
                  f"unfolded {r['unfolded']['words']:14s} settled {r['unfolded_settled']['words']:14s} "
                  f"layers {r['unfolded_settled']['layers']}", flush=True)
    (OUT / "results.json").write_text(json.dumps(rows))
