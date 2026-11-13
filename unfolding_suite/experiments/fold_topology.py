"""Can topology find the folds? Two detectors that use only the dots, judged against the truth.

The problem: where a sheet folds over, a dot's nearest dots can include dots from another
layer, close in 3-D but far along the paper ("jump links"). If we could SEE where that
happens from the dots alone, we could (a) avoid those links and (b) tell whether a
transformation of the cloud has made the problem better or worse.

Detector A, "one blob or two" (0-dimensional topology of a small ball):
    Take a dot's 40 nearest dots. Join them closest pair first (single linkage) and watch
    the pieces merge. Dots on one layer join up at about one dot spacing. If a second
    layer is in the ball, two big pieces stay apart until the join length reaches the gap
    between layers. Score = the longest join length at which two pieces of at least 5 dots
    are still apart, divided by the ball's typical join length. One layer: about 1.
    Two layers: about gap / spacing.

Detector B, "loops that never fill in" (1-dimensional persistent homology):
    A flat sheet with links drawn between nearby dots has no holes: every small loop
    is filled with triangles. A jump link between two layers closes a loop that goes
    round the fold, and nothing fills it. Persistent homology lists every loop with
    the link length where it appears ("birth") and the length where it fills in
    ("death"). Long-lived loops = folds. The link that gives birth to each loop shows
    where the fold is, and the shortest birth length is the longest hop that is safe.

Run (gudhi must be importable; it is not in .venv, see README):
  .venv/bin/python -m unfolding_suite.experiments.fold_topology blobs     # detector A vs truth
  .venv/bin/python -m unfolding_suite.experiments.fold_topology loops     # detector B vs truth
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

from unfolding_suite.diagnose import FAR
from unfolding_suite.sheets import SHEETS, make_instance

OUT = Path(__file__).resolve().parent.parent / "runs" / "fold_topology"
SEED = 100


# ------------------------------------------------------------------ truth (for judging only)
def paper(inst):
    """Paper position of each dot, in the order the method sees them."""
    return inst.uv[inst.order]


def jump_mask(inst, links):
    uv = paper(inst)
    return np.linalg.norm(uv[links[:, 0]] - uv[links[:, 1]], axis=1) > FAR


def knn_links(points, k=10):
    _, near = cKDTree(points).query(points, k=k + 1)
    return np.c_[np.repeat(np.arange(len(points)), k), near[:, 1:].ravel()]


def true_gap(inst):
    """Closest 3-D approach between two dots that are far apart on the paper."""
    pts, uv = inst.points, paper(inst)
    pairs = cKDTree(pts).query_pairs(0.3, output_type="ndarray")
    far = np.linalg.norm(uv[pairs[:, 0]] - uv[pairs[:, 1]], axis=1) > FAR
    if not far.any():
        return None
    return float(np.linalg.norm(pts[pairs[far, 0]] - pts[pairs[far, 1]], axis=1).min())


def surface_walk(inst, links, factor=2.0, refine=4):
    """Truth: walking distance along the real 3-D sheet between the ends of each link, divided by
    the link's straight length. The sheet is rebuilt on a grid `refine` times finer than the dots,
    walked with 16 step directions, so sheared patches are not mistaken for folds. Values above
    `factor` are capped at `factor` + 1: "a shortcut across empty space"."""
    g = (inst.grid - 1) * refine + 1
    axis = np.linspace(-1, 1, g)
    v, u = np.meshgrid(axis, axis, indexing="ij")
    xyz = inst.sheet.build(np.c_[u.ravel(), v.ravel()])
    idx = np.arange(g * g).reshape(g, g)
    pairs = []
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1), (1, 2), (2, 1), (1, -2), (2, -1)]:
        r0, r1 = max(0, -dr), g - max(0, dr)
        c0, c1 = max(0, -dc), g - max(0, dc)
        pairs.append(np.c_[idx[r0:r1, c0:c1].ravel(), idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc].ravel()])
    pairs = np.concatenate(pairs)
    w = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
    graph = csr_matrix((w, (pairs[:, 0], pairs[:, 1])), shape=(g * g, g * g))
    uv = paper(inst)
    node = cKDTree(np.c_[u.ravel(), v.ravel()]).query(uv)[1]               # each dot's nearest fine node
    a, b = node[links[:, 0]], node[links[:, 1]]
    straight = np.linalg.norm(inst.points[links[:, 0]] - inst.points[links[:, 1]], axis=1)
    ratio = np.full(len(links), factor + 1.0)
    limit = factor * straight.max()
    order = np.argsort(a)
    starts = np.searchsorted(a[order], np.unique(a))
    for src, sel in zip(np.unique(a), np.split(order, starts[1:])):
        d = dijkstra(graph, directed=False, indices=src, limit=limit)
        ratio[sel] = np.minimum(d[b[sel]] / straight[sel], factor + 1)
    return ratio


def spacing(points):
    return float(np.median(cKDTree(points).query(points, k=2)[0][:, 1]))


# ------------------------------------------------------------------ detector A: one blob or two
def blob_split(points, K=40, m=5):
    """Per dot: split score, which of its K nearest dots are in its own piece, and whether the
    other piece is STACKED over the dot (offset along the sheet's normal) or side by side.

    stacked = |cos| between the normal of my piece and the step from me to the nearest dot of
    the other pieces: about 1 for a layer above or below, about 0 for a row next to me.
    """
    n = len(points)
    _, near = cKDTree(points).query(points, k=K + 1)
    score = np.ones(n)
    stacked = np.zeros(n)
    mine = np.ones((n, K + 1), bool)
    for i in range(n):
        P = points[near[i]]
        T = minimum_spanning_tree(squareform(pdist(P))).tocoo()
        typical = np.median(T.data)
        parent = np.arange(K + 1)
        size = np.ones(K + 1, int)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        best, cut = 1.0, None
        for e in np.argsort(T.data):
            a, b = find(T.row[e]), find(T.col[e])
            if min(size[a], size[b]) >= m and T.data[e] / typical > best:
                best, cut = T.data[e] / typical, T.data[e]
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a
            size[a] += size[b]
        score[i] = best
        if cut is None:
            continue
        keep = T.data < cut
        g = np.zeros((K + 1, K + 1))
        g[T.row[keep], T.col[keep]] = 1
        label = connected_components(g, directed=False)[1]
        mine[i] = label == label[0]
        if mine[i].sum() >= 3:
            own = P[mine[i]]
            normal = np.linalg.svd(own - own.mean(0), full_matrices=False)[2][2]
            others = P[~mine[i]] - P[0]
            step = others[np.argmin(np.linalg.norm(others, axis=1))]
            stacked[i] = abs(step @ normal) / np.linalg.norm(step)
    return near, score, mine, stacked


def blobs(name):
    inst = make_instance(SHEETS[name], 65, SEED)
    near, score, mine, stacked = blob_split(inst.points)
    uv = paper(inst)
    far = np.linalg.norm(uv[near] - uv[:, None], axis=2) > FAR          # (n, 41)
    links, walk, _ = load_truth(name)                                     # same 10 nearest links, same order
    touched = (walk > 2).reshape(-1, 10).any(1)                           # a real shortcut among the 10 nearest
    in_reach = far.any(1) | (walk > 1.5).reshape(-1, 10).any(1)          # not clean: another layer may be near
    # does the split keep the right dots? among dots flagged, share of far dots dropped / near dots kept
    split = score > 1.0
    dropped_far = (far & ~mine)[split].sum() / max(far[split].sum(), 1)
    dropped_near = (~far & ~mine)[split].sum() / max((~far)[split].sum(), 1)
    return name, {
        "touched": int(touched.sum()), "in_reach": int(in_reach.sum()),
        "score_clean": np.percentile(score[~in_reach], [50, 99, 99.9, 100]).round(3).tolist() if (~in_reach).any() else None,
        "score_touched": np.percentile(score[touched], [5, 25, 50]).round(3).tolist() if touched.any() else None,
        "score": score.round(4).tolist(), "touched_mask": touched.tolist(), "in_reach_mask": in_reach.tolist(),
        "split_dropped_far": float(dropped_far), "split_dropped_near": float(dropped_near),
        "stacked": stacked.round(4).tolist(),
        "detect": {f"{t}{'_stacked' if st else ''}": detect_rates(score, stacked, touched, in_reach, t, st)
                   for t in (1.4, 1.6, 2.0) for st in (False, True)},
    }


def detect_rates(score, stacked, touched, in_reach, threshold, need_stacked):
    """Share of touched dots flagged (found) and of clean dots flagged (false alarms)."""
    flag = (score > threshold) & ((stacked > 0.7) if need_stacked else True)
    return {"found": float(flag[touched].mean()) if touched.any() else None,
            "false_alarm": float(flag[~in_reach].mean()) if (~in_reach).any() else None}


# ------------------------------------------------------------------ detector B: loops that never fill
def loop_bars(points, complex_kind="rips", reach=3.0, k=10):
    """H1 bars as (birth, death, birth edge). death = inf if the loop never fills in."""
    import gudhi
    s = spacing(points)
    if complex_kind == "rips":
        st = gudhi.RipsComplex(points=points, max_edge_length=reach * s).create_simplex_tree(max_dimension=2)
    elif complex_kind == "local":                    # every link measured in its own ends' local dot spacing
        links, scaled = local_scale_links(points, reach)
        st = gudhi.SimplexTree()
        for v in range(len(points)):
            st.insert([v], 0.0)
        st.insert_batch(links.T, scaled)
        st.expansion(2)
        s = 1.0                                      # filtration is already in local spacings
    else:                                            # the 10-nearest links only, filled with their triangles
        links = np.unique(np.sort(knn_links(points, k), 1), axis=0)
        st = gudhi.SimplexTree()
        for v in range(len(points)):
            st.insert([v], 0.0)
        length = np.linalg.norm(points[links[:, 0]] - points[links[:, 1]], axis=1)
        st.insert_batch(links.T, length)
        st.expansion(2)
    st.compute_persistence(persistence_dim_max=True)
    bars = []
    for b, d in st.persistence_pairs():
        if len(b) != 2:
            continue
        birth = st.filtration(b)
        death = st.filtration(d) if len(d) else np.inf
        bars.append((birth / s, death / s, tuple(b)))
    return bars, s


def local_spacing(points, k=8):
    """Distance to each dot's k-th nearest dot, divided so that a flat even grid gives about 1."""
    d = cKDTree(points).query(points, k=k + 1)[0]
    rho = d[:, k]
    return rho / np.median(rho / d[:, 1])


def local_scale_links(points, reach):
    """Pairs with |x - y| <= reach * sqrt(rho_x rho_y) (Berry & Sauer's continuous k-nearest links),
    and that ratio as the link's length."""
    rho = local_spacing(points)
    tree = cKDTree(points)
    pairs = []
    for i, nb in enumerate(tree.query_ball_point(points, reach * rho)):
        nb = np.asarray(nb)
        nb = nb[nb > i]
        pairs.append(np.c_[np.full(len(nb), i), nb])
    p = np.concatenate(pairs)
    p = np.unique(p, axis=0)                         # each pair can be found from both ends
    scaled = np.linalg.norm(points[p[:, 0]] - points[p[:, 1]], axis=1) / np.sqrt(rho[p[:, 0]] * rho[p[:, 1]])
    keep = scaled <= reach
    return p[keep], scaled[keep]


def loops(job):
    name, kind = job
    inst = make_instance(SHEETS[name], 65, SEED)
    bars, s = loop_bars(inst.points, kind)
    uv = paper(inst)
    gap = true_gap(inst)
    lived = long_lived(bars)
    walk = surface_walk(inst, np.array([b[2] for b in lived]), factor=2) if lived else []
    rows = []
    for (birth, death, (a, b)), w in zip(lived, walk):
        rows.append({"birth": birth, "death": death if np.isfinite(death) else None, "walk": float(w),
                     "paper_dist_of_birth_link": float(np.linalg.norm(uv[a] - uv[b])), "ends": [int(a), int(b)]})
    return name, kind, {"spacing": s, "true_gap": None if gap is None else gap / spacing(inst.points), "bars": rows}


def truths(name):
    """Truth per 10-nearest link: paper jump, and walk-along-the-real-sheet / straight length."""
    inst = make_instance(SHEETS[name], 65, SEED)
    links = knn_links(inst.points)
    paper_jump = jump_mask(inst, links)
    walk = surface_walk(inst, links, factor=10)
    np.savez_compressed(OUT / f"truth_{name}.npz", links=links, walk=walk, paper_jump=paper_jump)
    return name, {"paper_jumps": int(paper_jump.sum()),
                  "walk_hist_edges": WALK_BINS, "walk_hist": np.histogram(walk, WALK_BINS)[0].tolist(),
                  "shortcuts_over_2": int((walk > 2).sum()), "dots_touched_over_2": int(len(np.unique(links[walk > 2, 0])))}


WALK_BINS = [0, 1.05, 1.2, 1.5, 2, 3, 5, 11.1]


def load_truth(name, inst=None):
    t = np.load(OUT / f"truth_{name}.npz")
    return t["links"], t["walk"], t["paper_jump"]


# ------------------------------------------------------------------ the detectors as a judge
def long_lived(bars):
    return [b for b in bars if not np.isfinite(b[1]) or b[1] / b[0] > 2]


def fold_readout(Y):
    """Everything the detectors say about a cloud, using nothing but the cloud."""
    out = {}
    for kind in ("knn", "local"):
        lived = long_lived(loop_bars(Y, kind)[0])
        out[f"loops_{kind}"] = len(lived)
        out[f"first_birth_{kind}"] = min((b[0] for b in lived), default=None)
    _, score, _, stacked = blob_split(Y)
    out["split_dots"] = int((score > 1.6).sum())
    out["split_stacked_dots"] = int(((score > 1.6) & (stacked > 0.7)).sum())
    return out


def transforms(points):
    from unfolding_suite.experiments.pull_apart import pull, stretch_along
    from unfolding_suite.link_check import checked_links
    yield "as given", points
    for rank, f, label in [(2, 2, "thinnest axis x2"), (2, 4, "thinnest axis x4"), (0, 2, "longest axis x2")]:
        yield label, stretch_along(points, rank, f)
    yield "pulled apart (checked strings)", pull(points, checked_links(points, 10, 12, 0.5))[0]


JUDGE = ["flat", "uneven_roll", "s_curl", "h_roll5", "tight_roll", "crumple", "h_crumple29"]


def judge(name):
    inst = make_instance(SHEETS[name], 65, SEED)
    rows = []
    for label, Y in transforms(inst.points):
        links = knn_links(Y)
        walk = surface_walk(inst, links, factor=2)
        rows.append({"sheet": name, "transform": label, "true_shortcuts": int((walk > 2).sum()),
                     "true_shortcut_dots": int(len(np.unique(links[walk > 2, 0]))),
                     "paper_jumps": int(jump_mask(inst, links).sum()), **fold_readout(Y)})
    return rows


# ------------------------------------------------------------------ the detectors as flattening methods
def cut_loop_links(points, start_links, scaled=False, rounds=60):
    """Repeatedly find loops that never fill in and delete the link that closes each one."""
    import gudhi
    links = np.unique(np.sort(start_links, 1), axis=0)
    rho = local_spacing(points) if scaled else None
    removed = 0
    for _ in range(rounds):
        length = np.linalg.norm(points[links[:, 0]] - points[links[:, 1]], axis=1)
        if scaled:
            length = length / np.sqrt(rho[links[:, 0]] * rho[links[:, 1]])
        st = gudhi.SimplexTree()
        for v in range(len(points)):
            st.insert([v], 0.0)
        st.insert_batch(links.T, length)
        st.expansion(2)
        st.compute_persistence(persistence_dim_max=True)
        doors = set()
        for b, d in st.persistence_pairs():
            if len(b) != 2:
                continue
            birth = st.filtration(b)
            if not len(d) or st.filtration(d) / birth > 2:
                doors.add(tuple(sorted(b)))
        if not doors:
            break
        keep = np.array([tuple(l) not in doors for l in links.tolist()])
        removed += int((~keep).sum())
        links = links[keep]
    return links, removed


def blob_links(points, k=10, threshold=1.6, need_stacked=True):
    near, score, mine, stacked = blob_split(points)
    flag = (score > threshold) & ((stacked > 0.7) if need_stacked else True)
    src, dst = [], []
    for i in range(len(points)):
        nb = near[i][1:][mine[i][1:]] if flag[i] else near[i][1:]
        nb = nb[:k]
        src.append(np.full(len(nb), i))
        dst.append(nb)
    return np.c_[np.concatenate(src), np.concatenate(dst)]


def register_methods():
    from unfolding_suite.link_check import checked_links, reattach_stranded
    from unfolding_suite.methods import layout_from_links, method

    def wrap(make):
        def fn(points, dim, rng):
            links = reattach_stranded(points, make(points))
            flat, captured = layout_from_links(points, links, dim)
            return flat, {"edges": links, "captured": captured}
        return fn

    method("loopcut_knn")(wrap(lambda P: cut_loop_links(P, knn_links(P))[0]))
    method("loopcut_knn_local")(wrap(lambda P: cut_loop_links(P, knn_links(P), scaled=True)[0]))
    method("loopcut_checked")(wrap(lambda P: cut_loop_links(P, checked_links(P, 10, 12, 0.5))[0]))
    method("blob_k10")(wrap(lambda P: blob_links(P, need_stacked=False)))
    method("blob_stacked_k10")(wrap(lambda P: blob_links(P)))


FLATTEN = ["loopcut_knn", "loopcut_knn_local", "loopcut_checked", "blob_k10", "blob_stacked_k10"]


def flatten_one(job):
    import time
    from unfolding_suite.methods import METHODS
    from unfolding_suite.run import failure_words
    from unfolding_suite.scoring import score_layout, to_grid_order
    register_methods()
    sheet_name, method_name, seed = job
    inst = make_instance(SHEETS[sheet_name], 65, seed)
    start = time.time()
    try:
        flat, info = METHODS[method_name](inst.points.copy(), 2, np.random.default_rng(seed + 1000))
        r = score_layout(to_grid_order(flat, inst), inst.uv, inst.xyz, inst.grid, inst.sheet.family)
        r["jumps"] = int(jump_mask(inst, info["edges"]).sum())
        r["links"] = len(info["edges"])
    except Exception as e:                                                   # noqa: BLE001
        r = {"passed": False, "verdict": "crashed", "error": f"{type(e).__name__}: {e}"}
    keep = {k: r.get(k) for k in ("passed", "verdict", "fold_share", "overlap_share", "square_error_median",
                                  "roughness", "tear_share", "crush_share", "jumps", "links", "error")}
    keep["words"] = failure_words(r)
    keep["seconds"] = time.time() - start
    return job, keep


# ------------------------------------------------------------------ flat flap vs. loop that touches in one strip
SHAPE_GAPS = (0.005, 0.01, 0.02, 0.05, 0.10)


def shape_sheets():
    """The flap (folded flat over the base, close everywhere) and the loop (touches the base in one
    narrow strip) from experiments/fold_flatten.py, at more gaps."""
    from unfolding_suite.experiments.fold_flatten import bend_with, flap_curvature, loop_curvature
    from unfolding_suite.sheets import sheet
    names = []
    for g in SHAPE_GAPS:
        for kind, curvature in (("flap", flap_curvature), ("loop", loop_curvature)):
            name = f"{kind}_g{g}"
            if name not in SHEETS:
                sheet(name, "bent", "flap/loop check")(bend_with(curvature(g)))
            names.append(name)
    return names


def shapes(job):
    name, seed = job
    shape_sheets()
    inst = make_instance(SHEETS[name], 65, seed)
    links = np.unique(np.sort(knn_links(inst.points), 1), axis=0)
    wrong = jump_mask(inst, links)
    bars, s = loop_bars(inst.points, "knn")
    lived = long_lived(bars)
    on_wrong = [b for b in lived if jump_mask(inst, np.array([b[2]]))[0]]
    short = [b for b in bars if np.isfinite(b[1])]
    links_after, cuts = cut_loop_links(inst.points, links, rounds=400)
    return job, {"gap_spacings": float(name.split("_g")[1]) / s, "wrong": int(wrong.sum()),
                 "lasting": len(lived), "lasting_on_wrong": len(on_wrong),
                 "first_birth": min((b[0] for b in lived), default=None),
                 "short_loops": len(short),
                 "short_birth_death_median": [float(np.median([b[0] for b in short])), float(np.median([b[1] for b in short]))],
                 "cuts": cuts, "wrong_after_cutting": int(jump_mask(inst, links_after).sum())}


def fmt(x):
    return "  -  " if x is None else f"{100 * x:5.1f}%"


ALL = list(SHEETS)
KINDS = (sys.argv[2:] if sys.argv[1:2] == ['loops'] else []) or ["rips", "knn", "local"]

if __name__ == "__main__":
    part = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(4) as pool:
        if part == "blobs":
            res = dict(pool.map(blobs, ALL))
            for name, r in res.items():
                print(f"{name:14s} touched {r['touched']:5d} in reach {r['in_reach']:5d} | clean score p50/p99/p99.9/max "
                      f"{r['score_clean']} | touched p5/p25/p50 {r['score_touched']}")
                print("      " + "  ".join(f"{k}: found {fmt(v['found'])} false {fmt(v['false_alarm'])}"
                                         for k, v in r["detect"].items()))
            (OUT / "blobs.json").write_text(json.dumps(res))
        elif part == "truths":
            res = dict(pool.imap(truths, ALL))
            for name, r in res.items():
                print(f"{name:14s} paper jumps {r['paper_jumps']:5d} | walk/straight hist {WALK_BINS}: {r['walk_hist']}"
                      f" | over 2: {r['shortcuts_over_2']} links, {r['dots_touched_over_2']} dots", flush=True)
            (OUT / "truths.json").write_text(json.dumps(res, indent=1))
        elif part == "judge":
            rows = [r for rs in pool.imap(judge, JUDGE) for r in rs]
            for r in rows:
                print(f"{r['sheet']:12s} {r['transform']:30s} TRUE shortcuts {r['true_shortcuts']:5d} ({r['true_shortcut_dots']:4d} dots)"
                      f" | loops knn {r['loops_knn']:4d} local {r['loops_local']:4d} | split dots {r['split_dots']:4d}"
                      f" stacked {r['split_stacked_dots']:4d}", flush=True)
            (OUT / "judge.json").write_text(json.dumps(rows, indent=1))
        elif part == "flatten":
            methods = sys.argv[2:] or FLATTEN
            jobs = [(s_, m, seed) for s_ in ALL for m in methods for seed in (100, 101, 102)]
            res = {}
            for (s_, m, seed), r in pool.imap_unordered(flatten_one, jobs):
                res.setdefault(f"{s_} / {m}", {})[str(seed)] = r
                print(f"{'PASS' if r['passed'] else 'FAIL'} {s_:14s} {m:18s} {seed} jumps {r['jumps']} {r['words']}", flush=True)
            path = OUT / "flatten.json"
            old = json.loads(path.read_text()) if path.exists() else {}
            old.update(res)
            path.write_text(json.dumps(old, indent=1))
        elif part == "shapes":
            jobs = [(n, seed) for n in shape_sheets() for seed in (100, 101, 102)]
            res = {}
            for (n, seed), r in pool.imap(shapes, jobs):
                res.setdefault(n, {})[str(seed)] = r
                print(f"{n:12s} seed {seed} gap {r['gap_spacings']:.2f} sp | wrong {r['wrong']:5d} | lasting loops {r['lasting']:3d}"
                      f" ({r['lasting_on_wrong']} on wrong) first born {r['first_birth']} | small loops {r['short_loops']}"
                      f" born/filled {np.round(r['short_birth_death_median'], 2)} | cutting: {r['cuts']} cut,"
                      f" {r['wrong_after_cutting']} wrong left", flush=True)
            (OUT / "shapes.json").write_text(json.dumps(res, indent=1))
        elif part == "loops":
            res = {}
            for name, kind, r in pool.imap(loops, [(n, k) for n in ALL for k in KINDS]):
                res.setdefault(name, {})[kind] = r
                lived = r["bars"]
                jumpy = sum(b["paper_dist_of_birth_link"] > FAR for b in lived)
                real = sum(b["walk"] > 2 for b in lived)
                truth = json.loads((OUT / "truths.json").read_text())[name]
                first = min((b["birth"] for b in lived), default=None)
                print(f"{name:14s} {kind:5s} true shortcut links {truth['shortcuts_over_2']:5d} | long-lived loops {len(lived):4d},"
                      f" born on a real shortcut {real:4d}, on a paper jump {jumpy:4d} | first born at"
                      f" {'-' if first is None else f'{first:.2f}'} spacings, true gap {r['true_gap'] or 0:.2f}", flush=True)
            path = OUT / "loops.json"
            old = json.loads(path.read_text()) if path.exists() else {}
            for name, kinds in res.items():
                old.setdefault(name, {}).update(kinds)
            path.write_text(json.dumps(old))
