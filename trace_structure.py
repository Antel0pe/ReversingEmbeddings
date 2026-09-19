"""Recover the knobs of a complete on/off point cloud from the coordinates alone.

No generator, no pixel grid, no labels: just a set of 0/1 vectors that is known
to be complete. Three steps:

  1. STEPS. Join x and y when (a) they share at least one lit coordinate and
     (b) no third member is closer to both of them than they are to each other
     (the relative-neighbourhood rule). The data decides what one step is;
     there is no distance threshold.

     (a) is needed because distance SATURATES in pixel space: two strokes that
     do not overlap are always the same distance apart, however far they are
     moved, so without it far-apart pairs look like single steps.
  2. SQUARES. If move a then move b lands where b then a does, the two moves
     commute: they are independent knobs. The largest set of mutually commuting
     moves at a point is its local dimension.
  3. KNOBS. Opposite sides of a square are the same move made somewhere else.
     Merge those, then merge moves that continue each other in a straight line
     (the only shortest route) unless they are ever seen commuting -- commuting
     moves are different knobs by definition. What is left is one group per knob.
"""

import itertools
from collections import defaultdict
import numpy as np


class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def elementary_graph(V, cap):
    """Steps: pairs that share ink and have no member closer to both (Hamming <= cap)."""
    X = V.astype(np.float32); n = len(V); sq = X.sum(1)
    edges = set()
    for s in range(0, n, 1024):
        d = sq[s:s + 1024, None] + sq[None, :] - 2 * X[s:s + 1024] @ X.T
        share = (X[s:s + 1024] @ X.T) > .5
        for k in range(len(d)):
            i = s + k; row = d[k]
            J = np.where((row <= cap + .5) & share[k] & (np.arange(n) != i))[0]
            if not len(J): continue
            dJ = row[J]
            DJ = sq[J, None] + sq[None, J] - 2 * X[J] @ X[J].T      # distances among candidates
            for t, y in enumerate(J):
                if y < i: continue
                closer = (dJ < dJ[t] - .5) & (DJ[:, t] < dJ[t] - .5)
                if not closer.any():
                    edges.add((i, int(y)))
    return sorted(edges)


def analyse(V, cap):
    n = len(V)
    edges = elementary_graph(V, cap)
    eid = {e: k for k, e in enumerate(edges)}
    adj = defaultdict(set)
    for a, b in edges: adj[a].add(b); adj[b].add(a)
    E = lambda a, b: eid[(min(a, b), max(a, b))]
    ham = np.array([(V[a] != V[b]).sum() for a, b in edges])

    # squares: x-y-w-z-x with the diagonals not joined
    squares = []
    commute_at = defaultdict(set)
    for x in range(n):
        for y, z in itertools.combinations(sorted(adj[x]), 2):
            if z in adj[y]: continue
            for w in (adj[y] & adj[z]) - {x}:
                if w in adj[x]: continue
                if x < w:
                    squares.append((E(x, y), E(z, w), E(x, z), E(y, w)))
                commute_at[x].add((E(x, y), E(x, z)))
    uf = UF(len(edges))
    for a, b, c, d in squares: uf.union(a, b); uf.union(c, d)

    # classes that commute somewhere are different knobs
    comm = set()
    for a, b, c, d in squares:
        ca, cc = uf.find(a), uf.find(c)
        comm.add((min(ca, cc), max(ca, cc)))
    # straight continuation: y - x - z where x is the ONLY shortest route from y to z
    merged = 0
    for x in range(n):
        for y, z in itertools.combinations(sorted(adj[x]), 2):
            if z in adj[y]: continue
            if len(adj[y] & adj[z]) != 1: continue
            a, b = uf.find(E(x, y)), uf.find(E(x, z))
            if a != b and (min(a, b), max(a, b)) not in comm:
                uf.union(a, b); merged += 1
                comm = {(min(uf.find(p), uf.find(q)), max(uf.find(p), uf.find(q))) for p, q in comm}
    cls = np.array([uf.find(k) for k in range(len(edges))])

    # local dimension: largest set of mutually commuting moves at each point
    import networkx as nx
    dim = np.zeros(n, int)
    for x in range(n):
        g = nx.Graph(); g.add_nodes_from(E(x, y) for y in adj[x]); g.add_edges_from(commute_at[x])
        dim[x] = max((len(c) for c in nx.find_cliques(g)), default=0) if len(g) else 0
    return dict(edges=edges, adj=adj, ham=ham, squares=squares, cls=cls, dim=dim)


# ---------------------------------------------------------------------------
# Lattice axes: the minimum number of integer knobs (Eppstein 2005)
# ---------------------------------------------------------------------------
#
# A graph is a PARTIAL CUBE when every point can be given 0/1 coordinates whose
# Hamming distance equals the number of steps between points. Its steps then fall
# into CUTS: each cut is one yes/no threshold ("is the bar's left edge past column
# 4?"), and every step crosses exactly one cut. Cuts belonging to one knob are
# nested -- a chain of thresholds along a line. Eppstein: the fewest integer axes
# needed is (#cuts) - (maximum matching between cuts that can be chained).
# Each chain is one traced line of knob settings, and the position along it is
# that knob's value.

def lattice_axes(V, edges):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path, connected_components
    import networkx as nx
    n = len(V); a, b = np.array(edges).T
    G = csr_matrix((np.ones(len(a)), (a, b)), shape=(n, n)); G = G + G.T
    ncomp = connected_components(G, directed=False)[0]
    cls = -np.ones(len(edges), int); sides = []; ok = ncomp == 1; why = "" if ok else "not connected"
    for e, (u, v) in enumerate(edges):
        if cls[e] >= 0 or not ok: continue
        d = shortest_path(G, unweighted=True, indices=[u, v])
        if (d[0] == d[1]).any():
            ok, why = False, "odd cycle: some point is equally far from both ends of a step"; break
        side = d[0] < d[1]
        cross = np.where(side[a] != side[b])[0]
        if (cls[cross] >= 0).any():
            ok, why = False, "a step belongs to two different cuts"; break
        cls[cross] = len(sides); sides.append(side)
    out = dict(partial_cube=ok, why=why, cuts=len(sides), edge_cut=cls)
    if not ok:
        return out
    S = np.array(sides)                                   # cut i: S[i] = one side
    semis = np.vstack([S, ~S])                            # semicube 2i+0 / 2i+1 interleaved below
    k = len(S)
    comp = ~semis                                         # complement of each semicube
    inter = comp.astype(np.float32) @ comp.T.astype(np.float32)
    g = nx.Graph()
    for p in range(2 * k):
        for q in range(p + 1, 2 * k):
            if p % k != q % k and inter[p, q] < .5:       # different cuts, union = everything
                g.add_edge(p, q)
    M = nx.max_weight_matching(g, maxcardinality=True)
    link = {}
    for p, q in M: link[p] = q; link[q] = p
    # walk chains of cuts; orient so the "positive" sides are nested
    seen = set(); axes = []
    for start in range(k):
        if start in seen: continue
        # find an end of this chain: a cut with at most one linked semicube
        chain_cut = start; prev = None
        while True:
            nxt = [link[s] % k for s in (chain_cut, chain_cut + k) if s in link and link[s] % k != prev]
            if not nxt or nxt[0] == start: break
            prev, chain_cut = chain_cut, nxt[0]
            if chain_cut == start: break
        end = chain_cut
        # walk from the end
        order, pos_side = [], []
        cur, came_from_semi = end, None
        while True:
            seen.add(cur); order.append(cur)
            s0, s1 = cur, cur + k
            if came_from_semi is None:
                out_semi = s0 if s0 in link else (s1 if s1 in link else None)
                pos_side.append(~semis[out_semi] if out_semi is not None else semis[s0])
            else:
                pos_side.append(semis[came_from_semi])
                out_semi = s1 if came_from_semi == s0 else s0
                if out_semi not in link: out_semi = None
            if out_semi is None: break
            nxt_semi = link[out_semi]; nxt = nxt_semi % k
            if nxt in seen: break
            cur, came_from_semi = nxt, nxt_semi
        coord = np.sum(pos_side, axis=0)
        axes.append(dict(cuts=order, coord=coord))
    # verify: do the recovered integer coordinates reproduce every step distance?
    C = np.array([ax["coord"] for ax in axes]).T
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(n, 300), replace=False)
    dg = shortest_path(G, unweighted=True, indices=idx)
    l1 = np.abs(C[idx][:, None, :] - C[None, :, :]).sum(2)
    out.update(axes=axes, coords=C, matching=len(M), isometric=bool(np.allclose(dg, l1)))
    return out
