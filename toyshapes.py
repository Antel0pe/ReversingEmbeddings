"""A playground of small shapes with KNOWN answers, hidden in high dimensions.

Each shape is built in 2-D or 3-D where you can see it, then rotated into 50-D so
the measuring tools get no help from the picture. The point is to check a method
against a shape whose dimension, holes and singular spots you already know,
before trusting it on something like MNIST.

    swiss roll      a 2-D sheet, rolled up            dim 2, no holes  (PCA overstates it)
    sphere          a 2-D closed surface              dim 2, a void inside
    torus           a 2-D surface with 2 loops        dim 2, 2 loops
    drain           a funnel narrowing to a hole      dim 2, 1 loop, width shrinks
    cone            a 2-D sheet with a sharp tip      dim 2 except AT the tip
    crossing        two flat sheets meeting on a line dim 2 except ON the line

Tools (all work in any number of dimensions):
    local_dim       how many directions a small neighbourhood spreads along
    dim_vs_radius   the same, as the neighbourhood grows -- the fingerprint
    mapper          a stick-figure of the shape: cut it into overlapping slices,
                    cluster inside each slice, join clusters that share points
"""

import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


def swiss_roll(n, rng):
    t = 1.5 * np.pi * (1 + 2 * rng.random(n)); h = 21 * rng.random(n)
    return np.c_[t * np.cos(t), h, t * np.sin(t)] / 6


def sphere(n, rng):
    v = rng.normal(size=(n, 3)); return 2.4 * v / np.linalg.norm(v, axis=1, keepdims=True)


def torus(n, rng):
    u, v = 2 * np.pi * rng.random(n), 2 * np.pi * rng.random(n)
    R, r = 2.0, 0.8
    return np.c_[(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)]


def drain(n, rng):
    """A funnel: a surface of revolution that narrows as it goes down, with a hole."""
    rad = np.sqrt(rng.uniform(0.8 ** 2, 2.6 ** 2, n)); th = 2 * np.pi * rng.random(n)
    return np.c_[rad * np.cos(th), rad * np.sin(th), -1.6 / rad]


def cone(n, rng):
    rad = np.sqrt(rng.random(n)) * 2.2; th = 2 * np.pi * rng.random(n)
    return np.c_[rad * np.cos(th), rad * np.sin(th), 1.6 * rad]


def crossing(n, rng):
    a = rng.uniform(-2, 2, (n // 2, 2)); b = rng.uniform(-2, 2, (n - n // 2, 2))
    P1 = np.c_[a[:, 0], a[:, 1], np.zeros(len(a))]
    P2 = np.c_[b[:, 0], np.zeros(len(b)), b[:, 1]]
    return np.vstack([P1, P2])


SHAPES = {"swiss roll": swiss_roll, "sphere": sphere, "torus": torus,
          "drain": drain, "cone": cone, "crossing sheets": crossing}
TRUTH = {"swiss roll": "2-D, no holes", "sphere": "2-D, no loops (a void inside)",
         "torus": "2-D, 2 loops", "drain": "2-D, 1 loop, narrowing",
         "cone": "2-D, but the tip is singular", "crossing sheets": "2-D, but the join line is singular"}


def embed(X, dim, rng, noise=0.02):
    """Hide the shape: rotate it into `dim` dimensions and add a little noise."""
    Q = np.linalg.qr(rng.normal(size=(dim, dim)))[0][:, :X.shape[1]]
    return X @ Q.T + noise * rng.normal(size=(len(X), dim))


def local_dim(D, idx, k=60, floor=0.05):
    """Directions a neighbourhood spreads along: eigenvalues above `floor` of the largest."""
    out = []
    for a in idx:
        Z = D[np.argsort(((D - D[a]) ** 2).sum(1))[:k]]
        Z = Z - Z.mean(0)
        lam = np.linalg.eigvalsh(Z.T @ Z)[::-1].clip(0)
        out.append(int((lam > floor * lam[0]).sum()))
    return np.array(out)


def dim_vs_radius(D, ks, n_anchor=60, rng=None):
    rng = rng or np.random.default_rng(0)
    idx = rng.choice(len(D), n_anchor, replace=False)
    return {k: float(np.median(local_dim(D, idx, k=k))) for k in ks}


def mapper(D, bins=10, overlap=0.4, link=2.5):
    """Stick-figure of the shape. Slice along the widest direction, cluster inside
    each slice, and join clusters that share a point. Loops in the result are loops
    in the shape."""
    Z = D - D.mean(0)
    lens = Z @ np.linalg.eigh(Z.T @ Z)[1][:, -1]
    nn = np.median(np.sort(np.sqrt(np.maximum(((D[:, None, :] - D[None, :50, :]) ** 2).sum(-1), 0)), axis=1)[:, 1])
    lo, hi = lens.min(), lens.max(); w = (hi - lo) / bins / (1 - overlap)
    nodes, members = [], []
    for b in range(bins):
        c = lo + (hi - lo) * (b + .5) / bins
        m = np.where((lens >= c - w / 2) & (lens <= c + w / 2))[0]
        if len(m) < 3: continue
        lab = fcluster(linkage(pdist(D[m]), "single"), link * nn, criterion="distance")
        for c_ in np.unique(lab):
            members.append(set(m[lab == c_].tolist())); nodes.append((b, len(members[-1]), lens[m[lab == c_]].mean()))
    g = nx.Graph()
    for i, (b, sz, lv) in enumerate(nodes): g.add_node(i, size=sz, lens=lv)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if members[i] & members[j]: g.add_edge(i, j)
    loops = g.number_of_edges() - g.number_of_nodes() + nx.number_connected_components(g)
    return g, loops


def betti01(D, radius):
    """Pieces (H0) and loops (H1) at a connection radius, by filling triangles.
    Same routine as Part VIII of the notebook: cycles of the graph, minus the ones
    that triangles fill in. A loop that survives is a real hole."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(D)
    d = np.sqrt(np.maximum(((D ** 2).sum(1)[:, None] + (D ** 2).sum(1)[None, :] - 2 * D @ D.T), 0))
    A = (d <= radius) & (~np.eye(n, dtype=bool))
    e1, e2 = np.where(np.triu(A, 1)); E = len(e1)
    nc = connected_components(csr_matrix(A), directed=False)[0]
    cycles = E - n + nc
    eid = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(e1, e2))}
    piv, rank = {}, 0
    for r in range(E):
        i, j = int(e1[r]), int(e2[r])
        for k in np.where(A[i] & A[j])[0]:
            k = int(k)
            if k <= j: continue
            c = (1 << eid[(i, j)]) | (1 << eid[(i, k)]) | (1 << eid[(j, k)])
            while c:
                lo = c.bit_length() - 1
                if lo in piv: c ^= piv[lo]
                else: piv[lo] = c; rank += 1; break
    return nc, cycles - rank


def nn_scale(D):
    d = np.sqrt(np.maximum(((D ** 2).sum(1)[:, None] + (D ** 2).sum(1)[None, :] - 2 * D @ D.T), 0))
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(1)))
