"""The best simple unfolding built from this week's ideas (2026-11-15). Works in any dimension.

    1. Neighbours: each dot's k closest dots.
    2. Tilt check: fit the sheet's flat directions at every dot (its `dim` main directions among its
       12 closest dots). Drop a connection if more than half of its length leaves those directions
       at either end (30 degrees out of the surface, in 3-D). A true neighbour runs along the sheet.
    3. Facing check ("which way the paper faces", generalised): give every dot's flat directions a
       consistent handedness by carrying it outward along the most-parallel connections first. Drop
       a connection whose two ends are mirror images of each other (negative overlap determinant).
       Paper folded back over itself faces the other way; bending and rolling never flip it.
       In 3-D this is exactly the up-side check; in more dimensions there is no single "up", but
       handedness still exists.
    4. Walk along the connections that are left and lay the dots out to keep those walking distances
       (Isomap). Above LANDMARK_ABOVE dots, walks are measured only from a sample of landmark dots and
       every other dot is placed from its distances to them, so memory stays small.

Nothing here assumes 3-D, a square outline, or even spacing.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components, dijkstra, minimum_spanning_tree
from scipy.sparse.linalg import eigsh, splu
from scipy.spatial import cKDTree

from .link_check import reattach_stranded
from .methods import layout_from_links, method

PLANE_K = 12
LIMIT = 0.5                 # share of a connection's length allowed outside the local flat directions
LANDMARK_ABOVE = 9000
LANDMARKS = 1500


def nearest_links(points, k):
    _, near = cKDTree(points).query(points, k=k + 1)
    links = np.c_[np.repeat(np.arange(len(points)), k), near[:, 1:].ravel()]
    return np.unique(np.sort(links, 1), axis=0)


def flat_directions(points, dim, k=PLANE_K):
    """(n, D, dim): each dot's `dim` main directions among its k closest dots."""
    _, near = cKDTree(points).query(points, k=k + 1)
    patch = points[near]
    return np.transpose(np.linalg.svd(patch - patch.mean(1, keepdims=True), full_matrices=False)[2][:, :dim], (0, 2, 1))


def tilt_check(points, links, frames, limit=LIMIT):
    step = points[links[:, 1]] - points[links[:, 0]]
    step /= np.linalg.norm(step, axis=1, keepdims=True)

    def outside(f):
        inside = np.einsum("nDd,nD->nd", f, step)
        return np.sqrt(np.maximum(1 - (inside ** 2).sum(1), 0))

    return np.maximum(outside(frames[links[:, 0]]), outside(frames[links[:, 1]])) <= limit


def overlap_det(frames, links):
    return np.linalg.det(np.einsum("nDa,nDb->nab", frames[links[:, 0]], frames[links[:, 1]]))


def consistent_handedness(frames, links):
    """Flip dots' last direction so that connected dots agree, most-parallel connections first."""
    n = len(frames)
    agree = np.abs(overlap_det(frames, links))
    tree = minimum_spanning_tree(csr_matrix((1.0 - agree + 1e-6, (links[:, 0], links[:, 1])), shape=(n, n)))
    tree = tree + tree.T
    frames = frames.copy()
    done = np.zeros(n, bool)
    for root in range(n):
        if done[root]:
            continue
        order, parent = breadth_first_order(tree, root, directed=False)
        done[order] = True
        for i in order[1:]:
            if np.linalg.det(frames[parent[i]].T @ frames[i]) < 0:
                frames[i, :, -1] *= -1
    return frames


def choose_links(points, dim, k=10, tilt=True, facing=True):
    links = nearest_links(points, k)
    frames = flat_directions(points, dim)
    if tilt:
        links = links[tilt_check(points, links, frames)]
    if facing:
        frames = consistent_handedness(frames, links)
        links = links[overlap_det(frames, links) >= 0]
    return np.unique(np.sort(reattach_stranded(points, links), 1), axis=0)


def landmark_layout(points, links, dim, rng, m=LANDMARKS):
    """Walking distances from m landmark dots only; every dot placed from its distances to them."""
    n = len(points)
    lengths = np.linalg.norm(points[links[:, 0]] - points[links[:, 1]], axis=1)
    graph = csr_matrix((lengths, (links[:, 0], links[:, 1])), shape=(n, n))
    if connected_components(graph, directed=False)[0] > 1:
        raise RuntimeError("the neighbour graph broke into pieces")
    marks = rng.choice(n, m, replace=False)
    D = dijkstra(graph, directed=False, indices=marks) ** 2          # (m, n) squared walking distances
    Dm = D[:, marks]
    centred = -0.5 * (Dm - Dm.mean(0) - Dm.mean(1)[:, None] + Dm.mean())
    values, vectors = eigsh(centred, k=dim, which="LA", v0=np.random.default_rng(0).standard_normal(m))
    pinv = vectors / np.sqrt(np.maximum(values, 1e-12))
    return -0.5 * ((D - Dm.mean(1)[:, None]).T @ pinv)


def carry_vote_layout(points, links, dim):
    """The user's "carry two directions" idea, generalised: carry the flat directions outward from a
    central dot along the most-parallel connections, turning them as little as possible into each dot's
    own flat directions; write every connection's step in the carried directions; then place all dots
    at once so that every connection's step is matched as well as possible (each connection votes)."""
    n = len(points)
    frames = consistent_handedness(flat_directions(points, dim), links)
    agree = np.abs(overlap_det(frames, links))
    tree = minimum_spanning_tree(csr_matrix((1.0 - agree + 1e-6, (links[:, 0], links[:, 1])), shape=(n, n)))
    tree = tree + tree.T
    hops = dijkstra(tree, directed=False, indices=0, unweighted=True)
    a = int(np.argmax(np.where(np.isfinite(hops), hops, -1)))
    da = dijkstra(tree, directed=False, indices=a, unweighted=True)
    b = int(np.argmax(np.where(np.isfinite(da), da, -1)))
    db = dijkstra(tree, directed=False, indices=b, unweighted=True)
    root = int(np.argmin(np.maximum(da, db)))
    order, parent = breadth_first_order(tree, root, directed=False)
    E = frames.copy()
    for c in order[1:]:
        u, _, vt = np.linalg.svd(frames[c].T @ E[parent[c]])
        R = u @ vt
        if np.linalg.det(R) < 0:                         # turn only, never mirror
            u[:, -1] *= -1
            R = u @ vt
        E[c] = frames[c] @ R
    i, j = links.T
    d = points[j] - points[i]
    steps = 0.5 * (np.einsum("mD,mDc->mc", d, E[i]) + np.einsum("mD,mDc->mc", d, E[j]))
    steps *= (np.linalg.norm(d, axis=1) / np.maximum(np.linalg.norm(steps, axis=1), 1e-12))[:, None]
    m = len(links)
    A = csr_matrix((np.r_[-np.ones(m), np.ones(m)], (np.r_[np.arange(m), np.arange(m)], np.r_[i, j])), shape=(m, n))
    keep = np.ones(n, bool)
    keep[root] = False
    Ak = A[:, keep].tocsc()
    solve = splu((Ak.T @ Ak).tocsc()).solve
    flat = np.zeros((n, dim))
    for c in range(dim):
        flat[keep, c] = solve(Ak.T @ steps[:, c])
    return flat


def simple_unfold(points, dim, rng, k=10, tilt=True, facing=True, layout="walk"):
    """Closest dots, minus tilted and mirror-facing connections, then walk (any dimension)."""
    links = choose_links(points, dim, k, tilt, facing)
    if layout == "vote":
        return carry_vote_layout(points, links, dim), {"edges": links}
    if len(points) > LANDMARK_ABOVE:
        return landmark_layout(points, links, dim, rng), {"edges": links}
    flat, captured = layout_from_links(points, links, dim)
    return flat, {"edges": links, "captured": captured}


method("simple_unfold", k=10)(simple_unfold)
method("simple_tilt_only", k=10, facing=False)(simple_unfold)
method("simple_facing_only", k=10, tilt=False)(simple_unfold)
method("simple_vote", k=10, layout="vote")(simple_unfold)
method("simple_vote_no_checks", k=10, layout="vote", tilt=False, facing=False)(simple_unfold)
method("simple_vote_tilt_only", k=10, layout="vote", facing=False)(simple_unfold)
method("simple_vote_facing_only", k=10, layout="vote", tilt=False)(simple_unfold)
# The recommended method (2026-11-15): the facing check changed no pass or fail on any of the 32 sheets
# once the tilt check was in, so it is left out. Handedness is still used inside the vote layout, to
# carry the two directions without mirroring them.
method("best_simple", k=10, layout="vote", facing=False)(simple_unfold)
