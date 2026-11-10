"""Two simple methods that survived testing. Both start from the same checked links.

Link check: a link is only kept if it lies along the sheet at both of its ends. A true link
runs along the surface; a link that jumps to another layer leaves it at one end or the other.
The surface direction at a dot comes from a plane fitted to its nearest dots.

check_k10    Walking distances along the checked links, then the usual flat layout.
convex_edge  Lay out only the edge dots, push them onto a convex outline, and put each inside
             dot at the average of its neighbours. No all-pairs step, so it scales to many dots.
"""
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import splu
from scipy.spatial import ConvexHull, cKDTree

from .methods import layout_from_links, method


def checked_links(points, k, plane_k, limit):
    """k nearest links, minus any that leave the fitted surface at either end."""
    _, near = cKDTree(points).query(points, k=max(k, plane_k) + 1)
    patch = points[near[:, :plane_k + 1]]
    normals = np.linalg.svd(patch - patch.mean(1, keepdims=True), full_matrices=False)[2][:, 2]
    src = np.repeat(np.arange(len(points)), k)
    dst = near[:, 1:k + 1].ravel()
    step = points[dst] - points[src]
    step /= np.linalg.norm(step, axis=1, keepdims=True)
    tilt = np.maximum(np.abs((step * normals[src]).sum(1)), np.abs((step * normals[dst]).sum(1)))
    return reattach_stranded(points, np.c_[src, dst][tilt <= limit])


def reattach_stranded(points, links):
    """The check can cut off a few dots completely. Join each to the nearest dot of the main sheet."""
    n = len(points)
    labels = connected_components(csr_matrix((np.ones(len(links)), (links[:, 0], links[:, 1])), shape=(n, n)),
                                  directed=False)[1]
    main = labels == np.bincount(labels).argmax()
    if main.all():
        return links
    stranded, kept = np.where(~main)[0], np.where(main)[0]
    return np.r_[links, np.c_[stranded, kept[cKDTree(points[main]).query(points[stranded])[1]]]]


def walking_with_check(points, dim, rng, k=10, plane_k=12, limit=0.5):
    """Walking along checked links only, then the usual flat layout."""
    links = checked_links(points, k, plane_k, limit)
    flat, captured = layout_from_links(points, links, dim)
    return flat, {"edges": links, "captured": captured}


method("check_k10", k=10)(walking_with_check)


def edge_dots(points, links, plane_k=12, gap_degrees=100.0):
    """Dots whose linked neighbours leave a wide empty wedge around them."""
    n = len(points)
    _, near = cKDTree(points).query(points, k=plane_k + 1)
    patch = points[near]
    axes = np.linalg.svd(patch - patch.mean(1, keepdims=True), full_matrices=False)[2]
    both = np.r_[links, links[:, ::-1]]
    both = both[np.argsort(both[:, 0], kind="stable")]
    starts = np.searchsorted(both[:, 0], np.arange(n + 1))
    edge = np.zeros(n, bool)
    for i in range(n):
        offsets = points[both[starts[i]:starts[i + 1], 1]] - points[i]
        if len(offsets) < 3:
            edge[i] = True
            continue
        angle = np.sort(np.arctan2(offsets @ axes[i, 1], offsets @ axes[i, 0]))
        edge[i] = np.degrees(np.diff(np.r_[angle, angle[0] + 2 * np.pi]).max()) > gap_degrees
    return edge


def average_inside(points, links, flat, edge, by_distance=False):
    """Hold the edge dots still; put every inside dot at the average of its neighbours."""
    n = len(points)
    weight = 1 / np.linalg.norm(points[links[:, 0]] - points[links[:, 1]], axis=1) if by_distance else np.ones(len(links))
    A = csr_matrix((weight, (links[:, 0], links[:, 1])), shape=(n, n))
    A = A.maximum(A.T)
    L = diags(np.asarray(A.sum(1)).ravel()) - A
    inside = np.where(~edge)[0]
    outside = np.where(edge)[0]
    solved = flat.copy()
    rhs = -L[inside][:, outside] @ flat[outside]
    solve = splu(L[inside][:, inside].tocsc()).solve
    solved[inside] = np.c_[solve(rhs[:, 0]), solve(rhs[:, 1])]
    return solved


def layout_edge(points, links, edge, dim=2):
    """Flat positions for the edge dots alone, from walking distances between them."""
    n = len(points)
    lengths = np.linalg.norm(points[links[:, 0]] - points[links[:, 1]], axis=1)
    graph = csr_matrix((lengths, (links[:, 0], links[:, 1])), shape=(n, n))
    which = np.where(edge)[0]
    d2 = dijkstra(graph, directed=False, indices=which)[:, which] ** 2
    centred = -0.5 * (d2 - d2.mean(0) - d2.mean(1)[:, None] + d2.mean())
    value, vector = np.linalg.eigh(centred)
    return vector[:, -dim:] * np.sqrt(np.maximum(value[-dim:], 0))


def onto_convex_outline(xy):
    """Push each point straight out from the middle until it reaches the convex hull."""
    centre = xy.mean(0)
    hull = xy[ConvexHull(xy).vertices] - centre
    a = np.arctan2(hull[:, 1], hull[:, 0])
    order = np.argsort(a)
    hull, a = hull[order], a[order]
    p = xy - centre
    first = (np.searchsorted(a, np.arctan2(p[:, 1], p[:, 0])) - 1) % len(a)
    h0, h1 = hull[first], hull[(first + 1) % len(a)]
    side = h1 - h0
    t = (h0[:, 0] * side[:, 1] - h0[:, 1] * side[:, 0]) / (p[:, 0] * side[:, 1] - p[:, 1] * side[:, 0])
    return centre + p * t[:, None]


def outer_ring(points):
    """Edge dots, judged from more neighbours, keeping only the largest connected group."""
    n = len(points)
    wide = checked_links(points, 20, 12, 0.5)
    edge = edge_dots(points, wide, gap_degrees=130.0)
    both_on_edge = wide[edge[wide[:, 0]] & edge[wide[:, 1]]]
    labels = connected_components(csr_matrix((np.ones(len(both_on_edge)), (both_on_edge[:, 0], both_on_edge[:, 1])),
                                             shape=(n, n)), directed=False)[1]
    return edge & (labels == np.bincount(labels[edge]).argmax())


def convex_edge(points, dim, rng, k=10, by_distance=True):
    """Checked links; edge dots onto a convex outline; inside dots as neighbour averages."""
    links = checked_links(points, k, 12, 0.5)
    edge = outer_ring(points)
    flat = np.zeros((len(points), dim))
    flat[edge] = onto_convex_outline(layout_edge(points, links, edge, dim))
    return average_inside(points, links, flat, edge, by_distance), {"edges": links}


method("convex_edge", k=10)(convex_edge)
