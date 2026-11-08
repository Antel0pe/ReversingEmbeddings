"""Flattening methods. To add one, write a function and put @method("name") above it.

The contract, which is the whole interface:

    def my_method(points, dim, rng):
        points   (n, 3) array of 3-D positions, in shuffled order and at a random rotation
        dim      the sheet's true dimension, always 2 here
        rng      numpy random generator, for any randomness the method needs
        returns  (n, dim) array: where each point lands on the flat sheet, in the SAME
                 ORDER as `points`

Scale, rotation and mirroring of the answer do not matter. Raise an exception if the
method cannot produce an answer; the suite records that as a failure and moves on.

A method may instead return (flat, info), where info is a dict that helps the suite work
out WHY it failed. Everything in it is optional:

    info["edges"]     (m, 2) indices into `points`: the neighbour links the method relied on
    info["plane"]     (3, 2) the flat plane's two axes, for methods that project onto a plane
    info["captured"]  share (0-1) of the method's own distances that a flat layout can express
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

METHODS = {}


def method(name, **fixed):
    """Register a function; keyword arguments are passed through on every call."""

    def register(fn):
        METHODS[name] = (lambda points, dim, rng: fn(points, dim, rng, **fixed)) if fixed else fn
        return fn

    return register


@method("best_fit_plane")
def best_fit_plane(points, dim, rng):
    """Drop everything onto the flat plane closest to all the points (PCA)."""
    centred = points - points.mean(0)
    axes = np.linalg.svd(centred, full_matrices=False)[2][:dim].T
    return centred @ axes, {"plane": axes}


@method("tangent_plane")
def tangent_plane(points, dim, rng, k=20):
    """Pick a random point, take the plane touching the sheet there, drop everything onto it."""
    start = rng.integers(len(points))
    _, near = cKDTree(points).query(points[start], k=k)
    axes = np.linalg.svd(points[near] - points[near].mean(0), full_matrices=False)[2][:dim].T
    return (points - points[start]) @ axes, {"plane": axes}


def layout_from_links(points, links, dim):
    """Shortest walks along the given links, then the flat layout that keeps them best."""
    n = len(points)
    lengths = np.linalg.norm(points[links[:, 0]] - points[links[:, 1]], axis=1)
    graph = csr_matrix((lengths, (links[:, 0], links[:, 1])), shape=(n, n))
    pieces = connected_components(graph, directed=False)[0]
    if pieces > 1:
        raise RuntimeError(f"the neighbour graph broke into {pieces} pieces")
    d2 = shortest_path(graph, method="D", directed=False) ** 2
    centred = -0.5 * (d2 - d2.mean(0) - d2.mean(1)[:, None] + d2.mean())
    values, vectors = eigsh(centred, k=dim, which="LA")
    return vectors * np.sqrt(np.maximum(values, 0)), float(values.sum() / np.trace(centred))


def isomap(points, dim, rng, k=10):
    """Link each point to its k nearest; walk along links; lay out flat."""
    _, near = cKDTree(points).query(points, k=k + 1)
    links = np.c_[np.repeat(np.arange(len(points)), k), near[:, 1:].ravel()]
    flat, captured = layout_from_links(points, links, dim)
    return flat, {"edges": links, "captured": captured}


method("isomap_k6", k=6)(isomap)
method("isomap_k10", k=10)(isomap)
method("isomap_k20", k=20)(isomap)
