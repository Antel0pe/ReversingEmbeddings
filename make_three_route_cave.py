"""Build a genuinely three-dimensional manifold: a three-route cave volume.

The object is M = {(x,y,z) in R^3 : cave_score(x,y,z) >= 0}. Its interior is
three-dimensional. The displayed triangle mesh approximates only its wall,
the two-dimensional boundary cave_score = 0. The HTML has no 3D embedding or
projection from a higher-dimensional state space.

Run: python make_three_route_cave.py
Outputs: figures/three_route_cave_manifold.html and .png
"""

from base64 import b64encode
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import logsumexp
from skimage.measure import marching_cubes


FIGURES = Path("figures")
HTML = FIGURES / "three_route_cave_manifold.html"
PNG = FIGURES / "three_route_cave_manifold.png"
TEMPLATE = Path("three_route_cave_template.html")
TAU = .015
HUB_X = 2.05


def cave_score(x, y, z):
    """Positive exactly inside the designed cave; zero on its wall."""
    q = np.pi * (x + HUB_X) / (2 * HUB_X)
    sine = np.sin(q)
    centers = [
        (1.30 * sine + .12 * np.sin(3 * q),
         .35 * np.sin(2 * q) + .12 * np.sin(4 * q)),
        (-1.25 * sine + .10 * np.sin(2 * q),
         -.35 * np.sin(2 * q) + .11 * np.sin(3 * q)),
        (.25 * np.sin(2 * q) + .10 * np.sin(4 * q),
         1.42 * sine - .13 * np.sin(3 * q)),
    ]
    radii = [
        .36 + .04 * np.sin(4 * q)**2
        + .11 * np.exp(-((x + .55) / .45)**2),
        .36 + .06 * np.sin(3 * q + .4)**2
        + .09 * np.exp(-((x - .35) / .4)**2),
        .35 + .04 * np.cos(4 * q)**2
        + .10 * np.exp(-((x + .15) / .4)**2),
    ]
    end_penalty = .12 * (x / 2.2)**10
    scores = [
        radius**2 - (y - cy)**2 - (z - cz)**2 - end_penalty
        for (cy, cz), radius in zip(centers, radii)
    ]
    scores.extend([
        .66**2 - (x + HUB_X)**2 - y**2 - z**2,
        .70**2 - (x - HUB_X)**2 - y**2 - z**2,
    ])
    return TAU * logsumexp(np.stack(np.broadcast_arrays(*scores)) / TAU,
                           axis=0)


def make_mesh():
    x = np.linspace(-2.9, 2.9, 100)
    y = np.linspace(-2.2, 2.2, 80)
    z = np.linspace(-1.0, 2.2, 72)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    scalar = cave_score(xx, yy, zz).astype(np.float32)
    spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    vertices, faces, normals, _ = marching_cubes(
        scalar, level=0, spacing=spacing,
        gradient_direction="descent",
    )
    vertices += np.array([x[0], y[0], z[0]])
    if len(vertices) >= 65536:
        raise ValueError("The standalone WebGL mesh needs 16-bit indices")
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                    faces[:, [2, 0]])), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    adjacency = coo_matrix(
        (np.ones(len(unique_edges) * 2),
         (np.r_[unique_edges[:, 0], unique_edges[:, 1]],
          np.r_[unique_edges[:, 1], unique_edges[:, 0]])),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    components, _ = connected_components(adjacency, directed=False)
    euler = len(vertices) - len(unique_edges) + len(faces)
    residual = np.abs(cave_score(vertices[:, 0], vertices[:, 1],
                                 vertices[:, 2]))
    step = 1e-3
    slopes = []
    for axis in range(3):
        plus, minus = vertices.copy(), vertices.copy()
        plus[:, axis] += step
        minus[:, axis] -= step
        slopes.append((cave_score(*plus.T) - cave_score(*minus.T))
                      / (2 * step))
    gradient_norm = np.linalg.norm(np.stack(slopes), axis=0)
    summary = {
        "vertices": len(vertices), "triangles": len(faces),
        "mesh_components": int(components),
        "mesh_boundary_edges": int(np.sum(counts == 1)),
        "mesh_euler_characteristic": int(euler),
        "mesh_boundary_genus": int((2 - euler) // 2),
        "mesh_median_field_residual": float(np.median(residual)),
        "mesh_max_field_residual": float(np.max(residual)),
        "mesh_min_gradient_norm": float(np.min(gradient_norm)),
        "grid": [len(x), len(y), len(z)],
    }
    if components != 1 or np.any(counts != 2) or euler != -2:
        raise ValueError(f"Cave mesh topology changed: {summary}")
    return vertices, faces, normals, summary


def preview(vertices, faces, summary):
    fig = plt.figure(figsize=(14, 9.7), facecolor="white")
    fig.suptitle("A real 3D manifold: a cave with three winding routes",
                 fontsize=18, fontweight="bold", y=.985)
    fig.text(.5, .947,
             "A state is an ordinary position (x, y, z). Colored solid means "
             "inside the cave; the mesh shows its wall. The three passages "
             "join at both ends.", ha="center", fontsize=10.5)
    grid = fig.add_gridspec(2, 2, left=.055, right=.96, top=.91, bottom=.13,
                            height_ratios=[1.4, .9], hspace=.22, wspace=.20)
    for column, (elevation, azimuth) in enumerate(((25, -55), (22, 25))):
        ax = fig.add_subplot(grid[0, column], projection="3d")
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                        triangles=faces, color="#548f9b", alpha=.92,
                        linewidth=0, antialiased=False, shade=True)
        ax.set_box_aspect(np.ptp(vertices, axis=0))
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set(xlabel="x · chamber to chamber", ylabel="y · sideways",
               zlabel="z · height")
        ax.tick_params(labelsize=7)
        ax.set_title("Same 3D solid, rotated", fontsize=11)
    yy, zz = np.meshgrid(np.linspace(-2.2, 2.2, 260),
                         np.linspace(-1.0, 2.2, 220), indexing="xy")
    for column, xpos in enumerate((0.0, HUB_X)):
        ax = fig.add_subplot(grid[1, column])
        field = cave_score(xpos, yy, zz)
        ax.contourf(yy, zz, field >= 0, levels=[-.5, .5, 1.5],
                    colors=["#f3f6f7", "#62abb3"])
        ax.contour(yy, zz, field, levels=[0], colors=["#24434a"],
                   linewidths=1)
        ax.set(xlim=(-2.2, 2.2), ylim=(-1.0, 2.2), xlabel="y · sideways",
               ylabel="z · height")
        ax.set_aspect("equal")
        ax.set_title("x = 0: three separate passages" if column == 0
                     else "x = 2.05: passages meet in one chamber",
                     fontsize=11)
    fig.text(.5, .072,
             "Cyan = points inside M = {(x,y,z): score(x,y,z) ≥ 0}. "
             "White = outside. The wall is a 2D boundary of this 3D volume.",
             ha="center", fontsize=10)
    fig.text(.5, .043,
             f"Mesh check: {summary['mesh_components']} connected wall, "
             f"{summary['mesh_boundary_edges']} open edges, "
             f"boundary genus {summary['mesh_boundary_genus']} "
             "(two independent loops). Mesh triangles approximate the exact "
             "implicit boundary.", ha="center", fontsize=9.7)
    fig.savefig(PNG, dpi=145, facecolor="white")
    plt.close(fig)


def write_html(vertices, faces, normals, summary):
    positions = b64encode(np.asarray(vertices, dtype="<f4").tobytes()).decode()
    directions = b64encode(np.asarray(normals, dtype="<f4").tobytes()).decode()
    indices = b64encode(np.asarray(faces, dtype="<u2").tobytes()).decode()
    metadata = json.dumps(summary, separators=(",", ":"))
    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("__POSITIONS__", positions)
    html = html.replace("__NORMALS__", directions)
    html = html.replace("__INDICES__", indices)
    html = html.replace("__SUMMARY__", metadata)
    HTML.write_text(html, encoding="utf-8")


def main():
    FIGURES.mkdir(exist_ok=True)
    vertices, faces, normals, summary = make_mesh()
    preview(vertices, faces, summary)
    write_html(vertices, faces, normals, summary)
    print(f"Wrote {HTML} ({HTML.stat().st_size:,} bytes) and {PNG}")
    print(summary)


if __name__ == "__main__":
    main()
