"""Build the exact three-control valid-pose region for an inspection arm.

State (u,v,h) lives in ordinary R^3: shoulder angle = 95u degrees, relative
elbow angle = 150v degrees, and h is lift. Validity comes from forward
kinematics, a ring-shaped probe target, and obstacle clearance. The triangle
mesh is only a visualization of the exact rule boundary.

Run: python make_inspection_arm_manifold.py
"""

from base64 import b64encode
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import label
from skimage.measure import marching_cubes


FIGURES = Path("figures")
HTML = FIGURES / "inspection_arm_manifold.html"
PNG = FIGURES / "inspection_arm_manifold.png"
TEMPLATE = Path("inspection_arm_template.html")
LENGTHS = (1.0, 0.8)
TARGET = (1.3, 0.4)
TARGET_INNER, TARGET_OUTER = 0.20, 0.65
LOW_POST = (0.65, 0.5)
LOW_RADIUS, LOW_HEIGHT = 0.17, 0.55
ROUTE_TIP = (1.15, 0.70)


def arm_points(u, v):
    """Planar base, elbow, and probe tip for normalized controls u and v."""
    a = np.deg2rad(95 * u)
    b = np.deg2rad(150 * v)
    zero = np.zeros_like(a)
    base = (zero, zero)
    elbow = (np.cos(a), np.sin(a))
    tip = (elbow[0] + 0.8 * np.cos(a + b),
           elbow[1] + 0.8 * np.sin(a + b))
    return base, elbow, tip


def segment_distance(start, end, point):
    """Exact minimum planar distance from a point to a finite arm link."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    fraction = np.clip(((point[0] - start[0]) * dx
                        + (point[1] - start[1]) * dy) / (dx * dx + dy * dy),
                       0, 1)
    return np.hypot(point[0] - start[0] - fraction * dx,
                    point[1] - start[1] - fraction * dy)


def valid_score(u, v, h):
    """Nonnegative exactly for poses satisfying all stated rules.

    Score terms are normalized solely to balance mesh sampling; the score
    magnitude is not a physical distance. Zero is the exact validity boundary.
    """
    u, v, h = np.broadcast_arrays(u, v, h)
    base, elbow, tip = arm_points(u, v)
    tip_distance = np.hypot(tip[0] - TARGET[0], tip[1] - TARGET[1])
    terms = [1 - np.abs(u), 1 - np.abs(v), h, 1 - h,
             (TARGET_OUTER - tip_distance) / 0.8,
             (tip_distance - TARGET_INNER) / 0.8]
    for start, end in ((base, elbow), (elbow, tip)):
        planar_distance = segment_distance(start, end, LOW_POST)
        clearance = np.hypot(planar_distance,
                             np.maximum(0, h - LOW_HEIGHT)) - LOW_RADIUS
        terms.append(clearance / 0.8)
    return np.minimum.reduce(terms)


def inverse_kinematics(tip, elbow_sign):
    x, y = tip
    cosine = (x*x + y*y - 1 - .8**2) / 1.6
    elbow_angle = elbow_sign * np.arccos(cosine)
    shoulder_angle = np.arctan2(y, x) - np.arctan2(
        .8*np.sin(elbow_angle), 1+.8*np.cos(elbow_angle))
    return np.rad2deg(shoulder_angle), np.rad2deg(elbow_angle)


def route_vertices():
    """A continuous three-segment route between two valid low poses."""
    upper = np.array(inverse_kinematics(ROUTE_TIP, -1)) / [95, 150]
    lower = np.array(inverse_kinematics(ROUTE_TIP, +1)) / [95, 150]
    return np.array([[*upper, .35], [*upper, .80],
                     [*lower, .80], [*lower, .35]])


def route_checks():
    route = route_vertices()
    minima = []
    certified_lower_bounds = []
    t = np.linspace(0, 1, 2001)[:, None]
    for start, end in zip(route[:-1], route[1:]):
        points = start[None, :] * (1-t) + end[None, :] * t
        sampled_minimum = float(valid_score(*points.T).min())
        minima.append(sampled_minimum)
        delta = end - start
        shoulder_rate = np.deg2rad(95 * delta[0])
        elbow_rate = np.deg2rad(150 * delta[1])
        tip_speed_bound = (abs(shoulder_rate)
                           + .8 * abs(shoulder_rate + elbow_rate))
        # Every score term is Lipschitz in route parameter t. The finite-link
        # distance changes no faster than its fastest moving endpoint. A point
        # between samples is at most 1/4000 in t from a checked sample.
        score_speed_bound = max(abs(delta[0]), abs(delta[1]), abs(delta[2]),
                                tip_speed_bound / .8,
                                np.hypot(tip_speed_bound, delta[2]) / .8)
        certified_lower_bounds.append(sampled_minimum
                                      - score_speed_bound / 4000)
    low_line = route[0][None, :] * (1-t) + route[-1][None, :] * t
    low_direct_min = float(valid_score(*low_line.T).min())
    u, v = np.meshgrid(np.linspace(-1, 1, 500),
                       np.linspace(-1, 1, 500), indexing="ij")
    low_regions = label(valid_score(u, v, .35) > 0)[1]
    high_regions = label(valid_score(u, v, .80) > 0)[1]
    if (min(certified_lower_bounds) <= 0 or low_direct_min >= 0
            or low_regions != 2 or high_regions != 1):
        raise ValueError("The demonstrated route or its slice comparison changed")
    return {
        "route_min_score": min(minima),
        "route_certified_lower_score": min(certified_lower_bounds),
        "low_straight_min_score": low_direct_min,
        "low_slice_regions": low_regions,
        "high_slice_regions": high_regions,
    }


def make_mesh():
    u = np.linspace(-1.04, 1.04, 86)
    v = np.linspace(-1.04, 1.04, 86)
    h = np.linspace(-0.04, 1.04, 70)
    uu, vv, hh = np.meshgrid(u, v, h, indexing="ij")
    scalar = valid_score(uu, vv, hh).astype(np.float32)
    spacing = (u[1] - u[0], v[1] - v[0], h[1] - h[0])
    vertices, faces, normals, _ = marching_cubes(
        scalar, 0, spacing=spacing, gradient_direction="descent")
    vertices += np.array([u[0], v[0], h[0]])
    if len(vertices) >= 65536:
        raise ValueError("Standalone WebGL viewer uses 16-bit mesh indices")
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                    faces[:, [2, 0]])), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    adjacency = coo_matrix(
        (np.ones(2 * len(unique_edges)),
         (np.r_[unique_edges[:, 0], unique_edges[:, 1]],
          np.r_[unique_edges[:, 1], unique_edges[:, 0]])),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    components, _ = connected_components(adjacency, directed=False)
    euler = len(vertices) - len(unique_edges) + len(faces)
    residual = np.abs(valid_score(*vertices.T))
    summary = {
        "vertices": len(vertices), "triangles": len(faces),
        "mesh_components": int(components),
        "mesh_open_edges": int(np.sum(counts == 1)),
        "mesh_euler": int(euler),
        "boundary_genus": int((2 - euler) // 2),
        "mesh_median_score_residual": float(np.median(residual)),
        "mesh_max_score_residual": float(np.max(residual)),
        "grid": [len(u), len(v), len(h)],
    }
    summary.update(route_checks())
    if components != 1 or np.any(counts != 2) or euler != -2:
        raise ValueError(f"Unexpected topology of valid poses: {summary}")
    return vertices, faces, normals, summary


def preview(vertices, faces, summary):
    fig = plt.figure(figsize=(14, 9.3), facecolor="white")
    fig.suptitle("Robot inspection task: all valid poses, in their exact 3D control space",
                 fontsize=16.5, fontweight="bold", y=.985)
    fig.text(.5, .947,
             "Each point is (shoulder angle, elbow angle, lift). The probe must "
             "reach the target ring; the arm must clear the low post.",
             ha="center", fontsize=10.5)
    grid = fig.add_gridspec(2, 2, left=.08, right=.94, top=.86, bottom=.16,
                            hspace=.32, wspace=.24)
    ax = fig.add_subplot(grid[0, 0], projection="3d")
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    triangles=faces, color="#9bc9d0", linewidth=0,
                    antialiased=False, shade=True)
    route = route_vertices()
    ax.plot(route[:, 0], route[:, 1], route[:, 2],
            color="#dc5a8d", linewidth=3, zorder=8)
    ax.scatter(route[[0, -1], 0], route[[0, -1], 1],
               route[[0, -1], 2], color="#dc5a8d", s=28, zorder=9)
    ax.set(xlabel="shoulder / 95°", ylabel="elbow / 150°", zlabel="lift")
    ax.set_xticks([-.5, 0, .5, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_box_aspect((2, 2, 1.25))
    ax.view_init(elev=30, azim=-100)
    ax.set_title("Valid states; magenta = a route through them", fontsize=11)

    ax = fig.add_subplot(grid[0, 1])
    ax.add_patch(Circle(TARGET, TARGET_OUTER, facecolor="#daf0e9",
                        edgecolor="#3a977d", linewidth=1.6))
    ax.add_patch(Circle(TARGET, TARGET_INNER, facecolor="#68747c",
                        edgecolor="#37434a", linewidth=1.2))
    ax.add_patch(Circle(LOW_POST, LOW_RADIUS, facecolor="#da9f7b",
                        edgecolor="#ab5d3d", linewidth=1.5))
    labels = [("elbow above: valid", -1, "#bd742e"),
              ("elbow below: valid", +1, "#167d78")]
    for label, sign, color in labels:
        shoulder, elbow = inverse_kinematics(ROUTE_TIP, sign)
        points = arm_points(shoulder/95, elbow/150)
        px = [float(point[0]) for point in points]
        py = [float(point[1]) for point in points]
        ax.plot(px, py, marker="o", lw=3, color=color, label=label)
    ax.set(xlim=(-.05, 2.1), ylim=(-.55, 1.25), xlabel="workbench x",
           ylabel="workbench y")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Same probe tip; different arm posture\n"
                 "Green ring = target · gray = no-probe center · orange = low post",
                 fontsize=10.5)

    uu, vv = np.meshgrid(np.linspace(-1, 1, 280),
                         np.linspace(-1, 1, 280), indexing="xy")
    for index, height in enumerate((.35, .80)):
        ax = fig.add_subplot(grid[1, index])
        field = valid_score(uu, vv, height)
        ax.contourf(uu*95, vv*150, field >= 0,
                    levels=[-.5, .5, 1.5], colors=["#f2f5f6", "#5cabb0"])
        ax.contour(uu*95, vv*150, field, levels=[0], colors=["#26464c"],
                   linewidths=.8)
        for sign, color in ((-1, "#bd742e"), (+1, "#167d78")):
            shoulder, elbow = inverse_kinematics(ROUTE_TIP, sign)
            ax.scatter(shoulder, elbow, s=50, c=color, edgecolors="white",
                       linewidths=1.1, zorder=4)
        ax.set(xlim=(-95, 95), ylim=(-150, 150),
               xlabel="shoulder angle (degrees)",
               ylabel="relative elbow angle (degrees)")
        ax.set_title(f"Lift {height:.2f}: cyan = valid angle pairs", fontsize=11)
    fig.text(.5, .077,
             "At lift 0.35 the two valid poses lie in separate regions. "
             "Lift to 0.80, rotate within the valid region, then lower.",
             ha="center", fontsize=10)
    fig.text(.5, .041,
             f"The sampled wall has {summary['mesh_components']} component, "
             f"{summary['mesh_open_edges']} open edges and genus "
             f"{summary['boundary_genus']}. Only the wall is triangulated; "
             "the rules define membership continuously.",
             ha="center", fontsize=9.5)
    fig.savefig(PNG, dpi=145, facecolor="white")
    plt.close(fig)


def write_html(vertices, faces, normals, summary):
    html = TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        "__POSITIONS__": b64encode(np.asarray(vertices, dtype="<f4").tobytes()).decode(),
        "__NORMALS__": b64encode(np.asarray(normals, dtype="<f4").tobytes()).decode(),
        "__INDICES__": b64encode(np.asarray(faces, dtype="<u2").tobytes()).decode(),
        "__SUMMARY__": json.dumps(summary, separators=(",", ":")),
    }
    for key, value in replacements.items():
        html = html.replace(key, value)
    HTML.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    mesh = make_mesh()
    preview(mesh[0], mesh[1], mesh[3])
    write_html(*mesh)
    print(json.dumps(mesh[3], indent=2))
