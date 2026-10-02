"""Check full-knob path lengths against the repository's float64 coverage rule."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_global_span import coverage64
from grey_ones import RANGES


def length(points):
    image = coverage64(points)
    return float(np.linalg.norm(np.diff(image, axis=0), axis=1).sum())


def smooth_route(start, end, depth_height, depth_width, n=4096):
    t = np.linspace(0, 1, n + 1)
    p = start[None, :] * (1 - t[:, None]) + end[None, :] * t[:, None]
    p[:, 2] -= depth_height * np.sin(np.pi * t)
    p[:, 3] -= depth_width * np.sin(np.pi * t)
    assert np.all(p >= RANGES[:, 0] - 1e-12) and np.all(p <= RANGES[:, 1] + 1e-12)
    return p


def old_route_points(nodes, per_edge):
    nodes = np.asarray(nodes, np.float64)
    t = np.arange(per_edge) / per_edge
    points = [nodes[i, None, :] * (1 - t[:, None]) + nodes[i + 1, None, :] * t[:, None]
              for i in range(len(nodes) - 1)]
    return np.vstack(points + [nodes[-1:]])


def test_case(start, end, old_nodes=None):
    baseline = length(smooth_route(start, end, 0, 0, 8192))
    max_h = start[2] - RANGES[2, 0]
    max_w = start[3] - RANGES[3, 0]
    h_only = minimize_scalar(lambda h: length(smooth_route(start, end, h, 0, 2048)),
                             bounds=(0, max_h), method="bounded",
                             options={"xatol": 1e-5})
    both = minimize(lambda x: length(smooth_route(start, end, x[0], x[1], 2048)),
                    [h_only.x, 0], method="Nelder-Mead",
                    bounds=[(0, max_h), (0, max_w)],
                    options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 200})
    out = {
        "start": start.tolist(), "end": end.tolist(),
        "lean_only_length_8192": baseline,
        "sine_height_only": {"height_depth": float(h_only.x),
                             "length_8192": length(smooth_route(start, end, h_only.x, 0, 8192))},
        "sine_height_and_width": {"height_depth": float(both.x[0]), "width_depth": float(both.x[1]),
                                  "length_8192": length(smooth_route(start, end, *both.x, 8192))},
    }
    out["sine_height_and_width"]["gain_percent"] = 100 * (1 - out["sine_height_and_width"]["length_8192"] / baseline)
    if old_nodes is not None:
        points = old_route_points(old_nodes, 256)
        assert np.all(points >= RANGES[:, 0] - 1e-12) and np.all(points <= RANGES[:, 1] + 1e-12)
        out["prior_node_route_python_length_16384"] = length(points)
    return out


def main():
    old = json.loads(Path(__file__).with_name("generated_geometry_results.json").read_text())
    case = old["paths"]["leanCase"]
    results = [test_case(np.array(case["a"]), np.array(case["b"]), case["full"]["qs"])]
    results.append(test_case(np.array([14.5, 14.5, 20.5, 4.6, -10.0]),
                             np.array([14.5, 14.5, 20.5, 4.6, 35.0])))
    destination = Path(__file__).with_name("lean_shortcut_results.json")
    destination.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
