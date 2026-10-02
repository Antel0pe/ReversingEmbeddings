"""Measure one-sided tangent changes at three renderer event types."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grey_ones import RANGES, SUB


def jacobian64(q, sub=SUB):
    """Analytic 784-by-5 Jacobian in coordinates normalized to knob ranges."""
    cx, cy, height, width, lean = q
    slope = np.tan(np.deg2rad(lean))
    slope_rate = (1 + slope * slope) * np.pi / 180
    ylo = np.arange(28 * sub, dtype=np.float64) / sub
    yhi = ylo + 1 / sub
    ymid = ylo + .5 / sub
    top, bottom = cy - height / 2, cy + height / 2
    overlap_y = np.clip(np.minimum(yhi, bottom) - np.maximum(ylo, top), 0, 1 / sub)
    dy_cy = ((bottom < yhi).astype(float) - (top > ylo).astype(float)) * (overlap_y > 0)
    dy_height = .5 * ((bottom < yhi).astype(float) + (top > ylo).astype(float)) * (overlap_y > 0)
    center = cx - slope * (ymid - cy)
    left, right = center - width / 2, center + width / 2
    cols = np.arange(28)
    x_overlap = np.clip(np.minimum(right[:, None], cols + 1) - np.maximum(left[:, None], cols), 0, 1)
    right_active = ((right[:, None] < cols + 1) & (x_overlap > 0)).astype(float)
    left_active = ((left[:, None] > cols) & (x_overlap > 0)).astype(float)
    dc = right_active - left_active
    dw = (right_active + left_active) / 2
    terms = [
        dc * overlap_y[:, None],
        dc * slope * overlap_y[:, None] + x_overlap * dy_cy[:, None],
        x_overlap * dy_height[:, None],
        dw * overlap_y[:, None],
        -dc * (ymid - cy)[:, None] * slope_rate * overlap_y[:, None],
    ]
    j = np.stack([x.reshape(28, sub, 28).sum(axis=1).reshape(784) for x in terms], axis=1)
    return j * (RANGES[:, 1] - RANGES[:, 0])


def largest_angle(a, b):
    qa = np.linalg.qr(a)[0]
    qb = np.linalg.qr(b)[0]
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.rad2deg(np.arccos(np.clip(singular[-1], -1, 1))))


def cap_case(rng, integer_boundary):
    while True:
        q = RANGES[:, 0] + rng.random(5) * (RANGES[:, 1] - RANGES[:, 0])
        bottom_min = q[1] + RANGES[2, 0] / 2
        bottom_max = q[1] + RANGES[2, 1] / 2
        grid = np.arange(np.ceil(bottom_min * SUB), np.floor(bottom_max * SUB) + 1, dtype=int)
        grid = grid[(grid % SUB == 0) if integer_boundary else (grid % SUB != 0)]
        if len(grid):
            q[2] = 2 * (rng.choice(grid) / SUB - q[1])
            return q


def side_case(rng):
    while True:
        q = RANGES[:, 0] + rng.random(5) * (RANGES[:, 1] - RANGES[:, 0])
        slope = np.tan(np.deg2rad(q[4]))
        y = rng.integers(6 * SUB, 23 * SUB) / SUB + .5 / SUB
        sign = rng.choice([-1, 1])
        # center + sign*width/2 = integer at this sub-strip midpoint.
        offset = -slope * (y - q[1]) + sign * q[3] / 2
        integer = np.ceil(RANGES[0, 0] + offset)
        q[0] = integer - offset
        if RANGES[0, 0] + .01 < q[0] < RANGES[0, 1] - .01:
            return q


def run():
    rng = np.random.default_rng(20260930)
    data = {}
    classes = {
        "cap_crosses_pixel_row": lambda: cap_case(rng, True),
        "cap_crosses_subrow_only": lambda: cap_case(rng, False),
        "side_crosses_pixel_column_in_one_subrow": lambda: side_case(rng),
    }
    for name, draw in classes.items():
        states = np.stack([draw() for _ in range(300)])
        coordinate = 0 if name.startswith("side") else 2
        entries = []
        for epsilon in [1e-4, 1e-6, 1e-8]:
            angles = []
            for q in states:
                lo, hi = q.copy(), q.copy()
                lo[coordinate] -= epsilon
                hi[coordinate] += epsilon
                angles.append(largest_angle(jacobian64(lo), jacobian64(hi)))
            entries.append({"half_step_in_knob_units": epsilon,
                            "angle_degrees_quantiles": np.quantile(angles, [0, .1, .5, .9, 1]).tolist()})
        data[name] = {"states": len(states), "measurements": entries, "example_state": states[0].tolist()}
    destination = Path(__file__).with_name("tangent_event_results.json")
    destination.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    run()
