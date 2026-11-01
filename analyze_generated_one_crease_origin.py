"""Where the generated-1 creases come from, and what refining the grid does to them.

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python analyze_generated_one_crease_origin.py

`figures/linear_5d_to_784d` established that the exact decoder is piecewise and
pointed at the `clip/min/max` coverage calculation. This script asks the next
question: what exactly triggers each crease, and does a finer pixel grid help?

Four measurements, all on lean sweeps because lean moves both stroke edges
across the most pixels.

1. PREDICTED VERSUS OBSERVED. Every crease should sit where one of the
   renderer's decisions flips -- where a stroke edge crosses a pixel-column
   boundary at some sub-row. Those lean angles are solvable in closed form, with
   no rendering. Compare them to where creases are actually observed.
2. NOT ABOUT ZERO. Find the single worst crease over all 784 pixels and report
   the range that pixel's value occupies. A crease at a near-saturated pixel
   would rule out "a pixel switching on" as the mechanism.
3. GRID REFINEMENT. Census creases at 14, 28, 56 and 112 pixels across: how
   many, how hard each one kicks, and how long the pixel-space path gets.
4. A DECISION-FREE FAMILY. The same census on `smooth_ones`, which contains no
   min, max or clip anywhere, plus how far it sits from the coverage renderer.

Creases are detected from the DIRECTION of travel in pixel space, not from any
single pixel: normalise each step, then measure how hard the direction kicks
between consecutive steps. That is basis-free and ignores overall speed.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

import smooth_ones
from coverage_grids import render_coverage
from grey_ones import N_PIX, RANGES, SUB

OUT = Path(__file__).resolve().parent / "figures/generated_one_crease_origin"
BASE = np.array([14.4, 14.3, 19.9, 3.1, 8.0])   # a generic interior setting
WINDOW = 0.2                                     # degrees of lean swept
STEPS = 4001
KICK_THRESHOLD = 0.01                            # direction kick counted as a crease


def lean_sweep(base, window=WINDOW, steps=STEPS):
    P = np.tile(base, (steps, 1))
    P[:, 4] = base[4] + np.linspace(0.0, window, steps)
    return P, P[:, 4]


def direction_kicks(images):
    """How hard the direction of travel kicks, step to step. One per interior step."""
    flat = images.reshape(len(images), -1)
    moves = np.diff(flat, axis=0)
    norms = np.linalg.norm(moves, axis=1, keepdims=True)
    unit = moves / np.maximum(norms, 1e-300)
    return np.linalg.norm(np.diff(unit, axis=0), axis=1)


def predicted_decision_angles(base, window=WINDOW, n_pix=N_PIX, sub=SUB):
    """Lean angles at which a stroke edge crosses a pixel-column boundary.

    centre(y) = cx - tan(lean) * (y - cy), so an edge at centre -/+ width/2 meets
    a boundary at column coordinate c when tan(lean) = (cx -/+ w/2 - c) / (y - cy).
    Solved directly; nothing is rendered.
    """
    cx, cy, height, width, lean0 = base
    pitch = 28.0 / n_pix
    sub_mid = np.arange(n_pix * sub) * (pitch / sub) + 0.5 * pitch / sub
    active = sub_mid[(sub_mid > cy - height / 2) & (sub_mid < cy + height / 2)]
    boundaries = np.arange(n_pix + 1) * pitch
    angles = []
    for edge in (cx - width / 2, cx + width / 2):
        lever = active - cy
        lever = lever[np.abs(lever) > 1e-9]
        tangents = (edge - boundaries[:, None]) / lever[None, :]
        angles.append(np.degrees(np.arctan(tangents)).ravel())
    angles = np.concatenate(angles)
    return np.unique(angles[(angles >= lean0) & (angles <= lean0 + window)])


def census(images, leans):
    kicks = direction_kicks(images)
    flat = images.reshape(len(images), -1)
    return {
        "creases": int((kicks > KICK_THRESHOLD).sum()),
        "total_direction_change": float(kicks.sum()),
        "largest_kick": float(kicks.max()),
        "crease_angles": leans[1:-1][kicks > KICK_THRESHOLD],
        "kicks": kicks,
        "flat": flat,
    }


results = {"base_setting": BASE.tolist(), "lean_window_degrees": WINDOW,
           "steps": STEPS, "kick_threshold": KICK_THRESHOLD}
P, leans = lean_sweep(BASE)
step_size = WINDOW / (STEPS - 1)

# ---- 1. do creases land where the renderer's decisions flip? ----
coverage = render_coverage(P)
obs = census(coverage, leans)
predicted = predicted_decision_angles(BASE)
gaps = np.abs(obs["crease_angles"][:, None] - predicted[None, :]).min(1)
results["decision_flips"] = {
    "predicted_events_in_window": int(len(predicted)),
    "observed_creases": obs["creases"],
    "step_size_degrees": step_size,
    "worst_gap_in_steps": float(gaps.max() / step_size) if len(gaps) else None,
    "observed_creases_within_one_step_of_a_prediction":
        int((gaps <= step_size).sum()) if len(gaps) else 0,
}
print("1. Do creases land where a decision flips?")
print(f"   predicted decision flips in the window : {len(predicted)}")
print(f"   creases actually observed              : {obs['creases']}")
if len(gaps):
    print(f"   observed creases within one step of a prediction: "
          f"{(gaps <= step_size).sum()} of {len(gaps)}")
    print(f"   worst gap between a crease and a prediction: "
          f"{gaps.max() / step_size:.2f} steps of {step_size:.1e} deg")

# ---- 2. the worst crease is not at a pixel switching on ----
rates = (obs["flat"][2:] - obs["flat"][:-2]) / (2 * step_size)
abruptness = np.abs(np.diff(rates, axis=0))
at, pixel = np.unravel_index(int(np.argmax(abruptness)), abruptness.shape)
track = obs["flat"][:, pixel]
results["worst_crease_pixel"] = {
    "row": int(pixel // N_PIX), "col": int(pixel % N_PIX),
    "value_min": float(track.min()), "value_max": float(track.max()),
    "ever_exactly_zero": bool(track.min() == 0.0),
    "rate_before": float(rates[max(at - 2, 0), pixel]),
    "rate_after": float(rates[min(at + 2, len(rates) - 1), pixel]),
    "abrupt_rate_change": float(abruptness[at, pixel]),
    "as_percent_of_typical_rate": float(abruptness[at, pixel]
                                        / np.abs(rates).mean() * 100),
}
w = results["worst_crease_pixel"]
print(f"\n2. The worst crease sits at pixel (row {w['row']}, col {w['col']}).")
print(f"   its value never leaves [{w['value_min']:.6f}, {w['value_max']:.6f}]; "
      f"exactly zero anywhere? {'yes' if w['ever_exactly_zero'] else 'no'}")
print(f"   its rate of change goes {w['rate_before']:+.4f} -> {w['rate_after']:+.4f}, "
      f"a lurch of {w['as_percent_of_typical_rate']:.0f}% of a typical rate")

# ---- 3. what a finer grid does ----
print("\n3. Refining the pixel grid.")
print(f"   {'grid':>9} {'creases':>9} {'total dir. change':>19} "
      f"{'largest kick':>14} {'path speed':>12}")
results["grid_refinement"] = []
for n_pix in (14, 28, 56, 112):
    imgs = render_coverage(P, n_pix)
    c = census(imgs, leans)
    pitch = 28.0 / n_pix
    # speed of travel in units that do not depend on how many pixels there are
    speed = float(np.linalg.norm(np.diff(c["flat"], axis=0), axis=1).mean()
                  * pitch / step_size)
    row = {"n_pix": n_pix, "creases": c["creases"],
           "total_direction_change": c["total_direction_change"],
           "largest_kick": c["largest_kick"], "path_speed": speed}
    results["grid_refinement"].append(row)
    print(f"   {n_pix:>4}x{n_pix:<4} {c['creases']:>9} "
          f"{c['total_direction_change']:>19.4f} {c['largest_kick']:>14.4f} "
          f"{speed:>12.3f}")

# ---- 4. a family with no decisions in it ----
smooth_imgs = smooth_ones.render(P)
sc = census(smooth_imgs, leans)
box = RANGES[:, 0] + np.random.default_rng(3).random((300, 5)) * np.diff(RANGES).ravel()
cov_box, smooth_box = render_coverage(box), smooth_ones.render(box)
results["decision_free"] = {
    "creases": sc["creases"],
    "total_direction_change": sc["total_direction_change"],
    "largest_kick": sc["largest_kick"],
    "relative_rms_vs_coverage_over_box":
        float(np.linalg.norm(cov_box - smooth_box) / np.linalg.norm(cov_box)),
    "ink_correlation_vs_coverage_over_box":
        float(np.corrcoef(cov_box.ravel(), smooth_box.ravel())[0, 1]),
}
d = results["decision_free"]
print(f"\n4. The decision-free family, same 28x28 sweep.")
print(f"   creases {d['creases']}, total direction change "
      f"{d['total_direction_change']:.4f}, largest kick {d['largest_kick']:.2e}")
print(f"   distance from the coverage renderer over the knob box: "
      f"{d['relative_rms_vs_coverage_over_box'] * 100:.2f}% relative RMS, "
      f"ink correlation {d['ink_correlation_vs_coverage_over_box']:.4f}")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))
print(f"\nwrote {OUT / 'results.json'}")
