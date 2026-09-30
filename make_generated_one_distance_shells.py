"""Sample equal-pixel-distance shells of the controlled five-knob 1 family.

Run: python make_generated_one_distance_shells.py

The full shell is continuous. This script intersects deterministic rays in
normalized knob space with a 784-pixel Euclidean-distance level set and saves
the successful intersections, then displays a diverse finite subset.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtri
from scipy.stats import qmc

from grey_ones import KNOBS, RANGES, render, sample


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures" / "generated_one_distance_shells"
LOW = RANGES[:, 0]
SPAN = RANGES[:, 1] - LOW
RANDOM_SEED = 20260930
RADIUS = 4.5
COMPARISON_RADIUS = 2.5
LOCAL_RADIUS = 1.0
RAYS = 2048
DISPLAY_COUNT = 20
COLORS = {"ink": "#202733", "blue": "#236caa", "pale": "#eaf3fb"}


def pixels(settings):
    return render(settings).reshape(-1, 784).astype(np.float64)


def distances(seed_image, settings):
    return np.linalg.norm(pixels(settings) - seed_image, axis=1)


def sphere_directions(n, seed):
    assert n & (n - 1) == 0, "Sobol size must be a power of two"
    u = qmc.Sobol(5, scramble=True, seed=seed).random_base2(int(np.log2(n)))
    z = ndtri(np.clip(u, 1e-12, 1 - 1e-12))
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def shell(seed, radius, directions):
    """Find the first crossing from the seed on each ray, if one exists."""
    q0 = (seed - LOW) / SPAN
    positive = directions > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        boundary = np.where(positive, (1 - q0) / directions, -q0 / directions)
    tmax = np.min(boundary, axis=1)
    tgrid = np.linspace(0, 1, 9)
    dgrid = np.stack([
        distances(pixels(seed[None])[0], LOW + (q0 + t * tmax[:, None] * directions) * SPAN)
        for t in tgrid[1:]
    ], axis=1)
    dgrid = np.column_stack([np.zeros(len(directions)), dgrid])
    reached = dgrid >= radius
    feasible = reached.any(axis=1)
    ids = np.flatnonzero(feasible)
    first = np.argmax(reached[ids], axis=1)
    lo = tgrid[first - 1].copy()
    hi = tgrid[first].copy()
    for _ in range(18):
        mid = (lo + hi) / 2
        states = LOW + (q0 + mid[:, None] * tmax[ids, None] * directions[ids]) * SPAN
        d = distances(pixels(seed[None])[0], states)
        high = d >= radius
        hi[high] = mid[high]
        lo[~high] = mid[~high]
    frac = (lo + hi) / 2
    states = LOW + (q0 + frac[:, None] * tmax[ids, None] * directions[ids]) * SPAN
    exact_d = distances(pixels(seed[None])[0], states)
    # Captures obvious undersampling or a nonmonotone radial distance curve.
    nonmonotone = np.any(np.diff(dgrid, axis=1) < -1e-6, axis=1)
    return ids, states, exact_d, {
        "rays": len(directions),
        "crossings": len(ids),
        "fraction_with_crossing": float(feasible.mean()),
        "nonmonotone_grid_rays": int(nonmonotone.sum()),
        "max_distance_error": float(np.max(np.abs(exact_d - radius))),
    }


def diverse_subset(states, seed_image, count):
    """Greedy max-min separation in the actual 784-pixel metric."""
    images = pixels(states)
    mean_image = images.mean(axis=0)
    chosen = [int(np.argmax(np.linalg.norm(images - mean_image, axis=1)))]
    nearest_sq = np.sum((images - images[chosen[0]]) ** 2, axis=1)
    while len(chosen) < min(count, len(states)):
        nearest_sq[chosen] = -1
        j = int(np.argmax(nearest_sq))
        chosen.append(j)
        nearest_sq = np.minimum(nearest_sq, np.sum((images - images[j]) ** 2, axis=1))
    chosen = np.asarray(chosen)
    mutual = np.linalg.norm(images[chosen, None] - images[chosen][None], axis=2)
    mutual[np.diag_indices_from(mutual)] = np.inf
    return chosen, float(mutual.min())


def midpoint_deviation(seed, state):
    """Distance of a pixel midpoint to a numerically fitted generated image."""
    target = (pixels(seed[None])[0] + pixels(state[None])[0]) / 2
    q0 = np.clip(((seed + state) / 2 - LOW) / SPAN, 0, 1)

    def residual(q):
        return pixels((LOW + q * SPAN)[None])[0] - target

    fit = least_squares(residual, q0, bounds=(np.zeros(5), np.ones(5)),
                        max_nfev=70, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    return float(np.linalg.norm(fit.fun)), bool(fit.success)


def knob_path_ratio(seed, state, chord):
    t = np.linspace(0, 1, 65)
    images = pixels(seed[None] + t[:, None] * (state - seed)[None])
    return float(np.linalg.norm(np.diff(images, axis=0), axis=1).sum() / chord)


def show_image(ax, state, label, is_seed=False):
    ax.imshow(render(state[None])[0], cmap="gray_r", vmin=0, vmax=1,
              interpolation="nearest", origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(COLORS["blue"] if is_seed else "#aebdca")
        spine.set_linewidth(2.8 if is_seed else 0.8)
    ax.set_title(label, fontsize=10.5, color=COLORS["ink"], pad=5)


def plot_seed(index, seed, states, ds, selected, min_pair, meta):
    fig = plt.figure(figsize=(15, 19), facecolor="white")
    fig.text(.055, .971, f"Seed {index}: generated 1s at pixel distance {RADIUS:g}",
             fontsize=21, color=COLORS["ink"], weight="bold", va="top")
    fig.text(.055, .941,
             "Each gray square is a 28×28 coverage image (white = 0 ink, black = full ink). "
             "Distance uses all 784 pixel values.", fontsize=11, color=COLORS["ink"])
    seed_ax = fig.add_axes([.065, .81, .115, .115])
    show_image(seed_ax, seed, "START", is_seed=True)
    fig.text(.20, .901, "Start knob settings", fontsize=12, weight="bold", color=COLORS["blue"])
    fig.text(.20, .876,
             f"horizontal center {seed[0]:.3f} px   vertical center {seed[1]:.3f} px\n"
             f"height {seed[2]:.3f} px   width {seed[3]:.3f} px   lean {seed[4]:+.2f}°",
             fontsize=12, linespacing=1.7, color=COLORS["ink"])
    fig.text(.20, .816,
             f"Blue frame = start. Below: {len(selected)} spread-out examples from "
             f"{meta['crossings']:,} computed shell points. "
             f"Closest displayed pair: {min_pair:.2f} pixel units.",
             fontsize=11, color=COLORS["ink"])
    grid = fig.add_gridspec(5, 4, left=.045, right=.955, bottom=.075, top=.79,
                           hspace=.55, wspace=.20)
    for rank, j in enumerate(selected):
        cell = grid[rank // 4, rank % 4].subgridspec(2, 1, height_ratios=[4.2, 1.1], hspace=.05)
        ax = fig.add_subplot(cell[0])
        show_image(ax, states[j], f"{rank + 1:02d}  |  distance {ds[j]:.4f}")
        tx = fig.add_subplot(cell[1])
        tx.axis("off")
        p = states[j]
        tx.text(.5, .70, f"x {p[0]:.3f}    y {p[1]:.3f}    height {p[2]:.3f} px",
                ha="center", va="center", fontsize=9.5, color=COLORS["ink"])
        tx.text(.5, .22, f"width {p[3]:.3f} px    lean {p[4]:+.2f}°",
                ha="center", va="center", fontsize=9.5, color=COLORS["ink"])
    fig.text(.055, .039,
             "Fixed: this seed and the 4.5-unit distance. Varies: all five knobs "
             "within their allowed ranges.\n"
             "The shell is continuous; these are sampled points. Grid placement "
             "does not preserve distances between neighbors. Positive lean tilts the top right.",
             fontsize=10.4, color=COLORS["ink"], va="bottom", linespacing=1.5)
    path = OUT / f"seed_{index:02d}_radius_4p5.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_comparison(seed, radius, states, ds, selected, meta, midpoint_stats):
    fig = plt.figure(figsize=(13, 8.5), facecolor="white")
    fig.text(.04, .965, f"Radius {radius:g}: a closer shell around seed 1", fontsize=19,
             weight="bold", color=COLORS["ink"], va="top")
    fig.text(.04, .92,
             f"Same generated 1 and 784-pixel distance; radius {radius:g} instead of 4.5.\n"
             f"Seed: x {seed[0]:.3f}, y {seed[1]:.3f}, h {seed[2]:.3f}, "
             f"w {seed[3]:.3f}, lean {seed[4]:+.2f}°.",
             fontsize=11, color=COLORS["ink"], va="top", linespacing=1.4)
    grid = fig.add_gridspec(2, 5, left=.04, right=.96, top=.82, bottom=.19,
                           hspace=.47, wspace=.15)
    ax = fig.add_subplot(grid[0, 0])
    show_image(ax, seed, "START", is_seed=True)
    for k, j in enumerate(selected[:9]):
        ax = fig.add_subplot(grid[(k + 1) // 5, (k + 1) % 5])
        show_image(ax, states[j], f"{k+1:02d}  |  {ds[j]:.4f}")
        p = states[j]
        ax.text(.5, -.085, f"x {p[0]:.2f}  y {p[1]:.2f}  h {p[2]:.2f}\n"
                f"w {p[3]:.2f}  lean {p[4]:+.1f}°",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
    fig.text(.04, .095,
             f"{meta['crossings']:,} of {meta['rays']:,} sampled rays reached this shell. "
             f"For {midpoint_stats['n']} sampled seed-neighbor pairs, the pixel-average "
             f"midpoint's median distance to a fitted valid generated 1 was "
             f"{midpoint_stats['median']:.3f} (4.5-unit shell: "
             f"{midpoint_stats['main_median']:.3f}).",
             fontsize=10.5, color=COLORS["ink"], wrap=True)
    fig.text(.04, .035,
             "Images are 28×28 ink coverage. These are sampled examples, not every point on the shell. "
             "x/y = centers; h/w = height/width (pixels); positive lean tilts the top right.",
             fontsize=9.5, color=COLORS["ink"])
    path = OUT / f"seed_01_radius_{str(radius).replace('.', 'p')}_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    seeds = sample(5, rng)
    directions = sphere_directions(RAYS, RANDOM_SEED + 1)
    rows = []
    results = []
    all_main_deviations = []
    all_main_ratios = []
    main_success = []
    for i, seed in enumerate(seeds, 1):
        ids, states, ds, meta = shell(seed, RADIUS, directions)
        if len(states) < DISPLAY_COUNT:
            raise RuntimeError(f"Seed {i} has too few radius-{RADIUS} intersections")
        selected, min_pair = diverse_subset(states, pixels(seed[None])[0], DISPLAY_COUNT)
        path = plot_seed(i, seed, states, ds, selected, min_pair, meta)
        # A reproducible subset, distinct from the display selection, measures
        # midpoint behavior without claiming a universal reach threshold.
        diagnostic = np.linspace(0, len(states) - 1, 12, dtype=int)
        deviations, successes, ratios = [], [], []
        for j in diagnostic:
            deviation, success = midpoint_deviation(seed, states[j])
            deviations.append(deviation)
            successes.append(success)
            ratios.append(knob_path_ratio(seed, states[j], ds[j]))
        all_main_deviations += deviations
        all_main_ratios += ratios
        main_success += successes
        result = {"seed": i, "seed_knobs": seed.tolist(), "radius": RADIUS,
                  "figure": str(path.relative_to(ROOT)), "displayed": len(selected),
                  "minimum_displayed_pair_distance": min_pair,
                  "midpoint_fit_median": float(np.median(deviations)),
                  "direct_knob_path_to_chord_median": float(np.median(ratios)),
                  **meta}
        results.append(result)
        for j, (ray, p, d) in enumerate(zip(ids, states, ds)):
            rows.append([i, int(ray), int(j in selected), d, *p])
        print(json.dumps(result), flush=True)

    # Show one smaller radius to make the scale choice tangible.
    comp_ids, comp_states, comp_ds, comp_meta = shell(seeds[0], COMPARISON_RADIUS, directions)
    comp_selected, _ = diverse_subset(comp_states, pixels(seeds[0][None])[0], 9)
    comp_diagnostic = np.linspace(0, len(comp_states) - 1, 12, dtype=int)
    comp_deviations = []
    comp_success = []
    comp_ratios = []
    for j in comp_diagnostic:
        deviation, success = midpoint_deviation(seeds[0], comp_states[j])
        comp_deviations.append(deviation)
        comp_success.append(success)
        comp_ratios.append(knob_path_ratio(seeds[0], comp_states[j], comp_ds[j]))
    comparison_stats = {"n": len(comp_deviations), "median": float(np.median(comp_deviations)),
                        "main_median": results[0]["midpoint_fit_median"]}
    comparison_path = plot_comparison(seeds[0], COMPARISON_RADIUS, comp_states, comp_ds, comp_selected,
                                      comp_meta, comparison_stats)
    local_ids, local_states, local_ds, local_meta = shell(seeds[0], LOCAL_RADIUS, directions)
    local_selected, _ = diverse_subset(local_states, pixels(seeds[0][None])[0], 9)
    local_diagnostic = np.linspace(0, len(local_states) - 1, 12, dtype=int)
    local_deviations = []
    local_success = []
    local_ratios = []
    for j in local_diagnostic:
        deviation, success = midpoint_deviation(seeds[0], local_states[j])
        local_deviations.append(deviation)
        local_success.append(success)
        local_ratios.append(knob_path_ratio(seeds[0], local_states[j], local_ds[j]))
    local_stats = {"n": len(local_deviations), "median": float(np.median(local_deviations)),
                   "main_median": results[0]["midpoint_fit_median"]}
    local_path = plot_comparison(seeds[0], LOCAL_RADIUS, local_states, local_ds, local_selected,
                                 local_meta, local_stats)
    with (OUT / "sampled_radius_4p5_shells.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "ray", "displayed", "pixel_distance", *KNOBS])
        writer.writerows(rows)
    report = {
        "generator": "grey_ones.render; 28x28 coverage, five continuous knobs",
        "pixel_distance": "sqrt(sum over 784 pixels of squared coverage difference)",
        "seed_rng": RANDOM_SEED,
        "sampled_ray_method": "scrambled Sobol directions mapped to a 5D sphere, intersected with knob box",
        "radius": RADIUS,
        "seeds": results,
        "midpoint_diagnostic": {
            "pairs": len(all_main_deviations),
            "radius_4p5_median_distance_to_fitted_generated_image": float(np.median(all_main_deviations)),
            "radius_4p5_p90_distance_to_fitted_generated_image": float(np.percentile(all_main_deviations, 90)),
            "radius_4p5_direct_knob_path_to_chord_median": float(np.median(all_main_ratios)),
            "radius_4p5_fit_successes": int(sum(main_success)),
            "radius_2p5_seed1_pairs": len(comp_deviations),
            "radius_2p5_seed1_median_distance_to_fitted_generated_image": float(np.median(comp_deviations)),
            "radius_2p5_seed1_direct_knob_path_to_chord_median": float(np.median(comp_ratios)),
            "radius_2p5_fit_successes": int(sum(comp_success)),
            "radius_1p0_seed1_pairs": len(local_deviations),
            "radius_1p0_seed1_median_distance_to_fitted_generated_image": float(np.median(local_deviations)),
            "radius_1p0_seed1_direct_knob_path_to_chord_median": float(np.median(local_ratios)),
            "radius_1p0_fit_successes": int(sum(local_success)),
        },
        "comparison": {**comp_meta, "figure": str(comparison_path.relative_to(ROOT))},
        "local_comparison": {**local_meta, "figure": str(local_path.relative_to(ROOT))},
    }
    (OUT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["midpoint_diagnostic"], indent=2), flush=True)


if __name__ == "__main__":
    main()
