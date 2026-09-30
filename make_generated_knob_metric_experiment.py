"""Measure continuity, scale, direction, and joint motion of five-knob 1s.

Run: python make_generated_knob_metric_experiment.py
All distances and signed changes use the original 784 pixel coverages.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.stats import qmc

from grey_ones import KNOBS, RANGES, render


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures" / "generated_knob_metric"
LOW = RANGES[:, 0]
SPAN = RANGES[:, 1] - LOW
SEED = 20260930
N = 2048
PATH_CONTEXTS = 128
JOINT_N = 1024
JACOBIAN_N = 256
STEP = 0.1
BLUE = "#2166ac"
INK = "#202733"
NAMES = ["x center", "y center", "height", "width", "lean"]


def sobol(n, seed):
    assert n & (n - 1) == 0
    return qmc.Sobol(5, scramble=True, seed=seed).random_base2(int(np.log2(n)))


def images(q):
    return render(LOW + np.asarray(q) * SPAN).reshape(-1, 784).astype(np.float64)


def l2(v):
    return np.linalg.norm(v, axis=-1)


def summary(a):
    return {"p10": float(np.percentile(a, 10)), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "min": float(np.min(a)),
            "max": float(np.max(a))}


def single_steps():
    q = .9 * sobol(N, SEED + 1)
    base = images(q)
    norm_vectors, raw_distances = [], []
    for i in range(5):
        moved = q.copy()
        moved[:, i] += STEP
        norm_vectors.append(images(moved) - base)
        raw = q.copy()
        raw[:, i] += .1 / SPAN[i]
        raw_distances.append(l2(images(raw) - base))
    vectors = np.stack(norm_vectors, axis=1)
    norm_distances = l2(vectors)
    raw_distances = np.stack(raw_distances, axis=1)

    # The ideal real-valued coverage formula is continuous; this checks its
    # numerical scaling above the float32 rounding scale.
    sizes = np.array([.0125, .025, .05, .1, .2])
    qscale = .8 * sobol(N, SEED + 2)
    xscale = images(qscale)
    median_by_size = np.empty((5, len(sizes)))
    for i in range(5):
        for j, h in enumerate(sizes):
            moved = qscale.copy()
            moved[:, i] += h
            median_by_size[i, j] = np.median(l2(images(moved) - xscale))
    return q, base, vectors, raw_distances, norm_distances, sizes, median_by_size


def along_knob_paths():
    contexts = sobol(PATH_CONTEXTS, SEED + 3)
    data = np.empty((5, PATH_CONTEXTS, 10))
    turn = np.empty((5, PATH_CONTEXTS, 9))
    path_ratio = np.empty((5, PATH_CONTEXTS))
    for i in range(5):
        q = np.repeat(contexts[:, None, :], 11, axis=1)
        q[:, :, i] = np.linspace(0, 1, 11)[None]
        rendered = images(q.reshape(-1, 5)).reshape(PATH_CONTEXTS, 11, 784)
        changes = np.diff(rendered, axis=1)
        distance = l2(changes)
        data[i] = distance
        dots = np.sum(changes[:, :-1] * changes[:, 1:], axis=-1)
        cosine = dots / np.maximum(distance[:, :-1] * distance[:, 1:], 1e-12)
        turn[i] = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        path_ratio[i] = distance.sum(axis=1) / l2(rendered[:, -1] - rendered[:, 0])
    return data, turn, path_ratio


def joint_steps(q, base, vectors):
    q = q[:JOINT_N]
    base = base[:JOINT_N]
    vectors = vectors[:JOINT_N]
    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    cosine = np.full((5, 5), np.nan)
    interaction = np.full((5, 5), np.nan)
    direction_change = np.full((5, 5), np.nan)
    results = {}
    for i, j in pairs:
        qij = q.copy()
        qij[:, i] += STEP
        qij[:, j] += STEP
        actual = images(qij) - base
        vi, vj = vectors[:, i], vectors[:, j]
        sum_separate = vi + vj
        second = actual - sum_separate
        ni, nj = l2(vi), l2(vj)
        cos = np.sum(vi * vj, axis=1) / np.maximum(ni * nj, 1e-12)
        scale = np.sqrt(ni**2 + nj**2)
        rel = l2(second) / np.maximum(scale, 1e-12)
        after_j = actual - vj
        turn_cos = np.sum(vi * after_j, axis=1) / np.maximum(ni * l2(after_j), 1e-12)
        turn_angle = np.degrees(np.arccos(np.clip(turn_cos, -1, 1)))
        cosine[i, j] = cosine[j, i] = np.median(cos)
        interaction[i, j] = interaction[j, i] = np.median(rel)
        direction_change[i, j] = np.median(turn_angle)
        results[f"{KNOBS[i]}+{KNOBS[j]}"] = {
            "cosine": summary(cos),
            "interaction_over_individual_quadrature": summary(rel),
            "first_direction_turn_after_second_knob_degrees": summary(turn_angle),
            "actual_joint_distance": summary(l2(actual)),
            "separate_sum_distance": summary(l2(sum_separate)),
        }
    return pairs, cosine, interaction, direction_change, results


def all_five_steps(q, base, vectors):
    actual = images(q + STEP) - base
    separate = vectors.sum(axis=1)
    residual = actual - separate
    individual_quadrature = np.sqrt(np.sum(l2(vectors)**2, axis=1))
    outside = np.sum((base + separate < -1e-8) | (base + separate > 1 + 1e-8), axis=1)
    return {
        "actual_joint_distance": summary(l2(actual)),
        "separate_sum_distance": summary(l2(separate)),
        "nonlinear_residual_distance": summary(l2(residual)),
        "residual_over_actual_joint_distance": summary(l2(residual) / l2(actual)),
        "residual_over_individual_quadrature": summary(l2(residual) / individual_quadrature),
        "outside_coverage_pixel_count_if_summed": summary(outside),
        "fraction_with_any_outside_coverage_pixel_if_summed": float(np.mean(outside > 0)),
    }


def local_jacobian():
    q = .05 + .9 * sobol(JACOBIAN_N, SEED + 4)
    eps = .001
    j = np.empty((JACOBIAN_N, 784, 5))
    for i in range(5):
        up, down = q.copy(), q.copy()
        up[:, i] += eps
        down[:, i] -= eps
        j[:, :, i] = (images(up) - images(down)) / (2 * eps)
    singulars = np.linalg.svd(j, compute_uv=False)
    speeds = l2(np.moveaxis(j, 2, 1))
    return {
        "sample_count": JACOBIAN_N,
        "normalized_knob_tangent_speeds": {KNOBS[i]: summary(speeds[:, i]) for i in range(5)},
        "smallest_singular_value": summary(singulars[:, -1]),
        "largest_to_smallest_singular_ratio": summary(singulars[:, 0] / singulars[:, -1]),
    }


def plot_step_scale(raw, norm, sizes, median_by_size):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2), facecolor="white",
                             gridspec_kw={"width_ratios": [1, 1, 1.22]})
    fig.suptitle("What does a 0.1 knob change do to the image?", x=.035, ha="left",
                 fontsize=20, weight="bold", color=INK)
    fig.text(.035, .912, "Each dot is the median over 2,048 generated starting 1s; bars cover the middle 80%. "
             "Output = Euclidean distance across all 784 ink-coverages.", fontsize=10.5, color=INK)
    for ax, values, title in [
        (axes[0], raw, "A · Raw +0.1 knob unit\n0.1 px for the first four; 0.1° for lean"),
        (axes[1], norm, "B · +10% of each knob's range\n0.1, 0.1, 0.15, 0.28 px; 4.5°")
    ]:
        x = np.arange(5)
        med = np.median(values, axis=0)
        lo = np.percentile(values, 10, axis=0)
        hi = np.percentile(values, 90, axis=0)
        ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="o", color=BLUE,
                    ecolor="#88afd0", capsize=5, lw=2.2, markersize=8)
        ax.set_xticks(x, NAMES, rotation=25, ha="right")
        ax.set_ylabel("784-pixel distance")
        ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=13)
        ax.grid(axis="y", alpha=.25)
        ax.set_ylim(bottom=0)
    ax = axes[2]
    colors = ["#2166ac", "#b2182b", "#4d9221", "#8e44ad", "#d17a00"]
    for i in range(5):
        ax.plot(sizes, median_by_size[i], "o-", lw=2, color=colors[i], label=NAMES[i])
    anchor = median_by_size[0, 3]
    ax.plot(sizes, anchor * sizes / .1, ls="--", color="#697887", lw=1.3,
            label="straight scaling from x-center at 0.1")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes, [f"{x:g}" for x in sizes])
    ax.set_xlabel("Fraction of that knob's full range moved")
    ax.set_ylabel("Median 784-pixel distance")
    ax.set_title("C · Shrinking the step", loc="left", fontsize=13, weight="bold")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(alpha=.22, which="both")
    fig.text(.035, .026, "The two kinds of 0.1 have different input units. A pixel-space distance has no reason to equal the input number. "
             "All starts allow the indicated positive move without crossing a knob limit.", fontsize=10, color=INK)
    fig.subplots_adjust(left=.055, right=.975, top=.76, bottom=.19, wspace=.32)
    path = OUT / "step_scale.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_path_uniformity(data, turn, ratio):
    fig, axes = plt.subplots(5, 1, figsize=(14.5, 13.3), sharex=True,
                             facecolor="white")
    fig.suptitle("Does the same 10%-range step always travel the same distance?",
                 x=.06, ha="left", fontsize=19, weight="bold", color=INK)
    fig.text(.06, .939, "Each row moves one knob from 0% to 100% of its allowed range. "
             "The other four settings vary across 128 generated 1s.", fontsize=10.5, color=INK)
    x = np.arange(10) / 10
    for i, ax in enumerate(axes):
        lo, med, hi = (np.percentile(data[i], p, axis=0) for p in (10, 50, 90))
        ax.fill_between(x, lo, hi, color="#d7e8f6", label="10th–90th percentile" if i == 0 else None)
        ax.plot(x, med, "o-", color=BLUE, lw=2.4, ms=5,
                label="median" if i == 0 else None)
        ax.set_ylabel(f"{NAMES[i]}\npixel distance")
        ymin = max(0, np.min(lo) - .2 * (np.max(hi) - np.min(lo)))
        ymax = np.max(hi) + .2 * (np.max(hi) - np.min(lo))
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0, .9)
        ax.grid(alpha=.18)
        ax.text(1.025, .65,
                f"Median turn between\nnext 10% steps\n{np.median(turn[i]):.0f}°\n\n"
                f"Full-knob path\n/ endpoint chord\n{np.median(ratio[i]):.2f}×",
                ha="left", va="center", transform=ax.transAxes, fontsize=10,
                color=INK)
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    axes[-1].set_xlabel("Starting setting, fraction of this knob's allowed range")
    fig.text(.06, .029, "Each row has its own vertical scale; compare knob sizes in the step-scale figure. "
             "A flat line and narrow band would mean near-uniform pixel motion.\n"
             "Turn is the angle between consecutive signed pixel-change vectors. "
             "Path/chord compares ten valid steps with their direct endpoint distance.",
             fontsize=9.7, color=INK, linespacing=1.4)
    fig.subplots_adjust(left=.14, right=.72, top=.89, bottom=.12, hspace=.18)
    path = OUT / "along_knob_uniformity.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_joint_heatmaps(cosine, interaction, all_five):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.3), facecolor="white")
    fig.suptitle("Multiple knobs: directions and interaction", x=.06, ha="left",
                 fontsize=19, weight="bold", color=INK)
    fig.text(.06, .91, "Both knobs move +10% of their allowed ranges from each of 1,024 starting images. "
             "Each cell is the median over those images.", fontsize=10.5, color=INK)
    for ax, matrix, title, cmap, vmin, vmax, fmt in [
        (axes[0], cosine, "A · Alignment of separate pixel changes", "PuOr", -1, 1, "+.2f"),
        (axes[1], interaction, "B · True joint move minus separate sum", "magma", 0, .3, ".2f"),
    ]:
        masked = np.ma.masked_invalid(matrix)
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(5), NAMES, rotation=35, ha="right")
        ax.set_yticks(range(5), NAMES)
        ax.set_title(title, fontsize=12, weight="bold", loc="left")
        for i in range(5):
            for j in range(5):
                if i != j:
                    ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center",
                            fontsize=10, color=("white" if matrix[i, j] < .19 else INK)
                            if ax is axes[1] else INK)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.text(.06, .125,
             f"All five knobs +10% at once: median true distance "
             f"{all_five['actual_joint_distance']['median']:.2f}; separate-sum distance "
             f"{all_five['separate_sum_distance']['median']:.2f}; "
             f"nonlinear residual {all_five['nonlinear_residual_distance']['median']:.2f}. "
             f"The sum leaves [0,1] for "
             f"{100*all_five['fraction_with_any_outside_coverage_pixel_if_summed']:.1f}% of starting images.",
             fontsize=10.3, color=INK)
    fig.text(.06, .05,
             "A: +1 means similar signed pixel patterns, 0 perpendicular, −1 opposing. "
             "B: L2 interaction divided by the root-sum-square of the two separate L2 lengths.\n"
             "The interaction is a change in the response itself: "
             "F(both) − F(first) − F(second) + F(start).",
             fontsize=10, color=INK, linespacing=1.5)
    fig.subplots_adjust(left=.11, right=.96, top=.83, bottom=.24, wspace=.28)
    path = OUT / "joint_metric.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_joint_example(q, vectors, pair, pair_stats):
    i, j = pair
    q = q[:JOINT_N]
    vectors = vectors[:JOINT_N]
    qij = q.copy()
    qij[:, i] += STEP
    qij[:, j] += STEP
    base_all = images(q)
    joint_all = images(qij)
    actual_all = joint_all - base_all
    separate_all = vectors[:, i] + vectors[:, j]
    second_all = actual_all - separate_all
    scale = np.sqrt(l2(vectors[:, i])**2 + l2(vectors[:, j])**2)
    relative = l2(second_all) / scale
    target = np.median(relative)
    k = int(np.argmin(np.abs(relative - target)))
    q0 = q[k:k + 1]
    qi, qj = q0.copy(), q0.copy()
    qi[:, i] += STEP
    qj[:, j] += STEP
    x0, xi, xj, xij = [images(s)[0] for s in (q0, qi, qj, qij[k:k + 1])]
    vi, vj = xi - x0, xj - x0
    vi_after_j = xij - xj
    separate = vi + vj
    actual = xij - x0
    interaction = actual - separate
    maps = [vi, vi_after_j, interaction, vj, separate, actual]
    labels = [f"{NAMES[i]} at start", f"{NAMES[i]} after {NAMES[j]}",
              "response changed", f"{NAMES[j]} at start", "sum of separate changes",
              "actual joint change"]
    limit = max(np.max(np.abs(m)) for m in maps)
    fig = plt.figure(figsize=(18, 9.2), facecolor="white")
    fig.text(.04, .965, "The direction vector changes when another knob moves",
             fontsize=20, weight="bold", color=INK, va="top")
    fig.text(.04, .92,
             f"Controlled four-corner example: +10% {NAMES[i]} range = {STEP*SPAN[i]:.3g} "
             f"{'degrees' if i == 4 else 'pixels'}; +10% {NAMES[j]} range = {STEP*SPAN[j]:.3g} "
             f"{'degrees' if j == 4 else 'pixels'}. "
             f"Example interaction / separate lengths = {relative[k]:.3f} (near sample median).",
             fontsize=10.5, color=INK)
    gs = fig.add_gridspec(2, 6, left=.04, right=.96, bottom=.24, top=.84,
                          hspace=.44, wspace=.14, height_ratios=[1, 1])
    for col, (x, title) in enumerate([
        (x0, "start"), (xi, f"+ {NAMES[i]}"),
        (xj, f"+ {NAMES[j]}"), (xij, "both moved")
    ]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(x.reshape(28, 28), cmap="gray_r", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11)
    note = fig.add_subplot(gs[0, 4:])
    note.axis("off")
    note.text(0, .92,
              f"Start: x {(LOW + q0[0]*SPAN)[0]:.2f}, y {(LOW + q0[0]*SPAN)[1]:.2f},\n"
              f"h {(LOW + q0[0]*SPAN)[2]:.2f}, w {(LOW + q0[0]*SPAN)[3]:.2f},\n"
              f"lean {(LOW + q0[0]*SPAN)[4]:+.1f}°.\n\n"
              "Top: valid generated images.\n"
              "Bottom: signed pixel changes.\n"
              "Red gains ink; blue loses.\n"
              "All maps share one scale.\n\n"
              "Third map = change in the first\n"
              "knob's response after the second\n"
              "moved. It also equals actual\n"
              "joint change minus separate sum.",
              fontsize=10.3, va="top", color=INK, linespacing=1.28)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    for col, (v, title) in enumerate(zip(maps, labels)):
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(v.reshape(28, 28), cmap="RdBu_r", norm=norm,
                       interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10.5)
        ax.text(.5, -.07, f"L2 {l2(v):.3f}", ha="center", va="top",
                transform=ax.transAxes, fontsize=9.5)
    cax = fig.add_axes([.31, .15, .38, .018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Change in one pixel's ink coverage (red +, blue −)")
    invalid = int(np.sum((x0 + separate < -1e-8) | (x0 + separate > 1 + 1e-8)))
    fig.text(.04, .026,
             f"The separate sum would put {invalid} pixels outside [0, 1]. "
             "The true joint image follows the renderer, including its clipping and overlap rules.\n"
             "This example was chosen near the pair's median interaction, not as a worst case.",
             fontsize=9.7, color=INK, linespacing=1.3)
    path = OUT / "joint_response_example.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path, {"pair": [KNOBS[i], KNOBS[j]], "sample_index": k,
                  "normalized_knobs_at_start": q0[0].tolist(),
                  "knobs_at_start": (LOW + q0[0] * SPAN).tolist(),
                  "interaction_over_individual_quadrature": float(relative[k]),
                  "separate_sum_out_of_bounds_pixels": invalid,
                  "individual_first_distance": float(l2(vi)),
                  "individual_second_distance": float(l2(vj)),
                  "actual_joint_distance": float(l2(actual)),
                  "interaction_distance": float(l2(interaction))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    q, base, vectors, raw, norm, sizes, scale = single_steps()
    along, turn, ratio = along_knob_paths()
    pairs, cosine, interaction, direction_change, joint = joint_steps(q, base, vectors)
    all_five = all_five_steps(q, base, vectors)
    jacobian = local_jacobian()
    # An independently scrambled sample checks that reported medians are not
    # an accident of the first low-discrepancy sequence.
    qr = .9 * sobol(1024, SEED + 101)
    xr = images(qr)
    vr = []
    for i in range(5):
        moved = qr.copy()
        moved[:, i] += STEP
        vr.append(images(moved) - xr)
    vr = np.stack(vr, axis=1)
    both = qr.copy()
    both[:, 3] += STEP
    both[:, 4] += STEP
    replicated_width_lean = l2(images(both) - xr - vr[:, 3] - vr[:, 4]) / np.sqrt(
        l2(vr[:, 3])**2 + l2(vr[:, 4])**2)
    replication = {
        "sample_count": 1024,
        "normalized_step_medians": {KNOBS[i]: float(np.median(l2(vr[:, i]))) for i in range(5)},
        "width_lean_interaction_share_median": float(np.median(replicated_width_lean)),
    }
    chosen_pair = max(pairs, key=lambda ij: interaction[ij])
    fig1 = plot_step_scale(raw, norm, sizes, scale)
    fig2 = plot_path_uniformity(along, turn, ratio)
    fig3 = plot_joint_heatmaps(cosine, interaction, all_five)
    fig4, example = plot_joint_example(q, vectors, chosen_pair, joint)
    results = {
        "renderer": "grey_ones.render, 28x28 coverage images",
        "metric": "Euclidean distance over all 784 coverage values",
        "random_seed": SEED,
        "single_step_starts": N,
        "single_step_start_region": "normalized q in [0,0.9]^5, Sobol sample",
        "knob_names": KNOBS,
        "knob_ranges": RANGES.tolist(),
        "raw_plus_0p1_pixel_distances": {KNOBS[i]: summary(raw[:, i]) for i in range(5)},
        "range_normalized_plus_0p1_pixel_distances": {KNOBS[i]: summary(norm[:, i]) for i in range(5)},
        "step_scaling": {"fractions": sizes.tolist(),
                         "median_pixel_distances": {KNOBS[i]: scale[i].tolist() for i in range(5)}},
        "along_knob_paths": {KNOBS[i]: {
            "segment_medians_by_start_fraction": np.median(along[i], axis=0).tolist(),
            "segment_p10_by_start_fraction": np.percentile(along[i], 10, axis=0).tolist(),
            "segment_p90_by_start_fraction": np.percentile(along[i], 90, axis=0).tolist(),
            "median_consecutive_direction_turn_degrees": float(np.median(turn[i])),
            "full_knob_path_to_chord_ratio": summary(ratio[i]),
            "within_path_step_cv": summary(np.std(along[i], axis=1) / np.mean(along[i], axis=1)),
        } for i in range(5)},
        "joint_starts": JOINT_N,
        "joint_pairs": joint,
        "all_five_plus_0p1_range": all_five,
        "median_pair_cosine": [[None if not np.isfinite(v) else float(v) for v in row]
                               for row in cosine],
        "median_pair_interaction_share": [[None if not np.isfinite(v) else float(v) for v in row]
                                          for row in interaction],
        "median_first_direction_turn_after_second_knob_degrees": [
            [None if not np.isfinite(v) else float(v) for v in row]
            for row in direction_change],
        "local_jacobian": jacobian,
        "independent_scrambled_sobol_replication": replication,
        "illustrated_joint_example": example,
        "figures": [str(p.relative_to(ROOT)) for p in (fig1, fig2, fig3, fig4)],
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "raw_medians": {KNOBS[i]: round(float(np.median(raw[:, i])), 4) for i in range(5)},
        "normalized_medians": {KNOBS[i]: round(float(np.median(norm[:, i])), 4) for i in range(5)},
        "normalized_p10_p90": {KNOBS[i]: [round(float(np.percentile(norm[:, i], p)), 4)
                                          for p in (10, 90)] for i in range(5)},
        "path_cv_medians": {KNOBS[i]: round(float(np.median(np.std(along[i], axis=1) /
                                                            np.mean(along[i], axis=1))), 3)
                            for i in range(5)},
        "path_turn_medians": {KNOBS[i]: round(float(np.median(turn[i])), 1) for i in range(5)},
        "chosen_pair": [KNOBS[k] for k in chosen_pair],
        "chosen_pair_interaction_share": float(interaction[chosen_pair]),
        "all_five": all_five,
        "jacobian_min_singular_median": jacobian["smallest_singular_value"]["median"],
        "jacobian_condition_median": jacobian["largest_to_smallest_singular_ratio"]["median"],
        "replication": replication,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
