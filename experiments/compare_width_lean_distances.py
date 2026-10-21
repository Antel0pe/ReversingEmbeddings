"""Compare image gaps, vector gaps, equal angular steps, and the four corners.

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/compare_width_lean_distances.py
"""
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/width-lean-distances-mpl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grey_ones import render, RANGES
from experiments.width_vectors_by_lean import norm, angle

OUT = ROOT / "figures/generated_one_width_vectors_by_lean/distance_comparisons"
BASE = np.array([14., 14., 19.75, 3.2, 0.])
COLORS = plt.colormaps["turbo"]


def f(width, lean, dtype=np.float32):
    w, t = np.broadcast_arrays(width, lean)
    p = np.tile(BASE, (*w.shape, 1))
    p[..., 3], p[..., 4] = w, t
    return render(p.reshape(-1, 5), dtype=dtype).astype(np.float64).reshape(*w.shape, 784)


def color(lean):
    if lean == 0:
        return "#202a35"
    return {5:"#357fb0",10:"#319875",15:"#72972d",20:"#ad8a18",25:"#d27729",30:"#c24936",35:"#842e50",-10:"#6955a1",-5:"#4571ad",34:"#842e50"}.get(float(lean), COLORS((lean+10)/45))


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=150)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def decorate(ax, xlabel, ylabel, title, xlim):
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xlim=xlim)
    ax.grid(alpha=.17)


def width_lines(w, t, x, upright, root_path, mixed):
    fig = plt.figure(figsize=(14, 7.7), facecolor="white")
    fig.text(.055, .95, "One line per lean: how far apart are the images and the width vectors?", fontsize=18, weight="bold")
    fig.text(.055, .888, "Horizontal axis = width. Each colored line holds lean fixed. Black = upright (0°).\n"
             "Generated coverage images; center (14, 14), height 19.75 px fixed. Distance = L2 over all 784 pixels.", fontsize=11, linespacing=1.5)
    axes = [fig.add_axes([left, .34, .355, .39]) for left in [.075, .565]]
    selected = [0, 5, 10, 15, 20, 25, 30, 35]
    gap = norm(x-upright)
    residual = norm(mixed)
    for lean in selected:
        i = int(np.argmin(abs(t-lean)))
        for ax, values in zip(axes, [gap, residual]):
            ax.plot(w, values[i], color=color(lean), ls="--" if lean == 0 else "-", lw=2 if lean == 0 else 1.8,
                    label=(f"±{lean:g}°" if lean in [5,10] else f"{lean:+g}°") if lean else "0° reference")
    decorate(axes[0], "Width (horizontal pixels)", "Image distance to same-width upright 1", "1. Actual gap between the width sweeps", (w[0], w[-1]))
    decorate(axes[1], "Width (horizontal pixels)", "Distance between the two width-change vectors", "2. Does the upright width vector still work?", (w[0], w[-1]))
    axes[0].set_ylim(bottom=-.12)
    axes[1].set_ylim(bottom=-.12)
    for ax in axes:
        ax.axvline(3.2, color="#777", ls=":", lw=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .835), ncol=8, frameon=False, fontsize=10, handlelength=1.6, columnspacing=1.25)
    fig.text(.075, .17, "Left: distance between image(width, lean) and image(width, 0°). Compare leans at matching widths.\n"
             "Right: compare [image(width, lean) − image(3.2, lean)] with [image(width, 0°) − image(3.2, 0°)].\n"
             "Zero on the right means the same width-change vector works at both leans. It is forced to zero at width 3.2.", fontsize=10.5, linespacing=1.6)
    fig.text(.075, .055, "±5° and ±10° have identical distances here by left/right symmetry; their curves are combined. Widths: 1.8–4.6 px every 0.05 px.\n"
             "These are distances at corresponding settings, not nearest distances between entire curves. This is a two-knob slice, with no projection.", fontsize=10, linespacing=1.5)
    save(fig, "04_width_axis_distances")


def tick_plots(w, starts, ticks, tick_root, tick_upright, proximity_w, proximity_ratio):
    fig = plt.figure(figsize=(14, 10), facecolor="white")
    fig.text(.055, .952, "Does the same +1° lean step give a standard pixel-space vector?", fontsize=19, weight="bold")
    fig.text(.055, .898, "Each line starts at the labeled lean and adds exactly 1°. Horizontal axis = width. Black = the 0° → 1° step.\n"
             "A lean-step vector is image(width, lean + 1°) − image(width, lean). Center (14, 14), height 19.75 px fixed.", fontsize=11, linespacing=1.5)
    positions = [[.08, .565, .355, .25], [.58, .565, .355, .25],
                 [.08, .19, .355, .25], [.58, .19, .355, .25]]
    axes = [fig.add_axes(p) for p in positions]
    quantities = [norm(ticks), norm(ticks-tick_root[:, None]), norm(ticks-tick_upright)]
    for i, lean in enumerate(starts):
        for ax, values in zip(axes[:3], quantities):
            ax.plot(w, values[i], color=color(lean), ls="--" if lean == 0 else "-", lw=1.8, label=f"{lean:g}° → {lean+1:g}°")
        axes[3].plot(proximity_w, proximity_ratio[i], color=color(lean), ls="--" if lean == 0 else "-", lw=1.8)
    for ax in axes[:3]:
        ax.axvline(3.2, color="#888", ls=":", lw=1)
    titles = ["1. Amount of change in a +1° lean step", "2. Does that vector transfer across widths?",
              "3. Does that vector transfer across lean?", "4. Which chosen neighboring image is closer?"]
    ylabels = ["Step length\n(coverage L2)", "Gap to same step\nat width 3.2 (L2)",
               "Gap to upright's +1° step\nat same width (L2)", "+1° lean distance /\n+0.1 px width distance"]
    for j, ax in enumerate(axes):
        decorate(ax, "Starting width (pixels)", ylabels[j], titles[j], (w[0], w[-1] if j < 3 else proximity_w[-1]))
        ax.set_ylim(bottom=-.025 if j < 3 else 0)
    axes[3].axhline(1, color="#444", ls=":", lw=1.4)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .865), ncol=8, frameon=False, fontsize=9.7, handlelength=1.7, columnspacing=1.2)
    fig.text(.055, .045, "Panel 2: zero means the same +1° vector works at width 3.2 and the plotted width. Panel 3: zero means it works at both leans.\n"
             "Panel 4: ratio > 1 means the selected width neighbor is closer; ratio < 1 means the selected lean neighbor is closer.\n"
             "Local exception: the 0° → 1° vector matches across sampled widths 2.35–3.65 px (vector gap below 0.000001).\n"
             "Distances and vector gaps use all 784 coordinates. Step sizes matter: 0.1 px and 1° are explicit choices, not equivalent units.", fontsize=10.5, linespacing=1.5)
    save(fig, "05_equal_lean_steps")


def lean_lines(t, widths, curves, root_path, upright, corner):
    fig = plt.figure(figsize=(15.5, 7.6), facecolor="white")
    fig.text(.05, .95, "Now follow the lean line: what changes when you start from another width?", fontsize=18, weight="bold")
    fig.text(.05, .889, "Horizontal axis = lean. Each colored line holds width fixed. Black = root width 3.2 px; thick blue = maximum width 4.6 px.\n"
             "Lean-change vector = image(width, lean) − image(width, 0°). Center (14, 14), height 19.75 px fixed.", fontsize=11, linespacing=1.5)
    axes = [fig.add_axes([left, .35, .245, .38]) for left in [.065, .39, .715]]
    palettes = ["#ca7121", "#bba03b", "#202a35", "#4c978b", "#206b9c"]
    lean_vectors = curves-upright[None]
    values = [norm(lean_vectors), norm(lean_vectors-root_path[:, None]), norm(curves-f(3.2, t)[:, None])]
    for j, (width, c) in enumerate(zip(widths, palettes)):
        for ax, z in zip(axes, values):
            ax.plot(t, z[:, j], color=c, ls="--" if width == 3.2 else "-", lw=2.4 if width == 4.6 else 1.8,
                    label=f"width {width:g} px" + (" · reference" if width == 3.2 else ""))
    titles = ["1. How long is the lean-change vector?", "2. How different is it from the root-width vector?", "3. How far apart are same-lean images?"]
    ylabels = ["Distance from own upright image\n(coverage L2)", "Gap between lean-change vectors\n(coverage L2)", "Gap to width 3.2 at same lean\n(coverage L2)"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        decorate(ax, "Lean (degrees)", ylabel, title, (t[0], t[-1]))
        ax.axvline(0, color="#999", ls=":", lw=1)
        ax.set_ylim(bottom=-.12)
    axes[1].scatter([35], [corner["mixed_gap_L2"]], color=palettes[-1], s=40, zorder=5)
    axes[1].annotate(f"max width + lean: {corner['mixed_gap_L2']:.3f}", (35, corner["mixed_gap_L2"]), xytext=(-8, -26), textcoords="offset points", ha="right", fontsize=9.5,
                     arrowprops=dict(arrowstyle="-",color="#777"),bbox=dict(facecolor="white",edgecolor="none",pad=1))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .84), ncol=5, frameon=False, fontsize=10.5)
    fig.text(.065, .186, "Panel 2 removes each line's different upright starting image before comparing vectors. Panel 3 compares the actual images.\n"
             "The max-width line in panel 2 is also the error from adding the two original root vectors to predict the combined image.\n"
             "This same error measures the change in the width vector after leaning: both expressions are the same four-image mixed difference.", fontsize=10.5, linespacing=1.6)
    fig.text(.065, .055, "Sampled every 0.5° from −10° to 35°. Each line contains 91 measurements. Distances use the original 784 coverage coordinates.\n"
             "A zero vector gap would mean a transferable pixel vector; it is a stronger condition than reaching the same endpoint in either parameter order.", fontsize=10, linespacing=1.5)
    save(fig, "06_lean_axis_distances")


def csv_file(name, headers, rows):
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def page(summary):
    template = (ROOT / "experiments/width_lean_distance_comparisons.html").read_text()
    (OUT / "index.html").write_text(template.replace("__COMPARISON_DATA__", json.dumps(summary, separators=(",", ":"))))
    # Put the quantitative comparison first in the earlier viewer, retaining it.
    marker = '<div class="explain" id="distance-comparisons"><h2>Compare distances and vectors</h2><p><a href="distance_comparisons/">Open the distance plots and four-corner comparison</a>: one line per lean over width, equal +1° steps, and lean paths at different widths.</p></div>'
    for path in [ROOT / "experiments/width_vectors_by_lean.html", OUT.parent / "index.html"]:
        text = path.read_text()
        if 'id="distance-comparisons"' not in text:
            text = text.replace('<div class="controls">', marker+'\n<div class="controls">', 1)
            path.write_text(text)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size":10.5, "axes.titlesize":11.5, "axes.spines.top":False, "axes.spines.right":False})
    w = np.linspace(*RANGES[3], 57)
    t = np.linspace(*RANGES[4], 91)
    x = f(w[None], t[:, None])
    upright = f(w, 0)
    ref_images = f(3.2, t)
    root = f(3.2, 0)
    lean_root = ref_images-root
    mixed = (x-upright)-lean_root[:, None]
    other_mixed = (x-ref_images[:, None])-(upright-root)
    assert np.allclose(mixed, other_mixed, atol=1e-14)
    assert np.max(abs(mixed[np.argmin(abs(t))])) == 0
    assert np.max(abs(mixed[:, np.argmin(abs(w-3.2))])) < 1e-14

    # Four corners, all measured with the actual renderer.
    A, B, C, D = root, f(4.6, 0), f(3.2, 35), f(4.6, 35)
    W, L = B-A, C-A
    W_after, L_after = D-C, D-B
    interaction = L_after-L
    assert np.allclose(interaction, W_after-W, atol=1e-14)
    assert np.allclose(A+W+L_after, D, atol=1e-14)
    assert np.allclose(A+L+W_after, D, atol=1e-14)
    # Actual parameter updates commute, independently of the pixel-vector sums.
    params_wl = BASE.copy(); params_wl[3] = 4.6; params_wl[4] = 35
    params_lw = BASE.copy(); params_lw[4] = 35; params_lw[3] = 4.6
    assert np.array_equal(render(params_wl), render(params_lw))
    corner = dict(root_settings=BASE.tolist(), width_only_settings=params_wl.tolist()[:4]+[0.],
                  lean_only_settings=[14.,14.,19.75,3.2,35.], combined_settings=params_wl.tolist(),
                  width_from_root_L2=float(norm(W)), lean_from_root_L2=float(norm(L)),
                  lean_after_max_width_L2=float(norm(L_after)), width_after_max_lean_L2=float(norm(W_after)),
                  mixed_gap_L2=float(norm(interaction)), joint_displacement_L2=float(norm(D-A)),
                  additive_prediction_relative_L2_error=float(norm(interaction)/norm(D-A)),
                  width_vector_angle_before_after_lean_degrees=float(angle(W, W_after)),
                  lean_vector_angle_before_after_width_degrees=float(angle(L, L_after)),
                  parameter_endpoint_order_error_L2=0.,
                  max_width_vector_change_equals_max_width_lean_vector_change=True)

    starts = np.array([-10., -5., 0., 5., 10., 20., 30., 34.])
    ticks = f(w[None], starts[:, None]+1)-f(w[None], starts[:, None])
    tick_root = f(3.2, starts+1)-f(3.2, starts)
    tick_upright = f(w, 1)-f(w, 0)
    transfer_gaps = norm(tick_upright-(f(3.2,1)-root))
    transfer_mask = transfer_gaps < 1e-6
    transferable_widths = w[transfer_mask]
    assert np.all(np.diff(np.flatnonzero(transfer_mask)) == 1)
    tick64 = f(transferable_widths,1,np.float64)-f(transferable_widths,0,np.float64)
    tick64_root = f(3.2,1,np.float64)-f(3.2,0,np.float64)
    transfer64_max = float(norm(tick64-tick64_root).max())
    assert transfer64_max < 1e-12
    prox_w = np.linspace(1.8, 4.5, 55)
    prox_start = f(prox_w[None], starts[:, None])
    prox_lean = norm(f(prox_w[None], starts[:, None]+1)-prox_start)
    prox_width = norm(f(prox_w[None]+.1, starts[:, None])-prox_start)
    prox_ratio = prox_lean/prox_width

    # Denser leans verify ordering and the numerical ranges; no distances inferred
    # from an embedding. Compare both raw steps and equal range fractions.
    test_t = np.arange(-10, 34.001, .25)
    test_w = np.linspace(1.8, 4.5, 109)
    sx = f(test_w[None], test_t[:, None])
    lean_test = norm(f(test_w[None], test_t[:, None]+1)-sx)
    width_test = norm(f(test_w[None]+.1, test_t[:, None])-sx)
    equal_fraction_width_step = float(np.ptp(RANGES[3])/np.ptp(RANGES[4]))
    equal_width_test = norm(f(test_w[None]+equal_fraction_width_step, test_t[:, None])-sx)
    ratio = lean_test/width_test
    equal_ratio = lean_test/equal_width_test
    fine_w = np.linspace(1.8, 4.6, 113)
    fine_t = np.linspace(-10, 35, 181)
    fine_x = f(fine_w[None], fine_t[:, None])
    fine_mixed = (fine_x-f(fine_w, 0))- (f(3.2, fine_t)-root)[:, None]
    assert np.array_equal(fine_x[::2, ::2], x)
    assert np.allclose(fine_mixed[::2, ::2], mixed, atol=1e-14)
    assert np.all(ratio > 1) and np.all(equal_ratio > 1)
    # Check whether the matching-lean image gap increases with width: the answer
    # is not guaranteed, even for equal lean increments.
    gap1 = norm(f(w[None], test_t[:, None]+1)-f(w[None], test_t[:, None]))
    decrease_fraction = float(np.mean(np.diff(gap1, axis=1) < -1e-6))
    corner64 = [f(width, lean, np.float64) for width, lean in [(3.2,0),(4.6,0),(3.2,35),(4.6,35)]]
    mixed64 = corner64[3]-corner64[1]-corner64[2]+corner64[0]
    precision = float(abs(mixed64-interaction).max())
    assert precision < 2e-7

    measured_widths = np.array([1.8,2.5,3.2,3.9,4.6])
    lean_curves = f(measured_widths[None], t[:,None])
    lean_upright = f(measured_widths,0)
    selected = []
    for lean in [0,1,5,10,15,20,25,30,35]:
        i = int(np.argmin(abs(t-lean)))
        selected.append(dict(lean_degrees=lean,
                             lean_distance_at_width_3p2=float(norm(lean_root[i])),
                             lean_distance_at_width_4p6=float(norm(x[i,-1]-upright[-1])),
                             vector_gap_max_width_vs_root_width=float(norm(mixed[i,-1])),
                             width_after_lean_L2=float(norm(x[i,-1]-ref_images[i])),
                             width_vector_angle_degrees=float(angle(x[i,-1]-ref_images[i], W)),
                             lean_vector_angle_degrees=float(angle(x[i,-1]-upright[-1], lean_root[i])) if lean else None,
                             same_lean_gap_between_width_extremes=float(norm(x[i,-1]-x[i,0]))))
    summary = dict(fixed=dict(cx=14, cy=14, height=19.75), root=BASE.tolist(),
                   grid=dict(widths=57, width_step_px=.05, leans=91, lean_step_degrees=.5),
                   distance="L2 over all 784 pixel coverage entries; no projection; float32 rendering then float64 differences",
                   corner=corner, selected=selected,
                   equal_lean_step=dict(degrees=1, test_count=int(lean_test.size),
                                        vector_length_range=[float(lean_test.min()),float(lean_test.max())],
                                        matching_step_image_gap_decreases_with_width_fraction=decrease_fraction,
                                        upright_step_matching_sampled_width_interval=transferable_widths[[0,-1]].tolist(),
                                        upright_step_max_vector_gap_on_interval_float32=float(transfer_gaps[transfer_mask].max()),
                                        upright_step_max_vector_gap_on_interval_float64=transfer64_max,
                                        upright_step_length_width_3p2=float(norm(f(3.2,1)-root)),
                                        upright_step_length_width_4p6=float(norm(f(4.6,1)-B)),
                                        upright_step_angle_width_3p2_vs_4p6_degrees=float(angle(f(3.2,1)-root,f(4.6,1)-B))),
                   proximity=dict(raw_width_step_px=.1, lean_step_degrees=1,
                                  lean_distance_divided_by_width_distance_range=[float(ratio.min()),float(ratio.max())],
                                  equal_fraction_width_step_px=equal_fraction_width_step,
                                  equal_fraction_of_both_ranges=1/45,
                                  equal_fraction_ratio_range=[float(equal_ratio.min()),float(equal_ratio.max())],
                                  width_neighbor_closer_fraction=float(np.mean(ratio>1)),
                                  sampled_width_bounds=[1.8,4.5], sampled_lean_start_bounds=[-10,34],
                                  refined_width_step=.025, refined_lean_start_step=.25),
                   geometric_lean=dict(row_shift="(14-y) * [tan(lean+step)-tan(lean)], with angles in degrees",
                                       slope_step_0_to_1=float(np.tan(np.deg2rad(1))),
                                       slope_step_34_to_35=float(np.tan(np.deg2rad(35))-np.tan(np.deg2rad(34)))),
                   checks=dict(refined_grid_images=int(len(fine_w)*len(fine_t)),
                               matched_grid_images_identical=True,
                               two_expressions_for_mixed_difference_max_pixel_error=float(abs(mixed-other_mixed).max()),
                               corner_mixed_float32_vs_float64_max_pixel_difference=precision,
                               updated_vectors_reach_exact_corner_within_float64_roundoff=True,
                               parameter_order_endpoints_bit_identical=True),
                   understanding_check=dict(object="Distances between generated coverage images and between explicit pixel-change vectors",
                                            axes="Width or lean as labeled; lines fix the other knob; every y axis names its comparison",
                                            baseline="Upright lean and root width 3.2; reference lines and exact zeros are marked",
                                            evidence="Matching-setting image gaps, vector residuals, equal-step lengths, and explicit corner numbers",
                                            scope="Fixed center and height; corresponding-point distances, not nearest-curve distances; finite increments"))
    (OUT / "results.json").write_text(json.dumps(summary,indent=2)+"\n")
    csv_file("width_axis_distances.csv", ["lean_deg","width_px","image_gap_to_same_width_upright_L2","width_vector_gap_to_upright_L2","distance_from_root_L2"],
             ([lean,width,norm(x[i,j]-upright[j]),norm(mixed[i,j]),norm(x[i,j]-root)] for i,lean in enumerate(t) for j,width in enumerate(w)))
    csv_file("equal_lean_steps.csv", ["lean_start_deg","lean_end_deg","width_px","step_length_L2","vector_gap_to_same_step_at_width_3p2_L2","vector_gap_to_0_to_1_at_same_width_L2","angle_to_0_to_1_at_same_width_deg"],
             ([lean,lean+1,width,norm(ticks[i,j]),norm(ticks[i,j]-tick_root[i]),norm(ticks[i,j]-tick_upright[j]),angle(ticks[i,j],tick_upright[j])] for i,lean in enumerate(starts) for j,width in enumerate(w)))
    csv_file("lean_axis_distances.csv", ["lean_deg","width_px","own_upright_gap_L2","lean_vector_gap_to_width_3p2_L2","image_gap_to_width_3p2_at_same_lean_L2"],
             ([lean,width,norm(lean_curves[i,j]-lean_upright[j]),norm(lean_curves[i,j]-lean_upright[j]-lean_root[i]),norm(lean_curves[i,j]-ref_images[i])] for i,lean in enumerate(t) for j,width in enumerate(measured_widths)))
    csv_file("neighbor_proximity.csv", ["lean_start_deg","width_start_px","distance_of_1deg_lean_L2","distance_of_0p1px_width_L2","distance_ratio","equal_range_fraction_width_distance_L2","equal_range_fraction_ratio"],
             ([lean,width,lean_test[i,j],width_test[i,j],ratio[i,j],equal_width_test[i,j],equal_ratio[i,j]] for i,lean in enumerate(test_t) for j,width in enumerate(test_w)))
    width_lines(w,t,x,upright,lean_root,mixed)
    tick_plots(w,starts,ticks,tick_root,tick_upright,prox_w,prox_ratio)
    lean_lines(t,measured_widths,lean_curves,lean_root,lean_upright,corner)
    page(summary)
    (OUT / "README.md").write_text("""# Width and lean: distances and transferable vectors

The generated 1 has fixed center `(14,14)` and height `19.75 px`. Write `F(w,l)` for its 784 pixel coverage values at width `w` and lean `l`.
The normal root is `F(3.2,0)`. Width ranges 1.8–4.6 px; lean ranges −10°–35°.

## Three distinct comparisons

1. **Image gap:** `length(F(w,l)-F(w,0))`. How far apart are two 1s at the same width?
2. **Width-vector gap:** compare `F(w,l)-F(3.2,l)` with `F(w,0)-F(3.2,0)`.
3. **Lean-vector gap:** compare `F(w,l)-F(w,0)` with `F(3.2,l)-F(3.2,0)`.

The vector differences in 2 and 3 are identical. They both equal the mixed difference
`F(w,l)-F(w,0)-F(3.2,l)+F(3.2,0)`.
Subtracting each curve's own starting image matters: an image gap and a gap between vectors with different origins answer different questions.
All lengths are Euclidean over 784 values: square each entry, sum, and take the square root.

## Equal angular increments

`T(w,l) = F(w,l+1)-F(w,l)` always uses a +1° step.
We compare its length, its difference from `T(3.2,l)`, and its difference from `T(w,0)`.
Similar lengths do not imply similar vectors.
There is a local reuse interval: the 0-to-1-degree step matches across sampled widths 2.35-3.65 px, with vector gaps below 1e-6 after float32 storage and below 1e-12 before storage.
At widths 3.2 and 4.6 the same step has length 0.615 in both cases, but the two vectors are 90 degrees apart.
The canvas displacement at vertical coordinate `y` is `(14-y) * [tan(l+1)-tan(l)]` (angles in degrees), independent of width.
Equal degree steps are not exactly equal shear increments, and sampled pixel changes depend on edge positions and coverage overlaps.

## Four corners and commutativity

- `A = F(3.2,0)` normal root.
- `B = F(4.6,0)` max width only.
- `C = F(3.2,35)` max lean only.
- `D = F(4.6,35)` both maxima.

Known root vectors: `W = B-A`, `L = C-A`.
Recovered vectors: lean after widening is `D-B`; width after leaning is `D-C`.
Their changes satisfy `(D-B)-L = (D-C)-W = D-B-C+A`.
Both actual orders reach `D` exactly. Adding the original vectors predicts `B+C-A`; its error is the mixed difference.
Pixel-vector addition itself commutes. Its failure to match `D` is failure of a state-independent additive rule, not noncommutativity of those additions or of the known parameter setters.
The three images A, B, and C alone do not determine D; here the renderer supplies the fourth corner.
Distances alone cannot establish commutativity for an unknown learned direction field.

See `results.json` for measured corner lengths, angles, and checks, and the three CSV files for plotted data.
`neighbor_proximity.csv` additionally compares +0.1 px width against +1° lean, then +2.8/45 px width against +1° lean (equal fractions of the allowed ranges).
The sampled comparison uses starting widths 1.8–4.5 px and starting leans −10°–34°, so every endpoint remains in range.
These comparisons depend on specified step sizes; there is no unit-independent rule that all width neighbors are closer.

## Reproduce and scope

Run `OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/compare_width_lean_distances.py` from the repository root.
The script uses `grey_ones.render`, including its 16-subrow model and default float32 storage, and computes differences in float64.
The source grid has 57 widths × 91 leans; validation halves both spacings and checks every corresponding image.
The plots compare corresponding settings, not nearest distances between entire curves.
All claims apply to this sampled slice; no projection, decoder, or full-manifold equivalence is claimed.
""")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
