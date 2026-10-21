"""Measure full-width and small-width pixel vectors across the allowed lean range.

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/width_vectors_by_lean.py
"""

import base64
import csv
import gzip
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/generated-width-lean-mpl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grey_ones import KNOBS, RANGES, SUB, render

OUT = ROOT / "figures/generated_one_width_vectors_by_lean"
BASE = np.array([14., 14., 19.75, 3.2, 0.])


def norm(x):
    return np.linalg.norm(x, axis=-1)


def angle(a, b):
    cosine = np.sum(a * b, axis=-1) / (norm(a) * norm(b))
    # Avoid reporting a roundoff angle for identical vectors.
    cosine = np.where(np.abs(cosine - 1) < 1e-14, 1, cosine)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def grid(widths, leans):
    p = np.tile(BASE, (len(leans), len(widths), 1))
    p[:, :, 3], p[:, :, 4] = widths, leans[:, None]
    return render(p.reshape(-1, 5)).reshape(len(leans), len(widths), 784)


def metrics(x, widths, leans):
    x = x.astype(np.float64)
    full = x[:, -1] - x[:, 0]
    steps = np.diff(x, axis=1)
    length = norm(full)
    upright = int(np.argmin(abs(leans)))
    return dict(
        full=full, length=length, steps=steps,
        angle_upright=angle(full, full[upright]),
        mismatch=norm(full-full[upright]),
        arc=norm(steps).sum(axis=1),
        speed=norm(steps)/np.diff(widths),
        local_angle_upright=angle(steps, steps[upright]),
        local_angle_chord=angle(steps, full[:, None]),
        neighbor_angle=angle(full[2:], full[:-2]),
    )


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 150, "figure.facecolor": "white"})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", facecolor="white")
    plt.close(fig)


def image_patterns(x, widths, leans, m):
    fig = plt.figure(figsize=(16, 10.3))
    fig.text(.04, .958, "What does the min-to-max width vector look like at each lean?", fontsize=20, weight="bold")
    fig.text(.04, .914, "A generated 1 is a 28 × 28 coverage image: white = no ink (0), black = full ink (1).\n"
             "Center (14, 14) and height 19.75 px stay fixed. Width is horizontal; positive lean tilts the top right.", fontsize=11, linespacing=1.5)
    cases = [0, -10, 5, 15, 25, 35]
    row_y = [.690, .505, .320, .135]
    row_labels = ["START IMAGE\nwidth = 1.8 px", "END IMAGE\nwidth = 4.6 px",
                  "WIDTH VECTOR\nend − start", "VECTOR CHANGE\nthis vector − upright vector"]
    for y, label in zip(row_y, row_labels):
        fig.text(.035, y+.078, label, fontsize=11, weight="bold", va="center", linespacing=1.6)
    for col, lean in enumerate(cases):
        i = int(np.argmin(abs(leans-lean)))
        left = .23+col*.116
        arrays = [x[i, 0], x[i, -1], m["full"][i], m["full"][i]-m["full"][np.argmin(abs(leans))]]
        for row, (y, arr) in enumerate(zip(row_y, arrays)):
            ax = fig.add_axes([left, y, .096, .148])
            ax.imshow(arr.reshape(28, 28), cmap="gray_r" if row < 2 else "RdBu_r",
                      vmin=0 if row < 2 else -1, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_color("#202a35" if col == 0 else "#dddddd")
                s.set_linewidth(2 if col == 0 else .7)
            if row == 0:
                ax.set_title("Upright reference\n0°" if col == 0 else f"Lean {lean:+g}°", fontsize=11, pad=12)
            if row == 2:
                ax.text(.5, -.055, f"length {m['length'][i]:.3f}", transform=ax.transAxes, ha="center", va="top", fontsize=10)
            if row == 3:
                ax.text(.5, -.055, f"change {m['mismatch'][i]:.3f}", transform=ax.transAxes, ha="center", va="top", fontsize=10)
    cax = fig.add_axes([.94, .155, .013, .305])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-1, 1))
    fig.colorbar(sm, cax=cax, ticks=[-1, 0, 1], label="Coverage difference")
    fig.text(.035, .057, "Width-vector row: red = added ink; blue = removed ink (none occurs when widening here). White = no change.\n"
             "Bottom row: red/blue = a larger/smaller width response than upright, not direct ink gain/loss. Same scale; no clipping.\n"
             "Lengths and changes use all 784 pixel coordinates. This is a fixed-center, fixed-height slice, with no projection.", fontsize=10.5, linespacing=1.5)
    save(fig, "01_pixel_vectors")


def whole_vector_plots(leans, m):
    fig = plt.figure(figsize=(14, 9.5))
    fig.text(.055, .952, "How does the full width vector change as lean increases?", fontsize=20, weight="bold")
    fig.text(.055, .901, "At each lean: width vector = image(width 4.6) − image(width 1.8). Center (14, 14), height 19.75 px fixed.\n"
             "All measurements use 784 coverage values directly. The 0° upright vector is the reference; the sweep includes both bounds.", fontsize=11, linespacing=1.5)
    positions = [[.075, .555, .37, .265], [.575, .555, .37, .265],
                 [.075, .175, .37, .265], [.575, .175, .37, .265]]
    axes = [fig.add_axes(p) for p in positions]
    blue, orange = "#236c94", "#b85a27"
    axes[0].plot(leans, m["length"], color=blue, lw=2)
    axes[0].set(title="1. How big is the min-to-max change?", ylabel="Vector length (coverage L2)", ylim=(0, 7.25))
    axes[0].text(.03, .09, f"Range: {m['length'].min():.3f}–{m['length'].max():.3f}\n0 baseline shown; dashed = upright length", transform=axes[0].transAxes, fontsize=10)
    axes[0].axhline(m["length"][np.argmin(abs(leans))], color="#888", ls="--", lw=1)
    axes[1].plot(leans, m["angle_upright"], color=orange, lw=2)
    axes[1].set(title="2. How far has its direction turned from upright?", ylabel="Angle to upright width vector (degrees)", ylim=(-2, 92))
    axes[1].axhline(90, color="#aaa", ls=":", lw=1)
    axes[1].text(.03, .94, "0° = same direction; 90° = perpendicular", transform=axes[1].transAxes, fontsize=10, va="top")
    axes[2].plot(leans[1:-1], m["neighbor_angle"], color=orange, lw=2)
    axes[2].set(title="3. What does one more degree of lean do?", ylabel="Angle between width vectors (degrees)", ylim=(0, 12))
    axes[2].text(.03, .08, "Compare lean − 0.5° with lean + 0.5°\nThis is a finite 1° comparison, not a derivative.", transform=axes[2].transAxes, fontsize=10)
    ratio = m["arc"]/m["length"]
    axes[3].plot(leans, ratio, color=blue, lw=2)
    axes[3].axhline(1, color="#555", ls="--", lw=1)
    axes[3].set(title="4. Is widening itself a straight image-space path?", ylabel="Sampled path length / endpoint distance", ylim=(.98, 1.42))
    axes[3].text(.03, .075, "1 = straight, with no backtracking\nAbove 1 = intermediate width steps turn", transform=axes[3].transAxes, fontsize=10)
    for ax in axes:
        ax.axvline(0, color="#999", ls=":", lw=1)
        ax.set(xlabel="Lean (degrees); 0 = upright", xlim=(leans[0], leans[-1]))
        ax.grid(alpha=.17)
    fig.text(.055, .058, "57 widths every 0.05 px × 91 leans every 0.5°. Path length sums the 56 consecutive width-step lengths.\n"
             "The endpoint vector is one displacement between two images; it does not encode the intermediate widening path.\n"
             "These are measurements of this two-knob slice, not an embedding or a claim about the full five-knob family.", fontsize=10.5, linespacing=1.5)
    save(fig, "02_whole_vector_measurements")


def local_plots(widths, leans, m):
    fig = plt.figure(figsize=(14, 7.2))
    fig.text(.055, .94, "Does a small width step behave the same everywhere?", fontsize=20, weight="bold")
    fig.text(.055, .872, "Each cell measures image(width + 0.05) − image(width), at the cell's lean. Center (14, 14), height 19.75 px fixed.\n"
             "Horizontal position = midpoint of that width interval. Color compares full 784-value vectors, with no projection.", fontsize=11, linespacing=1.5)
    mid = (widths[:-1]+widths[1:])/2
    specs = [("speed", "How much pixel change per unit of width?", "Vector length / 0.05 px", "viridis", 0, float(m["speed"].max())),
             ("local_angle_upright", "How much has the small-step direction changed?", "Angle to same width step at lean 0°", "magma", 0, 90)]
    for left, (key, title, label, cmap, lo, hi) in zip([.075, .565], specs):
        ax = fig.add_axes([left, .31, .34, .435])
        mesh = ax.pcolormesh(mid, leans, m[key], shading="nearest", cmap=cmap, vmin=lo, vmax=hi)
        ax.axhline(0, color="white", ls="--", lw=1)
        ax.set(title=title, xlabel="Width-interval midpoint (pixels)", ylabel="Lean (degrees)")
        cax = fig.add_axes([left+.355, .31, .014, .435])
        fig.colorbar(mesh, cax=cax, label=label)
    fig.text(.075, .12, f"Left: speed ranges from {m['speed'].min():.2f} to {m['speed'].max():.2f} coverage-L2 units per width pixel.\n"
             "Right: 0° = same small-step direction as upright; 90° = perpendicular. Dashed row is the upright reference.\n"
             "A small-step vector depends on both width and lean. The full min-to-max vector adds these changing steps together.", fontsize=11, linespacing=1.7)
    fig.text(.075, .045, "Sampled grid: 56 width intervals × 91 leans. Positive lean tilts the top right. All sampled values are inside the shown color scales.\n"
             "This is a finite 0.05-pixel response, not an infinitesimal tangent. Other center/height choices may give different patterns.", fontsize=10, linespacing=1.5)
    save(fig, "03_small_width_steps")


def write_html(x, widths, leans, summary):
    payload = dict(widths=widths.tolist(), leans=leans.tolist(),
                   shape=list(x.shape), samples=base64.b64encode(gzip.compress(x.astype("<f4").tobytes())).decode(),
                   summary=summary)
    template = (ROOT / "experiments/width_vectors_by_lean.html").read_text()
    (OUT / "index.html").write_text(template.replace("__EXPERIMENT_DATA__", json.dumps(payload, separators=(",", ":"))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    widths = np.linspace(*RANGES[KNOBS.index("width")], 57)
    leans = np.linspace(*RANGES[KNOBS.index("lean")], 91)
    x = grid(widths, leans)
    m = metrics(x, widths, leans)
    # Meaningful checks: monotone widening, conserved ink area, and exact
    # telescoping of width steps to the endpoint vector, without a projection.
    assert x.shape == (91, 57, 784) and np.isfinite(x).all()
    assert x.min() >= 0 and x.max() <= 1
    assert m["steps"].min() >= 0
    mass_error = float(np.max(abs(x.astype(float).sum(axis=-1)-BASE[2]*widths)))
    assert mass_error < 2e-5
    telescoping = float(abs(m["steps"].sum(axis=1)-m["full"]).max())
    assert telescoping < 1e-12
    fine_widths = np.linspace(widths[0], widths[-1], 113)
    fine_leans = np.linspace(leans[0], leans[-1], 181)
    fine = grid(fine_widths, fine_leans)
    # Width refinement tests sampled path lengths, not only endpoint chords.
    fine_arc = norm(np.diff(fine[::2].astype(float), axis=1)).sum(axis=1)
    arc_change = abs(fine_arc-m["arc"])/fine_arc
    assert arc_change.max() < .003
    fine_d = fine[:, -1].astype(float)-fine[:, 0].astype(float)
    assert np.array_equal(fine[::2, ::2], x)
    fine_angles = angle(fine_d[4:], fine_d[:-4])
    assert np.allclose(fine_angles[::2], m["neighbor_angle"], atol=1e-10)
    # Float64 uses the same 16-subrow coverage model, before float32 storage.
    p = np.tile(BASE, (len(leans)*2, 1))
    p[:, 3] = np.tile(widths[[0, -1]], len(leans))
    p[:, 4] = np.repeat(leans, 2)
    d64 = np.diff(render(p, dtype=np.float64).reshape(len(leans), 2, 784), axis=1)[:, 0]
    precision = float(abs(d64-m["full"]).max())
    assert precision < 1e-7
    chosen = []
    for lean in [-10, 0, 5, 10, 15, 20, 25, 30, 35]:
        i = int(np.argmin(abs(leans-lean)))
        chosen.append(dict(lean_degrees=float(leans[i]), vector_length=float(m["length"][i]),
                           angle_from_upright_degrees=float(m["angle_upright"][i]),
                           vector_change_from_upright_L2=float(m["mismatch"][i]),
                           sampled_width_path_length=float(m["arc"][i]),
                           path_length_over_endpoint_distance=float(m["arc"][i]/m["length"][i])))
    summary = dict(root=BASE.tolist(), fixed_knobs=dict(cx=14, cy=14, height=19.75),
                   width_range_px=widths[[0, -1]].tolist(), lean_range_degrees=leans[[0, -1]].tolist(),
                   grid=dict(width_count=57, lean_count=91, width_step_px=.05, lean_step_degrees=.5, images=5187),
                   renderer=dict(subrows=SUB, storage="float32; differences and measurements in float64"),
                   width_vector_definition="render(cx,cy,h,max_width,lean) - render(cx,cy,h,min_width,lean)",
                   angle_definition="acos(dot(a,b)/(L2(a)*L2(b))) in degrees; all 784 coordinates, no projection",
                   length_range=[float(m["length"].min()), float(m["length"].max())],
                   one_degree_turn_range=[float(m["neighbor_angle"].min()), float(m["neighbor_angle"].max())],
                   half_degree_turn_range=[float(v) for v in [angle(m["full"][1:],m["full"][:-1]).min(),angle(m["full"][1:],m["full"][:-1]).max()]],
                   path_ratio_range=[float((m["arc"]/m["length"]).min()),float((m["arc"]/m["length"]).max())],
                   checks=dict(max_ink_mass_error=mass_error, telescoping_max_pixel_error=telescoping,
                               refined_grid_images=int(len(fine_widths)*len(fine_leans)),
                               refined_width_path_max_relative_change=float(arc_change.max()),
                               refined_lean_one_degree_turn_range=[float(fine_angles.min()),float(fine_angles.max())],
                               endpoint_vector_float32_vs_float64_max_pixel_difference=precision,
                               matched_refined_grid_images_identical=True, monotone_widening=True),
                   selected=chosen,
                   scope="Sampled width/lean slice at fixed center and height, not the whole five-knob manifold.",
                   understanding_check=dict(object="Source coverage images first, then their pixel differences",
                                            axes="Raw horizontal width pixels, lean degrees, coverage L2, and explicit angular references",
                                            baseline="Upright at lean 0 degrees, shown first and marked",
                                            evidence="Direct vector maps plus lengths, angles, and local grid heatmaps",
                                            limits="Finite sampled slice; same scales; no projection; no clipped displayed values"))
    (OUT / "results.json").write_text(json.dumps(summary, indent=2)+"\n")
    np.savez_compressed(OUT / "samples.npz", images=x.reshape(91,57,28,28), widths=widths, leans=leans, root=BASE,
                        full_width_vectors=m["full"].reshape(91,28,28))
    with (OUT / "whole_vectors.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lean_degrees","width_vector_length_L2","angle_to_upright_degrees","width_vector_change_from_upright_L2","sampled_width_path_length_L2","path_length_over_endpoint_distance","angle_between_width_vectors_one_degree_apart"])
        for i, lean in enumerate(leans):
            writer.writerow([lean,m["length"][i],m["angle_upright"][i],m["mismatch"][i],m["arc"][i],m["arc"][i]/m["length"][i],m["neighbor_angle"][i-1] if 0<i<len(leans)-1 else ""])
    with (OUT / "small_width_steps.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lean_degrees","width_start_px","width_end_px","L2_per_width_px","angle_to_same_step_at_upright_degrees","angle_to_full_width_vector_at_same_lean_degrees"])
        for i, lean in enumerate(leans):
            for j in range(len(widths)-1):
                writer.writerow([lean,widths[j],widths[j+1],m["speed"][i,j],m["local_angle_upright"][i,j],m["local_angle_chord"][i,j]])
    style()
    image_patterns(x, widths, leans, m)
    whole_vector_plots(leans, m)
    local_plots(widths, leans, m)
    write_html(x, widths, leans, summary)
    (OUT / "README.md").write_text(f"""# Width vectors across lean

Open `index.html` for the interactive comparison, or read the three numbered figures in order.

Center `(14,14)` and height `19.75 px` stay fixed. The root is `(14,14,19.75,3.2,0)`.
Sample 57 widths from {widths[0]:g} to {widths[-1]:g} px every 0.05 px and 91 leans from {leans[0]:g}° to {leans[-1]:g}° every 0.5°: 5,187 images.
The source is `grey_ones.render`, with 16 subrows and default float32 storage; subtract and measure in float64.

## What the vectors mean

At a fixed lean, **full width vector = image(width 4.6) − image(width 1.8)**.
Its 784 entries are the changes in pixel coverage between those two images.
Its length is `sqrt(sum(pixel_change**2))`.
The angle between two vectors is `acos(sum(a*b)/(length(a)*length(b)))`, in degrees.
For a direction-only comparison, lengths are divided out: 0° means the same direction; 90° means perpendicular.

`vector at lean − vector at upright` is a **second difference**: it describes how the width response changes, not direct ink gain or loss.
Its length measures the mismatch if the upright width vector were reused at the new lean.
The mixed difference is the same when computed as the change in the lean response across width.
Both parameter orders reach the same final rendered image; their intermediate pixel changes differ.

The full width vector sums the 56 consecutive width-step vectors exactly. It does not encode the curved intermediate path.
The sampled width-path length is the sum of their L2 lengths; dividing by the full-vector length gives 1 for a straight path without backtracking.

## Measured results

- Full-vector length ranges {m['length'].min():.3f}–{m['length'].max():.3f}.
- At lean 35°, its direction is {chosen[-1]['angle_from_upright_degrees']:.2f}° from upright.
- A finite 1° lean change turns the full width vector by {m['neighbor_angle'].min():.2f}–{m['neighbor_angle'].max():.2f}° over the sampled centers.
- Sampled widening path / endpoint distance ranges {(m['arc']/m['length']).min():.3f}–{(m['arc']/m['length']).max():.3f}; widening is not a straight pixel-space line.
- Angle from upright need not grow monotonically with lean, even though neighboring vectors keep changing. See `whole_vectors.csv`.

## Checks and limits

Refined grid: 113 widths × 181 leans (20,453 images), half the spacings.
Every matching image is identical. The largest relative path-length change after halving the width spacing is {arc_change.max():.4%}.
On the finer lean grid, finite 1° comparisons range {fine_angles.min():.3f}–{fine_angles.max():.3f}°.
Ink mass agrees with `height * width` to {mass_error:.2g}; all width steps add nonnegative ink; their sum agrees with each endpoint vector to {telescoping:.2g} per pixel.
Float32 versus float64 endpoint vectors differ by at most {precision:.2g} per pixel, retaining the same 16-subrow coverage model.

This is the requested width/lean slice at fixed center and height. It is a finite sampled experiment, not an exact map of the continuous region or the full five-knob manifold.
The 0.05-pixel and 1° vectors are finite differences, not analytic derivatives.
The direct vector images show all 784 entries; the scalar plots summarize them without projecting them.
Colors share explicit scales and have no clipped values. Differences from real MNIST are outside this experiment.

## Reproduce and reuse

Run `OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/width_vectors_by_lean.py` from the repository root.
`samples.npz` stores all images, settings, and full width vectors.
`whole_vectors.csv` and `small_width_steps.csv` store the measurements; `results.json` stores settings and verification results.
The standalone interactive viewer embeds the actual float32 samples and computes comparisons from them; it needs no server or network.
""")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
