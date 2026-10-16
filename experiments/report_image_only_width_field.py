"""Audit, explain, and plot the image-only width-field experiment."""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/image-only-width-field-mpl")

from pathlib import Path
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grey_ones import render
from experiments.image_only_width_field import predict_compact, metrics, STEP

OUT = ROOT / "figures/image_only_width_field"


def load_model(name):
    with np.load(OUT / f"{name}_compact_linear.npz") as saved:
        return {key: saved[key] for key in saved.files}


def audit(results):
    report = {}
    cases = list(results["cases"].items())
    cases += [(f"lean_extent_{float(k):g}", c) for k, c in results["lean_extent_sweep"].items()]
    for name, case in cases:
        with np.load(OUT / f"{name}_evaluation.npz") as saved:
            settings = saved["test_settings"]
            fresh_images = render(settings, dtype=np.float64).reshape(-1, 784)
            target_settings = settings.copy()
            target_settings[:, 3] += STEP
            fresh_target = render(target_settings, dtype=np.float64).reshape(-1, 784)
            fresh_change = fresh_target - fresh_images
            assert np.array_equal(fresh_images, saved["test_images"])
            assert np.array_equal(fresh_change, saved["true_changes"])
            prediction = predict_compact(fresh_images, load_model(name))
            assert np.max(np.abs(prediction - saved["compact_linear"])) < 1e-13
            stored_images = render(settings).reshape(-1, 784).astype(np.float64)
            stored_target = render(target_settings).reshape(-1, 784).astype(np.float64)
            storage_check = metrics(stored_target - stored_images,
                                    predict_compact(stored_images, load_model(name)))
            for model_name in case["metrics"]:
                recomputed = metrics(fresh_change, saved[model_name])
                assert recomputed == case["metrics"][model_name]
            outside = {}
            for model_name in ["compact_linear", "quadratic_5"]:
                target = fresh_images + saved[model_name]
                outside[model_name] = int(np.sum(np.any((target < -1e-12) | (target > 1 + 1e-12), axis=1)))
            report[name] = dict(states=len(settings), source_and_target_exactly_rechecked=True,
                                saved_metric_audit=True, predicted_targets_outside_unit_cube=outside,
                                default_float32_inputs_and_targets=storage_check)
    return report


def lean_diagnostics():
    diagnostics = {}
    for name, bounds, center in [
        ("tiny_tilted_patch", [17.299, 17.301], [14.43, 14.62, 19.73, 3.17, 17.3]),
        ("narrow_lean", [-.5, .5], [14., 14., 19.75, 3.2, 0.]),
    ]:
        p = np.tile(center, (101, 1))
        p[:, 4] = np.linspace(*bounds, len(p))
        x = render(p, dtype=np.float64).reshape(-1, 784)
        q = p.copy(); q[:, 3] += STEP
        y = render(q, dtype=np.float64).reshape(-1, 784) - x
        pred = predict_compact(x, load_model(name))
        diagnostics[name] = dict(maximum_width_pattern_difference_from_first=float(np.linalg.norm(y - y[0], axis=1).max()),
                                 prediction=metrics(y, pred))
        np.savez_compressed(OUT / f"{name}_lean_only.npz", lean=p[:, 4], true_changes=y,
                            predicted_changes=pred)
    lean = np.linspace(0., 5., 5001)
    p = np.tile([14., 14., 19.75, 3.8, 0.], (len(lean), 1))
    p[:, 4] = lean
    x = render(p, dtype=np.float64).reshape(-1, 784)
    q = p.copy(); q[:, 3] += STEP
    y = render(q, dtype=np.float64).reshape(-1, 784) - x
    pred = predict_compact(x, load_model("narrow_lean"))
    pixel_error = np.abs(pred - y).max(axis=1)
    support_changed = np.any((np.abs(y) > 1e-12) != (np.abs(y[0]) > 1e-12), axis=1)
    failure = np.flatnonzero(pixel_error > 1e-12)
    switch = np.flatnonzero(support_changed)
    diagnostics["dense_lean_sweep"] = dict(fixed_settings=[14., 14., 19.75, 3.8, 0.],
                                            lean_bounds=[0., 5.], step_degrees=.001,
                                            first_failed_sample_degrees=float(lean[failure[0]]),
                                            previous_passing_sample_degrees=float(lean[failure[0] - 1]),
                                            first_changed_update_index_set_degrees=float(lean[switch[0]]),
                                            extrapolation_beyond_training_lean=.5)
    np.savez_compressed(OUT / "dense_lean_sweep.npz", lean=lean, true_changes=y,
                        predicted_changes=pred, worst_pixel_error=pixel_error,
                        support_changed=support_changed)
    return diagnostics


def local_figure(results):
    name = "tiny_tilted_patch"
    saved = np.load(OUT / f"{name}_evaluation.npz")
    model = load_model(name)
    # Show the largest actual variation from the mean update, not a favorable fit.
    index = np.linalg.norm(saved["true_changes"] - model["anchor"], axis=1).argmax()
    x = saved["test_images"][index]
    truth = saved["true_changes"][index]
    prediction = saved["compact_linear"][index]
    residual = prediction - truth
    coefficients = (x - model["origin"]) @ model["reader"]
    corrections = coefficients[:, None] * model["patterns"]
    fig = plt.figure(figsize=(15, 10), facecolor="white")
    fig.text(.045, .968, "A local width rule needs three measured numbers", fontsize=22, weight="bold", va="top")
    fig.text(.045, .921, "Predict the full 784-pixel change for width +0.01 px from the current image only. No knob settings enter the predictor.", fontsize=11)
    fig.text(.045, .889, "Tiny tilted patch: center (14.43, 14.62), height 19.73 ±0.0001 px, width 3.17 ±0.0001 px, lean 17.3 ±0.001°.", fontsize=10)
    panels = [(x, "1 · Current generated 1"), (truth, "2 · True width change"),
              (prediction, "3 · Predicted width change"), (residual, "4 · Prediction error")]
    delta_limit = max(abs(truth).max(), abs(prediction).max())
    error_limit = max(abs(residual).max(), 1e-16)
    positions = [.055, .29, .525, .76]
    for j, ((data, title), left) in enumerate(zip(panels, positions)):
        ax = fig.add_axes([left, .61, .18, .23])
        if j == 0:
            ax.imshow(data.reshape(28, 28), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        else:
            limit = error_limit if j == 3 else delta_limit
            im = ax.imshow(data.reshape(28, 28), cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
            cax = fig.add_axes([left, .572, .18, .012])
            cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-limit, 0, limit])
            cb.ax.tick_params(labelsize=8)
            cb.ax.set_xticklabels([f"−{limit:.2g}", "0", f"+{limit:.2g}"])
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.text(.055, .548, "White = 0 ink; black = 1 ink.", fontsize=9)
    fig.text(.29, .53, "Change maps: red gains ink; blue loses ink. True and predicted changes share a scale; the error scale is magnified.", fontsize=9.5)
    fig.text(.045, .478, "Rule: width change = reference change + s₁ × pattern₁ + s₂ × pattern₂ + s₃ × pattern₃", fontsize=14, weight="bold")
    fig.text(.045, .443, "Each s is one fixed weighted sum of (current pixels − reference pixels). The three contributions for this image are shown below.", fontsize=10)
    correction_limit = np.abs(corrections).max()
    for j, left in enumerate([.075, .305, .535]):
        ax = fig.add_axes([left, .19, .17, .21])
        im = ax.imshow(corrections[j].reshape(28, 28), cmap="RdBu_r", vmin=-correction_limit, vmax=correction_limit, interpolation="nearest")
        ax.set_title(f"Correction {j + 1} · s = {coefficients[j]:+.3g}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.text(left + .085, .174, f"Largest pixel correction: {np.abs(corrections[j]).max():.2g}",
                 ha="center", fontsize=8)
    cax = fig.add_axes([.095, .153, .59, .012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=8)
    cb.set_label("Correction to the WIDTH-CHANGE pattern; all three use the same signed scale.", fontsize=9)
    ax = fig.add_axes([.78, .20, .18, .19])
    sweep = np.load(OUT / "tiny_tilted_patch_lean_only.npz")
    origin = sweep["true_changes"][0]
    ax.plot(sweep["lean"] - 17.3, 1e6 * np.linalg.norm(sweep["true_changes"] - origin, axis=1), color="#245c8c", label="Actual")
    ax.plot(sweep["lean"] - 17.3, 1e6 * np.linalg.norm(sweep["predicted_changes"] - origin, axis=1), "--", color="#c46b2b", label="Predicted")
    ax.set_xlabel("Lean offset from 17.3°", fontsize=8)
    ax.set_ylabel("Change of width pattern\n(millionths of pixel L2)", fontsize=8)
    ax.tick_params(labelsize=8); ax.grid(alpha=.2); ax.legend(fontsize=8)
    ax.set_title("Lean really changes the update", fontsize=9, pad=8)
    c = results["cases"][name]
    m = c["metrics"]["compact_linear"]
    fig.text(.045, .065, f"1,024 training pairs; {c['test_states']:,} independent test states. Largest pixel-change error: {m['worst_pixel_absolute_error']:.2g} (rounding scale).", fontsize=11)
    fig.text(.045, .032, "Scope: one tiny three-control patch and one width action. This is not a decoder of the whole manifold. Full-range controls fail (next figure).", fontsize=10)
    fig.savefig(OUT / "01_local_rule.png", dpi=150, facecolor="white")
    plt.close(fig)


def range_figure(results, diagnostics):
    fig = plt.figure(figsize=(15, 10), facecolor="white")
    fig.text(.045, .97, "The local width rule breaks at a pixel-set switch", fontsize=22, weight="bold", va="top")
    fig.text(.045, .92, "One fixed width increase: +0.01 px. Current image is the only prediction input. Values below are errors in the complete pixel-change vector.", fontsize=10.5)
    fig.text(.045, .885, "Left: extrapolate the successful ±0.5° rule along a dense lean sweep. Right: retrain separately on each larger range, then test unseen states.", fontsize=10)
    ax = fig.add_axes([.08, .51, .40, .30])
    sweep = np.load(OUT / "dense_lean_sweep.npz")
    relative = np.linalg.norm(sweep["predicted_changes"] - sweep["true_changes"], axis=1) / np.linalg.norm(sweep["true_changes"], axis=1)
    ax.plot(sweep["lean"], relative * 100, color="#b65b2a", lw=2)
    ax.axvspan(0, .5, color="#d9eadc", alpha=.7, label="Training lean range")
    first = diagnostics["dense_lean_sweep"]["first_failed_sample_degrees"]
    ax.axvline(first, color="#555", ls="--", lw=1)
    ax.text(first + .10, 58, f"First error / index switch:\n{first:.3f}°", fontsize=9)
    ax.set_xlim(0, 5); ax.set_ylim(-2, max(100, relative.max() * 110))
    ax.set_xlabel("Lean (degrees); sampled every 0.001°", fontsize=10)
    ax.set_ylabel("Error relative to true update (pixel L2, %)", fontsize=10)
    ax.set_title("Fixed center (14,14), height 19.75, width 3.8 px", fontsize=10, pad=12)
    ax.grid(alpha=.2); ax.legend(loc="upper left", fontsize=9)

    ax = fig.add_axes([.57, .51, .38, .30])
    entries = [("±0.5°", results["cases"]["narrow_lean"])]
    entries += [(f"±{float(k):g}°", c) for k, c in results["lean_extent_sweep"].items()]
    entries += [("Full 3D", results["cases"]["full_slice"]), ("Full 5D", results["cases"]["all_five"])]
    xx = np.arange(len(entries))
    for key, label, marker, color in [
        ("frozen", "Same change at every image", "s", "#888888"),
        ("quadratic_5", "Five pixel measurements + products", "o", "#245c8c"),
        ("best_fit_5_patterns", "Best fit to 5 patterns (true weights)", "^", "#b65b2a"),
    ]:
        error = [100 * c["metrics"][key]["pooled_relative_L2"] for _, c in entries]
        ax.plot(xx, error, marker + "-", color=color, label=label, lw=1.6, ms=5)
    ax.set_xticks(xx, [label for label, _ in entries], fontsize=9)
    ax.set_ylim(-2, 100); ax.set_ylabel("Pooled update error (pixel L2, %)", fontsize=10)
    ax.set_title("1,024 training pairs per range; independent tests", fontsize=10, pad=12)
    ax.grid(alpha=.2); ax.legend(loc="upper left", fontsize=8)

    rows = [
        ["Upright patch", results["cases"]["upright_patch"]],
        ["Tiny tilted patch", results["cases"]["tiny_tilted_patch"]],
        ["Height/width + lean ±0.5°", results["cases"]["narrow_lean"]],
        ["Height/width + lean ±1°", results["lean_extent_sweep"]["1.0"]],
        ["Full height/width/lean", results["cases"]["full_slice"]],
        ["Full five-control box", results["cases"]["all_five"]],
    ]
    ax = fig.add_axes([.055, .16, .89, .25]); ax.axis("off")
    cells = []
    for label, case in rows:
        m = case["metrics"]["quadratic_5"]
        worst = "< 0.000000001%" if m["worst_relative_L2"] < 1e-11 else f"{100*m['worst_relative_L2']:.1f}%"
        cells.append([label, str(case["test_states"]), f"{m['worst_pixel_absolute_error']:.2g}", worst])
    table = ax.table(cellText=cells, colLabels=["Separately fitted source range", "Test states", "Largest pixel-change error", "Worst update error (L2)"],
                     colWidths=[.40, .13, .25, .22], bbox=[0, 0, 1, 1], cellLoc="left")
    table.auto_set_font_size(False); table.set_fontsize(10)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_facecolor("#eeeeee"); cell.set_text_props(weight="bold")
        elif c == 2:
            cell.set_facecolor("#e4f0e5" if float(cells[r-1][2]) < 1e-12 else "#f8e6df")
    fig.text(.055, .115, "Green cells pass a 10⁻¹² absolute pixel-change tolerance; red cells fail. No predictions are clipped or corrected.", fontsize=10)
    fig.text(.055, .083, "The 5-pattern fit uses true held-out changes to choose weights: it checks available patterns, not a usable predictor. It has five correction patterns + a reference.", fontsize=9)
    fig.text(.055, .052, "±lean rows: centers (14,14), height 19.1–19.9, width 2.2–3.8. Full 3D: height 19–20.5, width 1.8–4.59, lean −10–35°.", fontsize=9)
    fig.text(.055, .023, "Full 5D also varies both centers 14–15. These are finite sampled tests; fixed-pattern failure does not rule out a simple nonlinear or piecewise rule.", fontsize=9)
    fig.savefig(OUT / "02_range_limits.png", dpi=150, facecolor="white")
    plt.close(fig)


def write_report(results, diagnostics, verification):
    lines = ["# Predicting a width update from the current image", "",
             "The predictor receives 784 pixel values and fixed fitted arrays. It predicts the complete pixel change for **width +0.01 px**. It receives no current control settings and does not call the renderer, detect edges, or invert the generator.", "",
             "**Result:** an affine rule compresses to a reference change plus three weighted patterns in a tiny tilted patch, matching all tested updates to rounding precision. A larger height/width patch remains accurate with lean from -0.5 to +0.5 degrees. Widening the lean range breaks the single global polynomial rule. Five fixed correction patterns are also insufficient across the full height/width/lean slice.", "",
             "![Local pixel-only rule](01_local_rule.png)", "",
             "![Range and boundary tests](02_range_limits.png)", "",
             "## The rule", "", "```text",
             "measured numbers = (current image - reference image) @ reader",
             "predicted width change = reference change + measured numbers @ patterns", "```", "",
             "Every column of `reader` is an explicit 784-number pixel-weight vector. Every row of `patterns` is an explicit 784-number correction to the width-change pattern. In the tiny tilted patch, there are three readers and three patterns. The values of those readers are useful for predicting this action; they are not claimed to be three coordinates that decode the whole image.", "",
             "The uncompressed comparison uses at most five linear measurements of the image and an explicit degree-two polynomial in them: one constant, five linear terms, and fifteen pair products including squares. Its output has at most 21 stored pixel patterns. The five measurements are selected by an SVD of training source images; no variance-retention claim is used to justify accuracy. All reported accuracy is measured against independent renderer outputs.", "",
             "## Training and information boundaries", "",
             "Each domain uses 1,024 scrambled Sobol training sources and their paired width-updated images (2,048 renderer images). Training sees source vectors and labeled-action change vectors only. Physical settings are used by the adapter to create data and by the checker to create fresh answers. All arrays are float64, using the original `grey_ones.render` coverage calculation before float32 storage.", "",
             "Test sources combine 512 new Sobol states, a lattice including corners, 128 near-box-face states, and 64 independent states selected for a large forward/backward width second difference. Screening candidates are never used for fitting. No test source equals any training source. Hard-case screening uses true renderer data for evaluation selection, not for prediction.", "",
             "Width sources in the full-range cases end at 4.59 so that the +0.01 target stays within the default 4.6 bound. Other default bounds are unchanged. Centers are fixed at (14,14) except for the explicitly tilted local patch and full-five control.", "",
             "## Results", "",
             "| Case | Tests | Compact affine correction patterns | Quadratic pooled L2 error | Quadratic worst L2 error | Largest pixel error |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    cases = list(results["cases"].items())
    cases += [(f"lean_extent_{float(k):g}", c) for k, c in results["lean_extent_sweep"].items()]
    for name, c in cases:
        m = c["metrics"]["quadratic_5"]
        lines.append(f"| {c['label']} | {c['test_states']} | {c['compact_linear_patterns']} | {100*m['pooled_relative_L2']:.4g}% | {100*m['worst_relative_L2']:.4g}% | {m['worst_pixel_absolute_error']:.3g} |")
    lines += ["", "Pooled error is the norm of all update errors divided by the norm of all true updates. Worst error is the largest individual ratio. Neither is a percentage of correct pixels. The comparison across full cases includes a much broader domain and different dimensions; it is not a uniform scale sweep.", "",
              "## Positive controls", "",
              f"The preceding four-image upright formula was newly checked against 1,024 independently rendered states. Its image-derived pixel reader predicted width changes with largest pixel error {results['four_image_control']['worst_pixel_absolute_error']:.3g}, and reconstructed current images with largest pixel error {results['four_image_control']['worst_source_reconstruction_pixel_error']:.3g}. Those are new renderer checks, not just values synthesized by the fitted formula.", "",
              f"The tiny tilted patch has source affine rank 5 and width-change affine rank 3 at the recorded thresholds. Compressing its affine predictor to three correction patterns gives largest held-out pixel error {results['cases']['tiny_tilted_patch']['metrics']['compact_linear']['worst_pixel_absolute_error']:.3g}. Varying lean alone genuinely changes the width-update vector by L2 {diagnostics['tiny_tilted_patch']['maximum_width_pattern_difference_from_first']:.3g}; this success is not explained by an unchanged target.", "",
              "In the wider upright height/width patch with lean +/-0.5 degrees, the width-update pattern is unchanged by lean within numerical accuracy. Its exact prediction is therefore an easier success than the tilted local result. It needs just one correction pattern tracking the height-dependent cap contribution.", "",
              "The estimated affine ranks describe the sampled arrays, not intrinsic dimension. Source singular values use threshold `max(largest*1e-12, 1e-12)`; change singular values use `max(largest*1e-10, 1e-12)`. An absolute floor excludes cancellation-level directions in the very small tilted patch.", "",
              "## Where the local rule first fails", "",
              f"A dense lean sweep fixes center (14,14), height 19.75, and width 3.8. The +/-0.5-degree compact rule stays below 1e-12 maximum pixel error through {diagnostics['dense_lean_sweep']['previous_passing_sample_degrees']:.3f} degrees; the first failing sampled state is {diagnostics['dense_lean_sweep']['first_failed_sample_degrees']:.3f} degrees. The set of actually changing pixels first changes at {diagnostics['dense_lean_sweep']['first_changed_update_index_set_degrees']:.3f} degrees. Resolution is 0.001 degree, so these are sampled event locations, not exact analytic thresholds.", "",
              "Retraining on wider lean ranges still fails, so this is not just reuse of an out-of-range fitted model. At +/-1 degree the quadratic rule has about 10.5% pooled error and 50.5% worst-update error. Generic pairwise products of five image measurements do not handle the newly changing pixel regimes at this budget.", "",
              "## Pattern capacity versus predicting weights", "",
              "For each domain, a separate diagnostic factors the training width-update vectors into shared patterns. On held-out updates it projects the true change into those patterns, giving their best attainable weights. This is not an inference algorithm: true target values are used. Across the full three-control slice, five correction patterns plus a reference still leave about 60.7% pooled error. Even a perfect nonlinear rule for those five weights could not remove that residual while keeping these same patterns.", "",
              "This result does not prohibit a compact five-coordinate nonlinear decoder. A simple function can move which pixel indices participate and consequently have a large global linear span. Piecewise rules or a nonlinear representation that changes the patterns themselves remain possibilities.", "",
              "The full affine image predictor is included as a diagnostic. It can extrapolate badly near sparsely observed box corners, particularly in the all-five case. Increasing the number of input patterns without a suitable state rule is not a cure; these results do not establish its best possible sample efficiency.", "",
              "## Verification and limits", "",
              "`verification.json` records exact re-rendering of every saved test source and target, recomputed metrics, and loading/evaluating each compact model. `results.json` includes pixel-order permutation invariance, separate test-group scores, singular values in the saved model files, and best-fit pattern counts. Predictions are not clipped; verification records targets that leave [0,1].", "",
              f"The main accuracy results use float64 coverage. As a storage control, the already fitted compact rules were also given default float32-rendered inputs and compared with float32-rendered targets. Largest pixel errors were {verification['upright_patch']['default_float32_inputs_and_targets']['worst_pixel_absolute_error']:.3g} in the upright patch, {verification['tiny_tilted_patch']['default_float32_inputs_and_targets']['worst_pixel_absolute_error']:.3g} in the tiny tilted patch, and {verification['narrow_lean']['default_float32_inputs_and_targets']['worst_pixel_absolute_error']:.3g} in the narrow-lean patch. Quantized input/target values cannot support the float64 rounding-scale claim.", "",
              "Only the positive width action of +0.01 px was fitted. This is a finite-action predictor, not an instantaneous derivative, a general step-size integrator, a global manifold-membership rule, or a five-coordinate decoder. No neural networks, shape features, or recovered physical settings are used at inference. A finite set of successful samples does not certify every point in a patch.", "",
              "The next targeted question is how to discover the switches between local image-to-change rules from numeric vectors. The failures identify a useful target for that work; they do not prove that a small exact global rule exists or cannot exist.", "",
              "## Reproduce", "", "```bash",
              "OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/image_only_width_field.py",
              "OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/report_image_only_width_field.py", "```", ""]
    (OUT / "README.md").write_text("\n".join(lines))


def main():
    results = json.loads((OUT / "results.json").read_text())
    verification = audit(results)
    diagnostics = lean_diagnostics()
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    (OUT / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    local_figure(results)
    range_figure(results, diagnostics)
    write_report(results, diagnostics, verification)
    print(json.dumps(diagnostics, indent=2))
    print("All saved renderer comparisons and metrics passed audit.")


if __name__ == "__main__":
    main()
