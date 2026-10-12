"""Explain the saved root-local SVD; run from the repository root.

OPENBLAS_NUM_THREADS=1 .venv/bin/python figures/twitter/svd_nine_directions/make_figure.py
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/generated-one-mpl")

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BG = "#101923"
PANEL = "#1c2936"
WHITE = "#f4f7fa"
MUTED = "#b7c4d1"
BLUE = "#63bfff"
RED = "#ff6470"
GREEN = "#a6df9f"
SIGNED = LinearSegmentedColormap.from_list("pixel_change", ["#0072b2", WHITE, "#cf153e"])
INK = LinearSegmentedColormap.from_list("ink", [WHITE, "#18212a"])
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": WHITE,
                     "savefig.facecolor": BG})


def main():
    archive = np.load(ROOT / "figures/root_one_local_svd/bases_and_samples.npz")
    report = json.loads((ROOT / "figures/root_one_local_svd/results.json").read_text())
    row = next(r for r in report["results"] if r["radius"] == 0.001)
    root = archive["root_image"]
    basis = archive["r_0p001_basis"].copy()
    changes = archive["r_0p001_images"] - root
    train, test = changes[:512], changes[512:]
    coefficients = train @ basis.T
    energy = np.sum(coefficients**2, axis=0)
    shares = 100 * energy / np.sum(train**2)
    expected = np.asarray(row["singular_values"][:9])**2
    assert np.allclose(energy, expected, rtol=1e-7, atol=1e-24)
    assert np.max(np.abs(basis @ basis.T - np.eye(9))) < 1e-12
    distances = np.linalg.norm(changes, axis=1)
    assert np.max(np.abs(distances - 0.001)) < 2e-10
    derivatives = archive["r_0p001_known_knob_derivatives"]
    derivatives /= np.linalg.norm(derivatives, axis=1)[:, None]
    order = [4, 3, 0, 1, 2]
    cosines = np.array([basis[i] @ derivatives[k] for i, k in enumerate(order)])
    assert np.all(np.abs(cosines) > 0.988)
    # Positive coefficients show increasing the closest named physical knob.
    for i, cosine in enumerate(cosines):
        if cosine < 0:
            basis[i] *= -1
    analytic = np.load(ROOT / "figures/generated_one_manifold_transport/analytic_geometry.npz")
    tangent = analytic["root_tangent"]
    normal = analytic["root_normal_mixed_basis"]
    extras = basis[5:] - (basis[5:] @ tangent) @ tangent.T
    extras /= np.linalg.norm(extras, axis=1)[:, None]
    normal_errors = np.linalg.norm(extras - (extras @ normal) @ normal.T, axis=1)
    assert normal_errors.max() < 1e-7
    errors = {}
    for k in [5, 9]:
        residual = test - (test @ basis[:k].T) @ basis[:k]
        errors[k] = float(np.max(np.linalg.norm(residual, axis=1) / np.linalg.norm(test, axis=1)))

    fig = plt.figure(figsize=(16, 12), facecolor=BG)
    def text(x, y, value, size=18, color=WHITE, **kw):
        return fig.text(x, y, value, fontsize=size, color=color, **kw)

    text(.045, .948, '9 SVD basis vectors', 38, weight="bold")
    text(.045, .895, 'Every neighbor is 0.001 away in pixel space.', 22, MUTED)
    text(.045, .850, 'Variation captured (%)', 19, MUTED)

    ax = fig.add_axes([.865, .828, .09, .12])
    ax.imshow(root.reshape(28, 28), cmap=INK, vmin=0, vmax=1, interpolation="nearest")
    ax.set_axis_off()
    text(.91, .804, 'Starting “1”', 14, ha="center")

    text(.045, .785, 'Physical knobs (approx.)', 25, GREEN, weight="bold")
    limit = float(np.abs(basis).max())
    names = ['Tilt', 'Width', 'Sideways', 'Up / down', 'Height']
    for i in range(9):
        col = i if i < 5 else i - 5
        y = .550 if i < 5 else .180
        x = .045 + col * .191
        ax = fig.add_axes([x, y, .152, .203])
        ax.imshow(basis[i].reshape(28, 28), cmap=SIGNED, vmin=-limit, vmax=limit,
                  interpolation="nearest")
        ax.set_axis_off()
        cx = x + .076
        label = f'{i+1} · {names[i]}' if i < 5 else f'Vector {i+1}'
        text(cx, y - .031, label, 21, ha="center", weight="bold")
        if i < 5:
            share = f'{shares[i]:.2f}%'
        else:
            power = int(np.floor(np.log10(shares[i])))
            mantissa = shares[i] / 10.**power
            share = rf'${mantissa:.2f}\times 10^{{{power}}}\%$'
        text(cx, y - .071, share, 24 if i < 5 else 21, GREEN if i < 5 else MUTED,
             ha="center", weight="bold")

    text(.045, .415, 'Extra directions · knob interactions', 25, BLUE, weight="bold")
    text(.24, .044, 'Red gains ink', 21, RED, weight="bold")
    text(.57, .044, 'Blue loses ink', 21, BLUE, weight="bold")
    text(.5, .012, 'Unit-length vectors · common color scale · percentages measure squared change about the root',
         12, MUTED, ha="center")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [t for t in fig.texts if t.get_text().strip()]
    overlaps = []
    for i, a in enumerate(labels):
        ba = a.get_window_extent(renderer)
        for b in labels[i+1:]:
            if ba.overlaps(b.get_window_extent(renderer)):
                overlaps.append([a.get_text(), b.get_text()])
    assert not overlaps, overlaps
    for t in labels:
        box = t.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, t.get_text()
    fig.savefig(OUT / "svd_nine_directions.png", dpi=125)
    fig.savefig(OUT / "svd_nine_directions.svg")
    plt.close(fig)
    with Image.open(OUT / "svd_nine_directions.png") as im:
        im.resize((800, 600), Image.Resampling.LANCZOS).save(OUT / "preview.png")
    checks = {
        "radius_pixel_L2": 0.001, "training_images": 512, "heldout_images": 256,
        "root_settings_cx_cy_height_width_lean": archive["root_settings"].tolist(),
        "energy_percent_by_vector": shares.tolist(),
        "closest_knob_order": names, "absolute_knob_cosines": np.abs(cosines).tolist(),
        "extra_normal_span_errors": normal_errors.tolist(),
        "heldout_worst_relative_error": errors,
        "common_unit_vector_color_limit": limit, "text_overlaps": overlaps,
        "understanding_check": {
            "object": "Starting 28x28 synthetic coverage image, followed by nine full pixel-space SVD directions.",
            "encoding": "Blue negative, white zero, red positive; percentages are uncentered displacement energy shares.",
            "baseline_operation": "Subtract the root from 512 generated neighbors at pixel L2 distance 0.001, then SVD.",
            "visible_result": "Five dominant knob-like directions; four very small interaction directions, all shown at unit length.",
            "scope": "One controlled root neighborhood. Labels checked afterward; first five are approximate knob associations. Last four span normal interaction effects, not four new independent knobs."
        }
    }
    (OUT / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
