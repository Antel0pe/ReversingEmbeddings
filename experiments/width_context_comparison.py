"""Compare the same finite width increase in two generated-one states.

Run from the repository root: .venv/bin/python experiments/width_context_comparison.py
"""

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grey_ones import KNOBS, RANGES, render


def main():
    out = ROOT / "figures" / "generated_one_width_context"
    out.mkdir(parents=True, exist_ok=True)
    root = np.array([14.0, 14.0, 19.75, 3.2, 0.0])
    adjustment = 0.10 * np.diff(RANGES, axis=1).ravel()
    width_step = float(adjustment[3])
    adjustment[3] = 0.0
    changed = root + adjustment
    starts = np.stack([root, changed])
    widened = starts.copy()
    widened[:, 3] += width_step
    settings = np.stack([root, widened[0], changed, widened[1]])
    imgs = render(settings)
    bases = imgs[[0, 2]]
    differences = imgs[[1, 3]].astype(float) - bases.astype(float)
    scale = float(np.abs(differences).max())
    tolerance = 1e-7
    metrics = []
    for base, delta, p in zip(bases, differences, starts):
        metrics.append({
            "start": dict(zip(KNOBS, p.tolist())),
            "width_increase": width_step,
            "image_change_l2": float(np.linalg.norm(delta)),
            "gaining_pixels": int(np.sum(delta > tolerance)),
            "losing_pixels": int(np.sum(delta < -tolerance)),
            "ink_increase": float(delta.sum()),
            "maximum_pixel_gain": float(delta.max()),
        })
    flat = differences.reshape(2, -1)
    cosine = float(np.dot(flat[0], flat[1]) / np.prod(np.linalg.norm(flat, axis=1)))
    result = {
        "scope": "Two selected states, actual finite width changes; no SVD or family discovery.",
        "adjustment_fraction_of_default_range": 0.10,
        "other_knob_adjustments": dict(zip(KNOBS, adjustment.tolist())),
        "panels": metrics,
        "width_change_vector_angle_degrees": float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
        "pixel_count_tolerance": tolerance,
        "color_scale_absolute_pixel_change": scale,
        "render_storage": "float32",
    }
    (out / "measurements.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(out / "images_and_changes.npz", settings=settings, images=imgs, differences=differences)

    red = np.array([0.82, 0.12, 0.14])
    blue = np.array([0.12, 0.34, 0.75])
    cmap = LinearSegmentedColormap.from_list("ink_change", [blue, [1, 1, 1], red])
    fig = plt.figure(figsize=(10.8, 11.8), facecolor="#fafaf7")
    fig.text(0.06, 0.956, "How does widening change when the starting 1 changes?", fontsize=18, weight="bold")
    fig.text(0.06, 0.919, "Read each row left to right. Both width changes are +0.28 px (10% of its default range).", fontsize=11)
    fig.text(0.06, 0.893, "Generated 28 × 28 coverage images: white = no ink, black = full ink; gray = partial ink.", fontsize=10.5)
    rects = [[0.14, 0.592, 0.284, 0.260], [0.61, 0.592, 0.284, 0.260],
             [0.14, 0.170, 0.284, 0.260], [0.61, 0.170, 0.284, 0.260]]
    titles = ["A · Normal centered 1", "B · Widen A", "C · Other four knobs moved", "D · Widen C"]
    captions = [
        "Center (14, 14) px · height 19.75 px\nWidth 3.20 px · lean 0°",
        "Width 3.20 → 3.48 px; all other knobs fixed\nColored pixels show the change from A",
        "Center (14.10, 14.10) px · height 19.90 px\nWidth 3.20 px · lean +4.5°",
        "Width 3.20 → 3.48 px; all other knobs fixed\nColored pixels show the change from C",
    ]
    for i, (rect, title, caption) in enumerate(zip(rects, titles, captions)):
        ax = fig.add_axes(rect)
        row = i // 2
        if i % 2 == 0:
            ax.imshow(bases[row], cmap="gray_r", vmin=0, vmax=1, interpolation="nearest", extent=(0, 28, 28, 0))
        else:
            base = bases[row].astype(float)
            delta = differences[row]
            rgb = np.repeat((1 - base)[..., None], 3, axis=2)
            alpha = np.abs(delta)[..., None] / scale
            color = np.where((delta > 0)[..., None], red, blue)
            rgb = rgb * (1 - alpha) + color * alpha
            ax.imshow(rgb, interpolation="nearest", extent=(0, 28, 28, 0))
        ax.set_xticks([0, 14, 28])
        ax.set_yticks([0, 14, 28])
        ax.tick_params(labelsize=8, length=2, colors="#69717b")
        ax.set_xticks(np.arange(29), minor=True)
        ax.set_yticks(np.arange(29), minor=True)
        ax.grid(which="minor", color="#a0a8b0", linewidth=0.25, alpha=0.3)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_color("#7c848b")
        ax.set_title(title, fontsize=13, loc="left", pad=10, weight="bold")
        ax.text(0.5, -0.12, caption, transform=ax.transAxes, ha="center", va="top", fontsize=9.6, linespacing=1.6)
        if i % 2:
            m = metrics[row]
            ax.text(0.5, -0.26, f"{m['gaining_pixels']} pixels gain ink · {m['losing_pixels']} lose ink", transform=ax.transAxes,
                    ha="center", va="top", fontsize=10, color="#8e2428")
    fig.text(0.06, 0.495, "C starts from A, then adds: right 0.10 px · down 0.10 px · height 0.15 px · lean 4.5°.", fontsize=10.5)
    fig.text(0.06, 0.475, "Each adjustment is 10% of that knob’s default allowed range. Width stays 3.20 px.", fontsize=9.8, color="#525b65")
    cax = fig.add_axes([0.12, 0.055, 0.24, 0.014])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(-scale, scale), cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([-scale, 0, scale], labels=[f"−{scale:.2f}", "0", f"+{scale:.2f}"])
    cb.ax.tick_params(labelsize=8, length=2)
    fig.text(0.12, 0.078, "Blue loses ink · red gains ink", fontsize=10)
    fig.text(0.40, 0.076, "Both overlays use the same color strength; unchanged ink stays black/gray.", fontsize=9.5)
    fig.text(0.40, 0.057, "Widening only adds ink here, so there are no blue pixels. Canvas axes are pixels.", fontsize=9.5)
    fig.text(0.40, 0.038, "These are finite pixel differences at two states, not a sample of the whole manifold.", fontsize=9.2, color="#525b65")
    fig.savefig(out / "comparison.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
