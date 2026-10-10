"""Fit root-anchored linear bases to equal image-distance local shells.

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python analyze_root_one_local_svd.py
Only image displacements enter SVD. Parameters generate samples and provide
an independent diagnostic afterward. No mean subtraction or pixel rescaling.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/generated-one-mpl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import qr
from scipy.special import ndtri
from scipy.stats import qmc

from grey_ones import KNOBS, RANGES, render

OUT = Path(__file__).resolve().parent / "figures/root_one_local_svd"
ROOT_SETTINGS = np.array([14., 14., 19.75, 3.2, 0.])
RADII = [1., .3, .1, .03, .01, .003, .001]
SCALE = np.diff(RANGES, axis=1).ravel()
# Explicitly extend position ranges left/up, so the root has a two-sided neighborhood.
LOW = np.array([13., 13., 19., 1.8, -10.])
HIGH = np.array([15., 15., 20.5, 4.6, 35.])
N_TRAIN, N_TEST = 512, 256


def pixels(p):
    return render(p, dtype=np.float64).reshape(-1, 784)


def directions(n, seed):
    u = qmc.Sobol(5, scramble=True, seed=seed).random_base2(int(np.log2(n)))
    z = ndtri(u)
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def shell(ray, radius, root, *, center=ROOT_SETTINGS, low=LOW, high=HIGH):
    velocity = ray * SCALE
    with np.errstate(divide="ignore"):
        limits = np.where(velocity > 0, (high - center) / velocity,
                          (low - center) / velocity)
    limits[velocity == 0] = np.inf
    maximum = limits.min(axis=1)
    # Bracket the first sampled crossing, rather than assume a globally monotone ray.
    fractions = np.linspace(0, 1, 17)
    previous = np.zeros(len(ray))
    lo = np.zeros(len(ray))
    hi = np.zeros(len(ray))
    reached = np.zeros(len(ray), dtype=bool)
    for fraction in fractions[1:]:
        t = maximum * fraction
        dist = np.linalg.norm(pixels(center + t[:, None] * velocity) - root, axis=1)
        new = ~reached & (dist >= radius)
        lo[new], hi[new] = previous[new], t[new]
        reached |= new
        previous = t
        if reached.all():
            break
    if not reached.all():
        raise RuntimeError(f"{np.sum(~reached)} rays failed to reach radius {radius}")
    for _ in range(32):
        mid = (lo + hi) / 2
        dist = np.linalg.norm(pixels(center + mid[:, None] * velocity) - root, axis=1)
        above = dist >= radius
        hi[above], lo[~above] = mid[above], mid[~above]
    settings = center + ((lo + hi) / 2)[:, None] * velocity
    images = pixels(settings)
    native = render(settings).reshape(-1, 784)
    assert np.array_equal(native, images.astype(np.float32))
    assert np.all(settings >= low - 1e-12) and np.all(settings <= high + 1e-12)
    error = np.abs(np.linalg.norm(images - root, axis=1) - radius).max()
    assert error < 2e-7
    return settings, images, native, float(error)


def residual_curve(vectors, basis):
    # Direct residuals avoid cancellation when errors approach machine precision.
    residual = vectors.copy()
    norms = np.linalg.norm(vectors, axis=1)
    rms, worst = [1.], [1.]
    for vector in basis:
        residual -= (vectors @ vector)[:, None] * vector
        relative = np.linalg.norm(residual, axis=1) / norms
        rms.append(float(np.sqrt(np.mean(relative ** 2))))
        worst.append(float(relative.max()))
    return np.array(rms), np.array(worst)


def first_below(values, threshold):
    indices = np.flatnonzero(values <= threshold)
    return int(indices[0]) if len(indices) else None


def figures(data, root):
    selected = data[-1]  # smallest sampled radius
    basis = selected["basis"][:5]
    fig = plt.figure(figsize=(14, 6.3), facecolor="white")
    fig.text(.04, .96, "What pixel-change vectors does a small neighborhood reveal?",
             fontsize=19, weight="bold", va="top")
    fig.text(.04, .885,
             f"Generated 1s: root (14, 14, 19.75, 3.2, 0°); 512 fitting images, each at image distance {selected['radius']:g}.\n"
             "Root is ink coverage: white = 0, black = 1. Basis panels show unit vectors, not complete images.",
             fontsize=11, va="top", linespacing=1.5)
    limit = float(np.abs(basis).max())
    axes = []
    for i in range(6):
        ax = fig.add_axes([.035 + i * .16, .34, .135, .38])
        axes.append(ax)
        if i == 0:
            ax.imshow(root.reshape(28, 28), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            label = "Root 1\nBaseline image"
        else:
            shown = ax.imshow(basis[i-1].reshape(28, 28), cmap="RdBu_r", vmin=-limit,
                              vmax=limit, interpolation="nearest")
            share = selected["singular"][i-1] ** 2 / np.sum(selected["singular"] ** 2)
            label = f"Basis vector {i}\n{100*share:.2f}% of fitting energy"
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(.5, -.09, label, transform=ax.transAxes, ha="center", va="top", fontsize=10)
    cax = fig.add_axes([.66, .15, .28, .025])
    fig.colorbar(shown, cax=cax, orientation="horizontal")
    cax.set_xlabel("Pixel change per unit coefficient (shared scale)", fontsize=9)
    fig.text(.04, .19, "Red gains ink; blue loses ink; white = zero change.\n"
             "Each basis vector has length 1; its overall sign is arbitrary.\n"
             "Vectors are learned mixtures, not named generator knobs.", fontsize=10, va="top", linespacing=1.5)
    fig.text(.04, .035,
             f"Five vectors: worst held-out relative error {100*selected['test_worst'][5]:.3f}% across 256 new neighbors. "
             "All 784 pixels shown; no mean subtraction.", fontsize=10)
    fig.savefig(OUT / "basis_vectors.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(11.5, 7.4), facecolor="white")
    fig.text(.07, .96, "How many fixed vectors reconstruct new local changes?",
             fontsize=18, weight="bold", va="top")
    fig.text(.07, .89,
             "Root: center (14, 14) px, height 19.75 px, width 3.2 px, lean 0°. All five parameters vary.\n"
             "Fit: 512 equal-distance neighbors. Check: 256 independent neighbors at that same distance.",
             fontsize=10.5, va="top", linespacing=1.5)
    ax = fig.add_axes([.10, .24, .65, .53])
    for item in data:
        n = min(15, len(item["test_worst"]) - 1)
        ax.plot(np.arange(n+1), np.maximum(item["test_worst"][:n+1] * 100, 1e-8),
                marker="o", markersize=4, label=f"Image distance {item['radius']:g}")
    ax.axhline(1, color="#555555", ls="--", lw=1)
    ax.axhline(.1, color="#999999", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 110)
    ax.set_xticks(range(16))
    ax.set_xlabel("Number of learned basis vectors kept", fontsize=11)
    ax.set_ylabel("Worst held-out error / distance to root (%)", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.grid(alpha=.15)
    fig.text(.07, .14,
             "Error = length of (observed pixel change − reconstructed pixel change). Dashed line: 1%; dotted: 0.1%.\n"
             "The null reconstruction uses zero vectors and has 100% error. Plot floor: 0.00000001%.",
             fontsize=10, va="top", linespacing=1.5)
    fig.text(.07, .035,
             "Euclidean distance uses 784 coverage values. Float64 storage of the same 16-subrow renderer avoids float32 rank artifacts.\n"
             "Sampled shells, not exhaustive sets; numerical linear rank is distinct from intrinsic dimension.", fontsize=9.5, linespacing=1.4)
    fig.savefig(OUT / "reconstruction_by_radius.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    root = pixels(ROOT_SETTINGS)[0]
    native_root = render(ROOT_SETTINGS).reshape(784).astype(float)
    train_rays = directions(N_TRAIN, 20261010)
    test_rays = directions(N_TEST, 20261011)
    results, saved, data = [], {"root_settings": ROOT_SETTINGS, "root_image": root}, []
    for radius in RADII:
        p, x, native, shell_error = shell(np.vstack([train_rays, test_rays]), radius, root)
        vectors = x - root
        train, test = vectors[:N_TRAIN], vectors[N_TRAIN:]
        _, singular, basis = np.linalg.svd(train, full_matrices=False)
        # Keep only numerically active vectors; unused null-space vectors have no learned meaning.
        rank = int(np.sum(singular > singular[0] * 1e-10))
        basis = basis[:rank]
        # Make signs deterministic for readable images. Sign does not affect the subspace.
        for b in basis:
            if b[np.argmax(np.abs(b))] < 0:
                b *= -1
        train_rms, train_worst = residual_curve(train, basis)
        test_rms, test_worst = residual_curve(test, basis)
        native_vectors = native.astype(float) - native_root
        _, native_singular, native_basis = np.linalg.svd(native_vectors[:N_TRAIN], full_matrices=False)
        native_rms, native_worst = residual_curve(native_vectors[N_TRAIN:], native_basis[:rank])
        # Select actual observed vectors as a second, nonorthogonal spanning basis.
        _, _, pivot = qr(train.T, mode="economic", pivoting=True)
        subset = train[pivot[:rank]]
        subset_coefficients = np.linalg.lstsq(subset.T, test.T, rcond=1e-10)[0].T
        subset_error = np.linalg.norm(test - subset_coefficients @ subset, axis=1) / np.linalg.norm(test, axis=1)
        cosine = np.abs((train / np.linalg.norm(train, axis=1)[:, None]) @ basis[:5].T).max(axis=0)
        # Independent known-knob probes are generated only after fitting.
        probe_rays = np.vstack([np.eye(5), -np.eye(5)])
        probe_radius = min(radius, .3)  # Pure height cannot reach distance 1 within its bounds.
        probe_p, probe_x, _, _ = shell(probe_rays, probe_radius, root)
        probe_rms, probe_worst = residual_curve(probe_x - root, basis)
        step = SCALE * 1e-5
        tangent = (pixels(ROOT_SETTINGS + np.diag(step)) -
                   pixels(ROOT_SETTINGS - np.diag(step))) / (2 * step[:, None])
        unit_tangent = tangent / np.linalg.norm(tangent, axis=1)[:, None]
        alignment = basis[:5] @ unit_tangent.T
        row = {
            "radius": radius, "train_count": N_TRAIN, "test_count": N_TEST,
            "max_shell_distance_error": shell_error,
            "numerical_rank_relative_1e_10": rank,
            "rank_by_relative_tolerance": {str(t): int(np.sum(singular > singular[0] * t))
                                           for t in [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]},
            "singular_values": singular.tolist(),
            "test_min_vectors_for_worst_error": {str(t): first_below(test_worst, t) for t in [.1, .01, .001, .0001]},
            "train_relative_rms_error_by_k": train_rms.tolist(),
            "test_relative_rms_error_by_k": test_rms.tolist(),
            "test_relative_worst_error_by_k": test_worst.tolist(),
            "five_vectors_test_worst_percent": float(test_worst[5] * 100),
            "five_vectors_native_float32_test_worst_percent": float(native_worst[5] * 100),
            "five_vectors_pure_knob_probe_worst_percent": float(probe_worst[5] * 100),
            "pure_knob_probe_radius": probe_radius,
            "first_five_basis_cosine_with_known_knob_derivatives": alignment.tolist(),
            "full_basis_test_worst_percent": float(test_worst[-1] * 100),
            "selected_sample_basis_test_worst_percent": float(subset_error.max() * 100),
            "first_five_basis_max_absolute_cosine_to_training_samples": cosine.tolist(),
        }
        assert np.allclose(basis @ basis.T, np.eye(rank), atol=1e-12)
        assert train_worst[-1] < 1e-8
        assert np.all(cosine < 1 - 1e-8)
        assert abs(test_worst[5] - native_worst[5]) < 2e-4
        key = "r_" + str(radius).replace(".", "p")
        saved.update({f"{key}_settings": p, f"{key}_images": x,
                      f"{key}_basis": basis, f"{key}_train_coefficients": train @ basis.T,
                      f"{key}_test_coefficients": test @ basis.T,
                      f"{key}_selected_sample_indices": pivot[:rank],
                      f"{key}_selected_sample_vectors": subset,
                      f"{key}_test_selected_sample_coefficients": subset_coefficients,
                      f"{key}_probe_settings": probe_p,
                      f"{key}_known_knob_derivatives": tangent})
        data.append({"radius": radius, "basis": basis, "singular": singular, "test_worst": test_worst})
        results.append(row)
        print(json.dumps({k: row[k] for k in ["radius", "numerical_rank_relative_1e_10",
                          "test_min_vectors_for_worst_error", "five_vectors_test_worst_percent",
                          "full_basis_test_worst_percent"]}), flush=True)
    np.savez_compressed(OUT / "bases_and_samples.npz", **saved)
    metadata = {"root_settings": ROOT_SETTINGS.tolist(), "knob_order": KNOBS,
                "sampling_lower": LOW.tolist(), "sampling_upper": HIGH.tolist(),
                "method": "Uncentered SVD of image minus root; no knob labels used in fitting.",
                "precision": "Canonical grey_ones.render, float64 output; cast exactly matches default float32.",
                "train_seed": 20261010, "test_seed": 20261011, "results": results}
    (OUT / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["radius", "numerical_rank_rtol_1e-10", "k_for_test_worst_1pct", "k_for_test_worst_0p1pct", "five_vectors_test_worst_percent"])
        for row in results:
            counts = row["test_min_vectors_for_worst_error"]
            writer.writerow([row["radius"], row["numerical_rank_relative_1e_10"], counts["0.01"], counts["0.001"], row["five_vectors_test_worst_percent"]])
    figures(data, root)


if __name__ == "__main__":
    main()
