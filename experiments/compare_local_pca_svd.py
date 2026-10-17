"""Compare centered PCA and start-anchored SVD on identical saved neighborhoods.

Run from the repo root:
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/compare_local_pca_svd.py
No rendering, resampling, hyperparameter selection, or new package is needed.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/local-pca-svd-mpl")

from pathlib import Path
import sys
import csv
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grey_ones import KNOBS
from analyze_generated_one_basis_followup import assignment, observable_matching

OUT = ROOT / "figures/generated_one_pca_vs_svd"
SOURCE = ROOT / "figures/generated_one_local_bases_across_starts"
FINE = ROOT / "figures/generated_one_scale_separation"


def worst_error(target, basis, offset):
    reconstructed = offset + ((target - offset) @ basis.T) @ basis
    error = np.linalg.norm(target - reconstructed, axis=1)
    return float(np.max(error / np.linalg.norm(target, axis=1))), float(error.max())


def compare(name, radius, source, images, targets, saved_basis):
    changes = images - source
    train, test = changes[:512], changes[512:]
    mean = train.mean(axis=0)
    _, singular, basis = np.linalg.svd(train, full_matrices=False)
    _, centered_singular, centered_basis = np.linalg.svd(train - mean, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * 1e-10))
    centered_rank = int(np.sum(centered_singular > centered_singular[0] * 1e-10))

    # Verify reproduction of the earlier leading frame, allowing signs/order.
    prior_order = assignment(saved_basis[:5] @ basis[:5].T)
    prior_cosines = np.abs(np.sum(saved_basis[:5] * basis[prior_order], axis=1))
    assert np.min(prior_cosines) > 1 - 1e-11

    # Centering subtracts precisely the outer product of the mean change.
    second = train.T @ train / len(train)
    covariance = (train - mean).T @ (train - mean) / len(train)
    identity_error = np.linalg.norm(second - covariance - np.outer(mean, mean)) / np.linalg.norm(second)
    assert identity_error < 1e-12

    order = assignment(basis[:5] @ centered_basis[:5].T)
    aligned = centered_basis[order].copy()
    aligned *= np.where(np.sum(basis[:5] * aligned, axis=1) < 0, -1, 1)[:, None]
    frame_change = np.linalg.norm(basis[:5] - aligned, axis=1)
    span_change = np.linalg.norm(basis[:5] - (basis[:5] @ centered_basis[:5].T) @ centered_basis[:5], axis=1)
    result = dict(name=name, radius=radius, training_samples=512, test_samples=256,
                  mean_change_relative_to_radius=float(np.linalg.norm(mean) / radius),
                  centering_identity_relative_error=float(identity_error),
                  max_matched_unit_axis_change=float(frame_change.max()),
                  max_svd_axis_outside_pca_five_span=float(span_change.max()))
    models = {}
    for label, b, s, n in [("svd", basis, singular, rank),
                            ("pca", centered_basis, centered_singular, centered_rank)]:
        offset = mean if label == "pca" else np.zeros(784)
        error, _ = worst_error(test, b[:5], offset)
        anchored_error, _ = worst_error(test, b[:5], np.zeros(784))
        full_relative, full_absolute = worst_error(test, b[:n], offset)
        products = np.abs(targets @ b[:5].T)
        knob_order = assignment(products)
        best_single_error = np.linalg.norm(
            targets - np.sum(targets * b[np.argmax(products, axis=1)], axis=1)[:, None]
            * b[np.argmax(products, axis=1)], axis=1)
        tangent_errors = np.linalg.norm(targets - (targets @ b[:5].T) @ b[:5], axis=1)
        result[label] = dict(
            rank_relative_1e_10=n,
            five_vector_worst_test_relative_error=error,
            five_vector_start_anchored_worst_test_relative_error=anchored_error,
            full_basis_worst_test_relative_error=full_relative,
            full_basis_worst_test_absolute_error=full_absolute,
            assigned_knob_cosines=dict(zip(KNOBS, products[np.arange(5), knob_order].tolist())),
            best_single_axis_knob_relative_errors=dict(zip(KNOBS, best_single_error.tolist())),
            five_span_knob_relative_errors=dict(zip(KNOBS, tangent_errors.tolist())),
            leading_normalized_singular_values=(s[:12] / s[0]).tolist(),
            zero_change_reconstruction_error=float(np.linalg.norm(offset - (offset @ b[:5].T) @ b[:5])))
        models[label] = b[:5]
        assert full_absolute < 1e-12
    return result, models, covariance, centered_basis


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archive = np.load(SOURCE / "bases_and_samples.npz")
    fine = np.load(FINE / "smaller_radius_samples.npz")
    names = [k.removesuffix("_start_settings") for k in archive.files if k.endswith("_start_settings")]
    rows, frames = [], {"svd": {}, "pca": {}}
    eigen_check = None
    for name in names:
        source = archive[name + "_start_image"]
        target = archive[name + "_r_0p001_unit_knob_derivatives"]
        for radius in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
            if radius >= 0.001:
                key = name + "_r_" + str(radius).replace(".", "p")
                data = archive
            else:
                key = name + "_r_" + str(radius)
                data = fine
                if key + "_images" not in data.files:
                    continue
            result, models, covariance, centered_basis = compare(
                name, radius, source, data[key + "_images"], target, data[key + "_basis"])
            rows.append(result)
            frames["svd"][(name, radius)] = models["svd"]
            frames["pca"][(name, radius)] = models["pca"]
            if name == "root" and radius == 0.001:
                values, vectors = np.linalg.eigh(covariance)
                eigen_basis = vectors[:, np.argsort(values)[::-1][:5]].T
                order = assignment(centered_basis[:5] @ eigen_basis.T)
                matched = eigen_basis[order]
                matched *= np.where(np.sum(matched * centered_basis[:5], axis=1) < 0, -1, 1)[:, None]
                eigen_check = dict(max_leading_axis_difference=float(np.linalg.norm(matched - centered_basis[:5], axis=1).max()))
                assert eigen_check["max_leading_axis_difference"] < 1e-11

    matching = {}
    for radius in [0.1, 0.01, 0.001]:
        matching[str(radius)] = {}
        for method in ["svd", "pca"]:
            # Reuse the earlier matcher verbatim; its archive keys fix radius 0.001.
            view = {}
            for name in names:
                view[name + "_start_image"] = archive[name + "_start_image"]
                view[name + "_r_0p001_basis"] = frames[method][(name, radius)]
                view[name + "_r_0p001_unit_knob_derivatives"] = archive[name + "_r_0p001_unit_knob_derivatives"]
            matching[str(radius)][method] = observable_matching([dict(name=n) for n in names], view)
    old = json.loads((ROOT / "figures/generated_one_basis_followup/results.json").read_text())["observable_matching"]
    assert matching["0.001"]["svd"]["observable_correct"] == old["observable_correct"] == 80
    assert matching["0.001"]["svd"]["pixel_correct"] == old["pixel_correct"] == 59

    result = dict(
        method=dict(neighborhoods=len(rows), starts=len(names), precision="float64 coverage",
                    radii=[0.1, 0.01, 0.001, 0.0001, 0.00001], training_samples=512, independent_checking_samples=256,
                    source_archives=["figures/generated_one_local_bases_across_starts/bases_and_samples.npz",
                                     "figures/generated_one_scale_separation/smaller_radius_samples.npz"],
                    svd="direct full SVD of training image-minus-start rows; decoder through the start",
                    pca="direct full SVD after subtracting training mean; affine decoder adds that mean",
                    fairness="PCA stores a mean vector as well as five basis vectors; anchored-only error also reported",
                    targets="Reuse prior tiny central-difference unit derivatives for the same evaluation protocol; at kinks these may average incompatible sides",
                    caveats=["No new neighbor sampling; held-out means independent of each basis fit, not previously unseen in project history",
                             "No pixel standardization, whitening, approximate SVD, or covariance eigensolver in the main comparison",
                             "Correct assigned category does not mean a pure physical knob or independent coordinate",
                             "Default bounds are extended for two-sided position neighborhoods as in source experiments",
                             "This comparison does not build or test a global nonlinear image decoder"]),
        pca_covariance_eigenvector_equivalence_check=eigen_check, matching=matching, neighborhoods=rows)
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    flat = []
    for r in rows:
        for m in ["svd", "pca"]:
            flat.append(dict(name=r["name"], radius=r["radius"], method=m,
                             worst_test_percent=100*r[m]["five_vector_worst_test_relative_error"],
                             worst_anchored_test_percent=100*r[m]["five_vector_start_anchored_worst_test_relative_error"],
                             rank=r[m]["rank_relative_1e_10"],
                             height_single_vector_error_percent=100*r[m]["best_single_axis_knob_relative_errors"]["height"]))
    with (OUT / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    for r in rows:
        if r["name"] in ["root", "lean_30", "spread_03"]:
            print(json.dumps({"name":r["name"], "radius":r["radius"],
                              "svd_percent":100*r["svd"]["five_vector_worst_test_relative_error"],
                              "pca_percent":100*r["pca"]["five_vector_worst_test_relative_error"],
                              "svd_height_error_percent":100*r["svd"]["best_single_axis_knob_relative_errors"]["height"],
                              "pca_height_error_percent":100*r["pca"]["best_single_axis_knob_relative_errors"]["height"]}))
    print(json.dumps({"neighborhoods":len(rows), "matching":{r:{m:{k:v for k,v in d.items() if k in ["pixel_correct","observable_correct"]} for m,d in methods.items()} for r,methods in matching.items()}, "eigen_check":eigen_check}))


if __name__ == "__main__":
    main()
