"""Compare local linear bases at multiple generated-one starting images.

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python analyze_generated_one_local_bases_across_starts.py
Sampling parameters never enter SVD or cross-start vector matching.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/generated-one-mpl")

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import qmc

from grey_ones import KNOBS, RANGES, render
from analyze_root_one_local_svd import directions, pixels, shell, residual_curve, SCALE

OUT = Path(__file__).resolve().parent / "figures/generated_one_local_bases_across_starts"
RADII = [.1, .01, .001]
N_TRAIN, N_TEST = 512, 256
# Extend positions on both sides of the root and all sampled starts.
LOW = np.array([13., 13., 19., 1.8, -10.])
HIGH = np.array([16., 16., 20.5, 4.6, 35.])
BASE = np.array([14., 14., 19.75, 3.2, 0.])


def starts():
    result = [("root", BASE.copy())]
    for name, axis, value in [
        ("right_0p3", 0, 14.3), ("down_0p3", 1, 14.3),
        ("height_20p1", 2, 20.1), ("width_2p2", 3, 2.2),
        ("width_4p2", 3, 4.2), ("lean_minus7", 4, -7.),
        ("lean_12p5", 4, 12.5), ("lean_30", 4, 30.),
    ]:
        p = BASE.copy()
        p[axis] = value
        result.append((name, p))
    # Interior samples avoid clipping neighborhoods against the parameter bounds.
    u = .1 + .8 * qmc.Sobol(5, scramble=True, seed=20261012).random_base2(3)
    for i, p in enumerate(RANGES[:, 0] + u * SCALE):
        result.append((f"spread_{i+1:02d}", p))
    return result


def sign_basis(basis):
    basis = basis.copy()
    for b in basis:
        if b[np.argmax(np.abs(b))] < 0:
            b *= -1
    return basis


def comparison(reference, candidate):
    products = reference[:5] @ candidate[:5].T
    r, c = linear_sum_assignment(-np.abs(products))
    order = c[np.argsort(r)]
    aligned = candidate[order].copy()
    signs = np.sign(np.sum(reference[:5] * aligned, axis=1))
    signs[signs == 0] = 1
    aligned *= signs[:, None]
    cosines = np.sum(reference[:5] * aligned, axis=1)
    principal_cosines = np.clip(np.linalg.svd(products, compute_uv=False), 0, 1)
    full_cosines = np.linalg.svd(reference @ candidate.T, compute_uv=False)
    projection_residuals = np.linalg.norm(reference[:5] -
        (reference[:5] @ candidate.T) @ candidate, axis=1)
    stacked_singular = np.linalg.svd(np.vstack([reference,candidate]), compute_uv=False)
    # Small singular directions amplify renderer roundoff when normalized.
    # This intersection diagnostic therefore uses a looser angle tolerance;
    # held-out reconstruction errors remain the precision test.
    union_rank = int(np.sum(stacked_singular > stacked_singular[0]*1e-6))
    return {
        "matched_candidate_indices": order.tolist(),
        "matched_absolute_cosines": cosines.tolist(),
        "principal_angle_cosines_top5": principal_cosines.tolist(),
        "principal_angles_degrees_top5": np.degrees(np.arccos(principal_cosines)).tolist(),
        "full_space_principal_angle_cosines": full_cosines.tolist(),
        "full_space_shared_directions_at_cosine_1_minus_1e_8": int(np.sum(full_cosines > 1 - 1e-8)),
        "full_space_shared_directions_by_stacked_rank_rtol_1e_6": int(len(reference)+len(candidate)-union_rank),
        "reference_first5_projection_relative_residual_in_candidate_full_space": projection_residuals.tolist(),
    }, aligned


def make_figures(all_starts, rows, arrays):
    # Seven controlled starts give a readable source-to-vector view. The summary
    # includes every start, including the eight spread samples.
    names = ["root", "right_0p3", "width_2p2", "width_4p2", "lean_minus7", "lean_12p5", "lean_30"]
    labels = {"root": "Root", "right_0p3": "Right by 0.3 px", "width_2p2": "Width 2.2 px",
              "width_4p2": "Width 4.2 px", "lean_minus7": "Lean −7°",
              "lean_12p5": "Lean 12.5°", "lean_30": "Lean 30°"}
    lookup = {(r["name"], r["radius"]): r for r in rows}
    matched = {n: arrays[f"{n}_r_0p001_matched_basis"] for n in names}
    limit = max(float(np.abs(b).max()) for b in matched.values())
    fig = plt.figure(figsize=(14, 14), facecolor="white")
    fig.text(.035, .975, "Do the same local pixel-change vectors work at different starting 1s?",
             fontsize=18, weight="bold", va="top")
    fig.text(.035, .937,
             "Start images: white = 0 ink, black = 1. Signed basis vectors: red gains ink; blue loses ink.\n"
             "Each local SVD uses 512 neighbors at image distance 0.001. Five leading vectors shown; all 784 pixels compared.\n"
             "Columns match by pixel similarity, allowing permutation and sign; no image shifting, warping, or rotation of the basis.",
             fontsize=10.5, va="top", linespacing=1.5)
    left, bottom, width, height = .18, .13, .79, .73
    grid = fig.add_gridspec(len(names), 6, left=left, right=left+width, bottom=bottom,
                           top=bottom+height, hspace=.42, wspace=.20)
    for i, name in enumerate(names):
        row = lookup[(name, .001)]
        for j in range(6):
            ax = fig.add_subplot(grid[i, j])
            if j == 0:
                ax.imshow(arrays[name+"_start_image"].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1,
                          interpolation="nearest")
                p = row["settings"]
                label = f"Linear rank {row['numerical_rank_relative_1e_10']}"
            else:
                shown = ax.imshow(matched[name][j-1].reshape(28, 28), cmap="RdBu_r", vmin=-limit,
                                  vmax=limit, interpolation="nearest")
                cosine = row["versus_root"]["matched_absolute_cosines"][j-1]
                label = f"Root similarity {cosine:.3f}"
            if i == 0:
                ax.set_title("Starting image" if j == 0 else f"Root vector {j}", fontsize=10, pad=10)
            ax.text(.5, -.09, label, transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
            ax.set_xticks([])
            ax.set_yticks([])
        cy = bottom + height * (1 - (i+.5) / len(names))
        fig.text(.025, cy, labels[name], fontsize=10.5, va="center")
    cax = fig.add_axes([.63, .068, .32, .014])
    fig.colorbar(shown, cax=cax, orientation="horizontal")
    cax.set_xlabel("Pixel change per unit coefficient (one scale for all vectors)", fontsize=9)
    fig.text(.035, .085, "Similarity = absolute dot product of unit vectors.\n"
             "1 = identical up to sign; 0 = orthogonal.\n"
             "Linear rank includes small vectors. Similarities are rounded;\n"
             "1.000 does not imply exact equality.", fontsize=9.5, va="top", linespacing=1.5)
    fig.text(.035, .015,
             "Controlled changes from root (14, 14, 19.75, 3.2, 0°); unlisted settings stay fixed. "
             "Finite samples; numerical rank uses s > 10⁻¹⁰ × largest singular value.", fontsize=9)
    fig.savefig(OUT / "matched_vectors.png", dpi=150)
    plt.close(fig)

    n = len(all_starts)
    rank_values = np.array([[lookup[(name, radius)]["numerical_rank_relative_1e_10"] for radius in RADII]
                           for name, _ in all_starts])
    similarities = np.array([lookup[(name, .001)]["versus_root"]["matched_absolute_cosines"]
                             for name, _ in all_starts])
    fig = plt.figure(figsize=(14, 10), facecolor="white")
    fig.text(.045, .97, "Local linear rank and vector matches across 17 starting 1s",
             fontsize=19, weight="bold", va="top")
    fig.text(.045, .915,
             "Generated 28×28 coverage images; root (14, 14, 19.75, 3.2, 0°) is the reference. All five parameters vary locally.\n"
             "Each cell: 512 fitting neighbors plus 256 independent checks. Spread 01–08 are interior samples across the default ranges.",
             fontsize=10.5, va="top", linespacing=1.5)
    ax1 = fig.add_axes([.16, .21, .26, .61])
    im1 = ax1.imshow(rank_values, cmap="viridis", aspect="auto", vmin=5, vmax=max(15, rank_values.max()))
    ax1.set_xticks(range(3), [str(r) for r in RADII])
    ax1.set_yticks(range(n), [name.replace("_", " ") for name, _ in all_starts], fontsize=9)
    ax1.set_xlabel("Distance from each starting image", fontsize=10)
    ax1.set_title("Number of retained linear vectors", fontsize=12, pad=12)
    for i in range(n):
        for j in range(3):
            ax1.text(j, i, str(rank_values[i, j]), ha="center", va="center", fontsize=10,
                     color="black" if rank_values[i,j] > (rank_values.max()+5)/2 else "white")
    ax2 = fig.add_axes([.56, .21, .39, .61])
    im2 = ax2.imshow(similarities, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax2.set_xticks(range(5), [f"Root {i}" for i in range(1,6)])
    ax2.set_yticks(range(n), [])
    ax2.set_xlabel("Matched leading vector at radius 0.001", fontsize=10)
    ax2.set_title("Similarity to the root's five leading vectors", fontsize=12, pad=12)
    for i in range(n):
        for j in range(5):
            ax2.text(j, i, f"{similarities[i,j]:.2f}", ha="center", va="center", fontsize=9,
                     color="white" if similarities[i,j] > .6 else "black")
    c1 = fig.add_axes([.16, .14, .26, .017])
    fig.colorbar(im1, cax=c1, orientation="horizontal").set_label("Numerical rank: s > 10⁻¹⁰ × s₁", fontsize=9)
    c2 = fig.add_axes([.56, .14, .39, .017])
    fig.colorbar(im2, cax=c2, orientation="horizontal").set_label("Absolute cosine: 0 = orthogonal, 1 = identical up to sign", fontsize=9)
    max_error = max(r["full_basis_test_max_absolute_L2_error"] for r in rows)
    fig.text(.045, .065,
             f"All 51 full bases pass a 10⁻¹² absolute L2 check; largest held-out error: {max_error:.2g}. Float64 storage of the canonical renderer.\n"
             "Matches allow signs and permutations, not image alignment. Ranks describe sampled linear displacements, not the number of nonlinear controls.",
             fontsize=10, va="top", linespacing=1.5)
    fig.savefig(OUT / "ranks_and_matches.png", dpi=150)
    plt.close(fig)


def boundary_diagnostic():
    """Compare a rasterizer kink with a nearby start away from that kink."""
    rays = np.vstack([directions(N_TRAIN,20261010),directions(N_TEST,20261011)])
    rows = []
    for name, cy in [("tilted_on_subrow_boundary",14.),("tilted_shifted_down_0p023",14.023)]:
        center = np.array([14.,cy,19.75,3.2,30.])
        source = pixels(center)[0]
        for radius in [.001,.0003,.0001]:
            _, x, _, _ = shell(rays,radius,source,center=center,low=LOW,high=HIGH)
            v = x-source
            _,s,b = np.linalg.svd(v[:N_TRAIN],full_matrices=False)
            rank = int(np.sum(s > s[0]*1e-10))
            _, worst = residual_curve(v[N_TRAIN:],b[:rank])
            error = np.linalg.norm(v[N_TRAIN:]-(v[N_TRAIN:]@b[:rank].T)@b[:rank],axis=1).max()
            row = dict(name=name,settings=center.tolist(),radius=radius,numerical_rank=rank,
                       singular_values=s[:15].tolist(),five_vectors_worst_relative_error=float(worst[5]),
                       full_basis_worst_relative_error=float(worst[-1]),full_basis_max_absolute_L2_error=float(error))
            jumps = {}
            for axis in [1,2]:
                for delta_size in [1e-5,1e-6]:
                    delta = np.zeros(5)
                    delta[axis] = delta_size
                    forward = (pixels(center+delta)[0]-source)/delta_size
                    backward = (source-pixels(center-delta)[0])/delta_size
                    jumps[f"{KNOBS[axis]}_step_{delta_size}"] = float(
                        np.linalg.norm(forward-backward)/np.linalg.norm((forward+backward)/2))
            row["relative_one_sided_derivative_jumps"] = jumps
            assert error < 1e-12
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='singular_values'}),flush=True)
    (OUT/"subrow_boundary_diagnostic.json").write_text(json.dumps(rows,indent=2)+"\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_starts = starts()
    rays = np.vstack([directions(N_TRAIN, 20261010), directions(N_TEST, 20261011)])
    arrays, rows, models = {}, [], {}
    for name, center in all_starts:
        source = pixels(center)[0]
        arrays[name+"_start_settings"] = center
        arrays[name+"_start_image"] = source
        for radius in RADII:
            p, x, native, shell_error = shell(rays, radius, source, center=center, low=LOW, high=HIGH)
            v = x - source
            train, test = v[:N_TRAIN], v[N_TRAIN:]
            _, singular, b = np.linalg.svd(train, full_matrices=False)
            rank = int(np.sum(singular > singular[0] * 1e-10))
            assert rank >= 5
            b = sign_basis(b[:rank])
            train_rms, train_worst = residual_curve(train, b)
            test_rms, test_worst = residual_curve(test, b)
            coefficients = test @ b.T
            error = np.linalg.norm(test - coefficients @ b, axis=1)
            assert error.max() < 1e-12
            row = {
                "name": name, "settings": center.tolist(), "radius": radius,
                "max_shell_distance_error": shell_error,
                "numerical_rank_relative_1e_10": rank,
                "rank_by_relative_tolerance": {str(t): int(np.sum(singular > singular[0]*t)) for t in [1e-6,1e-8,1e-10,1e-12]},
                "singular_values": singular.tolist(),
                "train_relative_worst_error_by_k": train_worst.tolist(),
                "test_relative_worst_error_by_k": test_worst.tolist(),
                "test_relative_rms_error_by_k": test_rms.tolist(),
                "five_vectors_test_worst_percent": float(test_worst[5]*100),
                "full_basis_test_max_absolute_L2_error": float(error.max()),
                "full_basis_test_worst_relative_error": float(np.max(error / np.linalg.norm(test,axis=1))),
                "full_basis_train_worst_relative_error": float(train_worst[-1]),
            }
            key = f"{name}_r_{str(radius).replace('.', 'p')}"
            arrays.update({key+"_settings": p, key+"_images": x, key+"_basis": b,
                           key+"_train_coefficients": train @ b.T,
                           key+"_test_coefficients": coefficients})
            # Diagnose known knob directions after fitting; no labels enter SVD.
            step = SCALE * 1e-6
            tangent = (pixels(center + np.diag(step)) - pixels(center - np.diag(step))) / (2*step[:,None])
            tangent /= np.linalg.norm(tangent, axis=1)[:,None]
            row["first_five_cosines_to_known_knob_derivatives"] = (b[:5] @ tangent.T).tolist()
            arrays[key+"_unit_knob_derivatives"] = tangent
            assert np.allclose(b @ b.T, np.eye(rank), atol=2e-12)
            assert train_worst[-1] < 1e-8
            assert np.array_equal(native, x.astype(np.float32))
            models[(name,radius)] = (b,test)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ["name","radius","numerical_rank_relative_1e_10",
                             "five_vectors_test_worst_percent","full_basis_test_max_absolute_L2_error"]}), flush=True)

    for row in rows:
        name, radius = row["name"], row["radius"]
        b, test = models[(name,radius)]
        ref = models[("root",radius)][0]
        report, matched = comparison(ref,b)
        projection_error = np.linalg.norm(test - (test @ ref.T) @ ref, axis=1)
        report["root_full_basis_test_worst_relative_error"] = float(np.max(projection_error / np.linalg.norm(test,axis=1)))
        # Fitting the same root span at radius 0.1 gives better numerical
        # conditioning for its smallest singular directions.
        stable_ref = models[("root",.1)][0]
        stable_error = np.linalg.norm(test-(test@stable_ref.T)@stable_ref,axis=1)
        report["root_radius_0p1_basis_test_max_absolute_L2_error"] = float(stable_error.max())
        report["root_radius_0p1_basis_test_worst_relative_error"] = float(
            np.max(stable_error/np.linalg.norm(test,axis=1)))
        row["versus_root"] = report
        key = f"{name}_r_{str(radius).replace('.', 'p')}"
        arrays[key+"_matched_basis"] = matched

    # Compare every pair, allowing sign/permutation for individual vectors and
    # arbitrary orthogonal changes of basis for principal-angle comparisons.
    pairwise = []
    for i, (name_a, _) in enumerate(all_starts):
        for name_b, _ in all_starts[i+1:]:
            a = models[(name_a,.001)][0]
            b = models[(name_b,.001)][0]
            report, _ = comparison(a,b)
            pairwise.append({"start_a": name_a, "start_b": name_b, **report})
    np.savez_compressed(OUT/"bases_and_samples.npz", **arrays)
    metadata = {"knob_order": KNOBS, "radii": RADII, "train_count": N_TRAIN, "test_count": N_TEST,
                "train_seed": 20261010,"test_seed": 20261011,"spread_seed": 20261012,
                "sampling_lower": LOW.tolist(),"sampling_upper": HIGH.tolist(),
                "precision": "float64 storage of canonical renderer; cast matches default float32 exactly",
                "method": "Uncentered SVD of neighbor minus its own start; no parameter labels in fitting or matching",
                "results": rows,"pairwise_comparisons_at_radius_0p001": pairwise}
    (OUT/"results.json").write_text(json.dumps(metadata, indent=2)+"\n")
    with (OUT/"summary.csv").open("w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["name",*KNOBS,"radius","numerical_rank","five_vectors_worst_percent",
                    "full_basis_max_absolute_L2_error","root_basis_worst_relative_error",* [f"root_vector_{j}_cosine" for j in range(1,6)]])
        for r in rows:
            c = r["versus_root"]
            w.writerow([r["name"],*r["settings"],r["radius"],r["numerical_rank_relative_1e_10"],
                        r["five_vectors_test_worst_percent"],r["full_basis_test_max_absolute_L2_error"],
                        c["root_full_basis_test_worst_relative_error"],*c["matched_absolute_cosines"]])
    make_figures(all_starts,rows,arrays)
    boundary_diagnostic()


if __name__ == "__main__":
    main()
