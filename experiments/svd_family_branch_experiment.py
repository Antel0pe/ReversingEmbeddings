"""One root, one fixed distance/angle matcher, and short SVD-family branches.

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/svd_family_branch_experiment.py
The generator supplies neighborhoods and valid moves. Physical family labels
are inspected only after the discovery graph and its continuation are frozen.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/svd-family-branch-mpl")
from pathlib import Path
import sys
import json
import csv
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grey_ones import KNOBS, RANGES, render
from analyze_generated_one_nearest_direction import local_fit
from analyze_generated_one_manifold_transport import jet, normalized

OUT = ROOT / "figures" / "generated_one_svd_family_branches"
RADIUS = 1e-4
SEED = 20261012
STEP = 0.3
DISCOVERY_HOPS = 6
CONTINUATION_HOPS = 3
BASE = np.array([14., 14., 19.75, 3.2, 0.])
PROBE_TICKS = np.arange(-2, 3)
UPPER = np.triu_indices(5, 1)
PAIR_SCALE = np.abs(PROBE_TICKS[:, None] - PROBE_TICKS[None, :])[UPPER] * STEP
NAMES = ["Horizontal position", "Vertical position", "Height", "Width", "Lean"]


def pixels(p):
    return render(p, dtype=np.float64).reshape(-1, 784)


def turns(images):
    changes = np.diff(images, axis=0)
    changes /= np.linalg.norm(changes, axis=1)[:, None]
    return np.arccos(np.clip(np.sum(changes[1:] * changes[:-1], axis=1), -1, 1))


def descriptors(probes):
    return np.array([squareform(pdist(x)) for x in probes]), np.array([turns(x) for x in probes])


def valid_canvas(p):
    p = np.atleast_2d(p)
    reach = p[:, 3] / 2 + p[:, 2] / 2 * np.abs(np.tan(np.deg2rad(p[:, 4])))
    return bool(np.all((p[:, 2] > 0) & (p[:, 3] > 0) &
                       (p[:, 0] - reach > 0) & (p[:, 0] + reach < 28) &
                       (p[:, 1] - p[:, 2] / 2 > 0) & (p[:, 1] + p[:, 2] / 2 < 28)))


def observe(p):
    fit = local_fit(p, RADIUS, SEED)
    basis = fit["basis"].copy()
    # Arbitrary deterministic signs, without physical labels.
    for b in basis:
        if b[np.argmax(np.abs(b))] < 0:
            b *= -1
    j = jet(p)[3]
    # Simulation-only retraction: realize the requested pixel direction by
    # its closest valid parameter ray. The matcher never receives this lift.
    lifts = np.linalg.lstsq(j, basis.T, rcond=None)[0].T
    projected = lifts @ j.T
    projection_error = np.linalg.norm(projected - basis, axis=1)
    lifts /= np.linalg.norm(projected, axis=1)[:, None]
    probe_settings = p[None, None, :] + STEP * PROBE_TICKS[None, :, None] * lifts[:, None, :]
    if not valid_canvas(probe_settings.reshape(-1, 5)):
        raise ValueError("Probe left the positive, unclipped stroke region.")
    probes = pixels(probe_settings.reshape(-1, 5)).reshape(5, 5, 784)
    distances, angles = descriptors(probes)
    return dict(p=p.copy(), image=pixels(p)[0], basis=basis, lifts=lifts,
                probes=probes, distances=distances, angles=angles,
                projection_error=projection_error, singular=fit["singular"],
                tail_energy=fit["tail_energy"], shell_error=fit["shell_error"],
                cap_kink=fit["cap_kink"])


def match(parent, child):
    """Only signed pixel frames and distance/turning descriptors enter here."""
    dot = parent["basis"] @ child["basis"].T
    signs = np.where(dot >= 0, 1., -1.)
    direction_cost = np.arccos(np.clip(np.abs(dot), 0, 1)) / (np.pi / 2)
    distance_cost = np.empty((5, 5))
    turn_cost = np.empty((5, 5))
    for i in range(5):
        for k in range(5):
            d = child["distances"][k]
            a = child["angles"][k]
            if signs[i, k] < 0:
                d = d[::-1, ::-1]
                a = a[::-1]
            distance_cost[i, k] = np.sqrt(np.mean(((d[UPPER] - parent["distances"][i][UPPER]) / PAIR_SCALE) ** 2))
            turn_cost[i, k] = np.sqrt(np.mean(((a - parent["angles"][i]) / (np.pi / 2)) ** 2))
    components = np.stack([direction_cost, distance_cost, turn_cost])
    cost = components.mean(axis=0)
    row, col = linear_sum_assignment(cost)
    order = col[np.argsort(row)]
    selected_signs = signs[np.arange(5), order]
    diagnostic = dict(cost=cost.copy(), components=components.copy(), order=order.copy(),
                      component_candidate_spread=np.ptp(components, axis=2).mean(axis=1))
    for key in ["basis", "lifts", "probes", "distances", "angles", "projection_error"]:
        child[key] = child[key][order].copy()
    child["basis"] *= selected_signs[:, None]
    child["lifts"] *= selected_signs[:, None]
    for i, sign in enumerate(selected_signs):
        if sign < 0:
            child["probes"][i] = child["probes"][i, ::-1]
            child["distances"][i] = child["distances"][i, ::-1, ::-1]
            child["angles"][i] = child["angles"][i, ::-1]
    return diagnostic


def build_graph():
    nodes = [observe(BASE)]
    edges = []
    children = {}
    nodes[0]["route"] = []

    def advance(parent_id, family, phase):
        parent = nodes[parent_id]
        end = parent["p"] + STEP * parent["lifts"][family]
        if not valid_canvas(end):
            raise ValueError("Path left the unclipped stroke region.")
        child = observe(end)
        diagnostic = match(parent, child)
        child["route"] = parent["route"] + [family]
        child_id = len(nodes)
        nodes.append(child)
        edge = dict(parent=parent_id, child=child_id, moved_family=family, phase=phase, **diagnostic)
        edges.append(edge)
        children[parent_id, family] = child_id
        return child_id

    first = [advance(0, f, "discovery") for f in range(5)]
    for parent in first:
        for f in range(5):
            advance(parent, f, "discovery")
    paths = []
    for f in range(5):
        path = [0, first[f], children[first[f], f]]
        while len(path) <= DISCOVERY_HOPS:
            path.append(advance(path[-1], f, "discovery"))
        paths.append(path)
    print(f"Discovery frozen: {len(nodes)} states and {len(edges)} branch edges.", flush=True)
    for f, path in enumerate(paths):
        for _ in range(CONTINUATION_HOPS):
            path.append(advance(path[-1], f, "continuation"))
    return nodes, edges, paths


def evaluate(nodes, edges, paths):
    root_unit_controls = normalized(jet(BASE)[3])
    root_coefficients = nodes[0]["basis"] @ root_unit_controls
    root_scores = np.abs(root_coefficients)
    row, col = linear_sum_assignment(-root_scores)
    labels = col[np.argsort(row)]
    root_signs = np.sign(np.sum(nodes[0]["basis"] * normalized(jet(BASE)[3]).T[labels], axis=1))
    for node in nodes:
        products = node["basis"] @ normalized(jet(node["p"])[3])
        cosine = np.abs(products)
        row, col = linear_sum_assignment(-cosine)
        node["evaluation"] = dict(assigned_labels=col[np.argsort(row)],
                                  expected_cosine=cosine[np.arange(5), labels],
                                  strongest_physical_labels=cosine.argmax(axis=1),
                                  signed_expected_cosine=products[np.arange(5), labels] * root_signs,
                                  all_cosines=cosine)
    rows = []
    for edge in edges:
        parent = nodes[edge["parent"]]
        child = nodes[edge["child"]]
        true_turn = np.rad2deg(np.arccos(np.clip(np.sum(normalized(jet(parent["p"])[3]) * normalized(jet(child["p"])[3]), axis=0), -1, 1)))
        for f in range(5):
            label = child["evaluation"]["assigned_labels"][f]
            rows.append(dict(parent=edge["parent"], child=edge["child"], phase=edge["phase"],
                             route="/".join(str(i + 1) for i in child["route"]), family=f + 1,
                             expected_knob=KNOBS[labels[f]], assigned_knob=KNOBS[label],
                             correct=bool(label == labels[f]),
                             cosine_to_expected=float(child["evaluation"]["expected_cosine"][f]),
                             signed_cosine_to_expected=float(child["evaluation"]["signed_expected_cosine"][f]),
                             correct_family_but_reversed=bool(label == labels[f] and child["evaluation"]["signed_expected_cosine"][f] < 0),
                             actual_followed_family=bool(f == edge["moved_family"]),
                             moved_using_correct_parent_label=bool(parent["evaluation"]["assigned_labels"][f] == labels[f]),
                             moved_using_correct_parent_family_and_sign=bool(parent["evaluation"]["assigned_labels"][f] == labels[f] and parent["evaluation"]["signed_expected_cosine"][f] > 0),
                             physical_family_turn_degrees=float(true_turn[labels[f]]),
                             tangent_projection_error=float(child["projection_error"][f]),
                             cost_of_selected=float(edge["cost"][f, edge["order"][f]])))
    references = []
    for f in range(5):
        p = BASE.copy()
        reference = [p.copy()]
        for _ in range(DISCOVERY_HOPS + CONTINUATION_HOPS):
            speed = np.linalg.norm(jet(p)[3][:, labels[f]])
            p = p.copy()
            p[labels[f]] += STEP * root_signs[f] / speed
            reference.append(p.copy())
        reference = np.array(reference)
        if not valid_canvas(reference):
            raise ValueError("Reference path left the unclipped stroke region.")
        references.append(reference)

    phases = {}
    for phase in ["discovery", "continuation"]:
        selected = [r for r in rows if r["phase"] == phase]
        moved = [r for r in selected if r["actual_followed_family"]]
        phases[phase] = dict(
            edges=len(moved), five_field_comparisons=len(selected),
            association_correct=sum(r["correct"] for r in selected),
            association_incorrect=sum(not r["correct"] for r in selected),
            actual_moves_with_correct_source_label=sum(r["moved_using_correct_parent_label"] for r in moved),
            actual_moves_with_correct_source_family_and_sign=sum(r["moved_using_correct_parent_family_and_sign"] for r in moved),
            correct_family_but_reversed_next_fields=sum(r["correct_family_but_reversed"] for r in selected),
            expected_cosine_below_0p9=sum(r["cosine_to_expected"] < .9 for r in selected),
            expected_cosine_median=float(np.median([r["cosine_to_expected"] for r in selected])),
            expected_cosine_min=float(min(r["cosine_to_expected"] for r in selected)),
            mean_cost_candidate_spread_by_component=np.mean([e["component_candidate_spread"] for e in edges if e["phase"] == phase], axis=0).tolist(),
        )
    path_metrics = []
    for f, path in enumerate(paths):
        found = np.array([nodes[i]["image"] for i in path])
        actual = pixels(references[f])
        record = dict(family=f + 1, reference_knob=KNOBS[labels[f]], sign=int(root_signs[f]),
                      root_cosine=float(root_scores[f, labels[f]]),
                      minimum_expected_cosine=float(min(nodes[i]["evaluation"]["expected_cosine"][f] for i in path)),
                      endpoint_knob_changes=(nodes[path[-1]]["p"] - BASE).tolist(),
                      discovery_endpoint_image_error=float(np.linalg.norm(found[DISCOVERY_HOPS] - actual[DISCOVERY_HOPS])),
                      continuation_endpoint_image_error=float(np.linalg.norm(found[-1] - actual[-1])),
                      discovery_path_distance_matrix_relative_error=float(np.linalg.norm(squareform(pdist(found[:7])) - squareform(pdist(actual[:7]))) / np.linalg.norm(squareform(pdist(actual[:7])))),
                      full_path_distance_matrix_relative_error=float(np.linalg.norm(squareform(pdist(found)) - squareform(pdist(actual))) / np.linalg.norm(squareform(pdist(actual)))),
                      discovery_actual_moves_correct=sum(nodes[path[k - 1]]["evaluation"]["assigned_labels"][f] == labels[f] for k in range(1, 7)),
                      discovery_actual_moves_correct_family_and_sign=sum(nodes[path[k - 1]]["evaluation"]["assigned_labels"][f] == labels[f] and nodes[path[k - 1]]["evaluation"]["signed_expected_cosine"][f] > 0 for k in range(1, 7)),
                      continuation_actual_moves_correct=sum(nodes[path[k - 1]]["evaluation"]["assigned_labels"][f] == labels[f] for k in range(7, 10)),
                      continuation_actual_moves_correct_family_and_sign=sum(nodes[path[k - 1]]["evaluation"]["assigned_labels"][f] == labels[f] and nodes[path[k - 1]]["evaluation"]["signed_expected_cosine"][f] > 0 for k in range(7, 10)),
                      actual_step_expected_knob_changes=(np.diff(np.array([nodes[i]["p"] for i in path]), axis=0)[:, labels[f]] * root_signs[f]).tolist(),
                      continuation_next_labels_correct=sum(nodes[path[k]]["evaluation"]["assigned_labels"][f] == labels[f] for k in range(7, 10)))
        path_metrics.append(record)
    failures = [r for r in rows if not r["correct"]]
    weak = sorted(rows, key=lambda r: r["cosine_to_expected"])[:10]
    reference_shapes = [squareform(pdist(pixels(p[:7]))) for p in references]
    shape_similarity = []
    for a in range(5):
        for b in range(a + 1, 5):
            shape_similarity.append(dict(first=KNOBS[labels[a]], second=KNOBS[labels[b]],
                                         relative_distance_matrix_difference=float(np.linalg.norm(reference_shapes[a] - reference_shapes[b]) / np.linalg.norm(reference_shapes[a]))))
    result = dict(
        method=dict(root=BASE.tolist(), neighborhood_radius=RADIUS, neighborhood_images=256,
                    svd="uncentered seed-relative displacements, equal weights, five leading rows",
                    precision="float64 coverage storage", sampling_seed=SEED,
                    sampling="same Sobol parameter-ray directions at every state, scaled by default knob ranges; rays solved to equal image radius",
                    step=STEP, step_definition="local projected pixel speed is one; a parameter-ray step of 0.3, without forcing its finite chord distance",
                    probes="five images at ray ticks -2,-1,0,1,2; 10 pairwise distances and three turn angles",
                    matching="one-to-one minimum total cost; equal average of direction-angle, relative pairwise-distance, and turning-angle mismatch",
                    normalizations="direction and turn angles divided by pi/2; pairwise differences divided by abs(tick difference)*0.3",
                    families="SVD axes only in this first experiment; arbitrary root signs; no fitted mixtures",
                    sign_policy="align each candidate to its parent vector by dot-product sign before comparing probes; root sign makes the largest absolute pixel positive",
                    graph="one root, all 5 first branches and all 25 second branches; five self-family paths extended to six hops, then three new continuation hops",
                    oracle_use="known generator and Jacobian provide valid moves and samples; physical labels used only in evaluation after paths freeze",
                    bounds="default box not imposed; positive, unclipped strokes required, including left/up moves from the root",
                    continuation="unchanged rule, fresh states and their local neighborhoods; not blind prediction without observing neighborhoods",
                    scope="one deterministic run, no alternative matchers, no global coordinates or decoder claim"),
        root=dict(assigned_knobs=[KNOBS[i] for i in labels], matching_cosines=root_scores[np.arange(5), labels].tolist(),
                  unit_knob_coefficients_in_svd=root_coefficients.tolist(),
                  coefficient_columns=KNOBS,
                  reconstructed_unit_knob_residuals=np.linalg.norm(root_unit_controls - nodes[0]["basis"].T @ root_coefficients, axis=0).tolist(),
                  singular_values=nodes[0]["singular"].tolist(), singular_values_relative=(nodes[0]["singular"] / nodes[0]["singular"][0]).tolist(),
                  tail_energy_after_five=nodes[0]["tail_energy"]),
        phases=phases, paths=path_metrics, failures=failures, ten_weakest_associations=weak,
        reference_path_shape_differences=shape_similarity,
        maximum_shell_relative_error=float(max(n["shell_error"] for n in nodes)),
        maximum_tangent_projection_error=float(max(n["projection_error"].max() for n in nodes)),
        maximum_tail_energy=float(max(n["tail_energy"] for n in nodes)),
        cap_kink_states=[i for i, n in enumerate(nodes) if n["cap_kink"]],
        state_count=len(nodes), default_box_outside_state_count=sum(np.any((n["p"] < RANGES[:, 0]) | (n["p"] > RANGES[:, 1])) for n in nodes),
    )
    return result, rows, np.array(references), labels, root_signs


def make_figure(mode, nodes, paths, references, labels, signs, result):
    continuation = mode == "continuation"
    n = DISCOVERY_HOPS + CONTINUATION_HOPS if continuation else DISCOVERY_HOPS
    ticks = [6, 7, 8, 9] if continuation else [0, 1, 2, 6]
    use_reference = mode == "reference"
    titles = dict(found="1 · Five paths found by the distance-and-angle matcher",
                  reference="2 · The five corresponding pure-control paths",
                  continuation="3 · Three further hops with the same matching rule")
    fig = plt.figure(figsize=(15, 13.4), facecolor="#fafaf7")
    fig.text(.035, .968, titles[mode], fontsize=19, weight="bold", va="top")
    common = "Each image is a generated 28 × 28 coverage array: white = no ink, black = full ink, gray = partial ink."
    fig.text(.035, .927, common, fontsize=10.5)
    if mode == "found":
        subtitle = "One root (14, 14, 19.75, 3.2, 0°). Neighborhood radius 0.0001; 256 samples per fit; six steps per displayed path."
        detail = "F1–F5 are arbitrary root SVD families. Control names are evaluation labels, added after the paths were frozen."
    elif mode == "reference":
        subtitle = "Only the named physical control moves in each row. Both reference and found paths use the same local step calibration."
        detail = "Orange dashed = found path; blue solid = pure control. Similar distance curves can accompany different pixel changes."
    else:
        subtitle = "Hop 6 is the previous endpoint. Hops 7–9 were not used to choose the rule; each new state supplies its local SVD neighborhood."
        detail = "Orange dashed = found path; blue solid = pure control. Shading marks continuation; this is a fresh walk, not blind pixel prediction."
    fig.text(.035, .901, subtitle, fontsize=10.3)
    fig.text(.035, .877, detail, fontsize=10.3, color="#525b65")
    grid = fig.add_gridspec(5, 6, left=.14, right=.975, bottom=.115, top=.823,
                           width_ratios=[1, 1, 1, 1, 2.15, 2.15], wspace=.40, hspace=.78)
    found_color, reference_color = "#bd5624", "#2467a1"
    all_angles = []
    for f in range(5):
        found = np.array([nodes[i]["image"] for i in paths[f]])
        truth = pixels(references[f])
        all_angles.extend(np.rad2deg(turns(found[:n + 1])))
        all_angles.extend(np.rad2deg(turns(truth[:n + 1])))
    angle_top = max(10, min(180, np.ceil(max(all_angles) / 10) * 10 + 5))
    for f in range(5):
        found = np.array([nodes[i]["image"] for i in paths[f][:n + 1]])
        truth = pixels(references[f, :n + 1])
        series = truth if use_reference else found
        parameters = references[f] if use_reference else np.array([nodes[i]["p"] for i in paths[f]])
        axes = []
        for k, tick in enumerate(ticks):
            ax = fig.add_subplot(grid[f, k])
            ax.imshow(series[tick].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#c4c8cb")
            value = parameters[tick, labels[f]]
            unit = "°" if labels[f] == 4 else " px"
            ax.text(.5, -.13, f"{value:.2f}{unit}", transform=ax.transAxes, ha="center", va="top", fontsize=8)
            if f == 0:
                ax.set_title(f"Hop {tick}" + (" · root" if tick == 0 else ""), fontsize=10, pad=11)
            axes.append(ax)
        pos = axes[0].get_position()
        sign = "+" if signs[f] > 0 else "−"
        name = NAMES[labels[f]].replace(" ", "\n")
        fig.text(.027, (pos.y0 + pos.y1) / 2, f"F{f + 1} · {name} {sign}\nRoot cosine {result['paths'][f]['root_cosine']:.3f}",
                 fontsize=10, va="center", linespacing=1.45)
        axd = fig.add_subplot(grid[f, 4])
        axt = fig.add_subplot(grid[f, 5])
        if mode != "found":
            axd.plot(np.arange(n + 1), np.linalg.norm(truth - truth[0], axis=1), color=reference_color, linewidth=2, label="Pure control")
            axt.plot(np.arange(2, n + 1), np.rad2deg(turns(truth)), color=reference_color, linewidth=2, label="Pure control")
        axd.plot(np.arange(n + 1), np.linalg.norm(found - found[0], axis=1), color=found_color,
                 linestyle="--" if mode != "found" else "-", marker=".", linewidth=1.6, label="Found")
        axt.plot(np.arange(2, n + 1), np.rad2deg(turns(found)), color=found_color,
                 linestyle="--" if mode != "found" else "-", marker=".", linewidth=1.6, label="Found")
        if f == 0:
            axd.set_title("Distance from root", fontsize=11, pad=11)
            axt.set_title("Turn between steps", fontsize=11, pad=11)
        axd.set_ylim(0, STEP * n * 1.12)
        axt.set_ylim(-1, angle_top)
        axd.set_ylabel("Pixel L2 distance", fontsize=8)
        axt.set_ylabel("Angle (degrees)", fontsize=8)
        for ax in [axd, axt]:
            ax.set_xlim(0, n)
            ax.set_xticks([0, 3, 6, 9] if continuation else [0, 2, 4, 6])
            ax.set_xlabel("Hop", fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=.2)
            if continuation:
                ax.axvspan(6, 9, color="#dfead9", alpha=.6)
                ax.axvline(6, color="#809472", linewidth=.8)
        if mode != "found" and f == 0:
            axd.legend(fontsize=7, loc="upper left", frameon=False)
    phase = result["phases"]["continuation" if continuation else "discovery"]
    if mode == "found":
        summary = f"Discovery tree: {phase['association_correct']}/{phase['five_field_comparisons']} family labels agree (sign ignored); {phase['actual_moves_with_correct_source_family_and_sign']}/50 moves use the expected family AND sign."
    elif mode == "reference":
        errors = [p["discovery_path_distance_matrix_relative_error"] * 100 for p in result["paths"]]
        summary = "All-pairs path-distance errors, F1 → F5: " + ", ".join(f"{v:.2f}%" for v in errors) + ". These measure shape, not pixel reconstruction."
    else:
        count = sum(p["continuation_actual_moves_correct_family_and_sign"] for p in result["paths"])
        summary = f"Continuation: {count}/15 executed steps use the expected family AND sign; {phase['association_correct']}/75 next-field family labels agree (sign ignored)."
    fig.text(.035, .066, summary, fontsize=10.5)
    fig.text(.035, .043, "One step follows a valid parameter ray with initial pixel speed 1 for an amount 0.3. Finite distances are measured, not forced equal.", fontsize=9.5)
    fig.text(.035, .022, "Only these five paths are shown. Labels may describe mixtures. Experimental moves can leave the default knob box; every stroke remains unclipped.", fontsize=9.5)
    filename = dict(found="01_found_paths.png", reference="02_true_control_paths.png", continuation="03_unseen_continuation.png")[mode]
    fig.savefig(OUT / filename, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def save(nodes, edges, paths, references, result, rows):
    (OUT / "results.json").write_text(json.dumps(result, indent=2, default=lambda x: x.item()) + "\n")
    with (OUT / "associations.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(OUT / "states_and_paths.npz",
                        settings=np.array([n["p"] for n in nodes]), images=np.array([n["image"] for n in nodes]),
                        bases=np.array([n["basis"] for n in nodes]),
                        probe_images=np.array([n["probes"] for n in nodes]),
                        probe_distances=np.array([n["distances"] for n in nodes]),
                        probe_turns=np.array([n["angles"] for n in nodes]),
                        singular_values=np.array([n["singular"] for n in nodes]),
                        paths=np.array(paths), reference_settings=references,
                        match_costs=np.array([e["cost"] for e in edges]),
                        match_components=np.array([e["components"] for e in edges]),
                        edge_states=np.array([[e["parent"], e["child"], e["moved_family"]] for e in edges]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    nodes, edges, paths = build_graph()
    result, rows, references, labels, signs = evaluate(nodes, edges, paths)
    save(nodes, edges, paths, references, result, rows)
    for mode in ["found", "reference", "continuation"]:
        make_figure(mode, nodes, paths, references, labels, signs, result)
    print(json.dumps({"root": result["root"], "phases": result["phases"], "paths": result["paths"],
                      "failures": result["failures"], "state_count": len(nodes), "seconds": time.monotonic() - start}, indent=2, default=lambda x: x.item()), flush=True)


if __name__ == "__main__":
    main()
