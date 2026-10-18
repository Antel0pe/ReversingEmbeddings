"""Observed-neighborhood SVD families: MNIST versus a size-matched generated cloud.

Run from the repository root:
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/mnist_svd_families/experiment.py
No transformations or synthetic interpolation are applied to MNIST images.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mnist-families-mpl')
from pathlib import Path
import sys
import json
import csv
import itertools
import time
import hashlib
import html
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import svd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from mnist import load, FILES, DEFAULT_ROOT, _md5
from grey_ones import render, RANGES, KNOBS

OUT = REPO / 'figures/mnist_svd_families'
SEED = 20261018
KS = [32, 64, 128, 256, 512]
MAIN_K = 128
N_ANCHORS = 64
DIMS = [1, 2, 3, 5, 10, 20, 30]
FEATURES = ['Ink center x', 'Ink center y', 'Vertical extent', 'Stroke thickness',
            'Lean angle', 'Row-center bend', 'Total ink']
COLORS = {'Generated cloud': '#276a9a', 'MNIST': '#b54435'}


def fit(d):
    # Omit identically zero columns only; retain all nonzero singular directions.
    active = np.any(d != 0, axis=0)
    _, s, vt = svd(d[:, active], full_matrices=False, lapack_driver='gesdd')
    basis = np.zeros((len(s), 784))
    basis[:, active] = vt
    signs = np.sign(basis[np.arange(len(s)), np.argmax(abs(basis), axis=1)])
    basis *= np.where(signs == 0, 1, signs)[:, None]
    return s, basis


def match(a, b, responses=None):
    products = a @ b.T if responses is None else responses
    row, col = linear_sum_assignment(-abs(products))
    order = np.empty(len(a), int)
    order[row] = col
    return order, np.where(products[np.arange(len(a)), order] >= 0, 1, -1), abs(products[np.arange(len(a)), order])


def features(x):
    """Evaluation descriptors only, never used for choosing roots or families."""
    a = x.reshape(-1, 28, 28)
    yy, xx = np.mgrid[:28, :28] + .5
    mass = a.sum((1, 2))
    cx = (a*xx).sum((1, 2))/mass
    cy = (a*yy).sum((1, 2))/mass
    dx, dy = xx-cx[:, None, None], yy-cy[:, None, None]
    vy = (a*dy**2).sum((1, 2))/mass
    vx = (a*dx**2).sum((1, 2))/mass
    cov = (a*dx*dy).sum((1, 2))/mass
    lean = np.rad2deg(np.arctan(-cov/vy))
    thickness = np.sqrt(12*np.maximum(vx-cov**2/vy, 0))
    bend = []
    rowmass = a.sum(2)
    for i in range(len(x)):
        use = rowmass[i] > .05*rowmass[i].max()
        y = np.arange(28)[use]+.5-cy[i]
        center = (a[i]*(np.arange(28)+.5)).sum(1)[use]/rowmass[i, use]
        design = np.column_stack([np.ones(len(y)), y])
        w = np.sqrt(rowmass[i, use])
        coeff = np.linalg.lstsq(design*w[:, None], center*w, rcond=None)[0]
        bend.append(np.sqrt(np.average((center-design@coeff)**2, weights=w**2)))
    return np.column_stack([cx, cy, np.sqrt(12*vy), thickness, lean, bend, mass])


def relative_curves(d, b):
    norm = np.linalg.norm(d, axis=1)
    assert np.all(norm > 0)
    residual = d.copy()
    values = []
    for i in range(min(max(DIMS), len(b))):
        residual -= (d@b[i])[:, None]*b[i]
        if i+1 in DIMS:
            values.append(np.linalg.norm(residual, axis=1)/norm)
    return np.array(values)


def choose_anchors(x):
    distances = cdist(x, x)
    np.fill_diagonal(distances, np.inf)
    kth = np.partition(distances, MAIN_K-1, axis=1)[:, MAIN_K-1]
    dense = int(np.argmin(kth))
    medoid = int(np.argmin(((x-x.mean(0))**2).sum(1)))
    # Cover the denser 80% without selecting roots for attractive SVD outcomes.
    allowed = kth <= np.quantile(kth, .8)
    selected = [dense] + ([medoid] if medoid != dense else [])
    nearest = distances[selected].min(0)
    nearest[selected] = 0
    while len(selected) < N_ANCHORS:
        chosen = int(np.argmax(np.where(allowed, nearest, -1)))
        selected.append(chosen)
        nearest = np.minimum(nearest, distances[chosen])
        nearest[selected] = 0
    # Separate local continuation from broad survey coverage. Only training
    # distances and shared-neighbor counts determine this connected exploration.
    nearest16 = np.argpartition(distances, 15, axis=1)[:, :16]
    connected = [dense]
    parents = []
    near = distances[dense].copy()
    near[dense] = 0
    frontier = set(nearest16[dense].tolist())
    while len(connected) < N_ANCHORS:
        candidates = [j for j in frontier if allowed[j] and j not in connected]
        if not candidates:
            raise RuntimeError('Connected observed-neighbor frontier exhausted')
        chosen = max(candidates, key=lambda j: (near[j], -j))
        parent = int(np.argmin(distances[chosen, connected]))
        parents.append((parent, len(connected)))
        connected.append(chosen)
        frontier.update(nearest16[chosen].tolist())
        near = np.minimum(near, distances[chosen])
        near[connected] = 0
    return np.array(selected), distances[np.array(selected)], kth, medoid, np.array(connected), parents


def graph_analysis(x, ids, indices, bases, method, extra_edges=()):
    distance = cdist(x[ids], x[ids])
    close = np.argsort(distance, axis=1)[:, 1:7]
    edges = sorted({tuple(sorted((i, int(j)))) for i in range(len(ids)) for j in close[i]} |
                   {tuple(sorted(e)) for e in extra_edges})
    transforms, signs, edge_rows = {}, {}, []
    weights = np.zeros_like(distance)
    for i, j in edges:
        shared = np.intersect1d(indices[i], indices[j])
        if len(shared) < 16:
            continue
        response = None
        if method == 'shared responses':
            z = x[shared]-x[shared].mean(0)
            u, v = z@bases[i].T, z@bases[j].T
            norms = np.linalg.norm(u, axis=0)[:, None]*np.linalg.norm(v, axis=0)[None, :]
            response = np.divide(u.T@v, norms, out=np.zeros_like(norms), where=norms > 1e-12)
        order, sign, score = match(bases[i], bases[j], response)
        transforms[i, j], signs[i, j] = order, sign
        inverse = np.argsort(order)
        transforms[j, i], signs[j, i] = inverse, sign[inverse]
        weights[i, j] = weights[j, i] = distance[i, j]
        pixel = abs(np.sum(bases[i]*bases[j, order], axis=1))
        edge_rows.append(dict(i=i, j=j, distance=float(distance[i, j]), shared=len(shared),
                              score=score.tolist(), pixel_cosine=pixel.tolist()))
    components, component = connected_components(csr_matrix(weights), directed=False)
    route, parent = shortest_path(csr_matrix(weights), directed=False, indices=0, return_predecessors=True)
    orders, orientations = {0: np.arange(5)}, {0: np.ones(5, int)}
    for j in np.argsort(route)[1:]:
        if not np.isfinite(route[j]):
            continue
        i = int(parent[j])
        orders[j] = transforms[i, int(j)][orders[i]]
        orientations[j] = orientations[i]*signs[i, int(j)][orders[i]]
    triangle, signed, triangle_rows = [], [], []
    for i, j, k in itertools.combinations(range(len(ids)), 3):
        if all(edge in transforms for edge in [(i, j), (j, k), (i, k)]):
            composed = transforms[j, k][transforms[i, j]]
            same = composed == transforms[i, k]
            comp_sign = signs[i, j]*signs[j, k][transforms[i, j]]
            signed_same = same & (comp_sign == signs[i, k])
            triangle.extend(same.tolist())
            signed.extend(signed_same.tolist())
            triangle_rows.append(dict(i=i, j=j, k=k, fraction=float(same.mean()),
                                      signed_fraction=float(signed_same.mean())))
    direct_agreement = []
    grouped = np.full_like(bases, np.nan)
    for j, order in orders.items():
        grouped[j] = bases[j, order]*orientations[j][:, None]
        if j:
            direct, _, _ = match(bases[0], bases[j])
            direct_agreement.extend((direct == order).tolist())
    root_component = np.flatnonzero(component == component[0])
    return dict(method=method, edges=edge_rows, components=int(components),
                root_component_size=len(root_component), triangles=triangle_rows,
                triangle_axis_agreement=float(np.mean(triangle)) if triangle else None,
                triangle_signed_agreement=float(np.mean(signed)) if signed else None,
                root_direct_route_agreement=float(np.mean(direct_agreement)) if direct_agreement else None,
                median_edge_cosine=float(np.median([e['pixel_cosine'] for e in edge_rows])),
                grouped=grouped, orders=orders, orientations=orientations,
                root_distances=distance[0], parents=parent, route_distances=route)


def run_cloud(name, x, test, original_ids, parameters=None):
    start = time.time()
    ids, distance, density, medoid, family_ids, family_parents = choose_anchors(x)
    train_order = np.argsort(distance, axis=1)[:, :max(KS)]
    test_distance = cdist(x[ids], test)
    test_order = np.argsort(test_distance, axis=1)
    fx, ft = features(x), features(test)
    records, arrays, frames, main_indices, correlations, stability = [], {}, [], [], [], []
    rng = np.random.default_rng(SEED)
    all_local = {}
    for ai, anchor in enumerate(ids):
        local = {}
        for k in KS:
            index = train_order[ai, :k]
            d = x[index]-x[anchor]
            s, b = fit(d)
            energy = np.cumsum(s*s)/np.sum(s*s)
            radius = float(distance[ai, index[-1]])
            # Every official-test image inside the TRAINING radius, no test-driven radius.
            ti = np.flatnonzero(test_distance[ai] <= radius)
            err = relative_curves(test[ti]-x[anchor], b) if len(ti) else np.empty((len(DIMS), 0))
            record = dict(dataset=name, anchor=ai, source_index=int(original_ids[anchor]), k=k,
                          radius=radius, nearest_distance=float(distance[ai, index[0]]),
                          training_dimension95=int(np.searchsorted(energy, .95)+1),
                          training_energy5=float(energy[4]), test_count=len(ti),
                          test_relative_median=np.median(err, axis=1).tolist() if len(ti) else None,
                          test_relative_p90=np.quantile(err, .9, axis=1).tolist() if len(ti) else None)
            records.append(record)
            local[k] = dict(basis=b[:30], singular=s, index=index, test_index=ti, errors=err)
            if k == MAIN_K:
                frames.append(b[:5])
                main_indices.append(index)
                # Correlation is held out, and descriptors are attached after SVD.
                c = (test[ti]-x[anchor])@b[:5].T
                delta = ft[ti]-fx[anchor]
                corr = np.zeros((5, len(FEATURES)))
                if len(ti) >= 8:
                    u, v = c-c.mean(0), delta-delta.mean(0)
                    norm = np.linalg.norm(u, axis=0)[:, None]*np.linalg.norm(v, axis=0)[None, :]
                    corr = np.divide(u.T@v, norm, out=np.full_like(norm, np.nan), where=norm > 1e-10)
                else:
                    corr[:] = np.nan
                correlations.append(corr)
                if ai < 20:
                    for trial in range(12):
                        subset = rng.choice(k, k, replace=True)
                        _, bb = fit(d[subset])
                        order, _, score = match(b[:5], bb[:5])
                        principal = np.linalg.svd(b[:5]@bb[:5].T, compute_uv=False)
                        stability.append(dict(anchor=ai, trial=trial, cosine=score.tolist(),
                                              space_overlap=float(np.mean(principal**2)),
                                              worst_principal_cosine=float(principal.min())))
        all_local[ai] = local
        if ai % 16 == 0:
            print(f'{name}: fitted {ai+1}/{len(ids)} anchors ({time.time()-start:.1f}s)', flush=True)
    frames, main_indices = np.array(frames), np.array(main_indices)
    pilot_graph = graph_analysis(x, ids, main_indices, frames, 'pixel cosine')
    family_frames, family_indices, family_correlations, family_singular = [], [], [], []
    family_test_distance = cdist(x[family_ids], test)
    family_train_distance = cdist(x[family_ids], x)
    family_train_distance[np.arange(len(family_ids)), family_ids] = np.inf
    for ai, anchor in enumerate(family_ids):
        ni = np.argsort(family_train_distance[ai])[:MAIN_K]
        s, b = fit(x[ni]-x[anchor])
        family_frames.append(b[:5]); family_indices.append(ni); family_singular.append(s)
        radius = family_train_distance[ai, ni[-1]]
        ti = np.flatnonzero(family_test_distance[ai] <= radius)
        corr = np.full((5, len(FEATURES)), np.nan)
        if len(ti) >= 8:
            u = (test[ti]-x[anchor])@b[:5].T
            v = ft[ti]-fx[anchor]
            u -= u.mean(0); v -= v.mean(0)
            norms = np.linalg.norm(u, axis=0)[:, None]*np.linalg.norm(v, axis=0)[None, :]
            corr = np.divide(u.T@v, norms, out=corr, where=norms > 1e-10)
        family_correlations.append(corr)
    family_frames, family_indices = np.array(family_frames), np.array(family_indices)
    graphs = {method: graph_analysis(x, family_ids, family_indices, family_frames, method, family_parents)
              for method in ['pixel cosine', 'shared responses']}
    graph = graphs['pixel cosine']
    corr_aligned = np.full((len(ids), 5, len(FEATURES)), np.nan)
    for ai, order in graph['orders'].items():
        corr_aligned[ai] = family_correlations[ai][order]*graph['orientations'][ai][:, None]
    # Honest whole-test atlas check: choose nearest TRAINING anchor, never lowest test error.
    atlas_assignment = np.argmin(test_distance, axis=0)
    atlas = {}
    global_s, global_b = fit(x-x.mean(0))
    for k in KS:
        errors = np.zeros((len(DIMS), len(test)))
        abs_errors = np.zeros_like(errors)
        test_radius_ratio = np.zeros(len(test))
        for ai in range(len(ids)):
            use = np.flatnonzero(atlas_assignment == ai)
            if not len(use):
                continue
            d = test[use]-x[ids[ai]]
            b = all_local[ai][k]['basis']
            errors[:, use] = relative_curves(d, b)
            abs_errors[:, use] = errors[:, use]*np.linalg.norm(d, axis=1)
            r = distance[ai, all_local[ai][k]['index'][-1]]
            test_radius_ratio[use] = test_distance[ai, use]/r
        global_delta = test-x.mean(0)
        global_relative = relative_curves(global_delta, global_b)
        global_absolute = global_relative*np.linalg.norm(global_delta, axis=1)
        atlas[k] = dict(median_relative=np.median(errors, axis=1).tolist(),
                        p90_relative=np.quantile(errors, .9, axis=1).tolist(),
                        mean_absolute=np.mean(abs_errors, axis=1).tolist(),
                        pixel_rmse=np.sqrt(np.mean(abs_errors**2, axis=1)/784).tolist(),
                        within_training_radius=float(np.mean(test_radius_ratio <= 1)),
                        global_mean_absolute=np.mean(global_absolute, axis=1).tolist(),
                        global_pixel_rmse=np.sqrt(np.mean(global_absolute**2, axis=1)/784).tolist())
        arrays[f'atlas_errors_{k}'] = errors
        arrays[f'atlas_absolute_errors_{k}'] = abs_errors
    arrays.update(anchor_images=x[ids], anchor_source_indices=original_ids[ids],
                  anchor_train_positions=ids, training_neighbor_indices=main_indices,
                  training_neighbor_source_indices=original_ids[main_indices],
                  bases=frames, grouped_bases=graph['grouped'],
                  family_anchor_images=x[family_ids], family_bases=family_frames,
                  family_anchor_source_indices=original_ids[family_ids],
                  family_neighbor_source_indices=original_ids[family_indices],
                  family_anchor_singular=np.array([np.pad(s, (0, MAIN_K-len(s))) for s in family_singular]),
                  descriptor_correlations=corr_aligned, descriptors_train=fx,
                  atlas_assignment=atlas_assignment, test_images=test,
                  anchor_radius=np.array([distance[i, main_indices[i, -1]] for i in range(len(ids))]),
                  root_distance=graph['root_distances'])
    for ai in range(len(ids)):
        arrays[f'anchor_{ai}_basis30'] = all_local[ai][MAIN_K]['basis']
        arrays[f'anchor_{ai}_singular'] = all_local[ai][MAIN_K]['singular']
        arrays[f'anchor_{ai}_test_indices'] = all_local[ai][MAIN_K]['test_index']
    if parameters is not None:
        from analyze_generated_one_manifold_transport import jet, normalized
        scores, named_cosines = [], []
        for ai, p in enumerate(parameters[family_ids]):
            true = normalized(jet(p)[3]).T
            order, _, cosine = match(true, family_frames[ai])
            named_cosines.append(cosine)
            scores.append(np.argsort(order))  # fitted basis index -> true-control label
        truth = np.array(scores)
        arrays['evaluation_true_labels'] = truth
        arrays['evaluation_true_cosines'] = np.array(named_cosines)
        for method, g in graphs.items():
            correctness = []
            for e in g['edges']:
                i, j = e['i'], e['j']
                if method == 'pixel cosine':
                    order, _, _ = match(family_frames[i], family_frames[j])
                else:
                    shared = np.intersect1d(family_indices[i], family_indices[j])
                    z = x[shared]-x[shared].mean(0)
                    u, v = z@family_frames[i].T, z@family_frames[j].T
                    norm = np.linalg.norm(u, axis=0)[:, None]*np.linalg.norm(v, axis=0)[None, :]
                    response = np.divide(u.T@v, norm, out=np.zeros_like(norm), where=norm > 1e-12)
                    order, _, _ = match(family_frames[i], family_frames[j], response)
                correctness.extend((truth[i] == truth[j, order]).tolist())
            g['true_category_edge_accuracy'] = float(np.mean(correctness))
        arrays['anchor_parameters'] = parameters[ids]
        arrays['family_anchor_parameters'] = parameters[family_ids]
    # Root descriptor readout: reconstruction uses only coefficients; no descriptor enters matching.
    readouts = []
    for ai in range(len(ids)):
        loc = all_local[ai][MAIN_K]
        ti, ni = loc['test_index'], loc['index']
        if len(ti) < 8:
            continue
        b = loc['basis'][:5]
        design = np.column_stack([np.ones(len(ni)), (x[ni]-x[ids[ai]])@b.T])
        target = fx[ni]-fx[ids[ai]]
        coeff = np.linalg.lstsq(design, target, rcond=None)[0]
        pred = np.column_stack([np.ones(len(ti)), (test[ti]-x[ids[ai]])@b.T])@coeff
        actual = ft[ti]-fx[ids[ai]]
        denominator = np.sum((actual-actual.mean(0))**2, axis=0)
        r2 = 1-np.divide(np.sum((pred-actual)**2, axis=0), denominator,
                         out=np.full(len(FEATURES), np.nan), where=denominator > 1e-10)
        readouts.append(r2)
    np.savez_compressed(OUT / ('mnist_arrays.npz' if name == 'MNIST' else 'generated_arrays.npz'), **arrays)
    result = dict(name=name, training_images=len(x), test_images=len(test), anchors=len(ids),
                  density_root_source_index=int(original_ids[ids[0]]), medoid_source_index=int(original_ids[medoid]),
                  root_density_radius=float(density[ids[0]]), global_dimension95=int(np.searchsorted(np.cumsum(global_s**2)/sum(global_s**2), .95)+1),
                  records=records, atlas=atlas, stability=stability,
                  correlations_median_absolute=np.nanmedian(abs(corr_aligned), axis=0).tolist(),
                  descriptor_readout_median_r2=np.nanmedian(readouts, axis=0).tolist(),
                  density_radius_quantiles=np.quantile(density, [0, .1, .5, .9, 1]).tolist(),
                  graphs={m: {k: v for k, v in g.items() if k not in ['grouped', 'orders', 'orientations', 'root_distances', 'parents', 'route_distances']} for m, g in graphs.items()})
    result['coverage_graph_pilot'] = {k: pilot_graph[k] for k in ['components', 'root_component_size', 'triangle_axis_agreement']}
    result['family_graph_anchor_count'] = len(family_ids)
    print(f'{name}: finished in {time.time()-start:.1f}s', flush=True)
    return result, arrays


def make_figure(title, subtitle, size=(16, 10)):
    fig = plt.figure(figsize=size, facecolor='white')
    fig.text(.035, .975, title, fontsize=19, weight='bold', va='top')
    fig.text(.035, .925, subtitle, fontsize=10.5, va='top', linespacing=1.5)
    return fig


def image_panel(ax, a, label, vector=False, limit=.4):
    ax.imshow(a.reshape(28, 28), cmap='RdBu_r' if vector else 'gray_r',
              vmin=-limit if vector else 0, vmax=limit if vector else 1, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(.5, -.09, label, ha='center', va='top', transform=ax.transAxes, fontsize=9)


def save(fig, filename):
    fig.savefig(OUT / filename, dpi=150, facecolor='white')
    plt.close(fig)


def plot_all(results, arrays, calibration):
    fig = make_figure('What do the first five local basis vectors change?',
        'Read left to right: actual starting image, then five unit pixel-change vectors. Black = ink 1; white = ink 0.\n'
        'Red gains ink; blue loses ink for a positive coefficient. All vector maps use one common scale; signs are arbitrary.', (16, 10))
    rows = [('Controlled tiny shell', calibration['image'], calibration['basis'], calibration['singular']),
            ('Generated finite cloud', arrays[0]['anchor_images'][0], arrays[0]['bases'][0], arrays[0]['anchor_0_singular']),
            ('Real MNIST', arrays[1]['anchor_images'][0], arrays[1]['bases'][0], arrays[1]['anchor_0_singular'])]
    limit = max(abs(row[2]).max() for row in rows)
    for r, (name, image, basis, singular) in enumerate(rows):
        y = .67-r*.24
        fig.text(.035, y+.08, name, fontsize=12, weight='bold', va='center')
        for c in range(6):
            ax = fig.add_axes([.23+c*.123, y, .105, .17])
            label = 'Starting image' if c == 0 else f'Axis {c}\n{100*singular[c-1]**2/sum(singular**2):.1f}% of fit energy'
            image_panel(ax, image if c == 0 else basis[c-1], label, c > 0, limit)
    fig.text(.035, .085, 'Tiny shell: 512 fit images exactly 0.001 from canonical root; float64. Finite clouds: 128 observed nearest neighbors.\n'
        'Finite-cloud roots minimize the 128th-neighbor radius in training data. No MNIST image is shifted, deskewed, or synthesized.', fontsize=10)
    cax = fig.add_axes([.74, .055, .2, .018])
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-limit, limit), cmap='RdBu_r'), cax=cax, orientation='horizontal')
    save(fig, '01_root_basis_vectors.png')

    fig = make_figure('Which local vectors were put in the same family?',
        'Each column F1–F5 inherits one root axis along the shortest supported graph route. Each edge uses a joint maximum-|cosine| assignment.\n'
        'Rows are sorted by root distance for display only. Cosine below 0.70 is marked amber: forced assignment can be weak.', (16, 17))
    limit = max(np.nanmax(abs(a['grouped_bases'])) for a in arrays)
    for block, (result, a) in enumerate(zip(results, arrays)):
        valid = np.flatnonzero(np.isfinite(a['grouped_bases']).all((1, 2)))
        sorted_ids = valid[np.argsort(a['root_distance'][valid])]
        selected = sorted_ids[np.round(np.linspace(0, len(sorted_ids)-1, 4)).astype(int)]
        for row, ai in enumerate(selected):
            y = .75-(block*4+row)*.088
            if row == 0:
                fig.text(.035, y+.07, result['name'], fontsize=12, weight='bold')
            fig.text(.035, y+.035, f'Anchor {ai}\nRoot gap {a["root_distance"][ai]:.2f}', fontsize=9, va='center')
            for c in range(6):
                ax = fig.add_axes([.23+c*.123, y, .105, .065])
                cosine = float(abs(a['grouped_bases'][0, c-1]@a['grouped_bases'][ai, c-1])) if c else 1
                label = 'Observed image' if c == 0 else f'F{c} · root cosine {cosine:.2f}'
                image_panel(ax, a['family_anchor_images'][ai] if c == 0 else a['grouped_bases'][ai, c-1], label, c > 0, limit)
                if c and cosine < .7:
                    for spine in ax.spines.values():
                        spine.set_color('#cf8b12'); spine.set_linewidth(2)
    fig.text(.035, .045, 'Images: white = ink 0; black = ink 1. Distance uses all 784 values. Vectors have unit L2 length; red gains and blue loses ink.\n'
        'A shared column means the graph assigned the same label; it does not establish a common physical knob. All maps share one color scale.', fontsize=10)
    save(fig, '02_vector_families.png')

    fig = make_figure('How accurately do local vectors reconstruct unseen images?',
        'Official MNIST test images are held out. The atlas chooses the nearest training anchor, then projects image-minus-anchor onto its basis.\n'
        'Curves show remaining L2 change / original L2 change; 0% is exact, 100% adds nothing beyond the anchor. Bands: 10th–90th percentiles.', (16, 12))
    for c, (result, a) in enumerate(zip(results, arrays)):
        ax = fig.add_axes([.08+c*.48, .53, .39, .28])
        err = a[f'atlas_errors_{MAIN_K}']
        median = np.median(err, axis=1)*100
        ax.plot(DIMS, median, 'o-', color=COLORS[result['name']], label='128-neighbor local atlas')
        ax.fill_between(DIMS, np.quantile(err, .1, axis=1)*100, np.quantile(err, .9, axis=1)*100,
                        color=COLORS[result['name']], alpha=.15)
        for k in [32, 512]:
            ax.plot(DIMS, np.median(a[f'atlas_errors_{k}'], axis=1)*100, '--', label=f'{k} neighbors', alpha=.7)
        ax.axvline(5, ls=':', color='gray'); ax.set_ylim(0, 100)
        ax.set(title=result['name'], xlabel='Number of fitted basis vectors retained', ylabel='Unexplained pixel change (%)')
        ax.legend(fontsize=8); ax.grid(alpha=.2)
        fig.text(.08+c*.48, .455, f'5 vectors: median {median[3]:.1f}%, 90th percentile {np.quantile(err[3], .9)*100:.1f}% remaining\n'
                 f'20 vectors: median {median[5]:.1f}% remaining. This is projection fidelity, not generated-image prediction.', fontsize=10)
        # Pick a median-error test image by rule, not visual appeal.
        ti = int(np.argsort(err[3])[len(err[3])//2])
        ai = int(a['atlas_assignment'][ti]); anchor = a['anchor_images'][ai]
        target = a['test_images'][ti]; b = a[f'anchor_{ai}_basis30']
        d = target-anchor
        for q, n in enumerate([None, 0, 5, 20]):
            image = target if n is None else anchor+(d@b[:n].T)@b[:n]
            axim = fig.add_axes([.07+c*.48+q*.103, .205, .09, .17])
            label = 'Held-out target' if n is None else ('Chosen anchor' if n == 0 else f'{n}-vector projection')
            if n:
                label += f'\nError {np.linalg.norm(image-target):.2f} L2'
            image_panel(axim, image, label)
            outside = (image < 0) | (image > 1)
            if outside.any():
                rgba = np.zeros((28, 28, 4)); rgba[outside.reshape(28, 28)] = [1, .62, 0, .8]
                axim.imshow(rgba)
            axim.text(.5, -.35, f'{outside.sum()} out-of-range pixels', ha='center', va='top', transform=axim.transAxes, fontsize=8)
    fig.text(.035, .065, 'Images use white = 0, black = 1. Amber marks projected pixels outside [0,1], which the grayscale display would otherwise clip.\n'
        'Example = median five-vector relative error in each test set. Error metrics always use unclipped projections. No test image trains a basis.', fontsize=10)
    save(fig, '03_heldout_reconstruction.png')

    fig = make_figure('What observable changes correlate with each assigned family?',
        'Each cell is the median absolute Pearson correlation across supported anchors: local basis coefficient versus descriptor change on held-out images.\n'
        'Root families are assigned using pixels only. Descriptors are added afterward; correlations describe associations, not identified controls.', (16, 8))
    for c, result in enumerate(results):
        ax = fig.add_axes([.12+c*.47, .24, .34, .49])
        values = np.array(result['correlations_median_absolute'])
        im = ax.imshow(values, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
        ax.set_yticks(range(5), [f'Family {i+1}' for i in range(5)])
        ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=35, ha='right', fontsize=9)
        ax.set_title(result['name'])
        for i, j in itertools.product(range(5), range(len(FEATURES))):
            ax.text(j, i, f'{values[i,j]:.2f}', ha='center', va='center', color='white' if values[i,j] > .65 else 'black', fontsize=9)
    cax = fig.add_axes([.945, .24, .015, .49]); fig.colorbar(im, cax=cax, label='|held-out correlation|')
    fig.text(.035, .05, 'Extent = √12 × vertical ink spread. Thickness = √12 × horizontal spread after linear lean removal. Lean = row-regression angle.\n'
        'Bend = weighted RMS row-center deviation from its best straight line. These are image measurements; handwriting has no known five-knob truth.', fontsize=10)
    save(fig, '04_family_correlations.png')

    fig = make_figure('Do labels survive resampling and alternative routes?',
        'Bootstrap: 12 resampled 128-neighbor fits at each of 20 training anchors. Route check: compose three edge matches around every graph triangle.\n'
        'Agreement is internal consistency, not MNIST knob accuracy. Graph edges require at least 16 shared training neighbors.', (16, 10))
    ax = fig.add_axes([.08, .55, .38, .27])
    for result in results:
        values = np.array([s['cosine'] for s in result['stability']]).ravel()
        ax.ecdf(values, label=result['name'], color=COLORS[result['name']])
    ax.set(xlabel='Matched unit-vector cosine after bootstrap (1 = identical)', ylabel='Fraction of matched axes', xlim=(0,1))
    ax.legend(fontsize=9); ax.grid(alpha=.2)
    ax = fig.add_axes([.57, .55, .37, .27])
    widths = .33
    for offset, result in enumerate(results):
        values = [result['graphs'][m]['triangle_axis_agreement']*100 for m in ['pixel cosine', 'shared responses']]
        ax.bar(np.arange(2)+(offset-.5)*widths, values, widths, label=result['name'], color=COLORS[result['name']])
        for j, v in enumerate(values):
            ax.text(j+(offset-.5)*widths, v+1, f'{v:.1f}%', ha='center', fontsize=9)
    ax.set_xticks([0,1], ['Pixel-cosine match', 'Shared-response match'])
    ax.axhline(20, color='gray', ls=':', label='Random permutation expectation')
    ax.set(ylabel='Axis labels closing a triangle (%)', ylim=(0,110), title='261 generated / 217 MNIST triangles'); ax.legend(fontsize=8)
    ax = fig.add_axes([.08, .15, .38, .27])
    for result in results:
        groups = [[r['radius'] for r in result['records'] if r['k'] == k] for k in KS]
        ax.plot(KS, np.median(groups, axis=1), 'o-', label=result['name'], color=COLORS[result['name']])
    ax.axhline(.001, color='gray', ls=':', label='Historical generated shell = 0.001')
    ax.set(xlabel='Observed training neighbors per SVD', ylabel='Median neighborhood radius (pixel L2)', yscale='log')
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    ax = fig.add_axes([.57, .15, .37, .27])
    for result in results:
        values = [[r['training_dimension95'] for r in result['records'] if r['k'] == k] for k in KS]
        ax.plot(KS, np.median(values, axis=1), 'o-', label=result['name'], color=COLORS[result['name']])
    ax.axhline(5, ls=':', color='gray'); ax.set(xlabel='Observed training neighbors per SVD', ylabel='Median axes for 95% fitting energy')
    ax.legend(fontsize=9); ax.grid(alpha=.2)
    fig.text(.035, .045, 'All samples remain at their observed distances. Neither equal-distance shells nor constant travel increments are imposed.\n'
        '95% fitting energy is shown as a diagnostic; held-out reconstruction and family stability are separate tests.', fontsize=10)
    save(fig, '05_stability_and_scale.png')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    mx, my = load('train'); tx, ty = load('test')
    mi, ti = np.flatnonzero(my == 1), np.flatnonzero(ty == 1)
    x, test = mx[mi].reshape(-1,784).astype(float)/255, tx[ti].reshape(-1,784).astype(float)/255
    assert x.min() >= 0 and x.max() <= 1
    # Reject exact duplicates across splits from scored test sets, if any.
    keys = {row.tobytes() for row in mx[mi]}
    keep = np.array([row.tobytes() not in keys for row in tx[ti]])
    test_ids = ti[keep]; test = test[keep]
    params = RANGES[:,0]+rng.random((len(x)+len(test),5))*np.diff(RANGES,axis=1).ravel()
    generated = render(params, dtype=np.float64).reshape(-1,784)
    results, arrays = [], []
    for name, train, testing, ids, par in [
            ('Generated cloud', generated[:len(x)], generated[len(x):], np.arange(len(x)), params[:len(x)]),
            ('MNIST', x, test, mi, None)]:
        result, a = run_cloud(name, train, testing, ids, par)
        results.append(result); arrays.append(a)
    old = np.load(REPO/'figures/root_one_local_svd/bases_and_samples.npz')
    root, basis, images = old['root_image'].reshape(784), old['r_0p001_basis'], old['r_0p001_images']
    singular, _ = fit(images[:512]-root)
    d = images[512:]-root
    error5 = np.linalg.norm(d-(d@basis[:5].T)@basis[:5],axis=1)/np.linalg.norm(d,axis=1)
    calibration = dict(image=root, basis=basis[:5], singular=singular)
    metadata = dict(seed=SEED, main_neighbors=MAIN_K, neighbors_sweep=KS, dimensions=DIMS,
                    anchors=N_ANCHORS, bootstrap_anchors=20, bootstrap_trials=12,
                    features=FEATURES, preprocessing='uint8/255 only; no alignment, denoising or augmentation',
                    svd='uncentered neighbor-minus-anchor rows, equal weights, right singular vectors',
                    root_selection='minimum training 128th-neighbor radius; second root nearest training mean; remaining roots farthest-point coverage of denser 80%',
                    graph='union of six nearest-anchor edges; require >=16 shared training neighbors; shortest pixel-distance paths from density root',
                    family_graph_sampling='64 anchors via 16-nearest-neighbor frontier exploration from density root, restricted to denser 80%; choose farthest frontier point; retain exploration edges plus six-nearest-anchor edges',
                    heldout='official MNIST test 1s; equal-size independent generated test cloud; atlas chooses nearest training anchor',
                    split_duplicate_test_images_removed=int((~keep).sum()), original_mnist_test_count=len(ti),
                    mnist_archive_md5={f:_md5(DEFAULT_ROOT/f) for f in FILES},
                    historical_calibration=dict(radius=.001, fitting=512, testing=256,
                        median_relative_error5=float(np.median(error5)), worst_relative_error5=float(error5.max()),
                        source='figures/root_one_local_svd/bases_and_samples.npz'))
    np.savez_compressed(OUT/'test_indices.npz', mnist_original_test_indices=test_ids)
    output = dict(method=metadata, datasets=results)
    (OUT/'results.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    with (OUT/'neighborhood_metrics.csv').open('w') as f:
        fields=['dataset','anchor','source_index','k','radius','nearest_distance','training_dimension95','training_energy5','test_count']
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for result in results for r in result['records'])
    plot_all(results, arrays, calibration)
    print(json.dumps(dict(calibration=metadata['historical_calibration'], results=[dict(name=r['name'],
          atlas=r['atlas'][128], descriptor_r2=r['descriptor_readout_median_r2'],
          graphs={m:{k:v for k,v in g.items() if k not in ['edges','triangles']} for m,g in r['graphs'].items()}) for r in results]),indent=2))


if __name__ == '__main__':
    main()
