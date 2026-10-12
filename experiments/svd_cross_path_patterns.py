"""Match bundles of SVD paths by cross-path profiles, from one root.

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/svd_cross_path_patterns.py
Known controls supply valid sampling rays; labels are used only after selection.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/svd-cross-path-mpl')
from pathlib import Path
import sys
import itertools
import json
import csv
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from matplotlib.lines import Line2D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grey_ones import KNOBS, RANGES, render
from analyze_generated_one_nearest_direction import local_fit
from analyze_generated_one_manifold_transport import jet, normalized

OUT = ROOT / 'figures' / 'generated_one_cross_path_patterns'
BASE = np.array([14., 14., 19.75, 3.2, 0.])
RADIUS = 1e-4
SEED = 20261012
OFFSETS = np.array([.2, .4, .6, .8])
UNSEEN = np.array([.3, .7])
TICKS = np.linspace(-2.4, 2.4, 33)
MID = len(TICKS) // 2
PERMS = np.array(list(itertools.permutations(range(5))))
NAMES = ['Horizontal position', 'Vertical position', 'Height', 'Width', 'Lean']
COLORS = ['#73529c', '#3669a2', '#249d91', '#9bbf21']


def pixels(p):
    return render(p, dtype=np.float64).reshape(-1, 784)


def valid(p):
    p = np.atleast_2d(p)
    reach = p[:, 3]/2 + p[:, 2]/2 * np.abs(np.tan(np.deg2rad(p[:, 4])))
    return np.all((p[:, 2] > 0) & (p[:, 3] > 0) & (p[:, 0]-reach > 0) &
                  (p[:, 0]+reach < 28) & (p[:, 1]-p[:, 2]/2 > 0) &
                  (p[:, 1]+p[:, 2]/2 < 28))


def observe(p):
    fit = local_fit(p, RADIUS, SEED)
    basis = fit['basis'].copy()
    for b in basis:
        if b[np.argmax(np.abs(b))] < 0:
            b *= -1
    j = jet(p)[3]
    lifts = np.linalg.lstsq(j, basis.T, rcond=None)[0].T
    projected = lifts @ j.T
    error = np.linalg.norm(projected-basis, axis=1)
    lifts /= np.linalg.norm(projected, axis=1)[:, None]
    settings = p[None, None, :] + TICKS[None, :, None]*lifts[:, None, :]
    assert valid(settings.reshape(-1, 5)), 'Probe left the unclipped positive-stroke region.'
    curves = pixels(settings.reshape(-1, 5)).reshape(5, len(TICKS), 784)
    return dict(p=p, basis=basis, lifts=lifts, curves=curves, settings=settings,
                singular=fit['singular'], tail=fit['tail_energy'],
                shell_error=fit['shell_error'], projection_error=error)


def profile(base, curve):
    """The reference figure's middle/right panels plus displacement angle."""
    base_change = base-base[MID]
    change = curve-curve[MID]
    gap = np.linalg.norm(curve-base, axis=1)
    mismatch = np.linalg.norm(change-base_change, axis=1)
    bn = np.linalg.norm(base_change, axis=1)
    cn = np.linalg.norm(change, axis=1)
    use = np.arange(len(TICKS)) != MID
    angle = np.zeros(len(TICKS))
    angle[use] = np.arccos(np.clip(np.sum(change[use]*base_change[use], axis=1) /
                                  (bn[use]*cn[use]), -1, 1))
    initial_gap = gap[MID]
    assert initial_gap > 1e-8
    # One scalar per offset is measured at t=0, NOT fitted to flatten curves.
    # Each of the three blocks receives equal total squared-error weight.
    feature = np.concatenate([gap/initial_gap/np.sqrt(len(TICKS)),
                              mismatch/initial_gap/np.sqrt(len(TICKS)),
                              angle[use]/(np.pi/2)/initial_gap/np.sqrt(use.sum())])/np.sqrt(3)
    return dict(gap=gap, mismatch=mismatch, angle=angle,
                initial_gap=initial_gap, feature=feature)


def candidates(base_curves, node):
    # Both orientations are rendered. Choose sign by displacement agreement
    # along the entire probe curve, rather than by one tangent-vector dot.
    features = np.empty((5, 5, 3*len(TICKS)-1))
    signs = np.ones((5, 5), int)
    profiles = [[None for _ in range(5)] for _ in range(5)]
    for f in range(5):
        for k in range(5):
            plus = profile(base_curves[f], node['curves'][k])
            minus = profile(base_curves[f], node['curves'][k, ::-1])
            sign = 1 if np.mean(plus['mismatch']**2) <= np.mean(minus['mismatch']**2) else -1
            chosen = plus if sign == 1 else minus
            features[f, k] = chosen['feature']
            signs[f, k] = sign
            profiles[f][k] = chosen
    return dict(features=features, signs=signs, profiles=profiles)


def select_bundle(observations):
    """Exact optimization for the stated nearest-offset anchored objective.

    Enumerate all 120 one-to-one assignments at the nearest branch offset.
    Conditional on each, Hungarian assignment solves the other three levels.
    No tangent cosine, physical labels, or linear-distance target enters cost.
    """
    x = np.array([o['features'] for o in observations])
    trials = []
    for anchor_perm in PERMS:
        template = x[0, np.arange(5), anchor_perm]
        orders = [anchor_perm.copy()]
        loss = 0.
        for j in range(1, len(x)):
            cost = np.sum((x[j]-template[:, None, :])**2, axis=2)
            rows, cols = linear_sum_assignment(cost)
            order = cols[np.argsort(rows)]
            orders.append(order)
            loss += cost[np.arange(5), order].sum()
        trials.append((float(loss), np.array(orders)))
    trials.sort(key=lambda z: (z[0], tuple(z[1].ravel())))
    loss, orders = trials[0]
    selected = x[np.arange(len(x))[:, None], np.arange(5)[None, :], orders]
    return dict(loss=loss, orders=orders, template=selected.mean(axis=0),
                anchor_template=selected[0],
                family_losses=np.sum((selected[1:]-selected[0])**2, axis=(0, 2)),
                second_loss=trials[1][0], alternative_anchor_orders=trials[1][1],
                near_optimal_anchor_assignments=sum(v[0] <= loss*1.01+1e-12 for v in trials))


def select_unseen(observation, template):
    cost = np.sum((observation['features']-template[:, None, :])**2, axis=2)
    rows, cols = linear_sum_assignment(cost)
    return cols[np.argsort(rows)]


def evaluate_node(node, orders, signs, root_labels, root_signs, context, offset, phase):
    product = node['basis'] @ normalized(jet(node['p'])[3])
    rows, cols = linear_sum_assignment(-np.abs(product))
    labels = cols[np.argsort(rows)]
    records = []
    for f, k in enumerate(orders):
        expected = root_labels[f]
        s = signs[f, k]
        signed = product[k, expected]*s*root_signs[f]
        records.append(dict(context_family=int(context+1), offset=float(offset), phase=phase,
                            family=int(f+1), expected_knob=KNOBS[expected],
                            selected_knob=KNOBS[labels[k]], selected_axis=int(k+1),
                            correct_family=bool(labels[k] == expected),
                            correct_family_and_sign=bool(labels[k] == expected and signed > 0),
                            cosine_to_expected=float(abs(product[k, expected])),
                            signed_cosine_to_expected=float(signed)))
    return records


def true_curves(node, labels, signs):
    j = jet(node['p'])[3]
    lifts = np.zeros((5, 5))
    for f in range(5):
        lifts[f, labels[f]] = signs[f]/np.linalg.norm(j[:, labels[f]])
    settings = node['p'][None, None, :] + TICKS[None, :, None]*lifts[:, None, :]
    assert valid(settings.reshape(-1, 5))
    return pixels(settings.reshape(-1, 5)).reshape(5, len(TICKS), 784), settings


def run():
    start = time.monotonic()
    root = observe(BASE)
    bundles = []
    for b in range(5):
        nodes = [observe(BASE+s*root['lifts'][b]) for s in OFFSETS]
        obs = [candidates(root['curves'], node) for node in nodes]
        selection = select_bundle(obs)
        fresh = [observe(BASE+s*root['lifts'][b]) for s in UNSEEN]
        fresh_obs = [candidates(root['curves'], node) for node in fresh]
        fresh_orders = [select_unseen(o, selection['template']) for o in fresh_obs]
        bundles.append(dict(context=b, nodes=nodes, obs=obs, selection=selection,
                            fresh=fresh, fresh_obs=fresh_obs, fresh_orders=fresh_orders))
        print(f'Context branch {b+1}/5 selected; loss {selection["loss"]:.6g}', flush=True)
    # Discovery and unseen assignments have all frozen before labels are read.
    products = root['basis'] @ normalized(jet(BASE)[3])
    rows, cols = linear_sum_assignment(-np.abs(products))
    labels = cols[np.argsort(rows)]
    signs = np.sign(products[np.arange(5), labels]).astype(int)
    actual_base, _ = true_curves(root, labels, signs)
    records = []
    details = []
    for bundle in bundles:
        b = bundle['context']
        oracle_profiles = []
        oracle_axis_profiles = []
        for j, (s, node, obs, order) in enumerate(zip(OFFSETS, bundle['nodes'], bundle['obs'], bundle['selection']['orders'])):
            records += evaluate_node(node, order, obs['signs'], labels, signs, b, s, 'discovery')
            actual, actual_settings = true_curves(node, labels, signs)
            node['actual'] = actual; node['actual_settings'] = actual_settings
            pp = [profile(actual_base[f], actual[f]) for f in range(5)]
            oracle_profiles.append(pp)
            p = node['basis'] @ normalized(jet(node['p'])[3])
            rr, cc = linear_sum_assignment(-np.abs(p))
            local_labels = cc[np.argsort(rr)]
            correct_order = np.array([np.flatnonzero(local_labels == labels[f])[0] for f in range(5)])
            oracle_axis_profiles.append(obs['features'][np.arange(5), correct_order])
        oracle_x = np.array([[p['feature'] for p in row] for row in oracle_profiles])
        oracle_axis_x = np.array(oracle_axis_profiles)
        truth_loss = np.sum((oracle_x[1:]-oracle_x[0])**2, axis=(0, 2))
        correct_axis_loss = np.sum((oracle_axis_x[1:]-oracle_axis_x[0])**2, axis=(0, 2))
        bundle['oracle_profiles'] = oracle_profiles
        bundle['truth_loss'] = truth_loss
        bundle['correct_axis_loss'] = correct_axis_loss
        for s, node, obs, order in zip(UNSEEN, bundle['fresh'], bundle['fresh_obs'], bundle['fresh_orders']):
            records += evaluate_node(node, order, obs['signs'], labels, signs, b, s, 'unseen')
        details.append(dict(context_family=b+1, context_knob=KNOBS[labels[b]],
                            selected_loss=bundle['selection']['loss'],
                            pure_control_loss=float(truth_loss.sum()),
                            correct_svd_label_loss=float(correct_axis_loss.sum()),
                            selected_family_losses=bundle['selection']['family_losses'].tolist(),
                            pure_control_family_losses=truth_loss.tolist(),
                            correct_svd_label_family_losses=correct_axis_loss.tolist(),
                            selected_orders=(bundle['selection']['orders']+1).tolist(),
                            second_best_anchor_loss=bundle['selection']['second_loss'],
                            near_optimal_anchor_assignments=bundle['selection']['near_optimal_anchor_assignments']))
    phases = {}
    for phase in ['discovery', 'unseen']:
        subset = [r for r in records if r['phase'] == phase]
        phases[phase] = dict(comparisons=len(subset), correct=sum(r['correct_family'] for r in subset),
                             correct_family_and_sign=sum(r['correct_family_and_sign'] for r in subset))
    confusion = {}
    for r in records:
        if not r['correct_family']:
            key = r['expected_knob']+' -> '+r['selected_knob']
            confusion[key] = confusion.get(key, 0)+1
    nodes = [root] + [n for b in bundles for key in ['nodes', 'fresh'] for n in b[key]]
    result = dict(method=dict(root=BASE.tolist(), radius=RADIUS, neighborhood_samples=256,
                    seed=SEED, precision='float64 renderer coverage',
                    branch_offsets=OFFSETS.tolist(), unseen_offsets=UNSEEN.tolist(),
                    path_ticks=TICKS.tolist(),
                    offset_units='amount along a root SVD ray with initial pixel speed 1; finite distances are measured, not fixed',
                    path_units='amount along each candidate local SVD ray, with initial pixel speed 1',
                    graph='one root; five context rays, four observed offsets each; five local candidate path rays per offset',
                    profiles=['distance between offset and base path at corresponding ticks',
                              'difference between their pixel-displacement vectors after subtracting their own starts',
                              'angle between those displacement vectors; undefined zero tick excluded'],
                    amplitude='divide all three profiles by observed start-image gap; angle first divided by 90 degrees',
                    objective='sum of equally weighted squared differences of normalized profiles across offsets, anchored at nearest offset',
                    optimization='enumerate 120 nearest-offset permutations; solve each other offset by Hungarian assignment; select global minimum for this anchored objective',
                    sign_rule='orientation with least mean squared displacement-vector mismatch to root path over all 33 ticks',
                    unseen='freeze mean selected normalized profiles; Hungarian selection at two new context offsets',
                    oracle='known renderer/Jacobian supply neighborhoods and valid parameter-ray moves; matcher receives only rendered curves, not knob settings or labels',
                    limitations=['fixed local parameter-ray probes, not adaptively tracked paths',
                                 'corresponding ticks use local pixel-speed calibration, not recovered common knob numbers',
                                 'SVD axes only, no mixtures fitted',
                                 'root context rays and default-range-shaped sampling are generator-assisted',
                                 'same root only; default box extended, strokes stay positive and unclipped',
                                 'profile scaling proportional to start gap is a tested assumption, not guaranteed manifold structure']),
                root=dict(labels=[KNOBS[k] for k in labels], signs=signs.tolist(),
                          matching_cosines=np.abs(products[np.arange(5), labels]).tolist()),
                phases=phases, confusion=confusion, contexts=details,
                minimum_selected_control_cosine=float(min(r['cosine_to_expected'] for r in records)),
                state_count=len(nodes), max_projection_error=float(max(n['projection_error'].max() for n in nodes)),
                max_tail_energy=float(max(n['tail'] for n in nodes)),
                max_shell_error=float(max(n['shell_error'] for n in nodes)),
                outside_default_box_states=sum(np.any((n['p'] < RANGES[:, 0]) | (n['p'] > RANGES[:, 1])) for n in nodes),
                seconds=time.monotonic()-start)
    return root, bundles, labels, signs, records, result


def plot_bundles(root, bundles, labels, signs, records, result, actual=False):
    # Each family gets a different other-family context. Width/lean both shown.
    shown_context_knobs = {0: 4, 1: 2, 2: 1, 3: 4, 4: 3}
    contexts = [int(np.flatnonzero(labels == shown_context_knobs[labels[f]])[0]) for f in range(5)]
    fig = plt.figure(figsize=(18, 15.3), facecolor='#fafaf7')
    title = '2 · Pure-control bundles on the same observed branches' if actual else '1 · Families selected by patterns across several paths'
    fig.text(.025, .975, title, fontsize=21, weight='bold', va='top')
    fig.text(.025, .940, 'Generated 28 × 28 images: white = no ink, black = full ink, gray = partial ink. Root: (14, 14, 19.75, 3.2, 0°).', fontsize=11)
    fig.text(.025, .917, 'Each row compares a root path with four paths starting farther along ONE other root direction. Curves can rise, fall, or peak.', fontsize=11)
    fig.text(.025, .894, 'Black dashed = base path. An amount of 1 has initial pixel speed 1; finite distances are measured, not forced equal.', fontsize=10.5)
    fig.legend([Line2D([0], [0], color=c, lw=2) for c in COLORS],
               [f's = {v:.1f}' for v in OFFSETS], loc='center right',
               bbox_to_anchor=(.97, .866), ncol=4, frameon=False, fontsize=10)
    grid = fig.add_gridspec(5, 6, left=.145, right=.97, bottom=.105, top=.84,
                           width_ratios=[.65, .65, .65, 1.65, 1.65, 1.65], wspace=.43, hspace=.78)
    for f in range(5):
        b = contexts[f]; bundle = bundles[b]
        source = root['curves'][f]
        if actual:
            source = true_curves(root, labels, signs)[0][f]
        last = bundle['nodes'][-1]
        order = bundle['selection']['orders'][-1, f]
        sg = bundle['obs'][-1]['signs'][f, order]
        curve = last['actual'][f] if actual else last['curves'][order][::sg]
        selected_records = [r for r in records if r['phase'] == 'discovery' and r['context_family'] == b+1 and r['family'] == f+1]
        good = sum(r['correct_family_and_sign'] for r in selected_records)
        axes = []
        for k, im in enumerate([source[MID], curve[MID], curve[-1]]):
            ax = fig.add_subplot(grid[f, k]); axes.append(ax)
            ax.imshow(im.reshape(28, 28), cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if f == 0:
                ax.set_title(['Root start', 'Offset start\ns = 0.8', 'Offset path\nt = +2.4'][k], fontsize=10, pad=10)
        pos = axes[0].get_position()
        label = f'F{f+1} · {NAMES[labels[f]]}\nOther: {NAMES[labels[b]]}'
        label += '\nPure control' if actual else f'\nCorrect: {good}/4'
        fig.text(.025, (pos.y0+pos.y1)/2, label, fontsize=10.5, va='center', linespacing=1.5)
        plots = [fig.add_subplot(grid[f, i]) for i in [3, 4, 5]]
        for j in range(4):
            if actual:
                p = bundle['oracle_profiles'][j][f]
            else:
                k = bundle['selection']['orders'][j, f]
                p = bundle['obs'][j]['profiles'][f][k]
            for ax, key, factor in zip(plots, ['gap', 'mismatch', 'angle'], [1, 1, 180/np.pi]):
                y = p[key]*factor
                if key == 'angle':
                    y = y.copy(); y[MID] = np.nan
                ax.plot(TICKS, y, color=COLORS[j], lw=1.7)
                if key != 'angle':
                    ax.scatter([0], [y[MID]], color=COLORS[j], s=17, zorder=4)
        losses = bundle['truth_loss'] if actual else bundle['selection']['family_losses']
        plots[1].text(.03, .96, f'Profile cost: {losses[f]:.3g}', transform=plots[1].transAxes, va='top', fontsize=8.5)
        for k, ax in enumerate(plots):
            ax.axhline(0, color='black', ls='--', lw=1)
            ax.axvline(0, color='#999999', ls=':', lw=.8)
            ax.grid(alpha=.18); ax.tick_params(labelsize=8)
            ax.set_xlabel('Path-ray amount t', fontsize=9)
            ax.set_ylabel('Degrees' if k == 2 else 'Pixel L2 distance', fontsize=9)
            ax.set_xlim(TICKS[0], TICKS[-1])
            ax.set_ylim(bottom=-.03)
            if f == 0:
                ax.set_title(['Gap at corresponding ticks', 'Displacement mismatch', 'Displacement-vector angle'][k], fontsize=11, pad=12)
    fig.text(.025, .058, 'Scoring compares whole profiles divided by each start gap; it does not reward straight paths or constant step distances. Lower profile cost = more agreement.', fontsize=10)
    fig.text(.025, .036, ('These are finite pure-control rays on the same branch starts; all other settings stay fixed within each path.' if actual else 'These are finite rays from local SVD directions, not adaptive walks. Names evaluate frozen choices; the selector does not receive them.'), fontsize=10)
    fig.text(.025, .016, 'Five of 25 family/context pairs are displayed; all 25 are scored. Angles at t = 0 are undefined and omitted. Local probes do not establish global coordinates.', fontsize=10)
    name = '02_pure_control_bundles.png' if actual else '01_selected_bundles.png'
    fig.savefig(OUT/name, dpi=145, facecolor=fig.get_facecolor()); plt.close(fig)
    return contexts


def plot_validation(root, bundles, labels, signs, records, result):
    fig = plt.figure(figsize=(15, 8.8), facecolor='#fafaf7')
    fig.text(.04, .965, '3 · Does cross-path coherence identify the physical families?', fontsize=20, weight='bold', va='top')
    fig.text(.04, .913, 'One root; five context branches. Each cell tests four training offsets or two fresh offsets, jointly assigning five SVD axes.', fontsize=11)
    grid = fig.add_gridspec(1, 3, left=.09, right=.96, top=.79, bottom=.24, width_ratios=[1, 1, 1.1], wspace=.48)
    for c, phase in enumerate(['discovery', 'unseen']):
        ax = fig.add_subplot(grid[0, c])
        count = len(OFFSETS) if phase == 'discovery' else len(UNSEEN)
        matrix = np.zeros((5, 5), int)
        for r in records:
            if r['phase'] == phase:
                matrix[r['family']-1, r['context_family']-1] += r['correct_family_and_sign']
        ax.imshow(matrix/count, cmap='RdYlGn', vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f'{matrix[i,j]}/{count}', ha='center', va='center', fontsize=12, color='#171717')
        ax.set_xticks(range(5), [KNOBS[k] for k in labels], rotation=40, ha='right')
        ax.set_yticks(range(5), [f'F{i+1} · {KNOBS[k]}' for i,k in enumerate(labels)])
        ax.set_xlabel('Direction used to vary the starting image', fontsize=10)
        ax.set_ylabel('Path family to identify', fontsize=10)
        n = result['phases'][phase]
        ax.set_title(('Training offsets' if phase == 'discovery' else 'Fresh offsets')+f'\n{n["correct_family_and_sign"]}/{n["comparisons"]} family + sign', fontsize=12, pad=14)
    ax = fig.add_subplot(grid[0, 2])
    x = np.arange(5)
    chosen = [b['selection']['loss'] for b in bundles]
    correct = [float(b['correct_axis_loss'].sum()) for b in bundles]
    pure = [float(b['truth_loss'].sum()) for b in bundles]
    ax.bar(x-.24, chosen, .24, label='Selected axes', color='#b25b35')
    ax.bar(x, correct, .24, label='Correctly labeled axes', color='#386a9b')
    ax.bar(x+.24, pure, .24, label='Pure controls', color='#299c8d')
    ax.set_xticks(x, [KNOBS[k] for k in labels], rotation=40, ha='right')
    ax.set_xlabel('Context branch', fontsize=10)
    ax.set_ylabel('Across-offset profile cost (lower = more coherent)', fontsize=10)
    ax.set_title('Selected vs known-control calibration', fontsize=12, pad=14)
    ax.legend(fontsize=8, frameon=False); ax.grid(axis='y', alpha=.2)
    fig.text(.04, .145, 'Green = all selections agree with the expected physical family AND sign; red = none. Names are evaluation labels only.', fontsize=10.5)
    fig.text(.04, .113, 'Fresh offsets 0.3 and 0.7 were excluded from selection and template fitting. Training offsets: 0.2, 0.4, 0.6, 0.8.', fontsize=10.5)
    fig.text(.04, .081, 'Blue bars force the correct SVD-family labels under the same sign rule. Teal bars use exact physical directions, on the same branch starts.', fontsize=10)
    fig.text(.04, .048, 'Orange and blue coincide here: every selected training axis has the expected physical-family label. Profile coherence is approximate, even for pure controls.', fontsize=10)
    fig.savefig(OUT/'03_validation.png', dpi=155, facecolor=fig.get_facecolor()); plt.close(fig)


def verify_invariance(root, bundles):
    """A semantic check: selections cannot depend on SVD index or sign."""
    rng = np.random.default_rng(20261013)
    comparisons = 0
    for bundle in bundles:
        shuffled = []
        maps = []
        for node in bundle['nodes']:
            order = rng.permutation(5)
            flips = rng.choice([-1, 1], size=5)
            curves = node['curves'][order].copy()
            for k in range(5):
                curves[k] = curves[k, ::flips[k]]
            shuffled.append(candidates(root['curves'], dict(curves=curves)))
            maps.append((order, flips))
        chosen = select_bundle(shuffled)
        assert abs(chosen['loss']-bundle['selection']['loss']) < 1e-10
        for j, obs in enumerate(shuffled):
            order, flips = maps[j]
            for f in range(5):
                k = chosen['orders'][j, f]
                original = bundle['selection']['orders'][j, f]
                assert order[k] == original
                assert obs['signs'][f, k]*flips[k] == bundle['obs'][j]['signs'][f, original]
                comparisons += 1
        for j, node in enumerate(bundle['fresh']):
            order = rng.permutation(5)
            flips = rng.choice([-1, 1], size=5)
            curves = node['curves'][order].copy()
            for k in range(5):
                curves[k] = curves[k, ::flips[k]]
            obs = candidates(root['curves'], dict(curves=curves))
            chosen_order = select_unseen(obs, chosen['template'])
            for f, k in enumerate(chosen_order):
                original = bundle['fresh_orders'][j][f]
                assert order[k] == original
                assert obs['signs'][f, k]*flips[k] == bundle['fresh_obs'][j]['signs'][f, original]
                comparisons += 1
    return dict(random_candidate_order_and_sign_invariance=True, selections_checked=comparisons)


def orient_frame(node, order, signs):
    out = dict(node)
    selected_signs = signs[np.arange(5), order]
    for key in ['basis', 'lifts', 'curves', 'settings', 'projection_error']:
        out[key] = node[key][order].copy()
    out['basis'] *= selected_signs[:, None]
    out['lifts'] *= selected_signs[:, None]
    for f, s in enumerate(selected_signs):
        if s < 0:
            out['curves'][f] = out['curves'][f, ::-1]
            out['settings'][f] = out['settings'][f, ::-1]
    return out


def walk_experiment(root, labels, signs):
    step = .3
    paths = []
    diagnostics = []
    for f in range(5):
        path = [root]
        for hop in range(1, 10):
            parent = path[-1]
            # Same cross-offset pattern rule, now centered at the current state.
            probes = [observe(parent['p']+s*parent['lifts'][f]) for s in OFFSETS]
            obs = [candidates(parent['curves'], n) for n in probes]
            fitted = select_bundle(obs)
            child = observe(parent['p']+step*parent['lifts'][f])
            child_obs = candidates(parent['curves'], child)
            order = select_unseen(child_obs, fitted['template'])
            child = orient_frame(child, order, child_obs['signs'])
            path.append(child)
            diagnostics.append(dict(path_family=f+1, hop=hop, training_loss=fitted['loss'],
                                    second_anchor_loss=fitted['second_loss'],
                                    source_settings=parent['p'].tolist(), child_settings=child['p'].tolist()))
        paths.append(path)
        print(f'Adaptive path {f+1}/5 frozen at nine hops.', flush=True)
    # Labels are evaluated after all five trajectories freeze.
    records = []
    references = []
    for f, path in enumerate(paths):
        reference = [BASE.copy()]
        for hop in range(1, 10):
            p = reference[-1].copy()
            p[labels[f]] += step*signs[f]/np.linalg.norm(jet(p)[3][:, labels[f]])
            reference.append(p)
            parent = path[hop-1]
            child = path[hop]
            pprod = parent['basis'] @ normalized(jet(parent['p'])[3])
            cprod = child['basis'] @ normalized(jet(child['p'])[3])
            rr, cc = linear_sum_assignment(-np.abs(pprod))
            plabels = cc[np.argsort(rr)]
            rr, cc = linear_sum_assignment(-np.abs(cprod))
            clabels = cc[np.argsort(rr)]
            for k in range(5):
                records.append(dict(path_family=f+1, hop=hop, phase='first_six' if hop <= 6 else 'last_three',
                                    family=k+1, expected_knob=KNOBS[labels[k]],
                                    selected_knob=KNOBS[clabels[k]],
                                    correct_family=bool(clabels[k] == labels[k]),
                                    correct_family_and_sign=bool(clabels[k] == labels[k] and cprod[k, labels[k]]*signs[k] > 0),
                                    executed_family=bool(k == f),
                                    executed_with_correct_source_family_and_sign=bool(plabels[f] == labels[f] and pprod[f, labels[f]]*signs[f] > 0)))
        references.append(np.array(reference))
    summary = {}
    for phase in ['first_six', 'last_three']:
        subset = [r for r in records if r['phase'] == phase]
        actual = [r for r in subset if r['executed_family']]
        summary[phase] = dict(comparisons=len(subset), correct_family=sum(r['correct_family'] for r in subset),
                              correct_family_and_sign=sum(r['correct_family_and_sign'] for r in subset),
                              executed_moves=len(actual), executed_moves_with_correct_source_family_and_sign=sum(r['executed_with_correct_source_family_and_sign'] for r in actual))
    errors = []
    for f, path in enumerate(paths):
        found = np.array([n['curves'][0, MID] for n in path])
        truth = pixels(references[f])
        errors.append(dict(family=f+1, knob=KNOBS[labels[f]],
                           endpoint_pixel_error=float(np.linalg.norm(found[-1]-truth[-1])),
                           endpoint_settings=path[-1]['p'].tolist(),
                           reference_endpoint_settings=references[f][-1].tolist()))
    summary['paths'] = errors
    summary['probe_rule'] = 'At each hop, fit four offset bundles along the moved family, select five axes jointly at the fresh 0.3 endpoint, reorient the frame, and repeat.'
    summary['continuation'] = 'Last three hops use the unchanged rule; new local templates are fitted at every hop, including these. Not frozen-template prediction.'
    summary['observation_count'] = 5*9*5
    return paths, np.array(references), records, diagnostics, summary


def turning_angles(images):
    d = np.diff(images, axis=0)
    d /= np.linalg.norm(d, axis=1)[:, None]
    return np.rad2deg(np.arccos(np.clip(np.sum(d[1:]*d[:-1], axis=1), -1, 1)))


def plot_walks(paths, references, labels, signs, records, summary):
    fig = plt.figure(figsize=(16, 13.8), facecolor='#fafaf7')
    fig.text(.03, .97, '4 · Repeated walks using the cross-path pattern rule', fontsize=21, weight='bold', va='top')
    fig.text(.03, .93, 'Generated 28 × 28 coverage images: white = no ink, black = full ink. Every hop observes a new local SVD neighborhood (radius 0.0001).', fontsize=10.5)
    fig.text(.03, .905, 'At each hop, four nearby offset bundles train a shared profile; the fresh 0.3 endpoint receives five jointly assigned families.', fontsize=10.5)
    fig.text(.03, .880, 'Orange dashed = found walk; blue solid = pure control. Shading marks the last three hops; the matching rule stays fixed.', fontsize=10.5)
    grid = fig.add_gridspec(5, 6, left=.15, right=.975, top=.82, bottom=.115,
                           width_ratios=[.8, .8, .8, .8, 2, 2], wspace=.43, hspace=.78)
    for f, path in enumerate(paths):
        found = np.array([n['curves'][0, MID] for n in path])
        truth = pixels(references[f])
        axes = []
        for j, hop in enumerate([0, 3, 6, 9]):
            ax = fig.add_subplot(grid[f, j]); axes.append(ax)
            ax.imshow(found[hop].reshape(28,28), cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if f == 0:
                ax.set_title(f'Hop {hop}', fontsize=11, pad=10)
        pos = axes[0].get_position()
        chosen = [r for r in records if r['path_family'] == f+1 and r['executed_family']]
        good = sum(r['executed_with_correct_source_family_and_sign'] for r in chosen)
        sg = '+' if signs[f] > 0 else '−'
        fig.text(.03, (pos.y0+pos.y1)/2, f'F{f+1} · {NAMES[labels[f]]} {sg}\nCorrect moves: {good}/9\nEndpoint error: {summary["paths"][f]["endpoint_pixel_error"]:.3f}', fontsize=10, va='center', linespacing=1.5)
        axd = fig.add_subplot(grid[f, 4]); axt = fig.add_subplot(grid[f, 5])
        for x, color, ls, name in [(truth, '#3669a2', '-', 'Pure control'), (found, '#b25b35', '--', 'Found')]:
            axd.plot(range(10), np.linalg.norm(x-x[0],axis=1), color=color, ls=ls, lw=1.8, label=name)
            axt.plot(range(2,10), turning_angles(x), color=color, ls=ls, lw=1.8)
        axd.set_ylabel('Pixel L2 distance', fontsize=9); axt.set_ylabel('Degrees', fontsize=9)
        if f == 0:
            axd.set_title('Distance from root', fontsize=11, pad=11)
            axt.set_title('Turn between steps', fontsize=11, pad=11)
            axd.legend(fontsize=8, frameon=False)
        for ax in [axd,axt]:
            ax.axvspan(6,9,color='#dfead9',alpha=.6)
            ax.set_xlim(0,9); ax.set_xticks([0,3,6,9]); ax.set_xlabel('Hop',fontsize=9)
            ax.set_ylim(bottom=-.03); ax.grid(alpha=.2); ax.tick_params(labelsize=8)
    train = summary['first_six']; test = summary['last_three']
    fig.text(.03, .070, f'First six hops: {train["executed_moves_with_correct_source_family_and_sign"]}/30 executed moves correct; {train["correct_family_and_sign"]}/150 next-field family + sign selections correct.', fontsize=10.5)
    fig.text(.03, .045, f'Last three hops: {test["executed_moves_with_correct_source_family_and_sign"]}/15 executed moves correct; {test["correct_family_and_sign"]}/75 next-field family + sign selections correct.', fontsize=10.5)
    fig.text(.03, .022, 'Endpoint error is pixel L2 distance to the calibrated pure-control walk. Correct labels allow mixtures. Valid probes still use the known renderer and Jacobian.', fontsize=10)
    fig.savefig(OUT/'04_repeated_walks.png',dpi=145,facecolor=fig.get_facecolor()); plt.close(fig)



def main():
    OUT.mkdir(parents=True, exist_ok=True)
    root, bundles, labels, signs, records, result = run()
    result['verification'] = verify_invariance(root, bundles)
    walks, walk_references, walk_rows, walk_diagnostics, walk_summary = walk_experiment(root, labels, signs)
    result['walks'] = walk_summary
    plot_walks(walks, walk_references, labels, signs, walk_rows, walk_summary)
    np.savez_compressed(OUT/'repeated_walks.npz', settings=np.array([[n['p'] for n in path] for path in walks]),
                        images=np.array([[n['curves'][0, MID] for n in path] for path in walks]), references=walk_references)
    with (OUT/'walk_associations.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(walk_rows[0])); writer.writeheader(); writer.writerows(walk_rows)
    (OUT/'walk_diagnostics.json').write_text(json.dumps(walk_diagnostics, indent=2)+'\n')
    contexts = plot_bundles(root, bundles, labels, signs, records, result)
    plot_bundles(root, bundles, labels, signs, records, result, actual=True)
    plot_validation(root, bundles, labels, signs, records, result)
    result['displayed_context_families'] = [c+1 for c in contexts]
    (OUT/'results.json').write_text(json.dumps(result, indent=2, default=lambda x: x.item())+'\n')
    with (OUT/'associations.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    np.savez_compressed(OUT/'paths_and_profiles.npz', ticks=TICKS, offsets=OFFSETS, unseen=UNSEEN,
                        root_settings=BASE, root_basis=root['basis'], root_curves=root['curves'],
                        settings=np.array([[n['p'] for n in b['nodes']] for b in bundles]),
                        candidate_curves=np.array([[n['curves'] for n in b['nodes']] for b in bundles]),
                        candidate_features=np.array([[o['features'] for o in b['obs']] for b in bundles]),
                        candidate_signs=np.array([[o['signs'] for o in b['obs']] for b in bundles]),
                        assignments=np.array([b['selection']['orders'] for b in bundles]),
                        templates=np.array([b['selection']['template'] for b in bundles]),
                        actual_curves=np.array([[n['actual'] for n in b['nodes']] for b in bundles]),
                        unseen_settings=np.array([[n['p'] for n in b['fresh']] for b in bundles]),
                        unseen_curves=np.array([[n['curves'] for n in b['fresh']] for b in bundles]),
                        unseen_assignments=np.array([b['fresh_orders'] for b in bundles]))
    print(json.dumps(result, indent=2, default=lambda x: x.item()), flush=True)


if __name__ == '__main__':
    main()
