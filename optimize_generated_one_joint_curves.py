"""Jointly optimize five learned pixel-space polylines on the full covering cloud.

The decoder remains an intercept plus five independent curve contributions.
Only data generation and initialization use the known renderer/knob settings.
Projection and every subsequent curve update use pixel-space distances alone.
"""
from pathlib import Path
import argparse
import json
import os
import time
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
from scipy import sparse
from scipy.linalg import solve
from scipy.stats import spearmanr
from grey_ones import KNOBS, RANGES
from analyze_generated_one_principal_curves import pixels, settings, error_stats

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures/generated_one_principal_curves/joint'
SOURCE = OUT.parent / 'followup/models_and_examples.npz'


def evaluate(c, t):
    u = np.clip(t, 0, 1) * (len(c)-1)
    j = np.minimum(u.astype(int), len(c)-2)
    f = u-j
    return c[j] + f[:, None]*(c[j+1]-c[j])


def project_index(x, c):
    """Exact nearest segment projection; parameter is fractional node index."""
    a = c[:-1]
    d = np.diff(c, axis=0)
    d2 = np.einsum('ij,ij->i', d, d)
    xd = x @ d.T - np.einsum('ij,ij->i', a, d)
    f = np.clip(xd / np.maximum(d2, 1e-20), 0, 1)
    dist = np.einsum('ij,ij->i', x, x)[:, None]
    dist = dist + np.einsum('ij,ij->i', a, a)[None, :] - 2*x @ a.T
    dist += f*f*d2 - 2*f*xd
    j = np.argmin(dist, axis=1)
    frac = f[np.arange(len(x)), j]
    t = (j+frac)/(len(c)-1)
    return a[j] + frac[:, None]*d[j], t


def encode(x, mean, curves, t=None, cycles=20):
    if t is None:
        residual = x-mean
        parts = []
        coords = []
        for c in curves:
            part, coord = project_index(residual, c)
            parts.append(part)
            coords.append(coord)
            residual -= part
        t = np.column_stack(coords)
    else:
        t = t.copy()
        parts = [evaluate(c, t[:, k]) for k, c in enumerate(curves)]
        residual = x-mean-np.sum(parts, axis=0)
    previous = np.sum(residual**2)
    for cycle in range(cycles):
        for k, c in enumerate(curves):
            target = residual+parts[k]
            part, coord = project_index(target, c)
            residual = target-part
            parts[k] = part
            t[:, k] = coord
        loss = np.sum(residual**2)
        if loss > previous+1e-7:
            raise RuntimeError('Nearest-point coordinate update increased squared error')
        if previous-loss < 1e-7*max(previous, 1):
            break
        previous = loss
    return x-residual, t


def design(t, n):
    u = np.clip(t, 0, 1)*(n-1)
    j = np.minimum(u.astype(int), n-2)
    f = u-j
    rows = np.broadcast_to(np.arange(len(t))[:, None], t.shape).ravel()
    cols = (1 + np.arange(5)[None, :]*n+j).ravel()
    return sparse.coo_matrix((np.r_[np.ones(len(t)), (1-f).ravel(), f.ravel()],
        (np.r_[np.arange(len(t)), rows, rows],
         np.r_[np.zeros(len(t), dtype=int), cols, cols+1])),
        shape=(len(t), 1+5*n)).tocsr()


def node_fit(x, t, n, penalty):
    """All nodes of all five curves are updated by a single least-squares solve."""
    b = design(t, n)
    d2 = np.diff(np.eye(n), n=2, axis=0)
    regularizer = sparse.block_diag([sparse.csr_matrix((1, 1))]+[d2.T@d2]*5).toarray()
    # A numerical ridge resolves the additive constant/gauge ambiguities.
    lhs = (b.T@b).toarray()+penalty*regularizer+1e-7*np.eye(b.shape[1])
    coef = solve(lhs, b.T@x, assume_a='pos')
    mean = coef[0].copy()
    curves = coef[1:].reshape(5, n, x.shape[1]).copy()
    for k in range(5):
        offset = evaluate(curves[k], t[:, k]).mean(0)
        curves[k] -= offset
        mean += offset
    return mean, curves


def interpolate_nodes(curves, n):
    return np.array([evaluate(c, np.linspace(0, 1, n)) for c in curves])


def optimize(x, mean, curves, t, stages, label):
    pred, t = encode(x, mean, curves, t, cycles=12)
    best_sse = float(np.mean(np.sum((x-pred)**2, axis=1)))
    best = (mean.copy(), curves.copy(), t.copy())
    history = []
    total_variance = np.sum((x-x.mean(0))**2)
    started = time.monotonic()
    for penalty, cycles in stages:
        stale = 0
        for cycle in range(cycles):
            # Fixed node-index interpolation avoids the additional resampling
            # approximation in the old solver; closest-point geometry is identical.
            new_mean, new_curves = node_fit(x, t, len(curves[0]), penalty)
            pred, new_t = encode(x, new_mean, new_curves, t, cycles=2)
            loss = float(np.mean(np.sum((x-pred)**2, axis=1)))
            row = dict(penalty=penalty, cycle=cycle+1, sse_per_image=loss,
                explained_percent=float(100*(1-loss*len(x)/total_variance)),
                lengths=np.linalg.norm(np.diff(new_curves, axis=1), axis=2).sum(1).tolist())
            history.append(row)
            mean, curves, t = new_mean, new_curves, new_t
            if loss < best_sse-1e-8:
                best_sse = loss
                best = (mean.copy(), curves.copy(), t.copy())
                stale = 0
            else:
                stale += 1
            if cycle % 5 == 0 or cycle == cycles-1:
                print(f'{label}: penalty {penalty:g}, cycle {cycle+1}, '
                      f'capture {row["explained_percent"]:.5f}%, '
                      f'elapsed {time.monotonic()-started:.1f}s', flush=True)
            if stale >= 8:
                break
        # Start each new regularization stage at the best reconstruction so far.
        mean, curves, t = (v.copy() for v in best)
    pred, t = encode(x, *best[:2], best[2], cycles=30)
    return (*best[:2], t, pred, history)


def summarize(x, mean, curves, pred, t, p):
    metrics = error_stats(x-pred, x, x.mean(0))
    # Fitting drops identically-zero columns for speed, but image metrics always
    # include all 784 pixels, including the zero-error pixels.
    metrics['rms_pixel_error'] = float(np.sqrt(np.sum((x-pred)**2)/(len(x)*784)))
    metrics['out_of_range_pixel_fraction'] = float(np.sum((pred < 0)|(pred > 1))/(len(x)*784))
    return dict(metrics=metrics, nodes_per_curve=len(curves[0]),
        lengths=np.linalg.norm(np.diff(curves, axis=1), axis=2).sum(1).tolist(),
        curve_coordinate_knob_spearman=[
            [float(spearmanr(t[:, k], p[:, j]).statistic) for j in range(5)]
            for k in range(5)])


def full_vectors(mean, curves, active):
    a = np.zeros(784)
    a[active] = mean
    c = np.zeros((5, curves.shape[1], 784))
    c[:, :, active] = curves
    return a, c


def encode_multistart(x, mean, curves, reference_x, reference_t):
    """Two pixel-only starts: sequential projection and nearest-image warm start."""
    from scipy.spatial import cKDTree
    _, nearest = cKDTree(reference_x).query(x, k=1, workers=4)
    greedy, gt = encode(x, mean, curves, cycles=30)
    warm, wt = encode(x, mean, curves, reference_t[nearest], cycles=30)
    use_warm = np.sum((x-warm)**2, axis=1) < np.sum((x-greedy)**2, axis=1)
    return np.where(use_warm[:, None], warm, greedy), np.where(use_warm[:, None], wt, gt), greedy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['65', '129', '257', '513', 'polish', 'evaluate'], default='65')
    parser.add_argument('--cycles', type=int, default=30)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    source = np.load(SOURCE)
    p = source['coverage_knobs']
    full_x = pixels(p)
    active = np.any(full_x != 0, axis=0)
    x = full_x[:, active]
    print('Full covering cloud:', len(x), 'images; nonconstant-zero pixels:', active.sum(), flush=True)
    results_path = OUT/'results.json'
    results = json.loads(results_path.read_text()) if results_path.exists() else dict(
        coverage_points=len(x), knob_order=KNOBS,
        score_definition='100 * (1 - total squared pixel error / total squared distance to cloud mean)',
        method='Five additive pixel-space polylines, exact segment projection and joint penalized node fitting',
        generator_use='Only samples, physical-path initialization, and diagnostics; absent from decoder and curve updates',
        candidates={})
    results.setdefault('run_cycles_per_penalty_stage', {})[args.phase] = args.cycles
    if args.phase == '65':
        m = source['mean'][active]
        c = source['curves'][:, :, active]
        baseline_pred, baseline_t = encode(x, m, c, cycles=30)
        results['baseline'] = summarize(x, m, c, baseline_pred, baseline_t, p)
        np.savez_compressed(OUT/'baseline.npz', mean=source['mean'], curves=source['curves'],
                            coordinates=baseline_t)
        starts = [('greedy', m, c, baseline_t)]
        # Knob-informed initialization 1: previously learned additive curves.
        # The known knob coordinates initialize t, but are then entirely free.
        starts.append(('knob_additive', source['additive_mean'][active],
                       source['additive_curves'][:, :, active],
                       (p-RANGES[:, 0])/np.ptp(RANGES, axis=1)))
        # Knob-informed initialization 2: physical lean slice, then residual PCA
        # directions. Exact rendering is used only to choose this starting path.
        from analyze_generated_one_curve_followup import fit_all
        residual = x-m
        lean = source['lean_initialized'][:, active]
        lean_part, _ = project_index(residual, lean)
        residual -= lean_part
        additional = []
        for k in range(4):
            z = residual-residual.mean(0)
            _, _, v = np.linalg.svd(z, full_matrices=False)
            score = z@v[0]
            initial = residual.mean(0)+np.linspace(score.min(), score.max(), 65)[:, None]*v[0]
            curve, _ = fit_all(residual, initial, 65, 1., 20, f'lean start residual {k+1}')
            additional.append(curve)
            residual -= project_index(residual, curve)[0]
        starts.append(('lean_seeded', m, np.array([lean]+additional), None))
        for label, mean, curves, t in starts:
            out = optimize(x, mean, curves, t,
                           [(1., 12), (.1, args.cycles), (.01, args.cycles)], label)
            mm, cc, tt, pred, hist = out
            fm, fc = full_vectors(mm, cc, active)
            np.savez_compressed(OUT/f'{label}_65.npz', mean=fm, curves=fc,
                                coordinates=tt, coverage_knobs=p)
            results['candidates'][f'{label}_65'] = summarize(x, mm, cc, pred, tt, p)
            (OUT/f'{label}_65_history.json').write_text(json.dumps(hist, indent=2)+'\n')
            results_path.write_text(json.dumps(results, indent=2)+'\n')
    elif args.phase in ['129', '257', '513']:
        n = int(args.phase)
        prior_n = (n+1)//2
        eligible = [(v['metrics']['explained_percent'], k) for k, v in results['candidates'].items()
                    if v['nodes_per_curve'] == prior_n]
        _, label = max(eligible)
        prior = np.load(OUT/f'{label}.npz')
        mean = prior['mean'][active]
        curves = interpolate_nodes(prior['curves'][:, :, active], n)
        t = prior['coordinates']
        # Relative second-difference penalty scales with node density cubed.
        scale = ((n-1)/64)**3
        mm, cc, tt, pred, hist = optimize(x, mean, curves, t,
            [(.01*scale, args.cycles), (.001*scale, args.cycles)], f'joint {n} nodes')
        fm, fc = full_vectors(mm, cc, active)
        label = f'joint_{n}'
        np.savez_compressed(OUT/f'{label}.npz', mean=fm, curves=fc, coordinates=tt, coverage_knobs=p)
        results['candidates'][label] = summarize(x, mm, cc, pred, tt, p)
        (OUT/f'{label}_history.json').write_text(json.dumps(hist, indent=2)+'\n')
        results_path.write_text(json.dumps(results, indent=2)+'\n')
    elif args.phase == 'polish':
        label = 'joint_513'
        model = np.load(OUT/f'{label}.npz')
        mm, cc, tt, pred, hist = optimize(x, model['mean'][active], model['curves'][:, :, active],
            model['coordinates'], [(.0512, args.cycles), (.00512, args.cycles)], '513-node final polish')
        fm, fc = full_vectors(mm, cc, active)
        np.savez_compressed(OUT/f'{label}.npz', mean=fm, curves=fc, coordinates=tt, coverage_knobs=p)
        results['candidates'][label] = summarize(x, mm, cc, pred, tt, p)
        path = OUT/f'{label}_history.json'
        prior = json.loads(path.read_text())
        path.write_text(json.dumps(prior+hist, indent=2)+'\n')
        results['final_polish_penalties'] = [.0512, .00512]
        results_path.write_text(json.dumps(results, indent=2)+'\n')
    else:
        # These extra points diagnose sampling resolution, with no train/test split
        # in fitting and no use for hyperparameter selection.
        from scipy.stats import qmc
        import itertools
        dense = settings(16384, 740)
        corners = np.array(list(itertools.product([0., 1.], repeat=5)))
        rng = np.random.default_rng(741)
        mid = (rng.random((2048, 5))+rng.random((2048, 5)))/2
        probes = {'dense_uniform': dense,
                  'all_32_corners': RANGES[:, 0]+corners*np.ptp(RANGES, axis=1),
                  'joint_midpoints': RANGES[:, 0]+mid*np.ptp(RANGES, axis=1)}
        best_label = max(results['candidates'], key=lambda k: results['candidates'][k]['metrics']['explained_percent'])
        labels = ['baseline', best_label]
        best65 = max((k for k, v in results['candidates'].items() if v['nodes_per_curve'] == 65),
                     key=lambda k: results['candidates'][k]['metrics']['explained_percent'])
        if best65 not in labels:
            labels.append(best65)
        results['sampling_checks'] = {}
        for label in ['baseline']+list(results['candidates']):
            model = np.load(OUT/f'{label}.npz')
            mm = model['mean'][active]
            cc = model['curves'][:, :, active]
            tt = model['coordinates']
            pred = mm+sum(evaluate(c, tt[:, k]) for k, c in enumerate(cc))
            stat = summarize(x, mm, cc, pred, tt, p)
            solo, _ = project_index(x-mm, cc[0])
            stat['first_curve_alone_explained_percent'] = error_stats(x-mm-solo, x, x.mean(0))['explained_percent']
            if label == 'baseline':
                results['baseline'] = stat
            else:
                results['candidates'][label] = stat
        for name, pp in probes.items():
            full_probe = pixels(pp)
            assert not np.any(full_probe[:, ~active]), 'New nonzero pixel outside fitting support'
            xx = full_probe[:, active]
            results['sampling_checks'][name] = {}
            for label in labels:
                model = np.load(OUT/f'{label}.npz')
                mean = model['mean'][active]
                curves = model['curves'][:, :, active]
                pred, tt, simple = encode_multistart(xx, mean, curves, x, model['coordinates'])
                results['sampling_checks'][name][label] = summarize(xx, mean, curves, pred, tt, pp)
                results['sampling_checks'][name][label]['sequential_start_capture_percent'] = \
                    error_stats(xx-simple, xx, xx.mean(0))['explained_percent']
                print('Sampling check:', name, label,
                      results['sampling_checks'][name][label]['metrics'], flush=True)
                if name == 'dense_uniform':
                    np.savez_compressed(OUT/f'{label}_dense.npz', knobs=pp, coordinates=tt,
                                        reconstructed=pred, active=active)
        results['best_candidate'] = best_label
        results['encoder'] = ('Nearest-point coordinate refinement from two starts: sequential projection '
                              'and the fitted coordinates of the closest covering image; both use pixels only')
        results_path.write_text(json.dumps(results, indent=2)+'\n')
    print('Candidate capture:', {k: v['metrics']['explained_percent']
          for k, v in results['candidates'].items()}, flush=True)


if __name__ == '__main__':
    main()
