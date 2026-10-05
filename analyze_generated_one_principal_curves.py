"""Smooth principal-curve experiment for the full generated-one range box.

Run: .venv/bin/python analyze_generated_one_principal_curves.py
Fits use only coverage pixels; physical knobs are used for interpretation later.
"""
from pathlib import Path
import itertools
import json
import time
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
from scipy.linalg import eigh, solve
from scipy.stats import qmc, spearmanr
from scipy.optimize import least_squares
from grey_ones import KNOBS, RANGES, render

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures/generated_one_principal_curves'
N_NODES = 65
PENALTIES = [1., 10., 100., 1000.]
MAX_ITERATIONS = 35


def settings(n, seed):
    u = qmc.Sobol(5, scramble=True, seed=seed).random_base2(int(np.log2(n)))
    return RANGES[:, 0] + u * np.ptp(RANGES, axis=1)


def pixels(p):
    return render(p).reshape(-1, 784).astype(np.float64)


def project(x, curve):
    """Exact nearest-point projection onto every segment, in all 784 pixels."""
    a, d = curve[:-1], np.diff(curve, axis=0)
    lengths2 = np.einsum('ij,ij->i', d, d)
    xa, xd = x @ a.T, x @ d.T
    ad = np.einsum('ij,ij->i', a, d)
    alpha = np.clip((xd - ad) / np.maximum(lengths2, 1e-20), 0, 1)
    dist2 = np.einsum('ij,ij->i', x, x)[:, None] + np.einsum('ij,ij->i', a, a)[None, :] - 2*xa
    dist2 += alpha**2*lengths2 - 2*alpha*(xd-ad)
    idx = np.argmin(dist2, axis=1)
    frac = alpha[np.arange(len(x)), idx]
    fitted = a[idx] + frac[:, None]*d[idx]
    arc = np.r_[0., np.cumsum(np.sqrt(lengths2))]
    t = (arc[idx]+frac*np.sqrt(lengths2[idx])) / max(arc[-1], 1e-20)
    return fitted, t, np.sum((x-fitted)**2, axis=1)


def resample(curve):
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    arc /= max(arc[-1], 1e-20)
    return np.column_stack([np.interp(np.linspace(0, 1, N_NODES), arc, c) for c in curve.T])


def smooth(x, t, penalty):
    # Penalized linear spline: local interpolation plus second-difference regularity.
    u = np.clip(t, 0, 1)*(N_NODES-1)
    j = np.minimum(u.astype(int), N_NODES-2)
    f = u-j
    b = np.zeros((len(x), N_NODES))
    b[np.arange(len(x)), j] = 1-f
    b[np.arange(len(x)), j+1] = f
    d2 = np.diff(np.eye(N_NODES), n=2, axis=0)
    lhs = b.T @ b + penalty*(d2.T @ d2) + 1e-8*np.eye(N_NODES)
    return solve(lhs, b.T @ x, assume_a='pos')


def linear_start(x, component=0):
    mean = x.mean(0)
    z = x-mean
    _, v = eigh(z.T @ z, subset_by_index=[783-component, 783-component])
    axis = v[:, 0]
    scores = z @ axis
    return mean + np.linspace(scores.min(), scores.max(), N_NODES)[:, None]*axis


def fit(x, validation, penalty, initial, name):
    curve = initial.copy()
    _, _, val_error = project(validation, curve)
    best_loss = val_error.mean()
    best_curve, best_iteration = curve.copy(), 0
    history = []
    stale = 0
    for iteration in range(1, MAX_ITERATIONS+1):
        _, t, _ = project(x, curve)
        # Smooth conditional means, then parameterize by pixel-space arc length.
        curve = resample(smooth(x, t, penalty))
        _, _, train_error = project(x, curve)
        _, _, val_error = project(validation, curve)
        loss = float(val_error.mean())
        history.append(dict(iteration=iteration, train_mse=float(train_error.mean()), validation_mse=loss,
                            length=float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())))
        if loss < best_loss-1e-7:
            best_loss, best_curve, best_iteration = loss, curve.copy(), iteration
            stale = 0
        else:
            stale += 1
        if iteration % 10 == 0:
            print(f'{name} penalty={penalty:g} iteration={iteration} validation SSE/image={loss:.5f}', flush=True)
        if stale >= 7:
            break
    return best_curve, dict(penalty=penalty, selected_iteration=best_iteration,
                           validation_mse=best_loss, history=history)


def error_stats(error, x, mean):
    e2 = np.sum(error**2, axis=1)
    total = np.sum((x-mean)**2)
    return dict(explained_percent=float(100*(1-e2.sum()/total)),
                rms_image_L2=float(np.sqrt(e2.mean())), median_image_L2=float(np.median(np.sqrt(e2))),
                p95_image_L2=float(np.percentile(np.sqrt(e2), 95)),
                max_image_L2=float(np.sqrt(e2.max())), max_pixel_error=float(np.abs(error).max()))


def encode(x, mean, curves):
    residual = x-mean
    ts, stats = [], []
    for curve in curves:
        fitted, t, _ = project(residual, curve)
        ts.append(t)
        residual = residual-fitted
        stats.append(error_stats(residual, x, mean))
    return np.column_stack(ts), x-residual, stats


def nearest_valid(im, initial):
    span = np.ptp(RANGES, axis=1)
    lo = RANGES[:, 0]
    def fun(u):
        return pixels(lo+np.clip(u, 0, 1)*span)[0]-im
    # Finite-difference step is large enough to survive the renderer's float32 output.
    sol = least_squares(fun, np.clip((initial-lo)/span, 1e-5, 1-1e-5), bounds=(0, 1),
                        diff_step=1e-3, ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=160)
    p = lo+sol.x*span
    return p, float(np.linalg.norm(fun(sol.x))), bool(sol.success)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    ptrain, pval, ptest = settings(4096, 731), settings(2048, 732), settings(4096, 733)
    train, val, test = pixels(ptrain), pixels(pval), pixels(ptest)
    mean = train.mean(0)
    z = train-mean
    eig, basis = eigh(z.T @ z)
    basis = basis[:, ::-1].T
    pca = []
    for k in range(1, 11):
        residual = test-mean-(test-mean) @ basis[:k].T @ basis[:k]
        pca.append(error_stats(residual, test, mean))
    curves, traces, chosen = [], [], []
    train_res, val_res = train-mean, val-mean
    for k in range(5):
        initial = linear_start(train_res)
        candidates = []
        for penalty in PENALTIES:
            curve, trace = fit(train_res, val_res, penalty, initial, f'curve {k+1}')
            candidates.append((trace['validation_mse'], curve, trace))
        _, curve, trace = min(candidates, key=lambda c:c[0])
        # Orient for interpretation only; this does not affect fit/projection.
        _, t, _ = project(train_res, curve)
        corr = [spearmanr(t, ptrain[:, j]).statistic for j in range(5)]
        dominant = int(np.nanargmax(np.abs(corr)))
        if corr[dominant] < 0:
            curve = curve[::-1]
        curves.append(curve)
        traces.append([c[2] for c in candidates])
        chosen.append(dict(penalty=trace['penalty'], iteration=trace['selected_iteration']))
        train_res -= project(train_res, curve)[0]
        val_res -= project(val_res, curve)[0]
        print(f'Selected curve {k+1}: {chosen[-1]}', flush=True)
    test_t, reconstructed, curve_stats = encode(test, mean, curves)
    train_t, _, _ = encode(train, mean, curves)
    correlations = [[float(spearmanr(test_t[:,k], ptest[:,j]).statistic) for j in range(5)] for k in range(5)]
    # A second PC initialization tests whether the first learned curve is unique.
    alternate = []
    for penalty in [10., 100.]:
        c, h = fit(train-mean, val-mean, penalty, linear_start(train-mean, 1), 'alternate PC2 start')
        alternate.append((h['validation_mse'], c, h))
    _, alternative, alternative_trace = min(alternate, key=lambda c:c[0])
    alt_t, _, alt_stats = encode(test, mean, [alternative])
    # Full-box lattice, all corners, near-face samples, and parameter midpoints.
    lattice_u = np.array(list(itertools.product(np.linspace(0, 1, 5), repeat=5)))
    corners_u = np.array(list(itertools.product([0., 1.], repeat=5)))
    rng = np.random.default_rng(734)
    faces_u = rng.random((1024, 5))
    for i in range(len(faces_u)):
        faces_u[i, i%5] = 1e-5 if i%2 else 1-1e-5
    mid_u = (rng.random((1024, 5))+rng.random((1024, 5)))/2
    probes = {}
    for label, u in [('lattice_5_per_knob', lattice_u), ('all_32_corners', corners_u),
                     ('near_faces', faces_u), ('joint_parameter_midpoints', mid_u)]:
        xx = pixels(RANGES[:, 0]+u*np.ptp(RANGES, axis=1))
        _, rr, ss = encode(xx, mean, curves)
        probes[label] = dict(count=len(xx), cumulative=ss,
                             out_of_range_pixel_fraction=float(np.mean((rr<0)|(rr>1))))
    lengths = [float(np.linalg.norm(np.diff(c,axis=0),axis=1).sum()) for c in curves]
    chords = [float(np.linalg.norm(c[-1]-c[0])) for c in curves]
    first_images = mean+curves[0]
    closest_knobs, validity = [], []
    for i, im in enumerate(first_images):
        idx = np.argmin(np.sum((test-im)**2, axis=1))
        best = min([nearest_valid(im, ptest[idx]), nearest_valid(im, RANGES.mean(axis=1))], key=lambda v:v[1])
        closest_knobs.append(best[0]); validity.append(best[1])
    first_fitted = project(test-mean, curves[0])[0]+mean
    out = dict(method='Penalized projection-and-smoothing principal polyline; sequential residual extension',
               distribution='Independent uniform knobs over grey_ones.RANGES, scrambled Sobol samples',
               knob_order=KNOBS, ranges=RANGES.tolist(), train_count=len(train), validation_count=len(val),
               test_count=len(test), nodes_per_curve=N_NODES, tested_penalties=PENALTIES,
               precision='float32 renderer output, float64 fitting and all-784-pixel scoring',
               variance_definition='100 * (1 - sum squared reconstruction error / sum squared distance to training mean)',
               selected=chosen, pca_cumulative=pca, curves_cumulative=curve_stats,
               marginal_percentage_points=np.diff([0]+[s['explained_percent'] for s in curve_stats]).tolist(),
               curve_coordinate_knob_spearman=correlations, curve_lengths=lengths, endpoint_chords=chords,
               length_to_chord=[l/c for l,c in zip(lengths,chords)], probes=probes,
               first_curve_nearest_renderer_L2=validity, first_curve_nearest_renderer_knobs=np.array(closest_knobs).tolist(),
               first_curve_out_of_range_pixel_fraction=float(np.mean((first_images<0)|(first_images>1))),
               five_curve_out_of_range_pixel_fraction=float(np.mean((reconstructed<0)|(reconstructed>1))),
               five_curve_pixel_min=float(reconstructed.min()), five_curve_pixel_max=float(reconstructed.max()),
               alternate_PC2_start=dict(selected_penalty=alternative_trace['penalty'],
                   selected_iteration=alternative_trace['selected_iteration'], cumulative=alt_stats,
                   knob_spearman=[float(spearmanr(alt_t[:,0],ptest[:,j]).statistic) for j in range(5)]),
               first_curve_nearest_endpoint_fraction=float(np.mean((test_t[:,0]<1e-8)|(test_t[:,0]>1-1e-8))),
               elapsed_seconds=float(time.monotonic()-start))
    (OUT/'results.json').write_text(json.dumps(out, indent=2)+'\n')
    (OUT/'fit_history.json').write_text(json.dumps(traces, indent=2)+'\n')
    np.savez_compressed(OUT/'model_and_samples.npz', mean=mean, basis=basis[:10], curves=np.array(curves),
                        alternate=alternative, train_knobs=ptrain, test_knobs=ptest, train_t=train_t,
                        test_t=test_t, test_images=test, reconstructed=reconstructed,
                        first_reconstructed=first_fitted, nearest_renderer_knobs=np.array(closest_knobs))
    print(json.dumps({k:out[k] for k in ['curves_cumulative','pca_cumulative','marginal_percentage_points',
          'curve_coordinate_knob_spearman','length_to_chord','elapsed_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
