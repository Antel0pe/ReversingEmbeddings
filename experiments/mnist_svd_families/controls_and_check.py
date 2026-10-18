"""Independent saved-array audit and two local fitting controls.

Run after experiment.py with the same Python environment and single-thread BLAS.
"""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys
import json
import numpy as np
from scipy.spatial.distance import cdist

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mnist import load
from grey_ones import RANGES, render
from experiment import fit, SEED, OUT


def main():
    report = json.loads((OUT/'results.json').read_text())
    mx, my = load('train')
    real = mx[my == 1].reshape(-1, 784).astype(float)/255
    rng = np.random.default_rng(SEED)
    params = RANGES[:, 0]+rng.random((len(real)+1135, 5))*np.diff(RANGES, axis=1).ravel()
    generated = render(params[:len(real)], dtype=np.float64).reshape(-1, 784)
    checks, controls = {}, {}
    for name, source, filename in [('Generated cloud', generated, 'generated_arrays.npz'),
                                    ('MNIST', real, 'mnist_arrays.npz')]:
        a = np.load(OUT/filename)
        test = a['test_images']
        assignment = a['atlas_assignment']
        anchor = a['anchor_images']
        bases = np.array([a[f'anchor_{i}_basis30'] for i in range(len(anchor))])
        assert np.max(abs(bases@bases.swapaxes(1, 2)-np.eye(30))) < 1e-10
        assert np.array_equal(anchor, source[a['anchor_train_positions']])
        # Independently reconstruct every saved test image, using direct summation.
        reconstructed = np.array([anchor[ai]+np.sum(
            ((test[t]-anchor[ai])@bases[ai, :5].T)[:, None]*bases[ai, :5], axis=0)
            for t, ai in enumerate(assignment)])
        absolute = np.linalg.norm(test-reconstructed, axis=1)
        relative = absolute/np.linalg.norm(test-anchor[assignment], axis=1)
        discrepancy = max(float(abs(absolute-a['atlas_absolute_errors_128'][3]).max()),
                          float(abs(relative-a['atlas_errors_128'][3]).max()))
        assert discrepancy < 1e-11
        actual_assignment = cdist(test, anchor).argmin(1)
        assert np.array_equal(assignment, actual_assignment)
        # Refit a few bases from original pixels and saved training indices.
        refit_overlap = []
        for i in [0, 7, 19, 43, 63]:
            ni = a['training_neighbor_indices'][i]
            assert a['anchor_train_positions'][i] not in ni
            d = source[ni]-anchor[i]
            _, _, v = np.linalg.svd(d, full_matrices=False)
            overlap = np.linalg.svd(v[:5]@bases[i, :5].T, compute_uv=False).min()
            refit_overlap.append(float(overlap))
            assert overlap > 1-1e-9
        checks[name] = dict(test_images_reconstructed=len(test), maximum_metric_discrepancy=discrepancy,
                            minimum_refit_subspace_cosine=min(refit_overlap),
                            saved_nearest_anchor_assignments_verified=True,
                            orthonormal_bases_verified=True)
        output = {}
        for variant in ['root-anchored SVD', 'angular-weighted SVD', 'mean-centered local PCA']:
            errors = {5: [], 20: []}
            for ai in range(len(anchor)):
                ni = a['training_neighbor_indices'][ai]
                d = source[ni]-anchor[ai]
                offset = d.mean(0) if variant == 'mean-centered local PCA' else np.zeros(784)
                matrix = d-offset
                if variant == 'angular-weighted SVD':
                    matrix = d/np.maximum(np.linalg.norm(d, axis=1), 1e-15)[:, None]
                _, b = fit(matrix)
                use = np.flatnonzero(assignment == ai)
                delta = test[use]-anchor[ai]-offset
                for n in errors:
                    residual = delta-(delta@b[:n].T)@b[:n]
                    errors[n].extend(np.linalg.norm(residual, axis=1).tolist())
            output[variant] = {n: dict(mean_pixel_l2=float(np.mean(e)),
                                       pixel_rmse=float(np.sqrt(np.mean(np.square(e))/784))) for n,e in errors.items()}
        controls[name] = output
    (OUT/'verification.json').write_text(json.dumps(checks, indent=2)+'\n')
    (OUT/'fitting_controls.json').write_text(json.dumps(controls, indent=2)+'\n')
    print(json.dumps(dict(verification=checks, controls=controls), indent=2))


if __name__ == '__main__':
    main()
