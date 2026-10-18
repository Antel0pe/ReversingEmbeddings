"""Observation acquisition only. Physical settings are exported separately."""
from pathlib import Path
import sys
import json
import itertools
import numpy as np
from scipy.stats import qmc, norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from grey_ones import render, RANGES

OUT = ROOT / 'figures/svd_blind_intermediates'
PRIVATE = Path('/tmp/svd_blind_intermediates_labels.npz')
SPAN = np.ptp(RANGES, axis=1)
SEED = 2026101801


def pixels(p):
    return render(p, dtype=np.float64).reshape(-1, 784)


def fit_frames(p, seed):
    rng = np.random.default_rng(seed)
    rays = rng.normal(size=(192, 5)) * SPAN
    rays /= np.linalg.norm(rays / SPAN, axis=1)[:, None]
    x = pixels(p)[0]
    probe = pixels(p + rays * 1e-5) - x
    t = 1e-4 / np.maximum(np.linalg.norm(probe, axis=1) / 1e-5, 1e-12)
    for _ in range(4):
        dx = pixels(p + t[:, None] * rays) - x
        distance = np.linalg.norm(dx, axis=1)
        t *= 1e-4 / np.maximum(distance, 1e-15)
    dx = pixels(p + t[:, None] * rays) - x
    result = []
    for centered in [False, True]:
        a = dx - dx.mean(0) if centered else dx
        active = np.any(a != 0, axis=0)
        _, s, vt = np.linalg.svd(a[:, active], full_matrices=False)
        b = np.zeros((5, 784)); b[:, active] = vt[:5]
        result.append(b)
    return result, float(np.max(abs(np.linalg.norm(dx, axis=1) / 1e-4 - 1)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = np.load(ROOT / 'figures/generated_one_svd_family_branches/states_and_paths.npz')
    public = dict(initial_x=old['images'], initial_b=old['bases'])
    labels = {}
    base = old['settings']
    rng = np.random.default_rng(SEED)
    errors = []
    for scope in ['local', 'broad']:
        def draw(n):
            if scope == 'local':
                return base[rng.integers(len(base), size=n)] + rng.normal(size=(n, 5)) * SPAN * .012
            return RANGES[:, 0] + rng.uniform(size=(n, 5)) * SPAN
        for split, n in [('train', 8192), ('val', 512), ('test', 512)]:
            p = draw(n)
            public[f'{scope}_{split}_x'] = pixels(p)
            labels[f'{scope}_{split}_p'] = p
        path = []
        for _ in range(12):
            if scope == 'local':
                a = draw(1)[0]; b = a + rng.normal(size=5) * SPAN * .04
            else:
                a, b = draw(2)
            path.append(a[None, :] + np.linspace(0, 1, 17)[:, None] * (b - a))
        path = np.array(path)
        public[f'{scope}_path_x'] = pixels(path.reshape(-1, 5))
        labels[f'{scope}_path_p'] = path.reshape(-1, 5)
        if scope == 'broad':
            u = np.array(list(itertools.product([0., 1.], repeat=5)))
            faces = rng.uniform(size=(100, 5))
            for i in range(100): faces[i, i % 5] = float((i // 5) % 2)
            edge = RANGES[:, 0] + np.concatenate([u, faces]) * SPAN
        else:
            edge = draw(132)
        public[f'{scope}_edge_x'] = pixels(edge)
        labels[f'{scope}_edge_p'] = edge
        for split, n in [('train', 256), ('val', 96), ('test', 96)]:
            p = draw(n)
            public[f'{scope}_{split}_frame_x'] = pixels(p)
            labels[f'{scope}_{split}_frame_p'] = p
            svd = []; pca = []
            for i, point in enumerate(p):
                fits, err = fit_frames(point, SEED + 10000 * (scope == 'broad') + 1000 * ['train','val','test'].index(split) + i)
                svd.append(fits[0]); pca.append(fits[1]); errors.append(err)
            public[f'{scope}_{split}_svd'] = np.array(svd)
            public[f'{scope}_{split}_pca'] = np.array(pca)
            print(scope, split, 'frames acquired', n, flush=True)
    np.savez_compressed(OUT / 'observations.npz', **public)
    np.savez_compressed(PRIVATE, **labels)
    (OUT / 'acquisition.json').write_text(json.dumps(dict(
        seed=SEED, frame_neighbors=192, image_shell_radius=.0001,
        maximum_relative_shell_error=max(errors),
        initial_families='Existing distance/angle assignments; copied images and bases only',
        simulation_prior='Sampling rays use the simulator control scaling; no analytic derivatives used',
        local_scope='Independent small joint perturbations around the 66 existing branch states',
        broad_scope='Independent full-range samples, plus corners/faces and joint paths',
        labels='Kept outside the approved discovery input archive'), indent=2))


if __name__ == '__main__': main()
