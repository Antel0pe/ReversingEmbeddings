"""Small image-only nonlinear bottleneck sweep on the generated 1 family.

Run: OPENBLAS_NUM_THREADS=1 python quick_generated_one_latents.py
This is a fixed-budget experiment, not an architecture search.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from grey_ones import RANGES, render


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures' / 'quick_generated_one_latents'
SEED = 20261001
WIDTHS = (3, 4, 5, 6, 8, 12, 16)
MAX_EPOCHS = 80
BATCH = 256
PATIENCE = 14
RATE = .002


def data():
    unit = qmc.Sobol(5, scramble=True, seed=SEED).random_base2(12)
    settings = RANGES[:, 0] + unit * (RANGES[:, 1] - RANGES[:, 0])
    images = render(settings).reshape(-1, 784).astype(np.float64)
    return settings, images


def score(reference, reconstruction, mean):
    diff = reference - reconstruction
    each = np.linalg.norm(diff, axis=1)
    denominator = np.sum((reference - mean) ** 2)
    return {
        'reconstructed_variance_percent': float(100 * (1 - np.sum(diff ** 2) / denominator)),
        'median_image_L2_error': float(np.median(each)),
        'p95_image_L2_error': float(np.percentile(each, 95)),
        'maximum_image_L2_error': float(each.max()),
        'p95_max_pixel_error': float(np.percentile(np.max(np.abs(diff), axis=1), 95)),
    }


class SmallAutoencoder:
    def __init__(self, features, bottleneck, seed):
        rng = np.random.default_rng(seed)
        sizes = (features, 128, bottleneck, 128, features)
        self.weights = [rng.standard_normal((a, b)) * np.sqrt(2 / a if i in (0, 2) else 1 / a)
                        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]))]
        self.biases = [np.zeros(b) for b in sizes[1:]]
        self.params = [v for pair in zip(self.weights, self.biases) for v in pair]
        self.first = [np.zeros_like(p) for p in self.params]
        self.second = [np.zeros_like(p) for p in self.params]
        self.steps = 0

    def forward(self, x, keep_cache=False):
        activations = [x]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i in (0, 2):
                x = np.maximum(x, 0)
            activations.append(x)
        return (x, activations) if keep_cache else x

    def train_batch(self, x):
        output, a = self.forward(x, True)
        grad = 2 * (output - x) / len(x)
        gradients = [None] * len(self.params)
        for i in range(3, -1, -1):
            gradients[2 * i] = a[i].T @ grad
            gradients[2 * i + 1] = grad.sum(0)
            if i:
                grad = grad @ self.weights[i].T
                if i - 1 in (0, 2):
                    grad *= a[i] > 0
        self.steps += 1
        for j, (p, g) in enumerate(zip(self.params, gradients)):
            self.first[j] = .9 * self.first[j] + .1 * g
            self.second[j] = .999 * self.second[j] + .001 * g * g
            corrected_first = self.first[j] / (1 - .9 ** self.steps)
            corrected_second = self.second[j] / (1 - .999 ** self.steps)
            p -= RATE * corrected_first / (np.sqrt(corrected_second) + 1e-8)

    def snapshot(self):
        return [p.copy() for p in self.params]

    def restore(self, params):
        for target, source in zip(self.params, params):
            target[:] = source


def train_autoencoder(train, validation, width):
    model = SmallAutoencoder(train.shape[1], width, SEED + width)
    rng = np.random.default_rng(SEED + 100 + width)
    best_loss = float('inf')
    best_epoch = 0
    best = None
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        permutation = rng.permutation(len(train))
        for ids in np.array_split(permutation, int(np.ceil(len(train) / BATCH))):
            model.train_batch(train[ids])
        estimate = model.forward(validation)
        loss = float(np.mean(np.sum((validation - estimate) ** 2, axis=1)))
        history.append(loss)
        if loss < best_loss - 1e-5:
            best_loss, best_epoch, best = loss, epoch, model.snapshot()
        if epoch - best_epoch >= PATIENCE:
            break
    model.restore(best)
    return model, {'best_epoch': best_epoch, 'epochs_run': epoch,
                   'validation_loss_normalized': best_loss, 'validation_history': history}


def make_figure(images, reconstructions, metrics):
    widths = list(WIDTHS)
    fig = plt.figure(figsize=(13, 10.5), facecolor='white')
    fig.text(.055, .965, 'Can image-only learning compress generated 1s to five numbers?',
             fontsize=19, weight='bold', va='top', color='#1f2937')
    fig.text(.055, .928, 'Five-knob 28×28 coverage renderer · 3,072 training, 512 validation, 512 unseen test images',
             fontsize=10.5, color='#374151')
    grid = fig.add_gridspec(2, 2, left=.07, right=.97, top=.875, bottom=.165,
                           height_ratios=[1.5, 1], hspace=.33, wspace=.22)
    ax = fig.add_subplot(grid[0, :])
    for method, color in [('PCA', '#2d70a2'), ('Small autoencoder', '#b05035')]:
        values = [metrics[method][str(k)]['reconstructed_variance_percent'] for k in widths]
        ax.plot(widths, values, 'o-', color=color, label=method, lw=2.5, markersize=6)
    for y in (95, 99):
        ax.axhline(y, color='#8a8a8a', linestyle='--', linewidth=1)
        ax.text(16.2, y, f'{y}%', va='center', fontsize=9, color='#555')
    ax.axvline(5, color='#888', alpha=.4)
    ax.set_xlim(2.7, 17); ax.set_ylim(65, 101)
    ax.set_xticks(widths)
    ax.set_xlabel('Coordinates per image')
    ax.set_ylabel('Unseen pixel variance reconstructed (%)')
    ax.legend(loc='lower right', frameon=False)
    ax.grid(axis='y', alpha=.15)
    ax.set_title('Fixed small training budget; same test images for every width', fontsize=12)
    # Use one representative test image, the same index for every method.
    index = 7
    targets = [('Original', images[index], None),
               ('5-variable AE', reconstructions['Small autoencoder']['5'][index], None),
               ('8-variable AE', reconstructions['Small autoencoder']['8'][index], None),
               ('5-variable PCA', reconstructions['PCA']['5'][index], None)]
    bottom = grid[1, :].subgridspec(1, 4, wspace=.23)
    for j, (title, image, _) in enumerate(targets):
        panel = fig.add_subplot(bottom[0, j])
        panel.imshow(image.reshape(28, 28), cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
        panel.set_xticks([]); panel.set_yticks([])
        panel.set_title(title, fontsize=10.5)
        for spine in panel.spines.values():
            spine.set_color('#bec8d1')
    fig.text(.055, .115, 'White = no ink; black = full ink. Below the curve is residual pixel information, '
             'not a count of missing generator knobs.', fontsize=10, color='#374151')
    fig.text(.055, .073, 'Autoencoder: 266 varying pixels → 128 → bottleneck → 128 → 266; '
             'up to 80 epochs, one initialization per width.', fontsize=10, color='#374151')
    fig.text(.055, .041, 'Known five knobs + renderer reconstruct all test images exactly; '
             'the learned methods only saw pixel arrays.', fontsize=10, weight='bold', color='#1f2937')
    fig.savefig(OUT / 'quick_latent_sweep.png', dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    settings, images = data()
    train = images[:3072]
    validation = images[3072:3584]
    test = images[3584:]
    mean = train.mean(0)
    active = train.var(0) > 1e-12
    center = train[:, active].mean(0)
    scale = float(np.sqrt(np.mean((train[:, active] - center) ** 2)))
    normalized = [(part[:, active] - center) / scale for part in (train, validation, test)]
    metrics = {'PCA': {}, 'Small autoencoder': {}}
    reconstructions = {'PCA': {}, 'Small autoencoder': {}}
    started = time.monotonic()
    with threadpool_limits(limits=1):
        pca = PCA(n_components=max(WIDTHS), svd_solver='full').fit(train)
        transformed = pca.transform(test)
        for width in WIDTHS:
            reconstruction = pca.mean_ + transformed[:, :width] @ pca.components_[:width]
            metrics['PCA'][str(width)] = score(test, reconstruction, mean)
            reconstructions['PCA'][str(width)] = reconstruction
        for width in WIDTHS:
            before = time.monotonic()
            model, diagnostics = train_autoencoder(normalized[0], normalized[1], width)
            predicted = model.forward(normalized[2]) * scale + center
            reconstruction = np.zeros_like(test)
            reconstruction[:, active] = predicted
            metrics['Small autoencoder'][str(width)] = {
                **score(test, reconstruction, mean), **diagnostics,
                'seconds': round(time.monotonic() - before, 2),
            }
            reconstructions['Small autoencoder'][str(width)] = reconstruction
            result = metrics['Small autoencoder'][str(width)]
            print(width, f"AE {result['reconstructed_variance_percent']:.3f}%",
                  f"PCA {metrics['PCA'][str(width)]['reconstructed_variance_percent']:.3f}%",
                  f"epoch {result['best_epoch']}", flush=True)
    # Knob values are used only here as a diagnostic oracle, never by PCA or AE.
    oracle = render(settings[3584:]).reshape(-1, 784).astype(np.float64)
    oracle_score = score(test, oracle, mean)
    result = {
        'dataset': 'Full controlled five-knob generated 1 family',
        'sample': '4096 scrambled Sobol knob points; 3072 train, 512 validation, 512 test',
        'seed': SEED, 'widths': WIDTHS, 'maximum_epochs': MAX_EPOCHS,
        'patience': PATIENCE, 'active_pixel_count': int(active.sum()),
        'knobs_visible_to_learned_methods': False,
        'score_definition': '100*(1 - sum_test ||image-reconstruction||^2 / sum_test ||image-train_mean||^2)',
        'metrics': metrics, 'known_knobs_and_renderer_oracle': oracle_score,
        'total_seconds': round(time.monotonic() - started, 2),
    }
    (OUT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    make_figure(test, reconstructions, metrics)
    print('Total seconds', result['total_seconds'], flush=True)


if __name__ == '__main__':
    main()
