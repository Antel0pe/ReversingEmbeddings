"""One small longer-budget follow-up to the image-only bottleneck sweep.

Run: MPLCONFIGDIR=/tmp/reversing-embeddings-mpl OPENBLAS_NUM_THREADS=1 \
     python quick_generated_one_latents_followup.py
"""

from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

import quick_generated_one_latents as sweep


WIDTHS = (5, 8, 9, 10, 11, 12, 16)
EPOCHS = 240


def plot(test, predictions, pca5, metrics, baseline, pca_thresholds):
    fig = plt.figure(figsize=(13, 9.5), facecolor='white')
    fig.text(.055, .965, 'How close to all the pixel variation in a short run?',
             fontsize=20, weight='bold', va='top', color='#1f2937')
    fig.text(.055, .925,
             'Full five-knob generated 1 family · 512 unseen images · methods see pixels only',
             fontsize=11, color='#374151')
    grid = fig.add_gridspec(2, 2, left=.075, right=.965, top=.865, bottom=.17,
                           height_ratios=[1.5, 1], hspace=.35, wspace=.2)
    ax = fig.add_subplot(grid[0, :])
    base = baseline['metrics']
    all_widths = list(sweep.WIDTHS)
    ax.plot(all_widths, [base['PCA'][str(w)]['reconstructed_variance_percent'] for w in all_widths],
            'o-', lw=2.1, color='#276c9c', label='PCA')
    ax.plot(all_widths, [base['Small autoencoder'][str(w)]['reconstructed_variance_percent'] for w in all_widths],
            'o-', lw=2.1, color='#b37555', label='Autoencoder, 80 epochs')
    ax.plot(WIDTHS, [metrics[str(w)]['reconstructed_variance_percent'] for w in WIDTHS],
            's-', lw=2.5, color='#a63224', label='Same autoencoder, 240 epochs')
    for line in (95, 99):
        ax.axhline(line, color='#757575', ls='--', lw=1)
        ax.text(16.17, line, f'{line}%', va='center', fontsize=9)
    ax.axvline(5, color='#aaa', alpha=.5)
    ax.set_xlim(2.7, 17); ax.set_ylim(65, 101)
    ax.set_xticks([3, 4, 5, 6, 8, 9, 10, 11, 12, 16])
    ax.set_xlabel('Coordinates per image')
    ax.set_ylabel('Unseen pixel variance reconstructed (%)')
    ax.grid(axis='y', alpha=.13)
    ax.legend(loc='lower right', frameon=False)
    index = 7
    images = [('Original', test[index]),
              ('5-variable AE, 240 epochs', predictions['5'][index]),
              ('12-variable AE, 240 epochs', predictions['12'][index]),
              ('5-variable PCA', pca5[index])]
    bottom = grid[1, :].subgridspec(1, 4, wspace=.2)
    for i, (name, image) in enumerate(images):
        panel = fig.add_subplot(bottom[0, i])
        panel.imshow(image.reshape(28, 28), cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
        panel.set_title(name, fontsize=10)
        panel.set_xticks([]); panel.set_yticks([])
        for spine in panel.spines.values():
            spine.set_color('#c9d0d7')
    fig.text(.055, .113, 'White = no ink; black = full ink. At 240 epochs, 11 was the first tested width to reach 99%.',
             fontsize=10, color='#374151')
    fig.text(.055, .078,
             f"PCA reaches 95% with {pca_thresholds['95']} components and 99% with {pca_thresholds['99']} "
             'on these same unseen images.', fontsize=10, color='#374151')
    fig.text(.055, .043,
             'Five known generator knobs reconstruct exactly. Learned images still have edge and gray-value errors.',
             fontsize=10, weight='bold', color='#1f2937')
    fig.savefig(sweep.OUT / 'quick_latent_followup.png', dpi=160)
    plt.close(fig)


def main():
    baseline = json.loads((sweep.OUT / 'results.json').read_text())
    _, images = sweep.data()
    train, validation, test = images[:3072], images[3072:3584], images[3584:]
    active = train.var(0) > 1e-12
    center = train[:, active].mean(0)
    scale = float(np.sqrt(np.mean((train[:, active] - center) ** 2)))
    normalized = [(part[:, active] - center) / scale for part in (train, validation, test)]
    sweep.MAX_EPOCHS = EPOCHS
    metrics = {}
    predictions = {}
    started = time.monotonic()
    with threadpool_limits(limits=1):
        for width in WIDTHS:
            model, diagnostics = sweep.train_autoencoder(normalized[0], normalized[1], width)
            prediction = np.zeros_like(test)
            prediction[:, active] = model.forward(normalized[2]) * scale + center
            metrics[str(width)] = {**sweep.score(test, prediction, train.mean(0)), **diagnostics}
            predictions[str(width)] = prediction
            print(width, round(metrics[str(width)]['reconstructed_variance_percent'], 4),
                  'best epoch', diagnostics['best_epoch'], flush=True)
        pca = PCA(n_components=int(active.sum()), svd_solver='full').fit(train)
        coordinates = pca.transform(test)
        total = np.sum((test - train.mean(0)) ** 2)
        cumulative = 100 * np.cumsum(np.sum(coordinates ** 2, axis=0)) / total
        thresholds = {str(t): int(np.searchsorted(cumulative, t) + 1)
                      for t in (95, 99, 99.9, 99.99)}
        print('PCA dimensions', thresholds, flush=True)
        # Reuse the same fitted PCA to render the comparison image.
        pca5 = pca.mean_ + coordinates[:, :5] @ pca.components_[:5]
    report = {
        'same_data_as': 'figures/quick_generated_one_latents/results.json',
        'widths': WIDTHS, 'maximum_epochs': EPOCHS, 'same_architecture_and_initialization_as_initial_sweep': True,
        'metrics': metrics, 'PCA_test_dimensions': thresholds,
        'threshold_note': 'The first tested nonlinear width reaching 99% was 11; this is specific to one initialization per width.',
        'total_seconds': round(time.monotonic() - started, 2),
    }
    (sweep.OUT / 'followup_results.json').write_text(json.dumps(report, indent=2) + '\n')
    plot(test, predictions, pca5, metrics, baseline, thresholds)
    print('Total seconds', report['total_seconds'], flush=True)


if __name__ == '__main__':
    main()
