"""Small image-only autoencoder diagnostic against known generated-1 knobs.

Run: OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/generated-one-ae-mpl \
     python compare_generated_one_autoencoder_knobs.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import PolynomialFeatures
from threadpoolctl import threadpool_limits

from grey_ones import KNOBS, RANGES, render
from quick_generated_one_latents import BATCH, SEED, SmallAutoencoder, data, score


OUT = Path(__file__).resolve().parent / 'figures' / 'generated_one_autoencoder_knobs'
CHECKPOINTS = (80, 160, 240)


def encode(model, normalized_images):
    return np.maximum(normalized_images @ model.weights[0] + model.biases[0], 0) @ model.weights[1] + model.biases[1]


def decode(model, images, active, center, scale):
    normalized = (images[:, active] - center) / scale
    answer = np.zeros_like(images)
    answer[:, active] = model.forward(normalized) * scale + center
    return answer


def probe(z_train, z_test, knobs_train, knobs_test, degree):
    zmean, zstd = z_train.mean(0), z_train.std(0)
    zstd[zstd == 0] = 1
    xtrain, xtest = (z_train - zmean) / zstd, (z_test - zmean) / zstd
    if degree == 2:
        transform = PolynomialFeatures(2, include_bias=False)
        xtrain = transform.fit_transform(xtrain)
        xtest = transform.transform(xtest)
    model = Ridge(alpha=1e-3).fit(xtrain, knobs_train)
    prediction = model.predict(xtest)
    return (1 - np.sum((prediction - knobs_test) ** 2, axis=0)
            / np.sum((knobs_test - knobs_train.mean(0)) ** 2, axis=0)).tolist()


def flexible_probe(z_train, z_test, knobs_train, knobs_test):
    """Nonparametric held-out readout check for information beyond a quadratic map."""
    model = ExtraTreesRegressor(n_estimators=250, min_samples_leaf=3,
                                max_features=1.0, random_state=SEED, n_jobs=1)
    model.fit(z_train, knobs_train)
    prediction = model.predict(z_test)
    return (1 - np.sum((prediction - knobs_test) ** 2, axis=0)
            / np.sum((knobs_test - knobs_train.mean(0)) ** 2, axis=0)).tolist()


def diagnostics(model, train, test, settings, images, active, center, scale):
    z_train = encode(model, train)
    z_test = encode(model, test)
    q_train = (settings[:3072] - RANGES[:, 0]) / np.ptp(RANGES, axis=1)
    q_test = (settings[3584:] - RANGES[:, 0]) / np.ptp(RANGES, axis=1)
    corr = np.corrcoef(np.column_stack([z_test, q_test]).T)[:5, 5:]
    reconstruction = decode(model, images[3584:], active, center, scale)
    metrics = score(images[3584:], reconstruction, images[:3072].mean(0))

    # Symmetric 8%-of-range knob moves, sampled away from parameter boundaries.
    selected = np.where(np.all((q_test > .1) & (q_test < .9), axis=1))[0][:96]
    start = settings[3584:][selected]
    true_moves = []
    predicted_moves = []
    latent_moves = []
    zstd = z_train.std(0)
    for j in range(5):
        step = np.zeros(5)
        step[j] = .04 * np.ptp(RANGES, axis=1)[j]
        minus = render(start - step).reshape(-1, 784).astype(np.float64)
        plus = render(start + step).reshape(-1, 784).astype(np.float64)
        true_moves.append(plus - minus)
        predicted_moves.append(decode(model, plus, active, center, scale)
                               - decode(model, minus, active, center, scale))
        latent_moves.append((encode(model, (plus[:, active] - center) / scale)
                             - encode(model, (minus[:, active] - center) / scale)) / zstd)
    true_moves = np.asarray(true_moves)
    predicted_moves = np.asarray(predicted_moves)
    latent_moves = np.asarray(latent_moves)
    numerator = np.sum(true_moves * predicted_moves, axis=2)
    denominator = np.linalg.norm(true_moves, axis=2) * np.linalg.norm(predicted_moves, axis=2)
    cosines = numerator / np.maximum(denominator, 1e-12)
    norm_ratio = np.linalg.norm(predicted_moves, axis=2) / np.linalg.norm(true_moves, axis=2)
    direction_l2 = np.linalg.norm(true_moves - predicted_moves, axis=2) / np.linalg.norm(true_moves, axis=2)
    metrics.update({
        'raw_latent_knob_pearson': corr.tolist(),
        'best_absolute_raw_axis_correlation_per_knob': np.max(np.abs(corr), axis=0).tolist(),
        'linear_probe_knob_R2': probe(z_train, z_test, q_train, q_test, 1),
        'quadratic_probe_knob_R2': probe(z_train, z_test, q_train, q_test, 2),
        'tree_probe_knob_R2': flexible_probe(z_train, z_test, q_train, q_test),
        'median_pixel_direction_cosine': np.median(cosines, axis=1).tolist(),
        'median_pixel_direction_norm_ratio': np.median(norm_ratio, axis=1).tolist(),
        'median_pixel_direction_relative_L2_error': np.median(direction_l2, axis=1).tolist(),
        'median_absolute_standardized_latent_response': np.median(np.abs(latent_moves), axis=1).T.tolist(),
        'direction_sample_count': int(len(selected)),
    })
    return metrics, reconstruction, selected


def make_figure(test_images, reconstructions, results):
    fig = plt.figure(figsize=(14, 11), facecolor='white')
    fig.text(.055, .965, 'What did five learned numbers encode?', fontsize=21,
             weight='bold', va='top')
    fig.text(.055, .929, 'Generated 1s only · same image-only autoencoder at 80, 160, 240 epochs · 512 unseen images',
             fontsize=11, color='#3c4856')
    gs = fig.add_gridspec(3, 3, left=.075, right=.96, top=.875, bottom=.225,
                          width_ratios=[1, 1.4, 1.5], hspace=.39, wspace=.3)
    image_index = 7
    for row, epoch in enumerate(CHECKPOINTS):
        result = results[str(epoch)]
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(reconstructions[epoch][image_index].reshape(28, 28), cmap='gray_r', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{epoch} epochs: {result["reconstructed_variance_percent"]:.2f}%\nheld-out pixel variance', fontsize=11)
        cor_ax = fig.add_subplot(gs[row, 1])
        corr = np.asarray(result['raw_latent_knob_pearson'])
        im = cor_ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        cor_ax.set_xticks(range(5), KNOBS, rotation=35, ha='right', fontsize=9)
        cor_ax.set_yticks(range(5), [f'z{i+1}' for i in range(5)], fontsize=9)
        cor_ax.set_title('Raw latent ↔ knob correlation', fontsize=11)
        for j in range(5):
            for k in range(5):
                cor_ax.text(k, j, f'{corr[j,k]:+.2f}', ha='center', va='center', fontsize=8,
                            color='white' if abs(corr[j,k]) > .65 else '#17202a')
        dir_ax = fig.add_subplot(gs[row, 2])
        xpos = np.arange(5)
        dirs = result['median_pixel_direction_cosine']
        dir_ax.bar(xpos, dirs, color='#39738e')
        dir_ax.set_ylim(0, 1.03)
        dir_ax.set_xticks(xpos, KNOBS, rotation=35, ha='right', fontsize=9)
        dir_ax.set_ylabel('Cosine: actual vs AE pixel change', fontsize=9)
        dir_ax.set_title('Does each knob move pixels correctly?', fontsize=11)
        dir_ax.grid(axis='y', alpha=.15)
    fig.text(.075, .17, 'Left: one fixed test image reconstructed at each checkpoint. Pixel value = ink coverage (white 0, black 1).', fontsize=10)
    fig.text(.075, .135, 'Middle: cell = correlation between a learned coordinate and a known knob. The coordinates are mixed, not named knobs.', fontsize=10)
    fig.text(.075, .10, 'Right: median alignment over 96 interior settings; one knob moves ±4% of its range. 1 = same signed pixel pattern.', fontsize=10)
    fig.text(.075, .06, 'Correlation only checks raw axes; a nonlinear decoder can still represent a knob without a strong one-axis correlation.', fontsize=10, color='#3c4856')
    fig.savefig(OUT / 'latent_meaning_summary.png', dpi=160)
    plt.close(fig)


def make_direction_figure(model, settings, test_images, active, center, scale):
    # Use a repeatable interior setting from the unseen test set as the baseline.
    unit = (settings[3584:] - RANGES[:, 0]) / np.ptp(RANGES, axis=1)
    eligible = np.where(np.all((unit > .1) & (unit < .9), axis=1))[0]
    idx = int(eligible[0])
    base = settings[3584 + idx]
    image = test_images[idx]
    actual, predicted, limits = [], [], []
    for j in range(5):
        step = np.zeros(5)
        step[j] = .04 * np.ptp(RANGES, axis=1)[j]
        minus = render(base - step).reshape(-1, 784).astype(np.float64)
        plus = render(base + step).reshape(-1, 784).astype(np.float64)
        actual.append((plus - minus)[0].reshape(28, 28))
        pred = (decode(model, plus, active, center, scale)
                - decode(model, minus, active, center, scale))[0].reshape(28, 28)
        predicted.append(pred)
        limits.append(float(max(np.abs(actual[-1]).max(), np.abs(pred).max())))
    fig = plt.figure(figsize=(15, 5.8), facecolor='white')
    fig.text(.04, .995, 'One small move in each known knob', fontsize=20, weight='bold', va='top')
    fig.text(.04, .91, 'Same generated 1 at the left; compare its true pixel change with the change made by the 240-epoch autoencoder.',
             fontsize=11, color='#3c4856')
    fig.text(.04, .85, 'Starting knobs: ' + ', '.join(f'{name}={value:.3f}' for name, value in zip(KNOBS, base)),
             fontsize=9.5, color='#3c4856')
    grid = fig.add_gridspec(2, 6, left=.04, right=.98, top=.79, bottom=.18,
                            width_ratios=[.9, 1, 1, 1, 1, 1], hspace=.38, wspace=.16)
    source = fig.add_subplot(grid[:, 0])
    source.imshow(image.reshape(28, 28), cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
    source.set_title('Starting 1\ncoverage', fontsize=11)
    source.set_xticks([]); source.set_yticks([])
    for j, knob in enumerate(KNOBS):
        for row, values, label in ((0, actual, 'True renderer'), (1, predicted, 'Autoencoder')):
            ax = fig.add_subplot(grid[row, j + 1])
            ax.imshow(values[j], cmap='RdBu_r', vmin=-limits[j], vmax=limits[j], interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f'{knob}\n±4% of range', fontsize=10)
            if j == 0:
                ax.set_ylabel(label, fontsize=10)
    fig.text(.04, .115, 'Each panel shows image(knob + step) − image(knob − step); red gains ink, blue loses ink. Pixel values are coverage changes.', fontsize=10)
    fig.text(.04, .075, 'Each knob column uses its own symmetric color limit, shared by the true and autoencoder rows; the printed cosine scores quantify alignment across 96 settings.', fontsize=10)
    fig.savefig(OUT / 'one_seed_knob_pixel_changes.png', dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    settings, images = data()
    training, validation, test = images[:3072], images[3072:3584], images[3584:]
    active = training.var(0) > 1e-12
    center = training[:, active].mean(0)
    scale = float(np.sqrt(np.mean((training[:, active] - center) ** 2)))
    normalized = [(part[:, active] - center) / scale for part in (training, validation, test)]
    model = SmallAutoencoder(int(active.sum()), 5, SEED + 5)
    rng = np.random.default_rng(SEED + 105)
    results, reconstructions = {}, {}
    with threadpool_limits(limits=1):
        for epoch in range(1, max(CHECKPOINTS) + 1):
            permutation = rng.permutation(len(training))
            for ids in np.array_split(permutation, int(np.ceil(len(training) / BATCH))):
                model.train_batch(normalized[0][ids])
            if epoch not in CHECKPOINTS:
                continue
            metrics, reconstruction, selected = diagnostics(
                model, normalized[0], normalized[2], settings, images, active, center, scale)
            results[str(epoch)] = metrics
            reconstructions[epoch] = reconstruction
            np.savez_compressed(OUT / f'checkpoint_{epoch}.npz',
                                **{f'weight_{j}': w for j, w in enumerate(model.weights)},
                                **{f'bias_{j}': b for j, b in enumerate(model.biases)},
                                active_pixels=active, pixel_center=center, pixel_scale=scale)
            print(epoch, f'{metrics["reconstructed_variance_percent"]:.4f}% variance',
                  'linear knob R2', np.round(metrics['linear_probe_knob_R2'], 3),
                  'quadratic knob R2', np.round(metrics['quadratic_probe_knob_R2'], 3),
                  'pixel direction cos', np.round(metrics['median_pixel_direction_cosine'], 3), flush=True)
    report = {
        'dataset': 'Controlled generated 1 family, not MNIST',
        'same_sample_and_model_initialization_as': 'quick_generated_one_latents.py',
        'training': '3072 images; 512 validation, 512 unseen test; no knob labels used by autoencoder',
        'architecture': '266 active pixels -> 128 ReLU -> 5 linear -> 128 ReLU -> 266 linear',
        'knob_order': KNOBS,
        'checkpoints': list(CHECKPOINTS),
        'probe_note': 'Linear and quadratic ridge readouts use training knob labels only after AE training; reported R2 is on the unseen test split.',
        'direction_note': 'Symmetric +/-4% of each full knob range at 96 interior test settings. Cosine 1 means identical signed pixel-change pattern; relative L2 error and norm ratio expose scale errors.',
        'results': results,
        'seconds': round(time.monotonic() - started, 2),
    }
    (OUT / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    make_figure(test, reconstructions, results)
    make_direction_figure(model, settings, test, active, center, scale)
    print('Finished in', report['seconds'], 'seconds', flush=True)


if __name__ == '__main__':
    main()
