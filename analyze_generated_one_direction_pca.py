"""PCA of seed-1 displacements to the saved radius-4.5 generated 1s.

Run: python analyze_generated_one_direction_pca.py
All source states come from the existing distance-shell experiment.
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from grey_ones import KNOBS, render

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'figures' / 'generated_one_distance_shells'
OUT = SOURCE / 'direction_pca'


def render64(states):
    """Identical 16-subrow coverage rule, retaining float64 before storage."""
    images = []
    y = np.arange(28 * 16) / 16
    ym = y + 1 / 32
    cols = np.arange(28)
    for cx, cy, h, w, a in np.atleast_2d(states):
        wy = np.clip(np.minimum(y + 1 / 16, cy + h / 2)
                     - np.maximum(y, cy - h / 2), 0, 1 / 16) * 16
        center = cx - np.tan(np.radians(a)) * (ym - cy)
        left, right = center - w / 2, center + w / 2
        overlap = np.clip(np.minimum(right[:, None], cols + 1)
                          - np.maximum(left[:, None], cols), 0, 1)
        images.append((overlap * wy[:, None]).reshape(28, 16, 28).mean(1))
    return np.asarray(images).reshape(-1, 784)


def decompose(displacements, centered):
    mean = displacements.mean(0) if centered else np.zeros(784)
    matrix = displacements - mean
    _, singular, basis = np.linalg.svd(matrix, full_matrices=False)
    variance = singular ** 2
    cumulative = np.cumsum(variance) / variance.sum()
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int(np.sum(singular > tolerance))
    scores = matrix @ basis.T
    stats = {
        'count': len(matrix), 'centered': centered,
        'components_95_percent': int(np.searchsorted(cumulative, .95) + 1),
        'numerical_rank': rank, 'singular_value_rank_tolerance': float(tolerance),
        'smallest_retained_singular_value': float(singular[rank - 1]),
        'largest_discarded_singular_value': float(singular[rank]) if rank < len(singular) else None,
        'five_components_percent': float(100 * cumulative[4]),
        'mean_direction_norm': float(np.linalg.norm(mean)),
        'cumulative_percent': (100 * cumulative).tolist(),
        'singular_values': singular.tolist(),
    }
    for k in sorted(set([5, stats['components_95_percent'], rank])):
        error = matrix - scores[:, :k] @ basis[:k]
        stats[f'reconstruction_{k}'] = {
            'rms_direction_L2_error': float(np.sqrt(np.mean(np.sum(error ** 2, axis=1)))),
            'max_direction_L2_error': float(np.linalg.norm(error, axis=1).max()),
            'max_pixel_absolute_error': float(np.abs(error).max()),
        }
    return stats, mean, basis, scores


def figure(seed, endpoint, sample_stats, all_stats):
    fig = plt.figure(figsize=(12.6, 9.4), facecolor='white')
    fig.text(.055, .96, 'How many linear coordinates describe these directions?',
             fontsize=19, weight='bold', va='top', color='#172a3a')
    fig.text(.055, .915, 'Generated 1s only · seed 1 · neighbors at pixel distance 4.5 · 28×28 ink coverage',
             fontsize=11)
    layout = fig.add_gridspec(2, 2, left=.075, right=.965, top=.85, bottom=.25,
                              height_ratios=[1, 1.5], hspace=.40, wspace=.27)
    top = layout[0, 0].subgridspec(1, 3, wspace=.25)
    images = [seed, endpoint, endpoint - seed]
    for i, (im, title) in enumerate(zip(images, ['Seed', 'One shell neighbor', 'Direction = neighbor − seed'])):
        ax = fig.add_subplot(top[0, i])
        if i == 2:
            ax.imshow(im.reshape(28, 28), cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
        else:
            ax.imshow(im.reshape(28, 28), cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(title, fontsize=9, pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#d0d5d9')
    tx = fig.add_subplot(layout[0, 1]); tx.axis('off')
    tx.text(0, .98, 'Images: white = 0 ink; black = full ink.\n'
            'One direction is a list of 784 pixel changes.\n'
            'Red adds ink; blue removes ink; white is zero.\n'
            'Change colors span −1 to +1 coverage.', fontsize=11, va='top', linespacing=1.55)
    tx.text(0, .28, 'Ordinary PCA subtracts the mean direction, then\n'
            'finds fixed pixel patterns that explain the variation.\n'
            'The plots below count those shared patterns.', fontsize=11, va='top', linespacing=1.55)
    for col, stats, title, color in [(0, sample_stats, 'The 20 displayed neighbors', '#2478a8'),
                                     (1, all_stats, 'All 619 sampled neighbors', '#a95030')]:
        ax = fig.add_subplot(layout[1, col])
        rank = stats['numerical_rank']; k95 = stats['components_95_percent']
        x = np.arange(1, rank + 1); y = np.array(stats['cumulative_percent'][:rank])
        ax.plot(x, y, color=color, lw=2.3)
        ax.axhline(95, color='#707070', linestyle='--', lw=1)
        ax.axvline(k95, color=color, alpha=.3)
        ax.scatter([5, k95], [y[4], y[k95 - 1]], color=color, s=35, zorder=3)
        ax.annotate(f'5 PCs: {y[4]:.1f}%', (5, y[4]), xytext=(5, y[4] - 14), fontsize=10)
        ax.text(.97, .04, f'95%: {k95} PCs\nAll: {rank} PCs*', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=12, weight='bold', color=color,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
        ax.set_title(title, fontsize=13, weight='bold', pad=12)
        ax.set_xlabel('Number of PCA components kept', fontsize=10)
        ax.set_ylabel('Variance retained (%)', fontsize=10)
        ax.set_ylim(0, 103); ax.grid(axis='y', alpha=.15)
        if rank > 20:
            ax.set_xscale('log'); ax.set_xticks([1, 5, 8, 20, 50, rank])
            ax.set_xticklabels([1, 5, 8, 20, 50, rank])
            ax.set_xlabel('Number of PCA components kept (log scale)', fontsize=10)
        else:
            ax.set_xticks([1, 5, 8, 12, 16, rank])
        ax.set_xlim(1, rank)
    fig.text(.055, .13, '*All = numerical rank of the same coverage calculation in float64. '
             'Stored float32 pixels add tiny rounding axes\n'
             'to the 619-point set (160 instead of 133). These ranks describe the sampled vectors, not the full continuous shell.',
             fontsize=10, linespacing=1.5)
    fig.text(.055, .04, 'Five knob changes still encode each direction exactly through the nonlinear renderer:\n'
             'direction = render(seed knobs + changes) − render(seed knobs).',
             fontsize=10.5, weight='bold', linespacing=1.5)
    fig.savefig(OUT / 'seed_01_direction_pca.png', dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    parent = json.loads((SOURCE / 'results.json').read_text())
    seed_knobs = np.array(parent['seeds'][0]['seed_knobs'])
    rows = [r for r in csv.DictReader((SOURCE / 'sampled_radius_4p5_shells.csv').open()) if r['seed'] == '1']
    states = np.array([[float(r[k]) for k in KNOBS] for r in rows])
    shown = np.array([r['displayed'] == '1' for r in rows])
    source32 = render(states).reshape(-1, 784)
    images = render64(states); seed = render64(seed_knobs)[0]
    # The higher-precision version must agree bit for bit after float32 storage.
    assert np.array_equal(images.astype(np.float32), source32)
    assert np.array_equal(seed.astype(np.float32), render(seed_knobs).reshape(784))
    displacement = images - seed
    displacement32 = source32.astype(float) - render(seed_knobs).reshape(784).astype(float)
    results = {'seed_knobs': seed_knobs.tolist(), 'knob_order': KNOBS,
               'float64_renderer_matches_float32_after_cast': True,
               'source': 'figures/generated_one_distance_shells/sampled_radius_4p5_shells.csv'}
    saved = {}
    with threadpool_limits(limits=1):
        for label, keep in [('displayed20', shown), ('sampled619', np.ones(len(rows), bool))]:
            results[label] = {}
            for centered, method in [(True, 'pca'), (False, 'seed_anchored_svd')]:
                stats, mean, basis, scores = decompose(displacement[keep], centered)
                raw_stats, _, _, _ = decompose(displacement32[keep], centered)
                stats['stored_float32_numerical_rank'] = raw_stats['numerical_rank']
                results[label][method] = stats
                if centered:
                    saved[label + '_mean'] = mean
                    saved[label + '_basis'] = basis[:stats['numerical_rank']]
                    if label == 'sampled619':
                        all_scores = scores[:, :8]
                print(label, method, '95%', stats['components_95_percent'], 'rank', stats['numerical_rank'],
                      '5 PCs %', round(stats['five_components_percent'], 4), flush=True)
    np.savez_compressed(OUT / 'pca_bases.npz', seed_knobs=seed_knobs, **saved)
    with (OUT / 'seed_01_coordinates.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['ray', 'displayed', *['delta_' + k for k in KNOBS],
                         *['PC' + str(k) for k in range(1, 9)]])
        for row, changes, scores in zip(rows, states - seed_knobs, all_scores):
            writer.writerow([row['ray'], row['displayed'], *changes, *scores])
    # Exact five-variable decode uses the existing renderer, not a fitted linear model.
    decoded = render(seed_knobs + (states - seed_knobs)).reshape(-1, 784)
    results['five_knob_decode_max_pixel_error'] = float(np.abs(decoded - source32).max())
    assert results['five_knob_decode_max_pixel_error'] == 0
    (OUT / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    figure(seed, images[np.flatnonzero(shown)[0]], results['displayed20']['pca'], results['sampled619']['pca'])


if __name__ == '__main__':
    main()
