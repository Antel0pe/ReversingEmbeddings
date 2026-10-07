"""Overlay actual lean changes at widely separated widths on one pixel grid.

Regenerate: .venv/bin/python make_generated_lean_pixel_overlay.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from grey_ones import render
from serve_generated_lean_pixel_overlay import auto_space, compute_overlay

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures/generated_lean_pixel_overlay'
BASE_WIDTH = 3.2
GAPS = np.arange(6., 10.01, 1.)
WIDTHS = np.unique(np.round([BASE_WIDTH + k*g for g in GAPS for k in range(3)], 8))
LEAN = np.arange(0, 10.01, .5)
FIXED = np.array([14.5, 14.5, 19.75])
EPS = 1e-7
RED = np.array([207, 49, 43]) / 255
BLUE = np.array([33, 102, 172]) / 255


def indices(gap):
    return [int(np.argmin(abs(WIDTHS - (BASE_WIDTH+k*gap)))) for k in range(3)]


def overlay(reference, changes):
    # Each original pixel receives its one actual signed change. Never split,
    # shift, blend, average, or stack values inside a pixel.
    assert np.max((np.abs(changes) > EPS).sum(0)) <= 1
    rgb = np.repeat((1-reference)[:, :, None], 3, axis=2)
    for delta in changes:
        mask = abs(delta) > EPS
        strength = abs(delta[mask])[:, None]
        target = np.where((delta[mask] > 0)[:, None], RED, BLUE)
        rgb[mask] = 1 + strength*(target-1)
    return rgb


def figure(settings):
    live = compute_overlay(settings, 2, 1)
    delta = np.array(live['deltas']).reshape(3, 28, 28)
    reference = np.array(live['reference']).reshape(28, 28)
    image = overlay(reference, delta)
    heights = [p[2] for p in settings[1:]]
    fig = plt.figure(figsize=(8, 9), facecolor='white')
    fig.text(.08, .95, 'Height changes at separated heights', fontsize=19, weight='bold')
    fig.text(.08, .89, 'The same +1 px height step, from three different starting heights.\nBlack = short reference. Red = gains ink; blue = loses ink.', fontsize=11)
    ax = fig.add_axes([.14, .18, .72, .64])
    ax.imshow(image, extent=[0, 28, 28, 0], interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for pos in range(29):
        ax.axvline(pos, color='#d8d8d8', lw=.35, alpha=.5)
        ax.axhline(pos, color='#d8d8d8', lw=.35, alpha=.5)
    for height in heights:
        ax.text(18, 14.5-height/2-.25, f'Height {height:g} → {height+1:g}', va='center', fontsize=9)
    fig.text(.08, .11, 'Starting heights: ' + ', '.join(f'{h:g}' for h in heights) + ' px. Width stays 3.2 px.\nOffsets were found by rendering: zero overlap, with a clear pixel between groups.', fontsize=10)
    fig.text(.08, .043, 'Fixed center (14.5, 14.5) px and lean 0°. Color strength = coverage change.\nEach colored mark is one real pixel value after the step minus its value before.', fontsize=9)
    fig.savefig(OUT/'overlay.png', dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = np.array([[*FIXED, w, t] for w in WIDTHS for t in LEAN])
    # Use the original 28x28 renderer, with real increased widths. Check the
    # geometric stroke lies inside the canvas even where border cells have ink.
    extent = p[:, 3]/2 + FIXED[2]/2*np.tan(np.radians(p[:, 4]))
    assert np.all(extent < min(FIXED[0], 28-FIXED[0]))
    x = render(p).reshape(len(WIDTHS), len(LEAN), 28, 28)
    assert x.dtype == np.float32 and x.min() >= 0 and x.max() <= 1
    assert np.max(abs(x.sum((2, 3), dtype=float)-WIDTHS[:, None]*FIXED[2])) < 1e-5
    max_shared = 0
    for gap in GAPS:
        selected = indices(gap)
        delta = x[selected].astype(float)-x[selected, 0].astype(float)[:, None]
        counts = (abs(delta) > EPS).sum(0)
        shared = np.sum(counts > 1, axis=(1, 2))
        assert np.all(shared == 0), (gap, LEAN[shared > 0])
        max_shared = max(max_shared, int(shared.max()))
        assert np.array_equal(x[selected, 0].astype(float)[:, None]+delta, x[selected].astype(float))
    sparse = []
    for row in x:
        sparse_row = []
        for image in row:
            flat = image.ravel()
            ids = np.flatnonzero(flat)
            values = flat[ids].astype(float).tolist()
            rebuilt = np.zeros(784, np.float32)
            rebuilt[ids] = values
            assert np.array_equal(rebuilt, flat)
            sparse_row.append([ids.tolist(), values])
        sparse.append(sparse_row)
    initial = auto_space([14.5, 14.5, 6.25, 3.2, 0], 2, 1, 2)
    data = dict(support_threshold=EPS, defaults=initial['knobs'], change_knob=2, amount=1, spread_knob=2)
    template = (ROOT/'experiments/lean_pixel_overlay/viewer.html').read_text()
    assert template.count('__RENDERED_DATA__') == 1
    (OUT/'index.html').write_text(template.replace('__RENDERED_DATA__', json.dumps(data, separators=(',', ':'))))
    figure(initial['knobs'])
    with (OUT/'step_metrics.csv').open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['width_gap_px', 'width_px', 'lean_deg', 'change_l2', 'changed_pixels', 'shared_changed_pixels'])
        for gap in GAPS:
            for wi in indices(gap):
                for ti, lean in enumerate(LEAN):
                    d = x[wi, ti].astype(float)-x[wi, 0].astype(float)
                    writer.writerow([gap, WIDTHS[wi], lean, float(np.linalg.norm(d)), int((abs(d)>EPS).sum()), 0])
    result = dict(default_widths_px=[3.2, 11.2, 19.2], default_width_offsets_px=[0, 8, 16],
                  default_lean_deg=5, lean_grid_deg=LEAN.tolist(), width_gaps_px=GAPS.tolist(),
                  rendered_frames=len(WIDTHS)*len(LEAN), max_shared_changed_pixels=max_shared,
                  no_canvas_clipping=True, sparse_roundtrip='exact', endpoint_reconstruction='exact',
                  renderer='grey_ones.render; float32 images, float64 subtraction',
                  scope='Fixed center and height. Expanded width range for physical separation.')
    result['viewer_default'] = dict(knobs=initial['knobs'], change_knob='height', amount=1, spacing_px=initial['gap'], shared_pixels=0, clear_pixel_gap=1)
    (OUT/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    (OUT/'README.md').write_text('# Pixel changes at different starting settings\n\nStart with `.venv/bin/python serve_generated_lean_pixel_overlay.py`, then\nopen `http://127.0.0.1:8765/generated_lean_pixel_overlay/index.html`.\nRegenerate the page and initial figure with\n`.venv/bin/python make_generated_lean_pixel_overlay.py`.\n\nChoose the knob to change and a signed amount in the toolbar. Each layer\'s\nfive controls are its starting settings. Its colored pixels are\n`render(start + selected_knob_step) - render(start)`. The small caption\nin each layer shows the exact before and after values. Red gains ink;\nblue loses ink. All colors use the same fixed +/-1 coverage-change scale.\nThe reference is a separate black context image and is not subtracted from\nall endpoints. A height change is visible at lean zero. The old version\nonly subtracted lean and silently erased this case.\n\n**Copy reference to layers** copies all five reference settings into the\nthree starting states, leaving the chosen change and amount intact.\n**Space by** selects the only knob that Auto space is allowed to vary.\n**Auto space** generates starting states from the reference and searches\nfor uniform positive, negative, or symmetric offsets on that knob\'s\nslider resolution. It renders actual before/after images, checks every\npair of changed-pixel masks, and requires at least one unchanged pixel\nbetween groups (8-neighbor dilation). Empty responses and strokes clipped\nby the canvas are excluded. It chooses the smallest successful tested\nspacing, preserving all other reference settings. On failure it reports\nwhy and leaves the current controls untouched; it never fakes separation\nor changes another knob to make the check pass.\n\nThe default uses the attached short-reference example: center (14.5, 14.5),\nheight 6.25, width 3.2, lean zero. The operation is Height +1 px. Auto space\nfinds starting heights 6.25, 11, and 15.75 (gap 4.75 px); the colored groups\nhave zero overlap and an unchanged pixel between them. `overlay.png` shows\nthis default. The older width/lean sample metrics remain in `step_metrics.csv`;\n`results.json` distinguishes those fixed-slice checks from the current default.\n\nEvery live image comes from `grey_ones.render`. Images are float32 and\nsubtraction uses float64. No JavaScript renderer, interpolation, averaging,\nnormalization, pixel subdivisions, or display translations are used.\nDifferences below 1e-7 are omitted from colored marks and support counts.\nThe spacing guarantee uses that threshold and applies to the computed\nsettings and selected finite change, not future manual edits or arbitrary\ncontinuous sweeps. Black reference ink can remain between change groups;\n"clear" refers to the absence of colored changes, not necessarily white ink.\n\nManual changes can cause overlap; the live count appears below the image,\nand higher-numbered visible layers overwrite earlier ones at shared pixels.\nShow toggles isolate layers. Strokes reaching past the 28x28 canvas are\nreported by name. The local server is required for arbitrary settings.\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
