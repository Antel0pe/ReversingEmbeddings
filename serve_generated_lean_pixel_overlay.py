"""Serve the pixel-change viewer and calculate real nonoverlapping offsets.

Run: .venv/bin/python serve_generated_lean_pixel_overlay.py
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation
from grey_ones import render

ROOT = Path(__file__).resolve().parent
EPS = 1e-7
NAMES = ['X center', 'Y center', 'Height', 'Width', 'Lean']
LIMITS = np.array([[4, 24, .1], [4, 24, .1], [2, 26, .05], [.5, 24, .1], [-35, 35, .5]])


def validate_knobs(knobs):
    p = np.asarray(knobs, dtype=float)
    if p.shape != (4, 5) or not np.all(np.isfinite(p)):
        raise ValueError('Provide four images with five finite knob values each.')
    if np.any(p < LIMITS[:, 0]-1e-8) or np.any(p > LIMITS[:, 1]+1e-8):
        raise ValueError('Knob values are outside the viewer limits.')
    return p


def operation(change_knob, amount):
    if isinstance(change_knob, bool) or int(change_knob) != change_knob or not 0 <= int(change_knob) < 5:
        raise ValueError('Choose one of the five knobs to change.')
    amount = float(amount)
    if not np.isfinite(amount) or abs(amount) > 20:
        raise ValueError('Use a finite change between -20 and +20.')
    return int(change_knob), amount


def clipped(p):
    p = np.atleast_2d(p)
    cx, cy, height, width, lean = p.T
    reach = width/2 + height/2*np.abs(np.tan(np.radians(lean)))
    return (cx-reach < -1e-8) | (cx+reach > 28+1e-8) | (cy-height/2 < -1e-8) | (cy+height/2 > 28+1e-8)


def valid_end(p):
    return (p[:, 2] > 0) & (p[:, 3] > 0) & (np.abs(p[:, 4]) <= 60)


def compute_overlay(knobs, change_knob=4, amount=5):
    p = validate_knobs(knobs)
    change_knob, amount = operation(change_knob, amount)
    after = p[1:].copy()
    after[:, change_knob] += amount
    if not np.all(valid_end(after)):
        raise ValueError('This change makes a stroke size nonpositive or its lean exceed 60°. Use a smaller change.')
    images = render(np.concatenate([p, after])).astype(float)
    differences = images[4:7] - images[1:4]
    cuts = clipped(p)
    cuts[1:] |= clipped(after)
    return dict(reference=images[0].ravel().tolist(),
                deltas=differences.reshape(3, 784).tolist(),
                previews=images[:4].reshape(4, 784).tolist(),
                endpoints=after.tolist(), clipped=cuts.tolist())


def auto_space(reference, change_knob, amount, spread_knob):
    reference = validate_knobs([reference]*4)[0]
    change_knob, amount = operation(change_knob, amount)
    spread_knob = int(spread_knob)
    if not 0 <= spread_knob < 5:
        raise ValueError('Choose a knob to space the layers by.')
    if abs(amount) < 1e-12:
        raise ValueError('Choose a nonzero change before spacing the layers.')
    lower, upper, step = LIMITS[spread_knob]
    anchor = reference[spread_knob]
    offsets = np.arange(int(np.ceil((lower-anchor)/step-1e-8)),
                        int(np.floor((upper-anchor)/step+1e-8))+1)
    bases = np.tile(reference, (len(offsets), 1))
    bases[:, spread_knob] = np.round(anchor+offsets*step, 8)
    ends = bases.copy()
    ends[:, change_knob] += amount
    fits = valid_end(ends) & ~clipped(bases) & ~clipped(ends)
    lookup = {}
    good = np.flatnonzero(fits)
    if len(good):
        imgs = render(np.concatenate([bases[good], ends[good]])).astype(float)
        masks = np.abs(imgs[len(good):]-imgs[:len(good)]) > EPS
        lookup = {int(offsets[i]): mask for i, mask in zip(good, masks) if np.any(mask)}
    if 0 not in lookup:
        raise ValueError('The reference change is empty or extends past the image edge. Adjust the reference or change amount first.')
    anchor_mask = lookup[0]
    # Adjacent cells also count as a conflict: success leaves a full empty
    # pixel between colored groups, with no display translations or masking.
    guard = binary_dilation(anchor_mask, structure=np.ones((3, 3)))
    chosen = None
    for n in range(1, int(max(abs(offsets)))+1):
        for moves in [[0, n, 2*n], [0, -n, -2*n], [0, -n, n]]:
            a, b = moves[1:]
            if a not in lookup or b not in lookup:
                continue
            if np.any(guard & lookup[a]) or np.any(guard & lookup[b]):
                continue
            if np.any(binary_dilation(lookup[a], structure=np.ones((3, 3))) & lookup[b]):
                continue
            chosen = (n, moves)
            break
        if chosen is not None:
            break
    if chosen is None:
        raise ValueError(f'No separated set fits by varying {NAMES[spread_knob].lower()} for this change. Try another spacing knob or a smaller change.')
    n, moves = chosen
    gap = float(round(n*step, 8))
    layers = np.tile(reference, (3, 1))
    layers[:, spread_knob] = np.round(anchor+np.array(moves)*step, 8)
    return dict(knobs=np.concatenate([reference[None], layers]).tolist(),
                spread_knob=spread_knob, gap=gap, values=layers[:, spread_knob].tolist(),
                shared_pixels=0, empty_pixel_gap=1,
                message=f'{NAMES[spread_knob]} spacing {gap:g}' + ('°' if spread_knob == 4 else ' px') + ' · One empty pixel between change groups')



class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def do_POST(self):
        if self.path not in ['/api/lean-overlay', '/api/auto-space']:
            self.send_error(404)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 8192:
                raise ValueError('Invalid request size.')
            payload = json.loads(self.rfile.read(length))
            if self.path == '/api/auto-space':
                result = auto_space(payload['reference'], payload['change_knob'], payload['amount'], payload['spread_knob'])
            else:
                result = compute_overlay(payload['knobs'], payload.get('change_knob', 4), payload.get('amount', 5))
            status = 200
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            result, status = dict(error=str(error)), 400
        data = json.dumps(result, separators=(',', ':'), allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(ROOT/'figures')))
    print(f'Viewer: http://127.0.0.1:{args.port}/generated_lean_pixel_overlay/index.html', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
