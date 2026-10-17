"""Conditional discovery. NumPy only; never import oracle or renderer here.

The supplied geometry/raster/program grammar is specified in PROTOCOL.md.
All public inputs are observations, anonymous action IDs, or frozen models.
"""
from fractions import Fraction
import itertools
import json
from pathlib import Path
import numpy as np

COUNTS = (1, 2, 4, 8, 16, 32, 64)
PAIRS = list(itertools.combinations_with_replacement(range(5), 2))


def encode_geometry(images):
    x = np.asarray(images, dtype=np.float64).reshape(-1, 28, 28)
    out = []
    for im in x:
        mass = im.sum(1)
        ink = np.flatnonzero(mass > 0)
        if len(ink) < 4:
            raise ValueError('not enough full stroke rows')
        first, last = ink[0], ink[-1]
        width = np.median(mass[first + 1:last])
        if width <= 0:
            raise ValueError('nonpositive width')
        top = first + 1 - mass[first] / width
        bottom = last + mass[last] / width
        centers = []
        used = (first + 1, last - 1)
        for row in used:
            cumulative = np.r_[0., np.cumsum(im[row])]
            k = np.argmin(np.abs(cumulative - width / 2))
            centers.append(k - cumulative[k] + width / 2)
        slope = (centers[0] - centers[1]) / (used[1] - used[0])
        intercept = centers[0] + slope * (used[0] + .5)
        out.append((top, bottom, intercept, width, slope))
    return np.asarray(out)


def decode_geometry(q, subrows, dtype=np.float64):
    q = np.atleast_2d(np.asarray(q, dtype=np.float64))
    out = np.empty((len(q), 28, 28), dtype=dtype)
    lo = np.arange(28 * subrows) / subrows
    mid = lo + .5 / subrows
    col = np.arange(28)
    for start in range(0, len(q), 64):
        t, b, a, w, s = [q[start:start+64, j, None] for j in range(5)]
        vy = np.maximum(0., np.minimum(lo + 1/subrows, b) - np.maximum(lo, t))
        center = a - s * mid
        hx = np.maximum(0., np.minimum((center+w/2)[:, :, None], col+1)
                        - np.maximum((center-w/2)[:, :, None], col))
        out[start:start+64] = (vy[:, :, None] * hx).reshape(-1, 28, subrows, 28).sum(2)
    return out


def features(q, degree=2):
    q = np.atleast_2d(q)
    if degree == 1:
        return q
    return np.column_stack([q] + [q[:, i]*q[:, j] for i, j in PAIRS])


def metrics(truth, predicted):
    d = np.asarray(predicted, dtype=np.float64)-np.asarray(truth, dtype=np.float64)
    l2 = np.linalg.norm(d.reshape(len(d), -1), axis=1)
    per = np.max(np.abs(d.reshape(len(d), -1)), axis=1)
    i = int(np.argmax(l2))
    j = int(np.argmax(per))
    return dict(max_pixel=float(per[j]), max_image_l2=float(l2[i]),
                worst_l2_index=i, worst_pixel_index=j,
                mean_image_l2=float(l2.mean()))


def fit(observations, pair_before, pair_after, action_ids):
    """No names, parameter values, actual range bounds or renderer internals."""
    q = encode_geometry(observations)
    raster_scores = {}
    for n in COUNTS:
        raster_scores[str(n)] = metrics(observations, decode_geometry(q, n))
    subrows = min(COUNTS, key=lambda n: raster_scores[str(n)]['max_image_l2'])
    if raster_scores[str(subrows)]['max_pixel'] > 2e-6:
        raise ValueError('supplied program grammar does not explain observations')
    qb, qa = encode_geometry(pair_before), encode_geometry(pair_after)
    coeffs, diagnostics, degrees = [], [], []
    for action in sorted(np.unique(action_ids)):
        record = []
        for degree in (1, 2):
            center = np.mean(np.vstack((qb, qa)), axis=0)
            unit = np.std(np.vstack((qb, qa)), axis=0)
            fb, fa = features((qb-center)/unit, degree), features((qa-center)/unit, degree)
            scale = np.std(np.vstack((fb, fa)), axis=0)
            scale = np.maximum(scale, 1e-12)
            delta = (fa-fb)[action_ids != action] / scale
            _, sig, vt = np.linalg.svd(delta, full_matrices=False)
            ratio = float(sig[-1]/sig[0])
            record.append(dict(degree=degree, singular_values=sig.tolist(), relative_smallest=ratio))
            if ratio < 2e-6:
                if sig[-2]/sig[0] < 2e-6:
                    raise ValueError('multiple invariant directions at chosen degree')
                standard = vt[-1] / scale
                c = np.zeros(len(standard))
                c[:5] = standard[:5]/unit
                if degree == 2:
                    for j, (u, v) in enumerate(PAIRS):
                        value = standard[5+j]/(unit[u]*unit[v])
                        c[5+j] = value
                        c[u] -= value*center[v]
                        c[v] -= value*center[u]
                c /= np.max(np.abs(c))
                nonzero = np.flatnonzero(np.abs(c) > 2e-5)
                if c[nonzero[0]] < 0:
                    c *= -1
                snapped = np.array([float(Fraction(float(v)).limit_denominator(16)) for v in c])
                if np.max(np.abs(c-snapped)) > 2e-5:
                    raise ValueError('invariant has no simple rational description')
                padded = np.zeros(20)
                padded[:len(snapped)] = snapped
                coeffs.append(padded)
                degrees.append(degree)
                break
        else:
            raise ValueError('no low-degree invariant for anonymous action')
        diagnostics.append(record)
    c = np.array(coeffs)
    b = c[:, :5]
    if np.linalg.matrix_rank(b) != 5:
        raise ValueError('linear part not invertible')
    # Executable inverse below needs nonlinear terms independent of intercept.
    for j, (u, v) in enumerate(PAIRS):
        if 2 in (u, v) and np.any(c[:, 5+j] != 0):
            raise ValueError('symbolic inverse rule is inapplicable')
    active = np.flatnonzero(np.any(c[:, 5:] != 0, axis=1))
    if len(active) != 1:
        raise ValueError('expected one triangular nonlinear equation')
    shift = np.linalg.inv(b)[:, active[0]]
    if np.max(np.abs(np.delete(shift, 2))) > 1e-10:
        raise ValueError('nonlinear correction is not triangular')
    return dict(schema=1, subrows=int(subrows), coefficients=c.tolist(),
                degrees=degrees, action_ids=sorted(map(int, np.unique(action_ids))),
                monomials=['t','b','a','w','s'] + [f'{i}*{j}' for i,j in PAIRS],
                raster_scores=raster_scores, invariance_diagnostics=diagnostics,
                training_images=int(len(observations)), training_pairs=int(len(action_ids)))


def encode(images, model):
    return features(encode_geometry(images)) @ np.asarray(model['coefficients']).T


def geometry_from_coordinates(z, model):
    c = np.asarray(model['coefficients'])
    inv = np.linalg.inv(c[:, :5])
    temp = np.atleast_2d(z) @ inv.T
    nonlinear = features(temp)[:, 5:] @ c[:, 5:].T
    return (np.atleast_2d(z) - nonlinear) @ inv.T


def decode(z, model, dtype=np.float64):
    return decode_geometry(geometry_from_coordinates(z, model), model['subrows'], dtype)


def save_model(model, path):
    Path(path).write_text(json.dumps(model, indent=2, allow_nan=False)+'\n')
