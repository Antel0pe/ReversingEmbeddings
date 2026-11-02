"""The data picks its own five motions out of a generic library.

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python analyze_generated_one_motion_discovery.py

`figures/generated_one_gradient_fields` showed that every knob direction lives in
the six affine fields built from an image's own two spatial gradients. But that
experiment was HANDED the affine library. Supplying "the generator slides,
stretches and shears the plane" is most of the answer, so the method was not yet
one a reader could apply to an unfamiliar generator.

This experiment removes that. The library here is TWELVE generic motions -- every
polynomial warp up to quadratic -- of which the five-knob family can only use
five. Nothing says which. Forward selection then adds whichever single motion
most reduces the leftover, repeatedly, and the leftover curve is read for a
plateau. If the selection picks the five affine motions first and then stalls,
the data has named its own vocabulary and counted it, with no knob labels and no
access to `grey_ones`.

What the analysis is allowed to see: images, and spatial gradients computed from
those images. Knob values are used ONLY to generate the dataset, never to fit or
to score. Two further things the method does need, both reported as controls:

  - DENSITY. Neighbour differences are fitted to first order, so the pairs must
    be close. A scattered sample's nearest neighbours in pixel distance are far
    enough apart that the fit is meaningless; that failure is measured below.
  - GRADIENTS. `grey_ones` coverage images have hard edges and no spatial
    derivatives, so they must be blurred first and differenced crudely. That
    tooling, not the creases, turns out to dominate the coverage leftover -- the
    same pipeline run on the crease-free family isolates how much.

Controls: drop the lean motion from the library; fit using a DIFFERENT image's
motions; fit against random pictures; shrink the step scale to confirm the
leftover is ordinary second-order error rather than a missing ingredient.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy.ndimage import gaussian_filter

import smooth_ones
from grey_ones import RANGES, render as render_coverage

OUT = Path(__file__).resolve().parent / "figures/generated_one_motion_discovery"

N_ANCHORS = 250
N_NEIGHBOURS = 8
STEP_SCALES = (0.02, 0.002, 0.0002)

WEIGHT_NAMES = ["1", "x", "y", "x^2", "xy", "y^2"]
FIELD_NAMES = [f"{w}*d/dx" for w in WEIGHT_NAMES] + [f"{w}*d/dy" for w in WEIGHT_NAMES]
# A five-knob affine family with no vertical shear can only use these five.
# Not told to the selection; this is the prediction being tested.
PREDICTED = [0, 1, 2, 6, 8]
KNOB_OF = {0: "cx", 1: "width", 2: "lean", 6: "cy", 8: "height"}


def weights(n_pix=smooth_ones.N_PIX):
    """Position weights, measured from the CANVAS centre, not the stroke's.

    Using the canvas keeps the library label-free: it needs no (cx, cy). Moving
    the origin only mixes `1*d/dx` with `x*d/dx` and so on, which stays inside
    the same span, so the prediction is unaffected.
    """
    x, y = smooth_ones.pixel_centres(n_pix)
    u, v = (x - 14.0) / 14.0, (y - 14.0) / 14.0
    return [np.ones_like(u), u, v, u * u, u * v, v * v]


def library(d_dx, d_dy, w):
    """The twelve candidate motion pictures, as (784, 12)."""
    return np.stack([b * d_dx for b in w] + [b * d_dy for b in w]).reshape(12, -1).T


def leftover(libs, diffs, cols, limit=None):
    """Median unexplained fraction of a neighbour difference, over all pairs."""
    cols = list(cols)
    out = []
    for i, (A, D) in enumerate(zip(libs, diffs)):
        if limit is not None and i >= limit:
            break
        M = A[:, cols]
        coef, *_ = np.linalg.lstsq(M, D.T, rcond=None)
        resid = np.linalg.norm(M @ coef - D.T, axis=0)
        norm = np.linalg.norm(D, axis=1)
        keep = norm > 1e-14
        out.append(resid[keep] / norm[keep])
    return float(np.median(np.concatenate(out)))


def dataset(kind, step_scale, seed, blur=None, scattered=False):
    """Anchors plus close-by neighbours. Returns (libraries, difference sets).

    `blur` forces the hard-edged pipeline (gaussian blur, then `np.gradient`) and
    is applied to BOTH renderers when set, so the tooling can be compared on a
    family that has no creases.
    """
    rng = np.random.default_rng(seed)
    span = RANGES[:, 1] - RANGES[:, 0]
    anchors = RANGES[:, 0] + rng.random((N_ANCHORS, 5)) * span
    if scattered:
        # No local structure: neighbours are simply other samples from the box,
        # found by pixel distance. This is the honest no-density case.
        others = RANGES[:, 0] + rng.random((N_ANCHORS * N_NEIGHBOURS, 5)) * span
        flat = np.concatenate([anchors, others])
    else:
        nb = anchors[:, None, :] + step_scale * span * rng.standard_normal(
            (N_ANCHORS, N_NEIGHBOURS, 5))
        flat = np.concatenate([anchors, nb.reshape(-1, 5)])

    if kind == "smooth":
        images = smooth_ones.render(flat)
    else:
        images = render_coverage(flat, dtype=np.float64)
    if blur is not None:
        images = np.stack([gaussian_filter(im, blur, mode="constant") for im in images])

    w = weights()
    if blur is None and kind == "smooth":
        grads = [smooth_ones.spatial_gradients(p) for p in anchors]
    else:
        grads = [tuple(np.gradient(im)[::-1]) for im in images[:N_ANCHORS]]
    libs = [library(gx, gy, w) for gx, gy in grads]

    flat_img = images.reshape(len(images), -1)
    if scattered:
        V = flat_img
        d2 = ((V ** 2).sum(1)[:, None] + (V ** 2).sum(1)[None, :] - 2 * V @ V.T)
        np.fill_diagonal(d2, np.inf)
        nearest = np.argsort(d2[:N_ANCHORS], axis=1)[:, :N_NEIGHBOURS]
        diffs = [V[nearest[i]] - V[i] for i in range(N_ANCHORS)]
    else:
        diffs = [flat_img[N_ANCHORS + i * N_NEIGHBOURS:
                          N_ANCHORS + (i + 1) * N_NEIGHBOURS] - flat_img[i]
                 for i in range(N_ANCHORS)]
    return libs, diffs


results = {}

for kind, blur, tag in (("smooth", None, "smooth_ones, exact gradients"),
                        ("coverage", 1.0, "grey_ones, blur 1.0 + np.gradient")):
    print(f"\n===== {tag} =====")
    libs, diffs = dataset(kind, 0.002, seed=5, blur=blur)

    print("  forward selection from the twelve generic motions")
    print("  (the five affine motions are the prediction, never an input)")
    chosen, remaining, curve = [], list(range(12)), []
    for round_ in range(8):
        best = min(remaining, key=lambda c: leftover(libs, diffs, chosen + [c], limit=60))
        chosen.append(best)
        remaining.remove(best)
        lo = leftover(libs, diffs, chosen)
        curve.append({"field": FIELD_NAMES[best], "leftover": lo,
                      "predicted": best in PREDICTED})
        note = f"(predicted: {KNOB_OF[best]})" if best in PREDICTED else "(NOT predicted)"
        print(f"     {round_ + 1}. add {FIELD_NAMES[best]:>10}"
              f"  -> leftover {lo * 100:7.3f}%   {note}")

    five = leftover(libs, diffs, PREDICTED)
    twelve = leftover(libs, diffs, list(range(12)))
    no_lean = leftover(libs, diffs, [c for c in PREDICTED if c != 2])
    borrowed = leftover(libs[1:] + libs[:1], diffs, PREDICTED)
    rng = np.random.default_rng(99)
    rand = leftover([rng.standard_normal((784, 5)) for _ in libs], diffs, range(5))
    print(f"  the five predicted motions alone     : {five * 100:7.3f}%")
    print(f"  all twelve generic motions           : {twelve * 100:7.3f}%")
    print(f"  CONTROL drop the lean motion (y*d/dx): {no_lean * 100:7.3f}%")
    print(f"  CONTROL motions from a DIFFERENT image: {borrowed * 100:7.3f}%")
    print(f"  CONTROL five random pictures         : {rand * 100:7.3f}%")

    sweep = {}
    print("  leftover for the five predicted motions as the steps shrink:")
    for e in STEP_SCALES:
        li, di = dataset(kind, e, seed=7, blur=blur)
        sweep[str(e)] = leftover(li, di, PREDICTED)
        print(f"     step scale {e:<8}: {sweep[str(e)] * 100:8.4f}%")

    sl, sd = dataset(kind, 0.002, seed=13, blur=blur, scattered=True)
    scattered = leftover(sl, sd, list(range(12)))
    print(f"  CONTROL no density, nearest of a scattered sample, all twelve: "
          f"{scattered * 100:7.3f}%")

    results[kind] = {"pipeline": tag, "selection": curve, "five_predicted": five,
                     "all_twelve": twelve, "drop_lean": no_lean,
                     "borrowed_fields": borrowed, "random_pictures": rand,
                     "step_sweep": sweep, "scattered_sample": scattered}

print("\n===== is the coverage leftover the creases, or the gradient tooling? =====")
print("  identical blur + np.gradient pipeline, on a family with NO creases\n")
print(f"  {'pipeline':>38} | " + " | ".join(f"step {e}" for e in STEP_SCALES))
tooling = {}
for kind in ("smooth", "coverage"):
    for blur in (1.0, 2.0):
        row = []
        for e in STEP_SCALES:
            li, di = dataset(kind, e, seed=21, blur=blur)
            v = leftover(li, di, PREDICTED)
            tooling[f"{kind}_blur{blur}_step{e}"] = v
            row.append(f"{v * 100:8.3f}%")
        name = f"{'smooth_ones' if kind == 'smooth' else 'grey_ones'}, blur {blur}"
        print(f"  {name:>38} | " + " | ".join(row))
print(f"  {'smooth_ones, exact gradients':>38} | "
      + " | ".join(f"{results['smooth']['step_sweep'][str(e)] * 100:8.3f}%"
                   for e in STEP_SCALES))
results["tooling_isolation"] = tooling

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "results.json").write_text(json.dumps(results, indent=2))
print(f"\nwrote {OUT / 'results.json'}")
