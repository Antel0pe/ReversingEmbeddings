"""All five knob directions are reweightings of two spatial gradients.

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python analyze_generated_one_gradient_fields.py

The claim: at any setting, what a knob does to the image can be built from just
two ingredients taken from that image alone -- how each pixel's brightness
changes if the picture slides sideways, and if it slides downward -- each
weighted by 1, by how far right of centre the pixel sits, and by how far below.
Six combinations. If the five knob directions all live in that six-dimensional
span, then the knobs' behaviour is readable off a single image, with no access to
the generator and no parameter labels.

Six fields inside a 784-dimensional space is a severe constraint, but three
controls keep the result from being vacuous:

  - BORROWED FIELDS. Fit a setting's knob direction using the six fields built
    from a DIFFERENT image. Should fail; the fields have to be the image's own.
  - DROPPED FIELD. Refit with one of the six removed. Should fail for whichever
    knob needed it.
  - THE UNUSED SIXTH. One field, sideways-gradient weighted by height (x*d/dy),
    is vertical shear, which this family cannot do. No knob should load on it.

The knob directions are measured by nudging a knob and subtracting, which is
itself slightly crooked: a before-and-after difference across a finite nudge
carries an error that theory says shrinks as the SQUARE of the nudge. So the
residual is re-measured at four nudge sizes. A residual that tracks the nudge
squared is measurement error; one that sits still is a real missing ingredient.

This needs `smooth_ones`, not `grey_ones`: spatial gradients only exist because
the edges are soft. See `figures/generated_one_crease_origin`.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

import smooth_ones
from grey_ones import KNOBS, RANGES

OUT = Path(__file__).resolve().parent / "figures/generated_one_gradient_fields"
FIELD_NAMES = ["d/dx", "d/dy", "x*d/dx", "y*d/dx", "x*d/dy", "y*d/dy"]
UNUSED = 4                      # x*d/dy, the vertical shear this family lacks
N_SETTINGS = 25
NUDGES = [1e-2, 1e-3, 1e-4, 1e-5]


def settings(n, seed):
    rng = np.random.default_rng(seed)
    return RANGES[:, 0] + rng.random((n, 5)) * np.diff(RANGES).ravel()


def knob_direction(p, knob, nudge):
    """What the image does when one knob is nudged. Central difference."""
    step = nudge * (1.0 + abs(p[knob]))
    lo, hi = p.copy(), p.copy()
    lo[knob] -= step
    hi[knob] += step
    return ((smooth_ones.render(hi)[0] - smooth_ones.render(lo)[0])
            / (2 * step)).ravel()


def fit(fields, target):
    """Best mix of the fields, and the fraction of the target left unexplained."""
    coef, *_ = np.linalg.lstsq(fields.T, target, rcond=None)
    residual = np.linalg.norm(fields.T @ coef - target) / np.linalg.norm(target)
    return coef, float(residual)


P = settings(N_SETTINGS, seed=7)
results = {"n_settings": N_SETTINGS, "field_names": FIELD_NAMES,
           "unused_field": FIELD_NAMES[UNUSED]}

# ---- 1. residual versus nudge size: is the leftover real, or is it my ruler? ----
print("1. Fraction of each knob's effect left unexplained by the six fields.")
print(f"   {'nudge':>8} " + " ".join(f"{k:>10}" for k in KNOBS)
      + "   | nudge squared")
table = []
for nudge in NUDGES:
    rows = []
    for p in P:
        fields = smooth_ones.affine_fields(p)
        rows.append([fit(fields, knob_direction(p, k, nudge))[1] for k in range(5)])
    med = np.median(np.array(rows), axis=0)
    table.append({"nudge": nudge, "median_residual": med.tolist()})
    print(f"   {nudge:>8.0e} " + " ".join(f"{v:>10.2e}" for v in med)
          + f"   | {nudge ** 2:>9.0e}")
results["residual_by_nudge"] = table
shrink = (np.array(table[0]["median_residual"])
          / np.array(table[2]["median_residual"]))
results["shrink_factor_per_100x_smaller_nudge"] = shrink.tolist()
print(f"   nudge 100x smaller shrinks the leftover by: "
      + ", ".join(f"{k}={v:.0f}x" for k, v in zip(KNOBS, shrink))
      + "  (10000x expected)")

# ---- 2. which fields each knob actually uses ----
best = NUDGES[2]
loads = np.zeros((5, 6))
for p in P:
    fields = smooth_ones.affine_fields(p)
    for k in range(5):
        coef, _ = fit(fields, knob_direction(p, k, best))
        loads[k] += np.abs(coef) / np.abs(coef).max()
loads /= N_SETTINGS
results["mean_relative_loading"] = loads.tolist()
print(f"\n2. Which of the six fields each knob loads on (mean relative weight).")
print(f"   {'':>8} " + " ".join(f"{n:>8}" for n in FIELD_NAMES))
for k, name in enumerate(KNOBS):
    print(f"   {name:>8} " + " ".join(f"{v:>8.4f}" for v in loads[k]))
results["max_loading_on_unused_field"] = float(loads[:, UNUSED].max())
print(f"   largest load any knob puts on {FIELD_NAMES[UNUSED]} "
      f"(vertical shear, absent from this family): {loads[:, UNUSED].max():.2e}")

# ---- 3. the two exact identities ----
cos = {}
for p in P:
    d_dx, d_dy = smooth_ones.spatial_gradients(p)
    for knob, field in (("cx", -d_dx.ravel()), ("cy", -d_dy.ravel())):
        g = knob_direction(p, KNOBS.index(knob), best)
        c = float(g @ field / (np.linalg.norm(g) * np.linalg.norm(field)))
        cos.setdefault(knob, []).append(c)
results["identity_cosines"] = {k: {"min": float(np.min(v)), "median": float(np.median(v))}
                               for k, v in cos.items()}
print(f"\n3. Two identities, as cosines against the raw gradients.")
for k, v in cos.items():
    print(f"   d(image)/d({k}) vs -(d/d{'x' if k == 'cx' else 'y'}): "
          f"median {np.median(v):.8f}, worst {np.min(v):.8f}")

# ---- 4. controls ----
print("\n4. Controls.")
borrowed, dropped = [], np.zeros((5, 6))
other = settings(N_SETTINGS, seed=99)
for p, q in zip(P, other):
    own, foreign = smooth_ones.affine_fields(p), smooth_ones.affine_fields(q)
    for k in range(5):
        g = knob_direction(p, k, best)
        borrowed.append(fit(foreign, g)[1])
        for j in range(6):
            keep = [i for i in range(6) if i != j]
            dropped[k, j] += fit(own[keep], g)[1] / N_SETTINGS
results["control_borrowed_fields_median_residual"] = float(np.median(borrowed))
results["control_dropped_field_residual"] = dropped.tolist()
print(f"   fields borrowed from another image: median residual "
      f"{np.median(borrowed):.4f}  (own fields: {np.median([r for t in table[2:3] for r in t['median_residual']]):.2e})")
print(f"   dropping one field, residual left for each knob:")
print(f"   {'':>8} " + " ".join(f"{n:>8}" for n in FIELD_NAMES))
for k, name in enumerate(KNOBS):
    print(f"   {name:>8} " + " ".join(f"{v:>8.4f}" for v in dropped[k]))

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "results.json").write_text(json.dumps(results, indent=2))
print(f"\nwrote {OUT / 'results.json'}")
