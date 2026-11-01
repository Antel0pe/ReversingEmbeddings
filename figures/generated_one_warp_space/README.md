# The family is flat when you describe WHERE ink moves, not HOW MUCH each pixel has

**Provenance.** These scripts and figures were produced in an earlier session and
were sitting in a temporary scratch directory. They are salvaged here unchanged
except for stripping hardcoded absolute paths. Re-verified in this session:
`warp5.py` reproduces its printed numbers from the cached fit, and `fig.py` and
`fig2.py` regenerate both PNGs. The expensive fit itself (`warp4.py`, cached in
`w4.npz`) was not re-run.

## The idea

Earlier work described an image as 784 numbers: how much ink sits at each fixed
place. This describes it instead as a **warp**: a smooth map saying where each
place came from, with 12 numbers per image, two degree-2 polynomials

```
x_src = a0 + a1*x + a2*y + a3*x^2 + a4*x*y + a5*y^2
y_src = b0 + b1*x + b2*y + b3*x^2 + b4*x*y + b5*y^2
```

Nothing affine was assumed. The six quadratic terms were free to be used, and
the fit had to discover on its own that they are not needed.

## `warp_vs_pixels.png` -- the dimension count

120 settings from the whole default knob box, rendered at 112x112.

- **In warp coordinates the family is a 5-dimensional flat subspace.** The
  spectrum of the exact warp coefficients falls to machine zero (1e-17) after
  the fifth component. Not "5 components explain most of it" -- the sixth and
  beyond are numerically absent.
- **In pixel coordinates it never flattens.** The pixel spectrum decays slowly;
  50 directions still leave 8% error.
- **Rebuilding 60 unseen 1s from k directions**, fitted on the other 60, same
  images and same metric for both curves: at k=5, warp space gives 0.086 and
  pixel space 0.389, a factor of 4.6. The 0.086 is a floor -- the cost of
  resampling a 28-pixel-wide picture through the warp, not model error.
- One held-out example: warp space with 5 directions leaves 0.061 error and a
  thin resampling rim; pixel space with 5 directions leaves 0.448 and a smeared
  average stroke, because a fixed set of pixel patterns cannot put ink in a
  place it was not already prepared to put it.

## `warp_identifiability.png` -- where fitting warps from pixels breaks

Fitting the warp from pixels alone recovers four knobs well and **cy badly**.
The cause is identifiability, not optimisation.

The three coefficients controlling the vertical map are not equally visible.
Take a step along their least-visible combination: the largest pixel changes by
0.0111 and the whole image by 0.25%. Take the **same-size** step along their
most-visible combination: largest pixel change 1.0000, whole image 20.11%. The
first is effectively invisible, so the data cannot pin it down.

The consequence is exact. Across all 120 fits, the error the fit makes in the
vertical-shift coefficient correlates at **-0.999** with the spurious `y <- y^2`
coefficient it invents. The fit never chooses between "slide the bar down" and
"curve the vertical map"; both produce the same picture.

Median error recovering each true knob, both methods seeing only pixels, same
120 images, same 112x112 grid:

| Knob | Warp fitted from pixels, then inverted | Six linear moments, closed form |
| --- | ---: | ---: |
| cx | 0.0717 | 7.6e-06 |
| cy | **0.4670** | 1.3e-04 |
| height | 0.0607 | 7.8e-04 |
| width | 0.0172 | 4.4e-05 |
| lean | 0.0036 | 6.4e-04 |

## How this sits with the other three directories

- `figures/generated_one_moment_chart` is the right-hand column above, developed
  on its own: six linear projections and a closed-form inversion, which has no
  identifiability hole because it never fits anything.
- `figures/generated_one_gradient_fields` supplies the affine field library and
  confirms the five knobs use exactly five of six affine fields. That assumes
  affine. **This directory is the version that does not**, which is why it
  matters: a free degree-2 warp basis rediscovers the dimension count by itself.
- `figures/generated_one_crease_origin` explains why pixel space behaves so
  badly here in the first place.

The open problem is unchanged: a generic warp fit gets the dimension count for
free but inherits an identifiability hole, and the moments close that hole only
because this shape is a symmetric parallelogram. Neither is yet a general method.

## Scripts

Run from the repo root, e.g. `.venv/bin/python figures/generated_one_warp_space/warp5.py`.

- `warpfit.py` -- the first fit, with the generic degree-2 warp basis.
- `warp2.py` -- renderer at arbitrary grid size plus the warp machinery; imported by the rest.
- `warp3.py`, `warp4.py` -- the fitting runs; `warp4.py` writes `w4.npz`.
- `warp5.py` -- closed-form knobs from the fitted coefficients, and the vertical degeneracy.
- `fig.py`, `fig2.py` -- the two figures, from the cached fits.
