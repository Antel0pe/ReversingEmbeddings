# Does a flat 5-D plane give each generated 1 a unique address?

Built by `analyze_generated_one_flat_plane.py`, `confirm_generated_one_flat_plane.py`
and `make_generated_one_flat_plane_figure.py`. Numbers are in `results.json` and
`fresh_confirmation.json`. Planes are in `planes.npz`.

The idea: pick one 1, take the flat 5-D plane touching the family there, project every
1 onto it, and use the five plane coordinates as an address. The coordinates need not
match the real knobs. So the only question is whether the projection is one-to-one.
No decoder is involved.

## Result

| Plane | 1s with a clearly different look-alike | Settings where plane is blind or flipped |
| --- | ---: | ---: |
| Gently curved paper (test check) | 0% | 0% |
| Crumpled paper (test check) | 33.5% | 29.5% |
| At root 1 | 16.7% | 93% |
| At the middle of the ranges | 17.7% | 59% |
| At the best of 200 points | 10.0% | 86% |
| Best-fit plane for all 1s | 1.2% | 49% |
| Random plane | 16.2% | 49% |
| Five weighted pixel sums | none found | none found |
| Searched from images, best of 6 | none found | none found |
| Searched from images, worst of 6 | 3.7% | 43% |

A look-alike has the same plane position to one part in a million and a picture at least
a quarter of the typical distance between two 1s away, and is rendered to confirm. One
pair proves failure. Finding none is only evidence. 600 1s and 1,500 settings per plane.

## Why the recipe fails

The plane at a 1 is built from that 1's knob directions, which are nonzero only near its
edges. It has weight on 44 to 58 of the 784 pixels, so ink elsewhere cannot affect the
five numbers. At a setting with a blind direction, a knob move changed the image while
the five numbers stayed the same to about 4e-16.

## What does pass

- **Five weighted sums**: total ink and the ink-weighted averages of x, y, y*y and x*y.
  A formula turns them into all five knobs, which is a proof for the continuous stroke.
  On the 28x28 grid the worst knob error is 3.9% of that knob's range, and re-rendering
  from the formula lands within 3.9% of a typical distance (median 0.4%). Five numbers are
  enough because width follows from total ink and height. The sixth sum in
  `figures/generated_one_moment_chart` is redundant.
- **Searched from images**: no knobs or formula, just a search for a plane that keeps
  different 1s apart. Only one of six tries came out clean, and I do not know why. Starting
  from the best-fit plane does not explain it: the other two tries from that start failed.

Because those two planes were picked as winners, they were re-checked on 150,000 new
images, 1,000 new targets and 4,000 new settings. Both again showed no look-alikes and no
blind or flipped settings. The near-miss starting points refined back onto the original
1 (gaps about 1e-13). A second try that looked nearly clean, random start with images C,
flipped at 3.5% of settings.

## Limits

Generated 1s within the default knob ranges only. Real MNIST 1s were not tested. Only the
weighted-sums plane has a formula, so the searched plane rests on evidence, not proof.
