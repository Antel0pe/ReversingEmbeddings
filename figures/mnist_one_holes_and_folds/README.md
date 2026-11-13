# Holes and folds in MNIST 1s: persistent homology + a local shell test

Scripts (repo root): `analyze_mnist_one_holes_and_folds.py` (main run, ~25 min),
`check_shell_test_touch.py` (toy follow-up), `make_mnist_one_holes_and_folds_figures.py`.
Needs `ripser` (`uv pip install --python .venv/bin/python ripser`).

## Tests
- **Loops**: Vietoris-Rips persistent homology (H1). Loop strength = death / birth.
  For the strongest loop, the birth edge (shortcut) is read from its cocycle and the
  long way round is a shortest path using only edges shorter than birth.
  Cavities (H2) on 600-point subsamples.
- **Shells**: for each centre, neighbours ranked 8-48. Score = the longest edge of the
  shell's minimum spanning tree whose removal leaves >= 5 points on each side, divided
  by the median MST edge.

## Results
| Cloud | Strongest loop | Cavity | Shell median | Flagged vs its Gaussian twin |
|---|---|---|---|---|
| Toy curl resting on itself | 8.52x (noise <= 2.78) | - | 2.19 | 0% within 0.3 of touch |
| Toy tube | 8.19x (noise <= 2.66) | - | 2.29 | - |
| MNIST 1s (6,742) | 1.33x | 1.11x | 1.12 | 69.5% |
| Gaussian twin of MNIST | 1.15x | 1.06x | 1.02 | 1.0% |
| Generated 1s (no holes/folds) | **1.70x** | 1.20x | 1.27 | **95.9%** |
| Gaussian twin of generated | 1.21x | 1.10x | 1.03 | 1.0% |

Toy shell test vs gap between layers (`touch_check.json`): gap 0 -> 0.0% / 0.4%
(density 160 / 480); ~1.3 spacings 2.3%; ~2.2 spacings 16.8%; ~3.2 spacings 40.5%;
~5.5 spacings at density 480 (gap wider than the shell) 4.0%.

## Reading
- PH is cheap here: H1 on all 6,742 MNIST 1s in ~10 min with cocycles; 2,000 in 3 s.
  3-D toys are the slow case (1,000 dots 40 s uncapped) -- cap with `thresh`.
- No MNIST loop exceeds the chance loops of the hole-free generated family.
  MNIST's strongest loop is a detour round an unsampled patch between two
  near-identical upright 1s.
- The shell test is blind to a genuine touch: the contact line passes through every
  nearby shell and joins its halves. It needs dot spacing < gap < shell radius.
- The Gaussian twin is the wrong null for the shell test: any lumpy real data beats it
  (the fold-free generated 1s are flagged 95.9%). The most split MNIST shells separate
  serif-and-base 1s from plain sticks -- possibly branching styles, not folds.
