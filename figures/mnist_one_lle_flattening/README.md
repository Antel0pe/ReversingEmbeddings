# Flattening MNIST 1s with locally linear pieces

`analyze_mnist_one_lle_flattening.py` (about 45 min on 4 cores) runs the unfolding suite's
`lle_k10` method unchanged, apart from piece size k and number of flat numbers d, on the
6,742 MNIST training 1s. It also runs a generated-1s control: the same count, uniform in the
default knob box. `make_mnist_one_lle_flattening_figure.py` makes `lle_flattening.png`.
`results.json` has every number. Each `*.npy` is one layout.

There is no true flat answer for MNIST, so every check uses only the images and the layout:
neighbourhoods kept (out of each image's 10 most similar), strangers (layout neighbours not
among the image's 100 most similar), and the rebuild miss (each image as the average of its 10
layout neighbours, median % of the image). "New" counts the layout numbers that the earlier
numbers do not predict (kNN R² < 0.5).

| MNIST, 15 numbers | rebuild | kept | strangers | new |
| --- | ---: | ---: | ---: | ---: |
| 10 most similar images (best possible) | 19.2% | | | |
| best flat plane (PCA) | 20.4% | 0.65 | 1.4% | 4 |
| walking along links (`isomap_k10`) | 22.1% | 0.52 | 5.2% | 8 |
| locally linear, k = 20 | 23.4% | 0.39 | 13.6% | 1 |
| locally linear, k = 40 | 23.5% | 0.35 | 17.1% | 1 |
| 10 random images | 67.3% | | | |

- **On MNIST, the locally linear method finds one direction, slant** (first number vs.
  measured slant |r| = 0.98). Every further number is a bent copy of it (R² 0.53-0.96 from the
  earlier numbers) at every k and d tried. It is worse than the best flat plane at every
  d >= 5 on every check.
- **Why:** the method's only rule against squashing everything to a point is that each
  number has the same spread and is uncorrelated with the others. A U-shaped copy of slant is
  uncorrelated with slant, so it obeys the rule and is cheaper than any genuinely new
  direction. Slant is by far the longest direction in MNIST 1s, so its copies fill every slot.
- **k = 10 with d >= 10 is not run:** a piece of 10 dots cannot span 10 or more directions.
- **Generated control (true answer: 5 knobs):** at 5 numbers the method recovers lean
  (R² 1.00), width (0.99) and left/right position (0.89-0.90), but not up/down position
  (0.05-0.23) or height (about 0). From the images directly, those two are 0.82 and 0.61, so
  the information is there. With 20 or 40 dots per piece, numbers 4 and 5 are copies of the
  first three (R² 0.97-0.99), so three directions are found: lean, width and left/right. The walking method gets up/down position to 0.52 at 10 numbers.
- **Double-layer check** (`verify.json`, `unfolding_suite/verify.py`): every layout reads
  `stacked` except the 15-number flat plane (`mild`, 6.4% flagged). At 15 numbers the locally
  linear layouts flag 54-63% of 1s (typically 2.1-2.3x too close), walking 31%. The check
  assumes the data were bent, not stretched. That is false here (lean moves many pixels,
  up/down very few), so read it only as a ranking of methods. It ranks them the same way
  as the strangers count does. Made by `verify_mnist_one_lle_flattening.py`.
  - **Floor, from the truth:** the generated 1s' true knobs, each stretched by how far it
    moves the image (fitted on nearest pairs), read 5% stacked at 2x (typically 1.68x). So
    about 5% at 2x is what a correct five-number answer looks like on images. Their knobs
    scaled only to their ranges read 100% stacked at 6x, so per-number stretch matters.
  - **Fair rescale, no truth** (`verify_per_number_stretch.json`): stretching each layout
    number separately to match nearby image distances barely changes anything. Generated at
    5 numbers: walking 57%, plane 78%, locally linear 80-99%. MNIST at 15 numbers: plane 4.6%
    (`mild`), walking 33%, locally linear 54-70%. So the method's same-spread-per-number output
    is not what makes it read `stacked`.
- **It is the pixels, not 2-D and not the method** (`check_mnist_one_lle_flat_controls.py`):
  on flat rectangles from 1:1 to 6:1 it finds both sides every time (short side R² 0.88-0.91).
  On the same generated 1s with 5 numbers and 20-dot pieces, given the dial settings or six ink
  averages instead of pixels, it finds all five dials at 0.97-0.98 with 5 of 5 numbers new. So
  the copies and the lost up/down and height come from how the 1s sit in pixel space: there a
  small piece is not flat (lean directions 10 degrees apart are perpendicular in pixels).
- **MNIST neighbourhoods are rough to begin with:** even an image's 10 most similar images
  rebuild it only to 19.2% (6.3% for the generated 1s), so no flattening of MNIST can rebuild
  better than that with this check.
