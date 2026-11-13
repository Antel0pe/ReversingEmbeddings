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
  the information is there. With 20 or 40 dots per piece, numbers 3-5 are copies of the first
  two (R² 0.97-0.99). The walking method gets up/down position to 0.52 at 10 numbers.
- **MNIST neighbourhoods are rough to begin with:** even an image's 10 most similar images
  rebuild it only to 19.2% (6.3% for the generated 1s), so no flattening of MNIST can rebuild
  better than that with this check.
