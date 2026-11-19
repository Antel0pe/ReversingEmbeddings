# The simplest learned decoder for MNIST 1s

`analyze_mnist_one_autoencoder.py` (parts `sweep`, `meaning`, `gaps`; `only=mnist` or
`only=generated`) and `make_mnist_one_autoencoder_figure.py` -> `autoencoder_summary.png`.

Model: plain autoencoder, 784 -> 256 -> k -> 256 -> 784 (ReLU, sigmoid output, squared
error, Adam, 200 epochs). The decoder half is a map from k numbers to a 28x28 image.
Data: 5,742 MNIST training 1s / 1,000 held out; control = 5,000 / 1,000 generated 1s
(`grey_ones.render`, five knobs uniform over `RANGES`). Error = whole-set relative error on
held-out images (`warp_charts.rel_error`). Seeds 0 and 1 agree within ~1 point.

## How many numbers

| k | generated 1s | MNIST 1s | PCA directions needed to match (MNIST) |
|---|---:|---:|---:|
| 1 | 38.1% | 46.2% | 3 |
| 3 | 10.9% | 29.8% | 13 |
| 5 | 6.0% | 24.2% | 23-25 |
| 8 | 4.4% | 20.2% | 38-39 |
| 12 | 4.4% | 17.9% | 50 |
| 20 | 4.3% | 16.0% | 62-63 |
| 30 | 4.4% | 15.0% | 70 |

Generated 1s: clear stop at 5-6, then flat (~4.4% is this small network's sharpness
floor, not missing knobs). MNIST: no stop; each doubling of k still buys 2-4 points, and
the train/held-out gap widens (2 points at k=5, 3 at k=30). Numbers line up with the warp
model (27.65% at 5, 15.6% at 20), so a generic network finds no shortcut the warp missed.

## What the numbers mean (k=5, main directions of the code cloud)

Generated control: the five directions are lean (0.98), width (0.96), cx (0.94), cy (0.75)
and height (linear 0.19, but read out quadratically at 0.77). So the method can find knobs.

MNIST: #1 slant (0.92); #2 is slant **squared** (0.84) -- the code cloud is a curved
banana and the second number is spent on that curve; #3-#5 are mixtures of ink amount,
left-right and width. Height and up-down are barely in the code (held-out R^2 0.25 and
0.32 even with a quadratic readout). At k=12 height and bend enter (0.81, 0.77).
Moving #2 alone, with #1 held typical, leaves the banana and decodes to a non-1 (panel B).

## Halfway between two real 1s (k=12)

Random pairs are 93% of an image apart. Decoded code-midpoints sit 0.274 from the nearest
real 1 (real held-out 1s: 0.251; plain pixel average: 0.437) with grey-ink share 0.44
(real 0.38; pixel average 0.76). Better than averaging, but visibly smudged for very
different pairs (panel C). On generated 1s the midpoint is as real as a reconstruction
(0.103-0.108 vs 0.100).
