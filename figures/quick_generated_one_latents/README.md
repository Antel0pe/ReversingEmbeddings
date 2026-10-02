# A small image-only bottleneck test on generated 1s

**Question.** How readily can a generic nonlinear model learn a compact image
code for the full controlled five-knob generated-1 family? This tests image
reconstruction from learned coordinates. It does not train on knob labels or
assert that a learned coordinate equals a physical knob.

The data are 4,096 distinct generated 28×28 coverage images from one scrambled
five-dimensional Sobol sequence, spread over the full allowed knob box.
The first 3,072 images train each method; 512 select the autoencoder checkpoint;
the final 512 are held out for every reported score. Image pixel values are
fractions of ink coverage from zero to one. There are 266 pixels that vary in
the training set; all other pixels are zero in the held-out set as well.

The common score is

`100 × (1 − sum_test ||image − reconstruction||² / sum_test ||image − training_mean||²)`.

Thus 100% means exact pixel reconstruction. This score is computed across
all 784 pixels of unseen images. The known five knob settings plus the shared
renderer exactly reconstruct the test images, providing a verified 100%
reference. The learned methods receive only pixel arrays.

## Fixed short sweep

PCA uses ordinary centered pixel vectors. The nonlinear method is a small
fully connected autoencoder with widths `266 → 128 → k → 128 → 266`, ReLU
hidden layers, and a linear bottleneck of `k` values. It uses up to 80 epochs,
batch size 256, and one fixed initialization per width. All widths use the
same training, validation, and test images.

| Coordinates `k` | PCA: unseen pixel variance reconstructed | Autoencoder, up to 80 epochs |
| ---: | ---: | ---: |
| 3 | 72.08% | 89.69% |
| 4 | 76.69% | 95.36% |
| 5 | 80.26% | 95.55% |
| 6 | 82.72% | 96.96% |
| 8 | 86.25% | 97.71% |
| 12 | 90.59% | 98.13% |
| 16 | 93.20% | 98.39% |

Most autoencoder validation scores were still improving near epoch 80. The
scores compare what this simple model learned within the small budget, not
the best possible result for each coordinate count.

## Targeted longer check

The same architecture, initialization, and data were used for a targeted
240-epoch check. Only selected widths were rerun. Validation selected the
checkpoint; the test set remained untouched until scoring.

| Coordinates `k` | Unseen pixel variance reconstructed | 95th-percentile image L2 error | 95th-percentile largest pixel error |
| ---: | ---: | ---: | ---: |
| 5 | 97.745% | 0.994 | 0.457 |
| 8 | 98.932% | 0.693 | 0.284 |
| 9 | 98.935% | 0.676 | 0.289 |
| 10 | 98.908% | 0.684 | 0.281 |
| 11 | 99.018% | 0.653 | 0.282 |
| 12 | 99.030% | 0.661 | 0.280 |
| 16 | 99.156% | 0.609 | 0.248 |

Here, 11 was the first **tested** width to pass 99%. This is a result for
one simple architecture and one initialization per width, not a lower bound
on the nonlinear dimension needed. The slight nonmonotonicity across widths
reflects finite training and optimization variability. Most validation
scores were still improving near epoch 240, so a better model or additional
training may improve the five-coordinate result.

On the same held-out images, PCA first reached 95% with 21 coordinates,
99% with 45, 99.9% with 93, and 99.99% with 152. Those are observed test
reconstruction thresholds, not an assertion about the full continuous
manifold's linear rank.

The image examples in [the follow-up figure](quick_latent_followup.png) show
why a number near 99% does not mean visually or pixelwise exact. The learned
images still have gray edge errors; at 11 coordinates, the largest error in
a typical high-error test image (95th percentile) is about 0.28 coverage.
The [first sweep figure](quick_latent_sweep.png) shows the 80-epoch results.

These runs test reconstruction, not preservation of distances, paths, or
one-to-one recovery of the named physical knobs. The held-out split checks
generalization to new settings from the same generator and knob range; it
does not test real MNIST digits.

Reproduce with `python quick_generated_one_latents.py` followed by
`python quick_generated_one_latents_followup.py`. The exact measurements,
validation histories, and settings are in `results.json` and
`followup_results.json`.
