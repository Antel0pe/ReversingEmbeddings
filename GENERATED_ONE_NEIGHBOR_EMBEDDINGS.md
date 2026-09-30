# Keeping generated “1” neighbors close in 3D

## Question and distance

Each controlled generated “1” is a 28 × 28 array of ink coverages. One pixel
has value 0 for no ink and 1 for full ink. We treat an image as a 784-number
point and define closeness by Euclidean distance over **all** 784 coverages:

\[
d_{784}(a,b)=\sqrt{\sum_{p=1}^{784}(a_p-b_p)^2}.
\]

The fit sees these pixels, not the five generator settings. It places each
sampled image at a 3D point. The score at neighborhood size \(K\) asks: of an
image's \(K\) closest other images by \(d_{784}\), what fraction remain among
its \(K\) closest 3D points? We average that fraction over all images. At
\(K=1\), it is the percentage whose *exact* nearest image remains nearest.
This set-overlap score does not require the retained neighbors to keep their
internal order, so we also measure average Spearman rank correlation among
each image's 30 true pixel neighbors.

## Results

The main sample has 4,329 images: 4,096 scrambled Sobol states, five one-knob
reference paths, and 32 corner states. The independent sample changes the
Sobol seed while keeping the sample size and ranges. Each method is **refitted**
on that second sample. The numbers below are percentages of true neighbors
retained in the 3D nearest-neighbor set.

| Method | Main K=1 | K=5 | K=15 | K=30 | Independent K=1 | K=5 | K=15 | K=30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t-SNE, perplexity 2 | 88.9 | 55.7 | 45.8 | 44.4 | 87.3 | 54.0 | 45.7 | 45.1 |
| t-SNE, perplexity 5 | 80.9 | 61.3 | 53.0 | 53.1 | 79.8 | 59.1 | 52.1 | 53.1 |
| t-SNE, perplexity 15 | 67.8 | 63.6 | 59.3 | 61.4 | 64.7 | 61.7 | 59.7 | 63.2 |
| Multi-scale t-SNE, 5 + 50 | 77.6 | 62.5 | 58.2 | 60.9 | 75.2 | 60.4 | 58.1 | 62.6 |
| Rank-refined Isomap | 48.7 | 58.5 | 66.6 | 71.9 | 44.8 | 57.5 | 66.8 | 72.6 |
| Original Isomap | 24.8 | 48.1 | 62.6 | 71.5 | 21.9 | 45.4 | 63.1 | 72.6 |

PCA and the previously generated UMAP layout were also checked on the main
sample. Their K=15 scores were 27.3% and 54.8%, respectively. The UMAP result
uses one parameter setting, so it is not a tuned UMAP benchmark.

The broader *ordering* of the true 30 neighbors also changes. Its average
Spearman correlation is 0.408 for t-SNE perplexity 2, 0.517 for multi-scale
t-SNE, 0.659 for rank-refined Isomap, and 0.603 for original Isomap on the main
sample. Thus the K-set winner at one scale need not be the winner for ordering
or another scale.

## What the methods do

- **Nearest-first:** t-SNE with perplexity 2 emphasizes extremely local pixel
  similarities. It retains 88.9% of exact closest matches on the main sample,
  while many of the next 5–30 neighbors move elsewhere in 3D.
- **Balanced local view:** multi-scale t-SNE averages its pixel-space affinity
  matrices at perplexities 5 and 50, then fits the result to 3D. In notation,
  \(P=0.5P_5+0.5P_{50}\). It was the best tested compromise under the declared
  score \(0.35R_1+0.30R_5+0.25R_{15}+0.10R_{30}\), where \(R_K\) is the
  neighbor-overlap fraction. Those weights express a preference for the very
  closest images; they are a choice, not a theorem. This is the viewer's
  default layout.
- **Broader-neighborhood view:** rank-refined Isomap begins with the original
  Isomap layout and adjusts the sampled 3D points using pixel-neighbor ranking
  constraints. It improves the main K=15 score from 62.6% to 66.6% and K=1
  from 24.8% to 48.7%. It is a direct layout of the sampled points; it does
  not define an out-of-sample coordinate function.

The chart `figures/generated_one_neighbor_comparison.png` shows overlap for
every K from 1 to 30. The interactive viewer
`figures/generated_one_3d.html` lets the reader rotate each layout, choose K,
color points by their local overlap, and compare the true nearest image with
the nearest point in 3D. `figures/generated_one_3d_overview.png` shows three
representative 3D layouts at one camera angle.

## Scope

These are finite-sample measurements using pixel Euclidean distance. A new
generated image is not automatically assigned a 3D coordinate by the saved
point layout; the second sample checks whether refitting reproduces the same
tradeoff. The full continuous five-dimensional generated family cannot be
embedded injectively and continuously in three-dimensional Euclidean space.
Accordingly, high neighbor overlap does not imply an exact 3D manifold.

Reproduce the exploration with `python benchmark_generated_one_neighbors.py`,
`python multiscale_generated_one_tsne.py`,
`python refine_generated_one_neighbors.py`,
`python replicate_generated_one_neighbors.py`,
`python make_generated_one_3d.py`, and
`python make_generated_one_neighbor_figure.py`. The multi-scale script uses a
private scikit-learn t-SNE optimizer API and may need adjustment with another
scikit-learn version.
