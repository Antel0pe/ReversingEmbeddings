# One five-dimensional principal manifold

Best fitted grid: **fine**, shape (3, 3, 3, 9, 65), 15,795 pixel-valued nodes. Exactly one manifold, with five continuous coordinates.

This is an elastic principal grid with continuous cell interpolation and projection, based on [Gorban and Zinovyev](https://arxiv.org/abs/cond-mat/0405648). It is an implementation variant, rather than a claim to reproduce a particular reference package.

## Decoder

The decoder locates the 5D grid cell containing coordinates u in [0,1]^5, then takes a weighted sum of its 32 corner nodes. For each corner, multiply its five one-axis interpolation weights. Weights sum to one. A node stores 283 active pixel values; the other 501 pixels are always zero. This is a joint tensor grid, so it models interactions between coordinates. It does not sum five independent curves. The decoder is continuous across shared cell faces, with possible slope changes at cell boundaries. No renderer, geometric stroke formula, or inverse-rendering solver occurs in the decoder.

## Fitting and knob information

All 14,389 covering images receive equal weight. Knob labels initialize u by subtracting each range minimum and dividing by its range width. Known dimension sets five axes. The grids prioritize width and lean based on their observed variation, while coarse center/height axes remain jointly present. These are heuristic allocations, not a proved optimal grid.

For fixed image coordinates, solve a sparse quadratic node fit: squared pixel reconstruction error + a bending penalty + weak stretching + a tiny numerical ridge. Bending is the squared second difference along each normalized grid axis, with spacing scaling. Then move image coordinates using bounded, damped Gauss–Newton steps and a line search that accepts only lower pixel error. The next node fit uses these updated coordinates.

Cold-start grids use penalty stages 0.01, 0.001, 0.0001. The fine grid interpolates the completed guided grid as its start, then uses two cycles at 0.001 and five at 0.0001, retaining the starting grid if an update worsens pixel capture. Each node solve uses pixel-column conjugate gradients with target relative residual 3e-6 and a finite iteration cap. The recorded solver history exposes any unconverged columns; this is an approximate numerical solve. Each coordinate refinement takes up to five steps per alternation, followed by twenty at the end. Nodes and coordinates are free after initialization; there is no knob-label loss or physical formula in updates. The box bounds and lattice connections preserve chart organization.

No samples are withheld from the covering-cloud fit. Additional uniform samples and joint midpoints check approximation between covering points. They are not used for selecting the winning grid; a preliminary guided-grid check was inspected during computation. All 32 corners are already part of the fitting lattice and are a boundary diagnostic. The encoder uses the fitted coordinates of the two nearest covering images in pixel distance as starts, takes up to twenty projection steps each, and chooses the smaller residual. If image L2 error exceeds 0.35, refinement uses eight nearest-image starts and up to sixty steps each, retaining only lower error. This policy also refines difficult covering points without changing learned nodes. Closest-point search is approximate and local.

## Results

| Model | Nodes | Covering-cloud capture | Pixel RMS |
|---|---:|---:|---:|
| Best five additive curves (513 nodes each) | 2565 | 99.733620% | 0.0087099 |
| One 5D manifold: uniform (5, 5, 5, 5, 5) | 3125 | 99.629164% | 0.0102767 |
| One 5D manifold: guided (3, 3, 3, 5, 33) | 4455 | 99.914917% | 0.0049225 |
| One 5D manifold: fine (3, 3, 3, 9, 65) | 15795 | 99.955447% | 0.0035620 |

| Additional points | Five curves | One 5D manifold |
|---|---:|---:|
| 16,384 additional uniform points | 99.466107% | 99.879293% |
| All 32 box corners | 99.857723% | 99.965508% |
| 2,048 joint midpoints | 99.388135% | 99.872925% |

Capture is 100 times one minus total squared reconstruction error divided by total squared distance to the evaluated cloud mean. All 784 unclipped coverage pixels enter the metric. Different clouds have different denominators. Five coordinates do not fix model complexity: grid nodes are separately reported.

## Limits and validation

All 514 checked fitted positions have five independent Jacobian columns at relative singular-value tolerance 1e-6. Extra samples and joint midpoints also have rank checks in results.json. This does not prove global injectivity, complete coverage, or exact agreement with the renderer.

The learned coordinates correlate with the initial physical knobs but drift during fitting. Feeding true knob fractions directly into the final decoder is a different operation and can have substantially larger error; those scores are recorded separately. The viewer uses learned image positions. Arbitrary slider combinations can leave the valid stroke family or coverage range. No clipping is used to improve reported scores.

The full fit is nonconvex. Five coordinates suffice in principle because the exact renderer has five inputs; that does not guarantee this finite, smooth grid and local optimizer find an exact manifold or global optimum. More nodes and alternate charts could reduce remaining approximation error.

## Reproduce

From the repository root, using the existing Python environment:
```sh
.venv/bin/python fit_generated_one_principal_manifold.py --phase uniform --cycles 5 --cg-iterations 200
.venv/bin/python fit_generated_one_principal_manifold.py --phase guided --cycles 7 --cg-iterations 200
.venv/bin/python fit_generated_one_principal_manifold.py --phase fine --cycles 5 --cg-iterations 160
.venv/bin/python fit_generated_one_principal_manifold.py --phase evaluate
.venv/bin/python fit_generated_one_principal_manifold.py --phase refine
.venv/bin/python make_generated_one_principal_manifold_view.py
```

The best NPZ contains nodes, grid shape, fitted image coordinates, active pixels, reference images, and knob labels. Only nodes, shape, and active pixels are needed to decode. decoder.json exports float32 nodes for the browser; measurements use full-precision fitted nodes.
