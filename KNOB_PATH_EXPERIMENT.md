# Knob paths: composition, coordinate criteria, and a local reconstruction experiment

## Scope and execution

This experiment concerns the continuous coverage-image family defined by the five controls in `grey_ones.py`.

The local command runner failed before launching a process. The numerical run therefore used the JavaScript execution tool and a float64 mirror of the 16-subrow coverage rule read earlier in this chat. The repository's Python renderer was **not executed in this run**.

A separately written dense implementation agreed with the faster mirror at every pixel on 100 random states and all 32 corners of the allowed control box: maximum difference 0 and zero float32 pixel mismatches. Full-row ink totals and total ink = width × height were checked to approximately 6e-14 or better. These checks support the mirror's internal consistency; they are not a new cross-runtime comparison against Python.

Saved code: [knob_path_experiment.mjs](knob_path_experiment.mjs). Detailed results: [knob_path_results.json](knob_path_results.json). Run `node knob_path_experiment.mjs` to regenerate the JSON.

## 1. Two separate changes have an exact relationship to the joint change

Let F(w,a) denote the full image with width w and lean angle a, with the other three controls fixed. For increments u and v, define:

- Width change W = F(w+u,a) − F(w,a).
- Lean change L = F(w,a+v) − F(w,a).
- Joint change J = F(w+u,a+v) − F(w,a).
- Interaction C = F(w+u,a+v) − F(w+u,a) − F(w,a+v) + F(w,a).

The exact relationship is J = W + L + C.

C is also the difference between the lean displacement measured after changing width and the lean displacement measured at the starting image. Equally, it is the corresponding difference in width displacements after changing lean.

The step can be computed exactly as width change at the initial image plus lean change at the widened image, or lean change at the initial image plus width change at the leaned image. Both arrive at F(w+u,a+v). Their intermediate image paths differ.

At the tested baseline (horizontal center 14.43, vertical center 14.62, height 19.73, width 3.17, lean 17.3°):

| Width increment | Lean increment | Error from adding starting-image displacements, relative to joint pixel L2 | Updated second change: endpoint error |
| --- | --- | --- | --- |
| 0.2 px | 5° | 17.758% | 0 |
| 0.02 px | 0.5° | 2.813% | 0 |
| 0.002 px | 0.05° | 0.504% | 0 |
| 0.0002 px | 0.005° | About 1.4e-11 percent | 0 |

The first pair had a joint change of 2.953923 in pixel L2, with a frozen-sum error of 0.524558. The lean displacement turned by about 10.307° after the width increment.

This is a finite-displacement comparison. It does not claim that the instantaneous derivative turns by exactly the same amount.

## 2. Some interactions come from formula boundaries; others exist inside a formula patch

With vertical center and height fixed, vertical strip coverage is fixed. Horizontal overlap is piecewise affine in width and lean slope s = tan(a). Consequently, width and lean slope combine additively inside one fixed overlap-formula region. Their finite interaction appears when the rectangle of control changes crosses one or more overlap boundaries.

Using angle rather than slope changes the speed along the lean path, but within one overlap region the image still lies on a straight segment as lean varies.

Width and height have a different mechanism. Their horizontal and vertical coverages multiply. Within a suitable formula region:

F(w+u,h+v) = A + B u + C v + D u v.

A, B, C, D are whole pixel arrays. The mixed term persists inside the patch. In the tested example, reducing both steps by a factor of ten reduced the interaction by a factor of one hundred:

| Width increment | Height increment | Interaction pixel L2 |
| --- | --- | --- |
| 0.02 px | 0.03 px | 3.0e-4 |
| 0.002 px | 0.003 px | 3.0e-6 |
| 0.0002 px | 0.0003 px | 3.0e-8 |

All single-control paths should not be described as piecewise straight in their original control coordinates. For example, changing vertical center changes both the cap coverage and the horizontal centerline position, and can produce quadratic segments.

There is a simpler common description. Use top edge, bottom edge, horizontal line intercept, width, and lean slope as the five coordinates. Inside a fixed overlap region, every pixel is a polynomial of degree at most two in those coordinates. Across regions, those formulas join continuously, although their derivatives can jump.

## 3. A knob is a family of curves across the manifold

Five curves drawn through one chosen image only specify five routes from that image. A full coordinate system supplies a width-like curve, lean-like curve, and so on through every state.

On a smooth patch, each knob has a direction field: an instantaneous pixel-change vector that depends on the current image or coordinate address. Simultaneous infinitesimal changes add their current direction vectors. Finite changes follow those changing vectors, and can contain interactions.

A useful set of independent coordinate knobs needs:

1. **Valid states:** permitted coordinate addresses decode to members of the target image family, within the stated tolerance.
2. **Coverage:** every target image has a coordinate address.
3. **Unique and stable addresses:** different target images do not collapse to one address; nearby images have nearby addresses.
4. **Independent local directions:** the coordinate directions span all local changes and do not omit an independent motion. For five independent coordinates on a regular patch, the decoder derivative has rank five.
5. **Consistent combinations:** changing coordinates in different orders reaches the same coordinate address and image. Moving a knob back reverses its coordinate change. On smooth patches, the coordinate vector fields commute.
6. **Consistent joins:** if multiple local charts are used, overlapping charts describe the same images through reversible coordinate changes. Piecewise paths may have corners while remaining continuous.

Coverage alone permits memorization or a traversal that visits many points without supplying stable, independent coordinates.

As a constructed counterexample, let one tangent motion change lean normally. Let a second change width with speed depending on lean. Both remain on the real manifold and are locally independent. A loop of +5° lean, a nominal +0.2 width step, −5° lean, and a nominal −0.2 width step drifted by 0.088889 in actual width, giving a pixel L2 difference of 0.260953. The ordinary coordinate-knob loop returned exactly to its starting image.

Those rescaled motions therefore are not independent coordinate increments with the specified step calibration. They could be redefined to obtain commuting coordinates; the example does not say that the manifold lacks coordinates.

The mathematical statement about commuting fields is local and assumes smoothness. At the renderer's formula boundaries, use finite coordinate updates and continuous joins instead of asserting a classical smooth derivative.

Primary source: [MIT notes on vector fields and integrability](https://math.mit.edu/classes/18.966/2014SP/965/class4.pdf).

## 4. Pixel-only local curved reconstruction

The fitting step received only images. For each supplied patch:

- 256 training images; 128 independently sampled held-out images.
- PCA supplies candidate coordinates from the training pixels.
- A least-squares decoder uses constants, the coordinates, their squares, and their pairwise products.
- Compare four-coordinate quadratic decoding, five-coordinate linear decoding, and five-coordinate quadratic decoding.
- No control values enter PCA, polynomial fitting, encoding, or decoding.

The nonlinear decoder has the form:

image(z) = A + sum_i B_i z_i + sum_{i<=j} C_ij z_i z_j.

Here z contains five learned coordinate values; A, B_i, and C_ij are fitted pixel arrays. Its direction for coordinate i changes when other coordinates change. The interaction terms encode that dependence.

Each patch is centered at the baseline above. A patch radius of r means each generating control was sampled independently within ±r times its full allowed range. That describes the data sampling, not information supplied to the fitter.

| Patch radius as fraction of each control's full range | 5-coordinate linear reconstruction | 5-coordinate quadratic reconstruction | Worst quadratic pixel error |
| --- | --- | --- | --- |
| ±3% | 99.08837% | 99.95938% | 0.014051 |
| ±1% | 99.71565% | 99.99159% | 0.0027715 |
| ±0.3% | 99.96969% | 99.99821% | 0.00024226 |
| ±0.1% | 99.99360% | 99.99874% | 0.000091307 |
| ±0.01% | 99.99999985% | Residual near numerical precision | 2.37e-11 |

At ±0.01%, quadratic held-out RMS image error was 5.58e-12 in pixel L2. A displayed score of 100% reflects numerical rounding, not a proof of an exact learned rule.

Four-coordinate quadratic reconstruction remained around 99.598% as the patch became very small. Meanwhile the fifth covariance eigenvalue stayed around 0.00446 times the largest; the sixth was approximately 1.07e-9 at ±0.01%, and shrank by another factor of 100 when the patch radius shrank by ten. This is direct local evidence distinguishing an independent fifth variation from higher-order bending.

**Limits:** dimension five was supplied as a candidate, and the patches were supplied by controlled sampling. This run did not discover neighborhoods from a global unordered point cloud, identify the original semantic controls, stitch the patches together, certify all between-sample addresses, or reconstruct the entire allowed family with a single learned five-coordinate map. Small-patch success is not that global result. The tiny residuals refer to the float64 coverage calculation before float32 image storage.

The relevant algorithmic family is local chart reconstruction followed by chart alignment; [Local Tangent Space Alignment](https://arxiv.org/abs/cs/0212008) is an established method for estimating and aligning local coordinates. This experiment used a local polynomial decoder rather than implementing LTSA globally.

