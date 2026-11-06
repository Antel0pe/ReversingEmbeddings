# Efficient Pattern Recognition Using a New Transformation Distance

https://proceedings.neurips.cc/paper_files/paper/1992/hash/26408ffa703a72e8ac0117e74ad46f33-Abstract.html

Patrice Simard, Yann Le Cun, John Denker (AT&T Bell Laboratories) ·
NIPS 5, 1992, pp. 50-58

**Read with:** 2026-11-05 · **Depth reached:** L1 · **Source read:** full text,
extracted from the NeurIPS PDF (figures not seen)

> **Which Simard paper is this.** The 1998 tutorial chapter *Transformation
> Invariance in Pattern Recognition — Tangent Distance and Tangent Propagation*
> (LNCS 1524, pp. 239-274, adds Victorri as co-author) is the expanded version
> and the one originally queued. Its only open copy is on HAL, which is behind
> bot protection and refused every fetch. This file is the **1992 original**,
> which carries the whole idea in nine pages. Grab the HAL PDF in a browser if
> the longer treatment is wanted.

---

## What it says (my notes)

<!-- Yours. -->

---

## Claude's running summary

### L1 — the gist

**The question.** To recognise a handwritten digit, the obvious move is to
compare it against stored examples pixel by pixel and pick the closest. This
works badly: shift a 3 by two pixels and it becomes, by that measure, further
from itself than from a 4. So the question is how to measure the distance
between two images in a way that ignores the differences that do not matter.

**The answer.** Write down the handful of ways a digit is allowed to move —
they list **seven**: left/right, up/down, rotate, scale, two kinds of shear,
and getting thicker or thinner. At any image, each of those gives you a
direction: a picture of which pixels change if you apply a little of that
motion. Seven directions span a small flat patch through the image. Now,
instead of measuring from image to image, measure from **patch to patch**:
the shortest distance between the two flat patches. That is *tangent distance*,
and it falls out of a small least-squares problem.

**The surprising thing.** This beat every neural network of its era, and tied
human accuracy, by **not learning anything**. 2.6% error on US Postal Service
digits against 3.3% for the best net of the day, 5.9% for plain pixel
comparison, and 2.5% for two humans labelling the same test set by hand. In the
1992 NIST competition — 45 algorithms, 26 companies, 7 countries — it finished
first among everyone who used the supplied training set. The knowledge is in
the seven hand-written motions, not in fitted parameters.

**Why it lands on your desk.** Their six geometric motions are chosen, in their
words, "to span the set of all possible linear coordinate transforms in the
image plane" — your six affine fields. Their seventh is line thickness — your
gain field. And their parenthetical is the warp-versus-pixel distinction
written down in 1992: those transforms are linear in the image plane but
"correspond to highly non-linear transforms in pixel space."

### Relevance to this repo

**A confirmation, nearly verbatim [paper, §3.1].** "Seven such image
transformations were identified: X and Y translations, rotation, scaling, two
hyperbolic transformations (which can generate shearing and squeezing), and
line thickening or thinning. **The first six transformations were chosen to
span the set of all possible linear coordinate transforms in the image plane
(nevertheless, they correspond to highly non-linear transforms in pixel
space).**" That is `figures/generated_one_gradient_fields/`'s six affine fields
and `figures/generated_one_warp_space/`'s flat-in-warp / curved-in-pixels
finding, in one sentence, in 1992. **[mine]** Note the repo uses five of the six
because a vertical stroke is blind to vertical shear; Simard keeps all six
because digits are not all 1s.

**A confirmation — the gain field is real but secondary [paper, §4].**
"For character recognition, thickness was one of the most important
transformations, reducing the error rate from 3.3% to 2.6%." So thickness is
worth about 0.7 points out of 3.3 in their setting. `warp_charts.py` found the
genuinely appearance-only part — ink density along the stroke — worth about a
point, and slightly negative on the generated family. Two independent estimates
that it is a real but minor knob. **[check]**

**A confirmation — pixel humps, as a footnote [paper, §1 fn.1].** "If the image
of a '3' is translated vertically upward, the middle top pixel will oscillate
from black to white three times." That is
`figures/generated_one_pixel_humps/`'s whole subject, noted in passing as the
reason the pixel representation is awkward. **[mine]**

**A method to steal — tangent distance as a replacement for pixel L2
[paper, §2].** Compare two images by the minimum distance between their tangent
planes, not between the images. Both planes move, so it is two-sided; the
optimum is a small linear system (their eqs. 8-11), and the per-pair cost is
about n(m_E+1)(m_P+1) + 3(m_E³ + m_P³) multiply-adds with the plane bases
precomputed and stored. This repo measures everything in raw pixel L2
(`AGENTS.md`, Geometry conventions). Tangent distance is the drop-in that makes
"how far apart are these two 1s" mean something closer to "how different are
their knobs". **[mine]**

**A method to steal — the Newton iteration, which is `fit_chart` [paper, §4].**
"Tangent distance can be viewed as one iteration of a Newton-type algorithm
which finds the points of minimum distance on the true transformation
manifolds." They then spell out the loop: read off the fitted coefficient,
*actually apply* that much real transformation to the image (a resampling),
recompute the tangent vectors at the new image, repeat. That is exactly the
rounds in `fit_chart`, and it is Wakin's multiscale Newton **without the scale
ladder**. Caveat they give: "the tangent distance needs to be normalized for
this iteration process to work."

**A diagnostic to steal — the validity-radius curve [paper, §3, Fig. 4].**
Translate one image by k pixels and plot both Euclidean and tangent distance
against k. Euclidean rises linearly from k = 0; tangent distance stays near
zero out to about **2 pixels** and only then climbs. That single curve measures
how far the linear model stays usable — a number this repo never measures
directly, and the one that would say whether MNIST's 24.8%-of-an-image nearest
neighbour is inside or outside the usable radius. **[mine]** They also note the
radius can be widened by smoothing, at the cost of blurring away real features.

**A warning — the scaling collapse [paper, §4].** If scale is one of your
motions, the minimum distance between two patches can be achieved by shrinking
both images toward a point. "The linear approximation of the scaling
transformation does not reach this extreme, but still yields a slight
degradation of the performance." Fix: normalise the tangent distance by the
norm of the vectors; it took USPS from 2.6% to 2.4%. `warp_charts.py` has
height and width knobs, so this failure mode is live here. **[check]**

**A number to compare against — how much blur [paper, §3].** "Smoothing was
done by convolution with a Gaussian of standard deviation σ = 0.75. This value,
which was estimated visually, turned out to be nearly optimal (but not
critical)." One blur level, picked by eye, no ladder — against Wakin's
elaborate scale sequence. The two are not in conflict: Simard needed about two
pixels of validity starting from roughly-aligned digits, Wakin needed to
converge to 1e-8 from a bad start. **[mine]** Worth noting σ = 0.75 px is tiny
compared with the blurs in Wakin's ladder.

**Measured: why high dimension makes the method trustworthy [paper, §4].**
The paper disposes of a safety constraint in one line — "tangent planes are
mostly orthogonal in high dimensional space and the norms of ‖a_E‖ and ‖a_P‖
are already small" — and that line is the reason the whole method works.
Checked here by simulation and on `grey_ones`:

| setting | a random gap shrinks to |
| --- | --- |
| two random 1-D patches in n = 3 | 46.3% |
| two random 5-D patches in n = 16 | 60.1% |
| two random 5-D patches in n = 64 | 91.7% |
| two random 5-D patches in n = 256 | 98.0% |
| two random 5-D patches in n = 784 | 99.4% |

Typical angle between two random directions: 60.6° at n = 3, 87.2° at n = 256,
88.4° at n = 784. On real generated 1s with the five knob tangents: a pair
differing by cx + 0.3 px collapses 1.875 → **0.000**; lean + 2° collapses
1.231 → **0.000**; all five knobs nudged, 2.167 → 0.126 (5.8% left); a far pair
(cx + 3 px, lean + 25°) keeps 89.7%; the same 1 with its pixels shuffled keeps
93.2% and matched random noise keeps 94.2%. So **0% left for a genuinely
related pair against ~90% for anything else** — and the margin exists because
10 ≪ 784. Real-image controls keep 90-94% rather than the simulation's 99.4%
because real differences are not random directions. **[mine]** Script:
`tangent_room.py` (scratchpad, not committed).

**A named concept.** *Tangent distance*; *tangent propagation* (the sibling
method, which pushes the same invariances into a network's training signal);
and the framing that carries the whole paper — "a priori knowledge can be very
effectively expressed in the form of tangent vectors."

**The honest limit.** This is a **classifier** paper. They never recover the
knob values; they only need distances, so the fitted coefficients are a
by-product they discard. The bridge to this repo's goal is the §4 Newton
iteration, which they describe and do not pursue.

### Spin-off experiment — knob subspace discovered from images alone

User's proposal (2026-11-05): use the patch machinery as a *discovery* tool
rather than a ruler — find a consistent knob without knowing the real ones.
The trap is circularity: Simard builds tangent vectors by nudging known
parameters. The fix is to estimate the patch by **local PCA of pixel
neighbours** (Wakin's §3.2 definition) and then convert each patch direction
into an **affine motion** (6 numbers, via the image's own gradients), because
motion coefficients are comparable across images while pixel directions are
not. Pipeline sees no knob value after rendering.

Sharp test: the 5-D patch inside the 6-D motion space has a **1-D orthogonal
complement** — the one motion the family cannot perform. Is it the same at
every image, and is it vertical shear?

| run | neighbour radius | leftover agreement across 14 images | distance from pure vertical shear |
| --- | --- | --- | --- |
| `grey_ones`, full box | 0.774 | median 17.7°, max 63.4° | 22.7° |
| `smooth_ones`, full box | 0.940 | median 24.4°, max 75.6° | 27.6° |
| `grey_ones`, quarter box | 0.196 | **median 6.8°, max 15.3°** | 18.3° |
| `smooth_ones`, quarter box | 0.238 | **median 6.3°, max 16.7°** | 23.5° |

Random 1-D directions in 6-D sit **71.9°** apart, so 6.3° is decisive.
Leftover composition, averaged: vertical shear 0.91, width 0.40, lean 0.07,
everything else under 0.03. Image norm ~6.9 for scale.

**Reading.** From pixels only, the method recovers that this family's motions
are 5 of the 6 affine ones and names the missing one as vertical shear —
`smooth_ones.py`'s docstring claim, rediscovered without being told. Density
is what matters, not the generator: quartering the box cuts disagreement from
~20° to ~6°, while grey vs smooth barely moves it. The ~30% patch residual and
the 18-23° offset from pure shear are both the `np.gradient` floor already
measured in `figures/generated_one_motion_discovery/` (23% on coverage, 16.7%
on smooth), not a missing motion. **[mine]**

**Why this matters beyond the result.** Motion coefficients are comparable
across images, so a basis fixed once at a reference image is valid everywhere —
which dissolves the transport/matching problem that
`analyze_generated_one_direction_history.py` and
`check_generated_one_direction_commutators.py` ran into in the pixel basis.
Any basis of the consistent 5-D subspace is a legitimate knob set, which is
exactly the user's criterion. Not yet done: integrating that frame into
coordinates and running the held-out decoder test `AGENTS.md` requires.
Script: `knob_discovery.py` (scratchpad, not committed).

### Open questions

- What is the validity radius (their Fig. 4 curve) for `grey_ones`, and for
  MNIST 1s? If MNIST's nearest neighbour sits outside it, that is a clean
  explanation for why linearised motion discovery failed there.
- Does the scaling collapse bite `warp_charts.py`, and does their
  normalisation fix it?
- They kept six geometric motions; the repo uses five. Does adding vertical
  shear back help on MNIST 1s, which bend?
- Simard needed one blur level; Wakin needs a ladder. Which regime is
  `fit_chart` in?

### Claims to check

- [ ] Tangent distance beats pixel L2 as the metric in this repo's shell and
      neighbour experiments. **[check]**
- [ ] Thickness worth ~0.7/3.3 points there vs ~1 point here — same knob,
      consistent size. **[check]**
- [ ] Scaling collapse is present in `warp_charts.py`'s height/width knobs.
      **[check]**

---

## Links

- Sibling: Simard, Victorri, LeCun & Denker, "Tangent Prop — A formalism for
  specifying selected invariances in an adaptive network", NIPS 4, 1992,
  pp. 895-903 **[paper, refs]**
- Expanded tutorial: "Transformation Invariance in Pattern Recognition —
  Tangent Distance and Tangent Propagation", LNCS 1524, 1998, pp. 239-274
  (HAL copy is bot-blocked)
- Follow-up that *learns* the tangent vectors: Hastie, Simard & Säckinger,
  NIPS 1994 — see the reading queue in `README.md`
- Read alongside: [[multiscale-structure-of-non-differentiable-image-manifolds]]
  — same Newton step, with a scale ladder bolted on and a proof of why one is
  needed.
- Repo: `figures/generated_one_gradient_fields/README.md`,
  `figures/generated_one_warp_space/README.md`,
  `figures/generated_one_pixel_humps/README.md`, `warp_charts.py`
