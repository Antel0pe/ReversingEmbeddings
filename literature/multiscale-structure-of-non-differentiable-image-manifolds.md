# The Multiscale Structure of Non-Differentiable Image Manifolds

https://doi.org/10.1117/12.617822 — author copy: https://people.mines.edu/mwakin/wp-content/uploads/sites/262/2024/06/wakin-spie-iams.pdf

Michael B. Wakin, David L. Donoho, Hyeokho Choi, Richard G. Baraniuk ·
Wavelets XI, SPIE Optics & Photonics, San Diego, July 2005 · doi:10.1117/12.617822

**Read with:** 2026-11-03 to 2026-11-04 · **Depth reached:** L5 (claim reproduced) · **Source read:** full text pasted by the user (figures described, not seen)

---

## What it says (my notes)

*My reading, in my own words, 2026-11-04. These are the parts that landed, not
a full account of the paper.*

**The setup.** Every image is a point in a space with one dimension per pixel.
Turn a knob (slide a disk sideways, say) and you get a different image, so the
whole family of images traces out a surface in that space. The standard
assumption is that this surface is smooth — that at any point there is a flat
plane hugging it that you can use to navigate. The paper asks whether that
plane is really there, and the answer is no, nowhere.

**Why sharp edges break it.** Move an edge sideways by a tiny amount. The
pixels that change are a thin sliver along the edge, and each one flips by a
*full* unit of ink. So the picture distance works out to the square root of the
sliver's area — roughly √(step). Shrink the step by 100 and the picture only
moves 10 times less. Divide picture change by knob change and you get 1/√step,
which runs away to infinity as the step shrinks.

**What "non-differentiable" means here.** It is the **infinite derivative**
kind, not the jumping-around kind. Like √x at zero: the curve is continuous,
you can draw it, but it goes straight up, so there is no slope. My instinct
that this is "basically a divide by zero" was right — the ratio really is
√δ/δ = 1/√δ. The paper's §5 occlusion cusps are the *other* kind, where the
left and right derivatives both exist but disagree: a corner, not a cliff. Two
different failures.

**Why nowhere, and why relabelling cannot fix it.** Every knob setting moves an
edge, so there is no good point to stand on — √x is only bad at zero, this is
bad everywhere. And chopping a trip into N steps gives a measured total of
about √N, which grows without bound, so *every path through the family has
infinite length*. Relabelling the knob just moves the tick marks along a path;
it cannot shorten an infinite one. (Important caveat I only got at the end:
this impossibility is about relabelling θ while measuring distance between
images pixel by pixel. See my idea below.)

**Twisting off — the part that took me longest.** This is **not** ordinary
curvature. Curvature is: fix the step size, move your *location*, and the plane
rotates — like how "up" points differently in Canada than at the equator.
Twisting is: fix your *location*, change the **step size**, and the plane
rotates. There is no Earth analogy, because on Earth measuring "up" with a 1 km
or a 1 m baseline gives the same answer.

Concretely: ask "which way does *move-right* push the picture?" by rendering
the disk at 0 and at ε and subtracting. You get a crescent **ε wide**. Ask with
ε/2 and you get a crescent half as wide, sitting inside half of the first one.
As pictures they barely overlap, so as directions they are far apart — and
halving again reproduces exactly the same geometry, so the disagreement never
shrinks. On a smooth surface the answers close in on one direction and *that*
is the tangent. Here they never close in, so there is nothing to call the
tangent.

Measured on an antialiased disk: 44.4°, 43.8°, 42.3°, 39.3° between each step
and half that step — basically constant. Compared against the ε = 1/4 answer it
accumulates 44.4° → 59.7° → 69.0° → 75.2°, heading for 90°, so the fine-step
answer is nearly *perpendicular* to the coarse-step one. A different answer,
not a refinement. Blur the same disk and the same numbers collapse (34 → 23 →
13 → 6.7°), i.e. a tangent plane appears.

**Dimension count — the thing I kept getting wrong.** At any single scale the
plane is still exactly d-dimensional; nothing grows. But each scale hands you a
direction nearly at right angles to the previous ones, so stacking ten scales
gives roughly 20 near-independent directions. The *union across scales* is
unbounded; each rung is not.

**Blur fixes it, and why.** Blurring turns the cliff into a ramp of width s. A
ramp rises 1 over a run of s, so the slope is at most about 1/s — finite, with
a ceiling set by how much you blurred. It will never shoot up from zero to
something. A cliff has no ceiling. So each blur level gives you a genuinely
smooth surface with real tangent planes, and the trade is: more blur is easier
to navigate but the answer is more wrong, because blurring destroyed the fine
detail.

**The algorithm, and what the twist analysis is for.** Recover the knobs from a
single image by blurring heavily, taking one step, blurring less, stepping
again — 1/2 → 1/4 → 1/16 → 1/256. No training, no network, no dataset; a fixed
four-step recipe. The twist analysis is the rule for how fast you are allowed
to sharpen: given the accuracy you have right now, how much blur you can safely
remove next. It is there to stop you going off the deep end into a landscape
your current guess cannot navigate. One step at scale s buys accuracy about s²,
which is exactly why the scale sequence squares each round.

### My idea — the pixel basis is the wrong basis

If someone shows me an image and asks how to widen it, I spread my fingers. I
am not naming 784 pixel values and I am not pointing down the edge pixel by
pixel. I am naming **one rule**: push each pixel outward, proportionally to how
far it is from the centre. Column 5 and column 6 are not two separate facts —
they are the same rule evaluated in two places.

So a "direction" should be a rule applied to an image, not a fixed list of
pixel changes. The crescent at one scale and the crescent at another scale are
90° apart *as pixel vectors* and yet they are obviously the same gesture. The
gesture is the real object; the pixel pattern is just what that gesture happens
to do to this particular picture.

Written out, the rule is a motion field and the pixel direction is that field
multiplied by the image's own gradient:

$$\tau \;=\; -\,v \cdot \nabla I$$

v is what I meant with my fingers and is the same at every image. ∇I is where
*this* picture has edges. Their product is what the pixels do, which is why the
pixel answer changes completely while the intended direction never changed.

The five knobs as five rules: cx → (1, 0); cy → (0, 1); width → (x − cx, 0);
height → (0, y − cy); lean → (−(y − cy), 0). There are six linear rules of this
kind and I use five — the unused one is vertical shear, which is why there are
five knobs and not six.

**Why this matters for the paper.** Its impossibility result relabels θ while
keeping distance measured pixel by pixel; it even says the knob space is
*isometric* to the picture space, so the roughness is baked into that distance.
Changing what an image is *represented as* — a warp rather than a list of ink
values — is a different move, and the theorem does not forbid it. That is why
`figures/generated_one_warp_space/` is flat to machine zero past the fifth
component while pixel space still leaves 8% at fifty directions.

**The honest limit.** This relocates the roughness rather than removing it. At
a true cliff, ∇I is a spike, so the pixel picture τ is still the infinitely
thin crescent. What changes is where the tidy object lives: **the knobs are
smooth as motions and rough as pictures.** The motion field sits in a fixed
six-dimensional space however sharp the edges are. The rules do not have to be
differentiable, but they do have to form a small fixed family — if every image
needed its own unrelated rule there would be nothing to recover.

**Next thing to try:** the paper's blur ladder run on top of a warp fit —
project the error onto the *motion fields* rather than the pixel patterns. That
is the combination neither the paper nor this repo has tried.

---

## Claude's running summary

### L1 — the gist

**The question.** Take a simple shape — a disk — and slide it around a picture.
Each picture is one point in a space with one dimension per pixel, and the
whole family of pictures traces out a surface in that space. Everyone working
on these problems assumes the surface is smooth, so that calculus applies: at
any point there is a flat plane that hugs the surface, and you can use that
plane to navigate. The paper asks whether that plane actually exists.

**The answer.** It does not. Not at one bad point — *nowhere*. A family of
images with sharp edges forms a surface that has no tangent plane anywhere on
it, and no clever relabelling of the knobs can fix that **[paper, §3.1]**. The
reason is almost embarrassingly simple: moving an edge by a tiny amount changes
a thin sliver of pixels by a *large* amount, so the picture changes fast
compared to the knob, and keeps changing faster the smaller the step you take.
Distance in picture space grows like the **square root** of the knob step
instead of proportionally to it, and a square root has infinite slope at zero
**[paper, §3.1]**.

**The surprising thing.** When you fit a best flat plane to a small
neighbourhood anyway, and then shrink the neighbourhood, the plane does not
settle down. It keeps rotating into fresh directions, by the *same* angle every
time you halve the radius — about 26.6° per halving for the disk, which they
derive in closed form **[paper, §4.2]**. So the surface is not "smooth but hard
to measure." It genuinely has a different flat approximation at every scale,
and scale becomes a real coordinate, exactly like it is in wavelet analysis.

**Why it matters here.** The last third of the paper turns the obstruction into
an algorithm, and that algorithm does what this repo wants: it recovers the
generating knobs from a single unlabelled image, with no training, no network,
and no learned latent space. Blur both the family and the target image by a lot,
take one Newton step on the blurred version where tangents *do* exist, then blur
less and step again. On a five-knob ellipse — two positions, one rotation, two
axis lengths — it lands the knobs to about 1e-5 in four steps **[paper, §6.4.2,
Table 3]**. It is the five-knob inversion problem, solved in 2005, by geometry.

### L3 — the machinery, in plain terms

**The sliver, which is the whole paper.** Move an edge sideways by a small
distance δ. The pixels that change are a sliver along the edge: length L, width
δ. Each one flips by a full unit of ink. Distance between the two pictures is
the square root of the changed area, so picture-distance ≈ √(L·δ) **[paper,
§3.1; arithmetic mine]**. That is the square root, and everything else follows.

**Non-differentiability here means infinite derivative, not a jump.** Speed is
picture-distance divided by knob-step, √δ/δ = 1/√δ, which grows without bound
as δ → 0. Concretely with L = 1: δ = 0.01 gives speed 10, δ = 0.0001 gives 100,
δ = 1e-6 gives 1000. The paper's §5 occlusion cusps are the *other* kind — left
and right derivatives both exist but differ, a corner rather than a cliff. Two
distinct failures, often conflated.

**Why no relabelling of the knobs fixes it [paper, §3.1].** Chop a trip through
the family into N equal knob-steps. Total measured distance ≥ c·N·√(1/N) = c√N,
which diverges. So **every path through the family has infinite length**. A
reparameterisation only moves the tick marks along a path; it cannot shorten an
infinite one. **[mine, derived from the paper's bound]** This is the same
statement as `figures/generated_one_crease_origin/`'s measured path-speed
divergence, so the repo has already observed the paper's central fact.

**Metric vs Euclidean distance is not curvature.** Metric distance = how far
apart two *pictures* are; Euclidean distance = how far apart two *knob
settings* are. "Non-Lipschitz" means no constant C satisfies
picture-distance ≤ C × knob-distance — the map stretches by an unbounded factor
as you zoom in. Curvature is unrelated.

**What twists, and what does not [paper, §3.2].** At radius ε the local-PCA
plane is still exactly d-dimensional; the dimension count does not grow. It is
a *different* d-dimensional plane, rotated about 30° between ε = 1/4 and 1/8.
The mechanism is visible in the basis images: they are annuli of shrinking
width and growing amplitude, so the ring measured at one radius barely overlaps
the ring measured at half that radius, making the two nearly perpendicular as
vectors. The tangent direction *is* the sliver, and the sliver's width is set by
the measuring radius.

**Regularisation is blur [paper, §3.3].** Convolve with a Gaussian of width s:
replace each pixel by a weighted average of its neighbours. A cliff becomes a
ramp of width s. Moving a ramp by δ changes s pixels' worth of area by δ/s
each, so picture-distance ≈ √(s·(δ/s)²) = δ/√s — **linear in δ**, so the
derivative exists. The price is the 1/√s prefactor: sharper blur means faster
speed, and s → 0 recovers the infinite speed. Same √ law, same exponent as the
repo's pixel-pitch result. `smooth_ones.py` is this operator.

**The test that separates twisting from ordinary curvature.** On any curved
surface a 1-unit step and a 3-unit step point different ways; that is just
curvature and every smooth manifold does it. The diagnostic is whether the
disagreement *shrinks* as both steps shrink. Smooth: the angle between the
direction at ε and at ε/2 falls roughly in proportion to ε. Here it is the
**same 26.57° at every ε** **[paper, §4.2]** — scale-invariant disagreement,
which is what "no tangent plane" means operationally. **[mine]**

**Dimension count: per-scale vs across-scale.** At any single scale the plane is
exactly d-dimensional. But the rings at ε, ε/2, ε/4 … barely overlap as
pictures, so the *union* of the planes over all scales spans unboundedly many
directions. Both readings of "new dimensions appear" are true of different
objects, and conflating them is the easy mistake. **[mine]** This predicts
something checkable: a shell of *finite thickness* mixes scales and should
therefore yield more than d directions, which is a candidate explanation for
`figures/root_one_local_svd/`'s nine-to-ten vectors and for the
"sampling-induced mixtures" already recorded in
`figures/generated_one_basis_followup/`. **[check]**

**Two separate budgets govern the scale ladder [paper, §6.3].**
(1) *Do not outrun your accuracy*: the tangent at blur s is trustworthy only
within about s of the truth, and one step at scale s buys accuracy ‖θ − θ⁽ᵏ⁾‖
< c·s², so the next scale may be as sharp as s² — which is exactly the paper's
own sequence 1/2 → 1/4 → 1/16 → 1/256. (2) *Do not outrun your geometry*: the
twist formula caps how far the plane rotates per rung, so α is chosen to keep
the carried-over tangent informative (α = 0.8 → 9.0°, α = 0.5 → 26.6°,
α = 0.25 → 46.7°). The squaring ladder is the ideal; the 3-D icosahedron needed
a gentler sequence **[paper, §6.4.3, one example]**. Noise sets a floor where
finer scales stop helping, and the paper has **no stopping rule** for it
**[paper, §7]** — directly relevant to MNIST, where the target is not in the
family at all.

### Relevance to this repo

**A method to steal — multiscale Gauss-Newton on the knobs [paper, §6.3].**
Minimise D(θ) = ‖f_θ − I‖² but only ever on a blurred copy. At blur scale s,
build the five tangent images τⁱ = ∂f/∂θᵢ, form J ᵢ = 2⟨f−I, τⁱ⟩ and
H ᵢⱼ = 2⟨τⁱ, τʲ⟩ (dropping the second-derivative term), and update
θ ← θ + H⁻¹J. Then shrink s and repeat, roughly squaring it each round
(1/2 → 1/4 → 1/16 → 1/256). Their ellipse has almost exactly this repo's knob
list, and `smooth_ones.py` already provides Φ_s. Compare against
`invert_grey.py` and the moment chart's closed-form inversion
(`figures/generated_one_moment_chart/`), which reach 0.001–0.009 units.

**A diagnostic to steal — the inter-scale twist angle [paper, §4.2, Thm 4.1].**
With s₁ = α·s₀, the angle between the tangent plane at the two scales converges
to a number depending only on α:

$$\cos\theta_{\text{twist}} \;=\; \frac{\sqrt{2\alpha}}{\sqrt{1+\alpha^{2}}}$$

α = ½ gives 26.57°, which they also measure at 26.4°. This repo compares local
bases *across starting points* but not *across radii at one start*. Measuring
it would turn `analyze_generated_one_scale_separation.py` from "does the rank
settle?" into a falsifiable prediction with a closed-form target. **[mine]**

**A warning — shrinking the radius will not clean up the basis
[paper, §3.2].** `analyze_generated_one_scale_separation.py` probes radii
1e-4 and 1e-5 looking for the five-dimensional first-order structure to
sharpen. For a sharp-edge family the paper says it cannot: the leading vectors
rotate into new directions instead of converging, and the basis images look
like "annuli of shrinking width and growing amplitude" that have no L² limit.
That is the mechanism behind the caveat already recorded in `AGENTS.md`, that
five leading vectors need not capture all first-order change as radius shrinks.
**[mine]** Their footnote grants the escape: on a *pixelised* image the
twisting stops at the finest scale, and the repo's 28×28, 16-subrow coverage
render is already partially regularised.

**A warning — plain Newton on the knobs is not just slow, it is wrong
[paper, §6.2].** The gradient of the fitting error is built from the tangent
images, and those do not exist. Any single-scale knob fit here inherits that.

**A confirmation — the ½ exponent, found twice [paper, §3.1].** The paper's
non-Lipschitz bound makes the difference quotient blow up like ‖Δθ‖^(−½).
`figures/generated_one_crease_origin/` independently measured path speed
diverging as one over the square root of the pixel pitch when the grid is
refined. Same exponent, same cause, different variable held fixed — the repo
refined the grid, the paper shrank the step. **[mine, [check]]**

**A confirmation / open puzzle — extra directions at a special view
[paper, §5.2].** A cube viewed face-on has 3 knobs but a **5-dimensional
tangent complex**: pitch and yaw each have different left and right tangents, so
the special view carries two extra directions. Generic views carry exactly 3.
Root 1 is the analogous accidental view — upright, lean = 0, perfectly centred —
and `figures/root_one_local_svd/` plus the manifold-transport work found nine or
ten vectors for five knobs, with four "mixed normal directions". Five knobs plus
four left/right splits is nine. **[mine, [check]]** If this is the same
phenomenon it also explains why
`analyze_generated_one_local_bases_across_starts.py` finds different ranks at
different starts: some starts are generic and some are not.

**Twisting off, measured here.** One-sided difference directions on a 256x256
antialiased translating disk, same start, step halved each time: consecutive
angles 44.4°, 43.8°, 42.3°, 39.3° — essentially constant. Accumulated from the
ε = 1/4 answer: 44.4°, 59.7°, 69.0°, 75.2°, marching toward 90°. The identical
disk blurred by 4 px gives 34.1°, 23.1°, 12.9°, 6.7° — converging, i.e. a
tangent plane. Note these are *finite-difference* angles; the paper's 26.57° is
the **blur-ratio** quantity of Thm 4.1 and its ~30° is local PCA. Different
estimators, same behaviour, different numbers — do not mix them. **[mine]**
Script: `twist_disk.py` / `twist_fig.py` (scratchpad, not committed).

**Contradiction RESOLVED — the impossibility is relative to the pixel metric
[paper, §3.1].** The paper's "no parameterization exists" relabels **θ** while
holding the image metric fixed at L² over pixels; it also states M = (Θ, μ) is
*isometric* to F, so the obstruction lives in μ. Changing the **representation
of the image** — from a list of ink values to a warp or an edge position — is a
different move and is not covered. That is why
`figures/generated_one_warp_space/` finds a flat 5-D subspace while pixel space
still leaves 8% at 50 directions, and it is not a counterexample to the paper.
**[mine]** The open part is only whether the repo's anti-aliasing *additionally*
places it on F_s rather than F.

**The paper writes down the motion/gradient factorisation and then walks away
from it [paper, §3.4].** Its tangent image is ∂f/∂θᵢ = ∫ φ_s(x−b)·σᵢ(b) db with
σᵢ(b) = ⟨vᵢ(b,θ), n(b)⟩ — the motion field dotted with the edge normal. Modern
form: τᵢ = −vᵢ·∇I. The motion field vᵢ is the **same rule at every image**; the
pixel-space direction differs only because ∇I differs. The paper notes vᵢ is
constant for translation and then returns to measuring in pixels, because its
goal was registration, not representation. This repo went the other way and got
`analyze_generated_one_gradient_fields.py` (every knob = a reweighting of the
image's own d/dx and d/dy, six affine fields, five used, unused one = vertical
shear) and `analyze_generated_one_motion_discovery.py`. **[mine]** The five
knobs as fields: cx → (1,0); cy → (0,1); width → (x−cx, 0); height →
(0, y−cy); lean → (−(y−cy), 0).

**A contradiction to resolve [paper, §3.1] vs `figures/generated_one_warp_space/`.**
The paper states flatly that no reparameterisation can make a sharp-edge family
differentiable. This repo found the generated-1 family is a *flat
five-dimensional subspace* in warp space, machine-zero past the fifth
component, with an exact piecewise-quadratic five-coordinate decoder
(`figures/generated_one_manifold_transport/`). Three candidate reconciliations,
none yet tested: the warp reparameterises the *image plane*, not θ, which the
paper's statement may not cover; "piecewise" quadratic is precisely the escape
hatch, since the pieces are where the derivative jumps; or the coverage
renderer's anti-aliasing is already a regularisation Φ_s and the repo has been
working on F_s all along, never on F. **[mine]** Superseded by the resolution above; kept for the record. Related:
[[warp-reframing]].

**Named concepts this repo has been describing in its own words.**
*Image appearance manifold (IAM)* — the object. *Approximate tangent plane
T(ε,θ)* via local PCA — what the repo's shell SVDs compute. *Inter-scale* and
*intra-scale twisting* — the scale and position dependence of those bases.
*Tangent complex* — the extra directions at a special point. *Generic view* —
why the root is a bad place to measure. *Multiscale Newton* / *coarse-to-fine
differential estimation* — the inversion algorithm, with a literature going
back to Lucas-Kanade 1981. Searching these terms will find the follow-up work.

**Twisting measured on the generated-1 family, and one open question closed
[mine].** `figures/pixel_metric_locality/` reproduces the paper's behaviour on
this repo's own object. On binary ink at 448x448 the largest principal angle
between the five-knob span at h and at h/2 is 55.9, 56.5, 53.4 degrees --
flat, which is the operational meaning of "no tangent plane". Under a Gaussian
blur encoding it is 11.7, 5.5, 2.7 degrees, halving per halving, so the tangent
plane exists on F_s. The speed law reproduces at slope **-0.500** for width.
Note these are one-sided finite-difference angles, NOT the Thm 4.1 blur-ratio
quantity (26.57 degrees) and NOT local PCA (~30 degrees); the three estimators
must not be mixed.

**Open question "is the repo already on F_s rather than F?" -- ANSWERED, yes,
below a step of about 0.06 [mine].** The same twist measurement on the 28x28
coverage render gives 38.5, 26.8, 3.6, 0.0, 0.0 degrees: exactly locally linear
once the normalised knob step drops under ~0.06. The anti-aliased coverage
renderer IS a regularisation, so sub-crease work in this repo has never been on
the sharp-edge manifold F.

**A cause the paper does not separate out: the sampling grid [mine].** Under
optimal transport a cx translation costs exactly h (4.0000, 2.0000, 1.0000 at
h = 4, 2, 1) with no blurring and nothing destroyed -- the sharp edges are not
an obstruction to OT at all. But the flat run ends exactly at the pixel pitch
(1.0, 0.5, 0.25 for 28^2, 56^2, 112^2 grids) and below it tracks 1/sqrt(h) to
three decimals. So on a SAMPLED image there are two blow-ups with different
causes: the edges, which OT removes for free, and the grid, which nothing
removes, because a sub-pixel shift of a sampled picture is a resampling and not
a translation. The paper works in the continuum and so never meets the second.

### Open questions

- Does the 26.57°-per-halving law hold for `grey_ones` at radii above the pixel
  grid, and where exactly does the 28×28 grid cut it off?
- Is the repo's warp flatness a genuine counterexample to §3.1, or is the repo
  already working on the smoothed manifold F_s rather than F?
- Does multiscale Gauss-Newton beat the moment chart's 0.001–0.009 units on the
  generated family, and does it survive on MNIST 1s where no small knob set
  exists?
- Root 1 is the non-generic view. Should the canonical reference for local-basis
  work move to a leaning, off-centre start?

### Claims to check

- [ ] 26.57° at α = ½ reproduced on `smooth_ones` with Gaussian Φ_s. **[check]**
- [ ] Path-speed ½ exponent and the paper's ½ exponent are the same law. **[check]**
- [ ] Root 1's nine-to-ten vectors = five knobs + four left/right splits. **[check]**
- [ ] The intra-scale result — twist angle constant when the knob step is
      proportional to the blur scale **[paper, §4.3, one example]** — holds here.

---

## Links

- Related memories: [[warp-reframing]], [[ones-manifold-goal]]
- Repo: `GENERATED_ONES_IMPLICIT_EQUATION.md`, `smooth_ones.py`,
  `figures/generated_one_crease_origin/README.md`,
  `figures/root_one_local_svd/README.md`,
  `figures/generated_one_warp_space/README.md`
- Companion paper by the same authors: "High-resolution navigation on
  non-differentiable image manifolds" (ICASSP) — not yet read.
- The paper's own key reference for the ½ exponent: Donoho & Grimes, "Image
  manifolds which are isometric to Euclidean space", J. Math. Imaging and
  Vision 23, 2005 **[paper, ref 11]** — this is where the detailed computations
  live, and is probably the next thing to read.
