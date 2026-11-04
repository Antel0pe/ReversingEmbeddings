# The Multiscale Structure of Non-Differentiable Image Manifolds

https://doi.org/10.1117/12.617822 — author copy: https://people.mines.edu/mwakin/wp-content/uploads/sites/262/2024/06/wakin-spie-iams.pdf

Michael B. Wakin, David L. Donoho, Hyeokho Choi, Richard G. Baraniuk ·
Wavelets XI, SPIE Optics & Photonics, San Diego, July 2005 · doi:10.1117/12.617822

**Read with:** 2026-11-03 · **Depth reached:** L1 · **Source read:** full text pasted by the user (figures described, not seen)

---

## What it says (my notes)

<!-- Yours. -->

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
working on F_s all along, never on F. **[mine]** This is the entry worth the
most. Related: [[warp-reframing]].

**Named concepts this repo has been describing in its own words.**
*Image appearance manifold (IAM)* — the object. *Approximate tangent plane
T(ε,θ)* via local PCA — what the repo's shell SVDs compute. *Inter-scale* and
*intra-scale twisting* — the scale and position dependence of those bases.
*Tangent complex* — the extra directions at a special point. *Generic view* —
why the root is a bad place to measure. *Multiscale Newton* / *coarse-to-fine
differential estimation* — the inversion algorithm, with a literature going
back to Lucas-Kanade 1981. Searching these terms will find the follow-up work.

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
