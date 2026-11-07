# A second set of five knobs that makes exactly the same 1s

`analyze_darmois_alternative_knobs.py` (numbers) · `make_darmois_figure.py` (figure)

## The question

Hyvärinen & Pajunen (1999) prove the generating map is not recoverable from the
data alone. Fine — but *how do you get even one of the alternatives?* Their
proof is constructive, so this builds one explicitly and checks it.

## The recipe

In general the construction is Darmois (1952) / Rosenblatt (1952): take the
conditional CDFs of the data,

    z₁ = F(x₁),  z₂ = F(x₂ | x₁),  z₃ = F(x₃ | x₁, x₂),  …

These are exactly independent, exactly uniform, and the map is invertible, so
`decoder(F⁻¹(z))` is a complete alternative generator. Every autoregressive
normalizing flow ever trained is a learned instance of this map.

Here `grey_ones.sample` draws the five knobs **independent uniform on a box**,
so the recipe collapses to: *any volume-preserving bijection of the unit cube*
sends independent uniforms to independent uniforms. Two are built, both mixing
normalized `cx` and `lean`:

- **wrap** — two cyclic shears on the torus. Exactly volume preserving, exact
  closed-form inverse, but discontinuous at the wrap.
- **flow** — time-1 map of the divergence-free field of H = (ε/π)·sin(πp)·sin(πq)
  on the unit square. Smooth, fixes the boundary, volume preserving to RK4 error.

## Results

| check | wrap | flow (ε=1.6) | true knobs (control) |
| --- | --- | --- | --- |
| round trip max abs error | 4.4e-16 | 5.4e-12 | — |
| **same images**, max pixel gap | **0.0** | **6.0e-8** (float32 rounding) | — |
| marginals, worst KS vs uniform | 0.017 | 0.012 | 0.011 |
| **independence**, worst pair distance correlation | **0.055** | **0.057** | **0.054** |
| new coord 1 vs true `cx`, Pearson r | +0.13 | −0.53 | +1.0 |

**[mine]** The alternative generators are not approximations. The flow decoder
reproduces every pixel of every image to float32 rounding, and the wrap decoder
reproduces them bit for bit. The new coordinates are as independent and as
uniform as the generator's own knobs — all three sit at the n=1200 sampling
floor, so no independence test can separate them at any sample size this repo
uses. And they are not relabelled knobs: the flow's first coordinate correlates
−0.53 with `cx` and mixes it with `lean`.

Sweeping ε gives a continuum: images identical throughout, r(z₁, cx) walking
+1.00 → +0.77 → +0.25 → −0.24 → −0.53 → −0.32 → −0.06 as ε goes 0 → 3.2.

### Tie-breaker 1 — additivity of one-knob image changes: **FAILED**

Residual of adding five one-coordinate image changes at the box centre, true
knobs vs flow knobs: 0.92x, 0.94x, 1.05x, 1.07x at normalized steps 0.1 down to
0.0125. **The twisted coordinates are exactly as flat as the true ones.** Local
linearity does not prefer the generator's parameterisation. (Only the largest
step, 0.2, shows 2.7x, and that is outside the locally linear regime.)

### Tie-breaker 2 — decoder simplicity: **WORKS**

Held-out error (%) of the best polynomial decoder of each degree, 4000 images,
1000 held out:

| degree (terms) | true knobs | flow knobs | wrap knobs |
| --- | --- | --- | --- |
| 1 (6) | 40.27 | 51.51 | 58.11 |
| 2 (21) | 28.11 | 47.59 | 53.91 |
| 3 (56) | 20.85 | 37.76 | 52.78 |
| 4 (126) | 16.65 | 35.95 | 51.10 |
| 5 (252) | **13.93** | **32.47** | **51.96** |

**[mine]** 2.3x and 3.7x worse at degree 5. The wrap knobs plateau near 52% —
a polynomial cannot represent a discontinuity at all. So the criterion that
separates the true knobs from the impostors is not statistical; it is **how few
terms the decoder needs**. That is the "sparsity/simplicity of the mixing
function" escape route named in the identifiability literature
(e.g. arXiv 2206.07751), measured here.

## Caveats

- Simplicity picks out *a* simple parameterisation, not *the* generator's. This
  repo already knows a parameterisation that beats the raw knobs: in warp space
  with `height·tan(lean)` as the fifth dial, the five lines add exactly
  (`figures/generated_one_line_interpolation/`). So degree-5 polynomial error is
  a *preference ordering*, not a proof of identity. **[mine]**
- The uniform-box prior is what makes the closed form available. For a general
  density you need the conditional CDFs, which is where flows come in.
- The flow's volume preservation is exact in the continuous field and
  approximate under RK4; the measured consequence is the 6e-8 pixel gap and the
  5.4e-12 round trip, both below float32 storage.

## Links

- Theorem: `literature/README.md` item 22 (Hyvärinen & Pajunen 1999).
- Independently confirms why `figures/mnist_one_knob_search/` needed five
  labelled examples to pin its five-number gauge: without an anchor the knobs
  are not determined, by theorem. **[mine]**
