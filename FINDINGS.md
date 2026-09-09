# Reversing a sentence embedding back into its phrase

Target: `BAAI/bge-large-en-v1.5`, phrase `"magic is real"`, embedding = L2-normalised CLS
row of the last hidden state. Constraint: deterministic mathematics only — no gradient
descent, no learned inverse model.

Everything below is measured in float64 against the true forward pass. Scripts:
`demo1_stages.py`, `demo2_full_chain.py`, `demo3_cls_bottleneck.py`, `demo4_branch_stats.py`.

## Short answer

**You cannot do it from the embedding alone, and the reason is not the one usually given.**
LayerNorm, GELU and the softmax all invert. What kills it is CLS pooling, which reports one
row of the final hidden state and discards the rest — for a 5-token phrase that is 4088
real dimensions the backward chain needs and never receives.

### What was actually run — read this before the numbers

**No phrase was ever recovered end to end, from anything.** Not from an embedding, and not
from the full hidden states either. Every "exact" figure below is a *single stage tested in
isolation, handed the true value from the forward pass as its input.* They have never been
chained into a working pipeline.

| Stage | Status | Measured | Input it was given |
|---|---|---|---|
| L2 normalisation | **exact, closed form** | 9.8e-15 | the real embedding |
| Feed-forward sublayer (GELU + LayerNorm) | exact on 2 of 6 layers tested | 1.8e-11 | true layer output |
| Attention sublayer (softmax + LayerNorm) | exact on 5 of 6 layers tested | ~1e-11 | true layer output |
| Embedding block → token ids | **exact, one matmul** | 1e-15 vs 1.0 for runner-up | true `h0` |
| CLS pooling | **not invertible** | 4088 missing dimensions | — |

The only complete 24-layer chain I ever ran (from the full hidden states, using an early
and since-abandoned solver) diverged to errors of 1e+62 and decoded to
`115 [unused650] [unused754] [unused754] [unused954]`. `demo2_full_chain.py` has since been
rewritten around the working solvers but **has never been run to completion**, so whether
the chain closes is untested. Error compounds across stages, and the branch problem below
gives good reason to expect it will not close without more work.

So: the pooling argument stands on a degrees-of-freedom count, which is solid and does not
depend on any of the solvers working. Everything else here is verified components, not a
verified system.

## The idea that makes any of this work

The obvious blocker is LayerNorm. Forwards it deletes the input's mean and variance — two
numbers per row — so one output corresponds to an infinite family of inputs. That is the
conclusion the earlier notebook in this repo reached, and read in isolation it is correct.

It stops being correct inside a transformer, because **every tensor you are trying to
recover is itself a LayerNorm output**, and therefore already satisfies two equations:

```
z = (x - beta) / gamma      mean(z) = 0      mean(z^2) = 1
```

Verified on `h_12`: `mean(z) = -1.4e-18`, `mean(z^2) - 1 = -3.8e-13`.

So the two discarded numbers are not free — they are pinned. Each sublayer inversion
becomes a **square** system: `d` unknowns for the row plus the 2 discarded scalars, against
`d` sublayer equations plus those 2 constraints. Square systems have isolated solutions.
That is what turns an infinite family into a finite candidate list, and everything else
rests on it.

One caveat that cost me a real bug: the recovered scale `s` is a standard deviation, so
`s > 0` is part of the problem. Roots with `s < 0` satisfy the constraint equations but not
the original `LN(x + delta(x)) = y`, and the solver happily converged to them until
positivity was enforced in the line search.

### L2 normalisation, in closed form

`emb = h/||h||` looks like it destroys the scale. But `h` is a LayerNorm output, so with
`h = s*u`, `mean(z) = 0` reads

```
s * mean(u/gamma) = mean(beta/gamma)
```

One linear equation, one unknown, no search. Recovered `s = 16.759219` against a true norm
of `16.759219`; max abs error **9.8e-15**. The second constraint `mean(z^2) = 1` is not
needed to solve and checks out to 6.8e-13 — it confirms the answer rather than producing it.

### Reading the tokens off the bottom, exactly

`h0 = LN(E[t] + P[pos] + T[0])` forces `E[t] + P[pos] + T[0]` into the known 2-D plane
`span{w, 1}`, `w = (h0-beta)/gamma`. Score all 30522 vocabulary rows by distance to it:

```
pos 0: [CLS] 2.2e-15   next '1842'    1.0e+00
pos 1: magic 6.6e-16   next 'magical' 1.1e+00
pos 2: is    1.9e-15   next 'was'     7.3e-01
pos 3: real  5.1e-16   next 'actual'  1.1e+00
pos 4: [SEP] 1.3e-15   next '1888'    9.9e-01
```

A fifteen-order-of-magnitude gap, from one matmul per position. Given `h0`, the tokens are
not guessed — they are read. No search over sequences.

## Why naive backward iteration cannot work

Rearranging a sublayer into a fixed point and iterating diverges, even when started at the
true answer plus 1e-3. Power iteration on the frozen-pattern attention operator gives the
reason: **`rho(T_P) > 1` in 20 of 24 layers** (up to 1.50). The attention map is expansive,
so backward Picard iteration is unstable by construction. A genuine solve is required.

- **Feed-forward sublayer.** Rows are independent (the FFN touches one position at a time),
  so each is a 1026x1026 square system, solved by Newton with the explicit Jacobian
  `I + t*W2 diag(gelu'(W1 a + b1)) W1`. Quadratic convergence, ~6 iterations.
- **Attention sublayer.** Rows are coupled — but attention is *linear in `h` through the
  value path* once the softmax probabilities `P` are frozen. So alternate: read `P` off the
  current `h` (16xnxn numbers, a tiny object), solve the sublayer equation exactly with `P`
  fixed (an affine system plus the 2n manifold constraints, reduced to a 2n-dimensional
  Newton solve), recompute `P`. Every inner step is exact linear algebra; the only iteration
  is on the attention pattern.

## Branch multiplicity — claimed, then largely retracted

I originally reported that the feed-forward sublayer has multiple genuine preimages: GELU is
not monotone, so several inputs can produce the identical output. Layer 22 on a one-token
sequence converged to `|F| = 4.1e-14` at a point `2.27` from the true `a`, and layer 4
produced an answer replaying to `5.1e-13` while sitting `0.12` from the truth.

**A multistart test does not support that claim.** Firing Newton at the same equation from
18 different random points on the LayerNorm manifold, at layers 4 and 23:

```
layer  4, row 0: 18 random starts -> 1 DISTINCT valid preimage, and it is the TRUE one
                 (distance from truth 3.1e-10, replays output to 1.6e-14)
layer 23, row 0: 18 random starts -> 1 DISTINCT valid preimage, and it is the TRUE one
                 (distance from truth 1.7e-11, replays output to 8.9e-16)
```

Every start that converged with scale `s > 0` landed on the same root. So the earlier
"alternative preimages" were most likely **continuation artifacts** — traces truncated by the
120-step budget, returning a point that solves `a + t*FFN(a) = ...` for some `t < 1` rather
than the real equation — not a genuine multiplicity. 18 starts on one row of two layers is
evidence, not proof, and GELU's non-monotonicity does still permit multiplicity in principle.
But the working assumption should now be that the feed-forward inverse is **unique in
practice**, and that plain multistart plus a replay check finds it.

That reorders the problem list: the remaining blocker is not branching, it is the attention
solver failing to converge to a root that provably exists (see below).

Two things help, and they are not interchangeable:

1. **Continuation.** Scale the sublayer contribution by `t`, start at `t=0` where the system
   is linear and the root unique, and walk `t` to 1 with a tangent predictor and adaptive
   step control. This rescued layer 23's FFN (0.49 error → 1.1e-11). It does not fix
   everything: the branch folds at some layers, so the path from `t=0` does not always
   connect to the branch the forward pass took. And on the *attention* sublayer it often
   makes things worse rather than better.
2. **Replay-and-keep.** A candidate is checkable without knowing the answer: push it
   forwards and see whether it reproduces the observed output. So run both strategies and
   keep whichever replays. This is verification, not search — two candidates, one test — and
   it is what makes the attention inversion usable.

Measured over 6 layers (`demo4_branch_stats.py`, input = the true layer output, so any error
is branch selection rather than accumulation):

```
layer    ffn err  ffn replay     ffn   attn err  attn replay    attn
   23    1.8e-11     9.6e-14    TRUE    1.3e+01      9.8e+00  FAILED
   20    5.2e-10     6.2e-13    TRUE    1.0e-11      1.3e-11    TRUE
   16    3.2e+00     4.5e-01  FAILED    9.9e-12      6.0e-12    TRUE
   12    1.4e+00     2.8e-01  FAILED    1.5e-11      4.2e-12    TRUE
    8    8.2e-01     1.1e-01  FAILED    6.4e-12      5.2e-12    TRUE
    4    1.2e-01     5.1e-13   other    7.6e-12      1.6e-12    TRUE

feed-forward sublayers landing on the true preimage: 2/6
attention   sublayers landing on the true preimage: 5/6
```

Reading this honestly:

- **Attention: 5/6 exact**, to ~1e-11. Only layer 23 resists.
- **FFN's three `FAILED` rows are my step budget, not mathematics.** I capped the
  continuation at 120 steps to bound runtime; those traces were truncated mid-path and
  never reached `t=1`. Raising the cap should fix them, at a cost I did not pay.
- **Layer 4's `other` is the real phenomenon**: it replays the output to 5.1e-13 while
  sitting 0.12 from the truth. A genuine second preimage. Nothing *local* distinguishes it —
  the only thing that rules it out is failing to bottom out on the token manifold twenty
  layers later, which is a search over an exponentially branching tree, not a formula.

So layer-by-layer inversion is *sound* and mostly *works*, but it is not yet a push-button
24-layer chain. That is an engineering gap on top of the mathematical one below.

## The real wall: CLS pooling

Degrees of freedom for a 5-token phrase, `d = 1024`:

```
final hidden state the backward chain needs : 5 x 1024 = 5120 numbers
  each row is a LayerNorm output, so free   : 5 x 1022 = 5110
what the sentence embedding gives           : 1 unit vector = 1023 free
undoing the L2 norm recovers row 0 exactly  : row 0 fully known
rows never observed                         : 4 x 1022 = 4088 unknowns
equations available to pin them             : 0
```

**The backward chain is underdetermined by 4088 real dimensions.** Not by LayerNorm, not by
GELU, not by the softmax — those are all handled above. Purely because pooling reports one
row and discards four.

This is not a Shannon-information problem. Naming 5 tokens from a 30522 vocabulary needs 74
bits; 1023 float64 coordinates carry tens of thousands. A collision probe over 60 one-token
perturbations found none (max cosine 0.888). The map is injective and **the inverse exists
as a function** — there is just no formula for it, because computing it by peeling requires
rows that were never reported. Existence of an inverse is not a construction of one.

## What a deterministic attack on the pooled vector actually achieves

The strongest non-gradient, non-learned attack I found: linearise the whole 24-layer network
in its token-embedding inputs via the exact Jacobian (autodiff computes a derivative — that
is not training), then solve for tokens by matching pursuit over the vocabulary,
re-linearising each sweep.

```
sweep 0: ['[CLS]', 'optimistic', 'radios',     'berman',     '[SEP]']
sweep 1: ['[CLS]', 'hugging',    'mistakenly', '##jong',     '[SEP]']
sweep 2: ['[CLS]', 'beyond',     'lighted',    'perfection', '[SEP]']
```

Final cosine to the target: **0.555**. It fails, and instructively: a first-order model of
the network is not valid across a jump as large as swapping one token for another, so the
proposals are uninformative and the sweeps wander. Note also that this is a *search* over
30522 candidates per position — on the other side of the line the question drew.

## Bottom line

- **Nothing was recovered end to end.** The components below work in isolation; the
  pipeline has never closed.
- The step everyone flags as fatal — LayerNorm — **is not fatal**. Its two lost numbers per
  row come back, because its own output manifold supplies exactly two equations. That single
  observation is what makes deterministic inversion of a post-LN transformer well posed.
- Individual sublayer inversions are exact — ~1e-11 for both types, verified by replay —
  *when handed the true layer output.* Feeding them their predecessor's reconstruction
  instead is the untested part.
- Two things stand in the way. The soft one is **branch multiplicity** — preimages are
  isolated but not unique, and choosing among them is not a local decision, though
  replay-and-keep handles most cases. The hard one is **CLS pooling**, which deletes 4088
  dimensions no algebra can reconstruct.
- **From a sentence embedding alone, exact deterministic inversion is impossible** in the
  strict sense asked for. Anything that recovers the phrase from that vector must search.

## Where I would go next

- Raise the FFN continuation step budget and close the three truncated layers; then actually
  run `demo2_full_chain.py` to completion. Until that happens the central engineering claim
  — that these stages compose — is unverified. Expect error compounding to bite.
- Layer 23's attention sublayer resists both strategies — worth understanding why, since it
  is the entry point of any backward pass.
- The `n=1` case is the clean end-to-end demonstration: with a single token, attention cannot
  mix positions, so the pooled row *is* the whole state and the 4088-dimension deficit
  vanishes. That path is blocked today only by the FFN branch problem, not by pooling.
