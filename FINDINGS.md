# Reversing a sentence embedding back into its phrase

Target: `BAAI/bge-large-en-v1.5`, phrase `"magic is real"`, embedding = L2-normalised CLS
row of the last hidden state. Constraint: deterministic mathematics only — no gradient
descent, no learned inverse model.

Everything below is measured, in float64, against the true forward pass. Run
`demo1_stages.py`, `demo2_full_chain.py`, `demo3_cls_bottleneck.py`, `demo4_branch_stats.py`.

## Short answer

The pipeline splits into four stages. Three of them invert exactly. The fourth is where
it dies, and it is **not** the one everybody expects.

| Stage | Status | Measured error |
|---|---|---|
| L2 normalisation | **exact, closed form** | 9.8e-15 |
| Feed-forward sublayer (incl. GELU, LayerNorm) | **exact**, but multiple preimages | 1.7e-11 on the true branch |
| Attention sublayer (incl. softmax, LayerNorm) | **exact**, but multiple preimages | 3.8e-11 on the true branch |
| Embedding block → token ids | **exact, one matmul** | 1e-15 vs 1.0 for the runner-up |
| CLS pooling | **not invertible** | 4088 missing dimensions |

So: given the *full* final hidden state matrix, the algebra recovers the phrase. Given a
sentence embedding, it cannot — because pooling throws away four of the five rows, and no
rearrangement of the equations puts them back.

## The idea that makes any of this work

The obvious blocker is LayerNorm. Forwards it deletes the input's mean and variance — two
numbers per row — so one output corresponds to an infinite family of inputs. That is the
conclusion the earlier notebook in this repo reached, and read in isolation it is correct.

It stops being correct inside a transformer, because **every tensor you are trying to
recover is itself a LayerNorm output**, and therefore satisfies two equations you already
know:

```
z = (x - beta) / gamma      mean(z) = 0      mean(z^2) = 1
```

Verified on `h_12`: `mean(z) = 1e-17`, `mean(z^2) - 1 = 4e-16`.

So the two numbers LayerNorm discarded are not free — they are pinned by two extra
constraints. Each sublayer inversion becomes a **square** system: `d` unknowns for the row
plus the 2 discarded scalars, against `d` sublayer equations plus those 2 constraints.
Square systems have isolated solutions. That is what turns an infinite family into a
finite list of candidates, and it is the step everything else rests on.

### The same trick inverts L2 normalisation in closed form

`emb = h/||h||` looks like it destroys the scale. But `h` is a LayerNorm output, so with
`h = s*u`, `mean(z) = 0` reads

```
s * mean(u/gamma) = mean(beta/gamma)
```

— one linear equation, one unknown, no search. Recovered `s = 16.759219` against a true
norm of `16.759219`, max abs error **9.8e-15**. The second constraint `mean(z^2) = 1` is
not needed to solve, and checks out to 6.8e-13, confirming the answer independently.

### Reading the tokens off the bottom is also exact

`h0 = LN(E[t] + P[pos] + T[0])` forces `E[t] + P[pos] + T[0]` to lie in the known 2-D
plane `span{w, 1}` with `w = (h0-beta)/gamma`. Score all 30522 vocabulary rows by distance
to that plane:

```
pos 0: [CLS] 2.2e-15   next '1842'    1.0e+00
pos 1: magic 6.6e-16   next 'magical' 1.1e+00
pos 2: is    1.9e-15   next 'was'     7.3e-01
pos 3: real  5.2e-16   next 'actual'  1.1e+00
pos 4: [SEP] 1.3e-15   next '1888'    1.0e+00
```

A fifteen-order-of-magnitude gap, from one matmul per position. Given `h0`, the tokens are
not guessed — they are read. No search over sequences.

## What the sublayer inversions actually need

**Naive backward iteration does not work, and there is a clean reason.** Rearranging each
sublayer into a fixed point and iterating diverges — even when started at the true answer
plus 1e-3. Power iteration on the frozen-pattern attention operator explains it:
`rho(T_P) > 1` in **20 of 24 layers** (up to 1.50). The residual stream's attention map is
expansive, so backward Picard iteration is unstable by construction. A genuine solve is
required, not a rearrangement.

- **Feed-forward sublayer.** Rows are independent (the FFN touches one position at a
  time), so each is a 1026x1026 square system solved by Newton with the explicit Jacobian
  `I + t*W2 diag(gelu'(W1 a + b1)) W1`. Converges quadratically, ~6 iterations.
- **Attention sublayer.** Rows are coupled. But attention is *linear in `h` through the
  value path* once the softmax probabilities `P` are frozen. So we alternate: read `P` off
  the current `h` (16xnxn numbers — a tiny object), solve the sublayer equation exactly
  with `P` held fixed (an affine system plus the 2n manifold constraints, reduced to a
  2n-dimensional Newton solve), recompute `P`. Every inner step is exact linear algebra;
  the only iteration is on the attention pattern.

Both reach ~1e-11 against the true hidden states — *when they land on the right branch*.

## The second obstruction: preimages are isolated but not unique

Square systems have isolated roots, not unique ones. GELU is not monotone and softmax is
not affine, so a sublayer can have several preimages that **all reproduce the observed
output to machine precision**. Example, layer 22, one-token sequence: Newton converged to
`|F| = 4.1e-14` at a point `2.27` away from the true `a`. That is not a solver failure. It
is a different, equally valid preimage.

Continuation helps: scale the sublayer contribution by `t`, start at `t=0` where the
system is linear and the root unique, and walk `t` to 1 with a tangent predictor and
adaptive step control. This fixed layer 23 (0.49 error → 1.1e-11). It does not fix
everything — the solution branch folds at some layers, so the path from `t=0` does not
always connect to the branch the forward pass used.

`demo4_branch_stats.py` counts how often the trace lands on the true preimage. Nothing
*local* distinguishes the branches; the only thing that rules a wrong one out is that it
fails to bottom out on the token manifold twenty layers later. That is a search over an
exponentially branching tree, not a formula.

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

**The backward chain is underdetermined by 4088 real dimensions.** Not by LayerNorm, not
by GELU, not by the softmax — all of those are handled above. Purely by the fact that
pooling reports one row and discards four.

This is not a Shannon-information problem. Naming 5 tokens from a 30522 vocabulary needs
74 bits; 1023 float64 coordinates carry tens of thousands. A collision probe over 60
one-token perturbations found none (max cosine 0.888). The map is injective and **the
inverse exists as a function** — there is simply no formula for it, because computing it
by peeling requires rows that were never reported. Existence of an inverse is not a
construction of one.

## What a deterministic attack on the pooled vector actually achieves

The strongest non-gradient, non-learned attack I found: linearise the whole 24-layer
network in its token-embedding inputs via the exact Jacobian (autodiff computes a
derivative — that is not training), then solve for tokens by matching pursuit over the
vocabulary, re-linearising each sweep.

```
sweep 0: ['[CLS]', 'optimistic', 'radios',     'berman',     '[SEP]']
sweep 1: ['[CLS]', 'hugging',    'mistakenly', '##jong',     '[SEP]']
sweep 2: ['[CLS]', 'beyond',     'lighted',    'perfection', '[SEP]']
```

Final cosine to the target: **0.555**. It fails, and instructively: a first-order model of
the network is not valid across a jump as large as swapping one token for another, so the
linear proposal is not informative and the sweeps wander. Note also that this is a
*search* over 30522 candidates per position, not a reversal — it is on the other side of
the line the question drew.

## Bottom line

- Deterministic reversal of `bge-large-en-v1.5` down to exact token ids **is** possible,
  and this repo does it, **given the full final hidden-state matrix**.
- The step everyone flags as fatal — LayerNorm — is not fatal. Its two lost numbers per
  row are recoverable, because its output manifold supplies exactly two equations.
- Two things do block it. The soft one is branch multiplicity: sublayer preimages are
  isolated but not unique, and choosing among them is not a local decision. The hard one
  is CLS pooling, which deletes 4088 dimensions that no algebra can reconstruct.
- From a sentence embedding alone, exact deterministic inversion is **impossible** in the
  strict sense asked for. Anything that recovers the phrase from that vector must search.
