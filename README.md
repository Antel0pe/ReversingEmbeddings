# ReversingEmbeddings

Given the output of a BERT submodule, recover the inputs that could have produced it.

`ReversingBertEmbeddings.ipynb` walks the last encoder layer of `BAAI/bge-large-en-v1.5`
backwards using closed-form inversions only — pseudoinverse + nullspace for the linear
pieces, `arctanh` for the pooler, Newton's method for GELU, `log(y) + c` for softmax.

Ported from `FantasyArchetypesInSemanticSpace/notebooks/SimilarityBetweenHiddenLayers.ipynb`.
Gradient-descent search, brute-force token sweeps, layer-similarity measurements and the
geometry plots were left behind.

## Setup

```
uv venv --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
```

Then select `.venv` as the notebook kernel. No GPU needed — a forward pass on a 5-token
phrase is ~60ms on CPU. The model (~1.3GB) downloads from HuggingFace on first run.

## Runtime

Around 6 minutes end to end. Nearly all of it is one cell: `find_z_for_exact_match` on
`output.dense` does a 4096x4096 pseudoinverse per token.

## What reverses and what doesn't

Measured on the phrase "magic is real". The key distinction is **structural** reversibility
(guaranteed by the shape of the operation) versus **contingent** reversibility (holds only
because these particular trained weights cooperate).

### Structural — true for any weights

| Piece | Cost |
|---|---|
| Tokenizer | Nothing |
| Softmax | Exactly 1 constant per row |
| LayerNorm | Exactly 2 scalars per token (mean, std); direction fully preserved |

### Contingent, and robust

| Piece | Condition | Margin |
|---|---|---|
| `intermediate.dense` (1024 -> 4096) | Full column rank, well conditioned | cond 34, err 9.5e-06 |
| Embedding lookup | No duplicate rows | 0 collisions, 24% of a row norm apart |
| LayerNorm affine | No near-zero gamma | min 0.048 |

### Contingent, and fragile

| Piece | Condition | Why fragile |
|---|---|---|
| Q/K/V projections | Full rank | Only at tol=1e-10; at float32 tolerance 13-18 of 24 layers lose 1-2 dims. cond up to 7.6e5 |
| `attention.output.dense` | Full rank | 10/24 deficient at float32 tolerance |
| Pooler `tanh(Wx+b)` | No saturation | margin to 1.0 is 5.5e-05, cond 6.3e4 |
| Attention value path | SxS probs invertible | Holds at 5 tokens; longer/peakier attention would not |

### Not reversible

| Piece | Why |
|---|---|
| GELU | Non-injective below -0.7517, and **89.7% of real activations are there** |
| `output.dense` (4096 -> 1024) | 3072-dim nullspace by shape (though well conditioned on what it keeps) |
| Attention scores | Quadratic in `X` |
| Residual connections | `y = x + f(x)`; never attempted |

**The honest summary:** only the FFN up-projection is unconditionally and robustly invertible.
The attention projections cling to full rank by a margin thinner than float32 can represent —
an earlier claim that all 72 of them are full rank was an artifact of a tolerance set 1000x
tighter than the arithmetic warrants.
