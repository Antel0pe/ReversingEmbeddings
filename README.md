# ReversingEmbeddings

Given a sentence embedding from `BAAI/bge-large-en-v1.5`, recover the phrase that produced
it — using deterministic mathematics only. No gradient descent, no learned inverse model.

**[FINDINGS.md](FINDINGS.md) has the full write-up and all measured numbers.**

## Short version

**No phrase was ever recovered end to end** — not from an embedding, not from the full
hidden states. The figures below are single stages tested in isolation, each handed the true
value from the forward pass. They have never been chained into a working pipeline; see
FINDINGS.md. The pooling result is a degrees-of-freedom argument and does not depend on the
solvers working.

| Stage | Status | Measured error |
|---|---|---|
| L2 normalisation | exact, closed form | 9.8e-15 |
| Feed-forward sublayer (GELU + LayerNorm) | exact, but several valid preimages | 1.8e-11 on the true branch |
| Attention sublayer (softmax + LayerNorm) | exact on 5 of 6 layers tested | ~1e-11 |
| Embedding block → token ids | exact, one matmul | 1e-15 vs 1.0 for the runner-up |
| CLS pooling | **not invertible** | 4088 missing dimensions |

The key move: **LayerNorm is not the wall it looks like.** It deletes the input's mean and
variance, but every tensor you need to recover is itself a LayerNorm output, so those two
numbers are pinned by two equations you already know (`mean(z)=0`, `mean(z^2)=1`). That
turns each sublayer inversion into a *square* system with isolated solutions instead of an
infinite family.

What actually blocks the goal is CLS pooling: for a 5-token phrase it reports one row of
the final hidden state and discards four, leaving the backward chain underdetermined by
`4 x 1022 = 4088` real dimensions. The inverse map exists (the pooled map is injective —
no collisions found), but there is no formula for it. Recovering the phrase from the
embedding alone requires search.

## Files

| File | What it does | Runtime |
|---|---|---|
| `inversion.py` | The library: forward pass rebuilt from raw weights, plus every inversion primitive | — |
| `demo1_stages.py` | Verifies each stage that inverts exactly, against ground truth | ~5 min |
| `demo2_full_chain.py` | Walks all 24 layers backwards from the full final hidden states | slow |
| `demo3_cls_bottleneck.py` | Degrees-of-freedom accounting, collision probe, and the strongest deterministic attack on the pooled vector | ~25 min |
| `demo4_branch_stats.py` | Counts how often a sublayer inversion lands on the true preimage vs a different valid one | ~20 min |
| `ReversingBertEmbeddings.ipynb` | The earlier per-submodule exploration this grew out of | ~6 min |

The notebook reached the common conclusion that LayerNorm is a hard wall and the attention
block is not analytically invertible. `FINDINGS.md` revisits both claims: the first is
wrong inside a transformer, the second is right about the *method* but for a different
reason than stated.

## Setup

```
uv venv --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
```

Everything runs on CPU in float64 — float32 is not accurate enough to peel 24 layers. The
model (~1.3 GB) downloads from HuggingFace on first run.
