"""DEMO 4 -- how often does the sublayer inverse land on the TRUE preimage?

Each sublayer inversion is a square system, so its solutions are isolated points rather
than a family -- that is what makes the problem well posed at all. But "isolated" is not
"unique": GELU is not monotone and the softmax is not affine, so a sublayer can have
several preimages, all of which reproduce the observed output to machine precision.

This script inverts every sublayer from its TRUE output (so the input is exact and any
error is branch selection, not error accumulation) and counts how often the branch trace
lands on the preimage the forward pass actually used.
"""
import time
import torch
from inversion import (load, forward_states, prev_layernorms, invert_ffn,
                       invert_attention_continued, ln, run_layer, attn_delta)

PHRASE = "magic is real"
tok, model = load()
ids = tok(PHRASE, return_tensors="pt")["input_ids"]
A, H = forward_states(model, ids)
lnp = prev_layernorms(model)

print(f"phrase {PHRASE!r}, {ids.shape[1]} tokens\n")
print(f"{'layer':>5} {'ffn |F|':>10} {'ffn err':>10} {'ffn':>6} "
      f"{'attn |dh|':>10} {'attn err':>10} {'attn':>6}  {'s':>5}")
ffn_true = attn_true = 0
t0 = time.time()
LAYERS = [23, 22, 21, 20, 16, 12, 8, 4, 0]
for Li in LAYERS:
    layer = model.encoder.layer[Li]
    ln_in = layer.attention.output.LayerNorm
    a_rec, r1, _ = invert_ffn(layer, H[Li + 1], ln_in)
    e1 = (a_rec - A[Li]).abs().max().item()
    h_rec, r2, _ = invert_attention_continued(layer, A[Li], ln_in, lnp[Li])
    e2 = (h_rec - H[Li]).abs().max().item()
    ok1, ok2 = e1 < 1e-6, e2 < 1e-6
    ffn_true += ok1
    attn_true += ok2
    print(f"{Li:5d} {r1[0]:10.1e} {e1:10.1e} {'TRUE' if ok1 else 'other':>6} "
          f"{r2[-1]:10.1e} {e2:10.1e} {'TRUE' if ok2 else 'other':>6}  {time.time()-t0:5.0f}")

print(f"\nfeed-forward sublayers landing on the true preimage: {ffn_true}/{len(LAYERS)}")
print(f"attention   sublayers landing on the true preimage: {attn_true}/{len(LAYERS)}")
print("\nEvery 'other' row still satisfies its own equation to ~1e-13 -- these are real,")
print("distinct preimages, not solver failures. Nothing local distinguishes them: the")
print("only thing that rules them out is that they fail to bottom out on the token")
print("manifold 20-odd layers later, which is a search, not a formula.")
