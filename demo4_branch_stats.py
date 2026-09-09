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
                       invert_attention_best, ln, run_layer, attn_delta, ffn_delta)

PHRASE = "magic is real"
tok, model = load()
ids = tok(PHRASE, return_tensors="pt")["input_ids"]
A, H = forward_states(model, ids)
lnp = prev_layernorms(model)

print(f"phrase {PHRASE!r}, {ids.shape[1]} tokens\n")
print(f"{'layer':>5} {'ffn err':>10} {'ffn replay':>11} {'ffn':>7} "
      f"{'attn err':>10} {'attn replay':>12} {'attn':>7}  {'s':>5}")
def verdict(err, replay):
    if err < 1e-6: return 'TRUE'
    return 'other' if replay < 1e-8 else 'FAILED'
ffn_true = attn_true = 0
t0 = time.time()
LAYERS = [23, 20, 16, 12, 8, 4]
for Li in LAYERS:
    layer = model.encoder.layer[Li]
    ln_in = layer.attention.output.LayerNorm
    a_rec, r1, _ = invert_ffn(layer, H[Li + 1], ln_in)
    e1 = (a_rec - A[Li]).abs().max().item()
    # replay: a wrong-but-valid preimage still reproduces the output; a failed solve does not
    p1 = (ln(a_rec + ffn_delta(layer, a_rec), layer.output.LayerNorm) - H[Li + 1]).abs().max().item()
    h_rec, r2, _ = invert_attention_best(layer, A[Li], ln_in, lnp[Li])
    e2 = (h_rec - H[Li]).abs().max().item()
    p2 = (ln(h_rec + attn_delta(layer, h_rec), ln_in) - A[Li]).abs().max().item()
    ok1, ok2 = e1 < 1e-6, e2 < 1e-6
    ffn_true += ok1
    attn_true += ok2
    print(f"{Li:5d} {e1:10.1e} {p1:11.1e} {verdict(e1,p1):>7} "
          f"{e2:10.1e} {p2:12.1e} {verdict(e2,p2):>7}  {time.time()-t0:5.0f}")

print(f"\nfeed-forward sublayers landing on the true preimage: {ffn_true}/{len(LAYERS)}")
print(f"attention   sublayers landing on the true preimage: {attn_true}/{len(LAYERS)}")
print("\nTRUE   = landed on the preimage the forward pass used.")
print("other  = a DIFFERENT preimage: it replays the observed output to ~1e-13, so it is")
print("         a real solution of the same equations, not a solver failure. Nothing")
print("         local distinguishes it; the only thing that rules it out is failing to")
print("         bottom out on the token manifold 20-odd layers later -- a search, not a")
print("         formula.")
print("FAILED = the solve did not reach a solution at all (does not replay the output).")
