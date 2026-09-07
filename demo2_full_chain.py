"""DEMO 2 -- walk all 24 encoder layers backwards and recover the phrase exactly.

This one is given the FULL final hidden-state matrix (n x 1024), not just the pooled CLS
row. That is the honest boundary of the method: given everything the encoder produced,
the encoder stack is exactly invertible by algebra alone. Demo 3 shows what pooling
takes away.
"""
import time
import torch
from inversion import (load, forward_states, prev_layernorms, invert_ffn,
                       invert_attention_continued, recover_tokens, ln, run_layer)

PHRASE = "magic is real"

tok, model = load()
ids = tok(PHRASE, return_tensors="pt")["input_ids"]
true_toks = tok.convert_ids_to_tokens(ids[0])
A, H = forward_states(model, ids)
lnp = prev_layernorms(model)
n = ids.shape[1]
print(f"phrase   : {PHRASE!r}")
print(f"tokens   : {true_toks}")
print(f"given    : final hidden states, shape {tuple(H[-1].shape)}\n")

y = H[-1].clone()
t0 = time.time()
for Li in range(23, -1, -1):
    layer = model.encoder.layer[Li]
    ln_in = layer.attention.output.LayerNorm
    a_rec, _, ok1 = invert_ffn(layer, y, ln_in)
    h_rec, _, ok2 = invert_attention_continued(layer, a_rec, ln_in, lnp[Li])
    # forward-check: does the recovered state actually reproduce what we were given?
    chk = (run_layer(layer, h_rec)[1] - y).abs().max()
    print(f"layer {Li:2d}  ffn ok={int(ok1)} err={(a_rec-A[Li]).abs().max():.1e}"
          f" | attn ok={int(ok2)} err={(h_rec-H[Li]).abs().max():.1e}"
          f" | replays to within {chk:.1e}   [{time.time()-t0:5.0f}s]")
    y = h_rec

print(f"\nreconstructed h0: max abs err {(y - H[0]).abs().max():.3e}")
got = []
for pos, (idx, resid) in enumerate(recover_tokens(y, model)):
    name = tok.convert_ids_to_tokens([idx[0]])[0]
    got.append(idx[0])
    flag = "OK " if name == true_toks[pos] else "MISS"
    print(f"  pos {pos}  {flag} recovered {name!r} at {resid[0]:.1e}"
          f"  (runner-up {tok.convert_ids_to_tokens([idx[1]])[0]!r} at {resid[1]:.1e})")
print(f"\n>>> RECOVERED {tok.decode(got, skip_special_tokens=True)!r} | TRUE {PHRASE!r}"
      f" | {'MATCH' if got == ids[0].tolist() else 'MISMATCH'}   ({time.time()-t0:.0f}s)")
