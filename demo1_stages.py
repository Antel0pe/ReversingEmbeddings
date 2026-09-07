"""DEMO 1 -- the pieces that invert exactly, verified one at a time.

Everything here is checked against the true forward pass, so the numbers are honest
errors, not self-consistency. Runtime a few minutes.
"""
import time
import torch
from inversion import (load, embed, forward_states, sentence_embedding, ln, run_layer,
                       undo_l2, invert_ffn, invert_attention_continued, recover_tokens,
                       prev_layernorms, attn_delta, ffn_delta)

PHRASE = "magic is real"
tok, model = load()
ids = tok(PHRASE, return_tensors="pt")["input_ids"]
toks = tok.convert_ids_to_tokens(ids[0])
A, H = forward_states(model, ids)
emb = sentence_embedding(H)
lnp = prev_layernorms(model)
n, d = H[0].shape

print("=" * 74)
print("0. The forward pass, rebuilt from raw weights (nothing to invert yet)")
print("=" * 74)
ref = model(**tok(PHRASE, return_tensors="pt"), output_hidden_states=True)
print(f"  our h0  vs transformers': {(H[0]-ref.hidden_states[0][0]).abs().max():.2e}")
print(f"  our h24 vs transformers': {(H[-1]-ref.hidden_states[-1][0]).abs().max():.2e}")
print("  -> we are inverting the real model, in float64, not an approximation of it.")

print("\n" + "=" * 74)
print("1. Undo the L2 normalisation -- EXACT, closed form")
print("=" * 74)
print("  The pooled vector is h/||h||, so the norm looks lost. But h is a LayerNorm")
print("  output: z = (h-beta)/gamma has mean 0. Writing h = s*u gives one linear")
print("  equation  s * mean(u/gamma) = mean(beta/gamma)  in the one unknown s.")
h_cls, s, check = undo_l2(emb, model.encoder.layer[-1].output.LayerNorm)
print(f"  recovered scale s = {s:.6f}   true ||h|| = {H[-1][0].norm():.6f}")
print(f"  max abs error = {(h_cls-H[-1][0]).abs().max():.2e}")
print(f"  the second LayerNorm constraint mean(z^2)=1 was not used; it checks out")
print(f"  to {check:.1e}, confirming the solution rather than constraining it.")

print("\n" + "=" * 74)
print("2. LayerNorm is NOT the wall it looks like")
print("=" * 74)
print("  Read forwards, LayerNorm destroys the input's mean and variance -- 2 numbers")
print("  per row, an infinite family of inputs per output. Read backwards inside a")
print("  transformer, those 2 numbers come back, because every tensor we need is")
print("  ITSELF a LayerNorm output and so obeys 2 equations:")
print("        mean(z) = 0        mean(z^2) = 1,      z = (x - beta)/gamma")
z = (H[12] - model.encoder.layer[11].output.LayerNorm.bias) / model.encoder.layer[11].output.LayerNorm.weight
print(f"  check on h_12: mean(z) = {z.mean():.2e}, mean(z^2)-1 = {(z.pow(2).mean()-1):.2e}")
print("  So each sublayer inversion is a SQUARE system: d unknowns for the row plus the")
print("  2 discarded scalars, against d sublayer equations plus those 2 constraints.")
print("  That is the move that makes the rest of this possible.")

print("\n" + "=" * 74)
print("3. Invert the embedding block -- EXACT, one matmul, no sequence search")
print("=" * 74)
print("  h0 = LN(E[t] + P[pos] + T[0]) forces E[t] + P[pos] + T[0] into the known 2-D")
print("  plane span{w, 1}, w = (h0-beta)/gamma. Score all 30522 rows by distance to it.")
for pos, (idx, resid) in enumerate(recover_tokens(H[0], model)):
    names = [tok.convert_ids_to_tokens([i])[0] for i in idx]
    print(f"  pos {pos}: true {toks[pos]:8s} -> {names[0]:10s} at {resid[0]:.1e} | "
          f"next {names[1]:10s} at {resid[1]:.1e}")
print("  A 15-order-of-magnitude gap. Given h0 the tokens are not guessed, they are read.")

print("\n" + "=" * 74)
print("4. Invert the two sublayer types (input = the true layer output)")
print("=" * 74)
t0 = time.time()
for Li in (23, 20):
    layer = model.encoder.layer[Li]
    ln_in = layer.attention.output.LayerNorm
    a_rec, r1, _ = invert_ffn(layer, H[Li + 1], ln_in)
    replay = ln(a_rec + ffn_delta(layer, a_rec), layer.output.LayerNorm)
    print(f"  layer {Li} FFN : |F|={r1[0]:.1e}  err vs true a = {(a_rec-A[Li]).abs().max():.2e}"
          f"  replays output to {(replay-H[Li+1]).abs().max():.1e}")
    h_rec, r2, _ = invert_attention_continued(layer, A[Li], ln_in, lnp[Li])
    replay = ln(h_rec + attn_delta(layer, h_rec), ln_in)
    print(f"  layer {Li} ATTN: |dh|={r2[-1]:.1e} err vs true h = {(h_rec-H[Li]).abs().max():.2e}"
          f"  replays output to {(replay-A[Li]).abs().max():.1e}")
print(f"  [{time.time()-t0:.0f}s]")
print("\n  Both sublayer types invert to machine precision when the trace follows the")
print("  right branch. demo4_branch_stats.py measures how often it does.")
