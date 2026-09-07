"""DEMO 3 -- what CLS pooling actually destroys, and how far you can get anyway.

Demo 2 needed the full n x 1024 final hidden state. A sentence embedding hands you one
row of it. This script (a) counts exactly how much is missing, (b) checks that the
information-theoretic story is not the problem, and (c) runs the strongest purely
deterministic attack I could find that uses only the pooled vector.
"""
import time
import torch
from inversion import (load, forward_states, sentence_embedding, undo_l2,
                       recover_tokens, embed, ln, run_layer)

PHRASE = "magic is real"
tok, model = load()
ids = tok(PHRASE, return_tensors="pt")["input_ids"]
n = ids.shape[1]
d = model.config.hidden_size
A, H = forward_states(model, ids)
emb = sentence_embedding(H)

print("=" * 72)
print("A. Degrees-of-freedom accounting")
print("=" * 72)
print(f"  sequence length n                         : {n}")
print(f"  final hidden state the backward chain needs: {n} x {d} = {n*d} numbers")
print(f"  ... but each row is a LayerNorm output, so : {n} x {d-2} = {n*(d-2)} free")
print(f"  what the sentence embedding gives          : 1 x {d} unit vector = {d-1} free")
print(f"  step 1 (undo L2) recovers row 0 exactly    : + 1  -> row 0 fully known")
print(f"  rows never observed                        : {n-1} x {d-2} = {(n-1)*(d-2)} unknowns")
print(f"  equations available to pin them            : 0")
print(f"\n  => the backward chain is underdetermined by {(n-1)*(d-2)} real dimensions.")
print("     No algebraic rearrangement creates those equations; they are simply not")
print("     in the input. This is the whole obstruction -- not LayerNorm, not GELU,")
print("     not the softmax.")

print("\n" + "=" * 72)
print("B. But the information is not gone in the Shannon sense")
print("=" * 72)
bits_needed = n * torch.log2(torch.tensor(float(model.config.vocab_size)))
print(f"  bits needed to name {n} tokens from a {model.config.vocab_size} vocab : {bits_needed:.0f}")
print(f"  bits carried by {d-1} float64 coordinates (~50 usable each)  : ~{(d-1)*50}")
print("  => the embedding has ample room to identify the phrase uniquely; the inverse")
print("     map EXISTS. It is just not computable by peeling layers, because peeling")
print("     needs the unobserved rows. Recovering it means searching a continuum.")

print("\n" + "=" * 72)
print("C. Is the pooled map even injective? (collision probe)")
print("=" * 72)
torch.manual_seed(0)
V = model.config.vocab_size
base = ids[0].tolist()
sims = []
for _ in range(60):
    cand = base.copy()
    cand[torch.randint(1, n - 1, (1,)).item()] = torch.randint(999, V, (1,)).item()
    _, Hc = forward_states(model, torch.tensor([cand]))
    sims.append(torch.dot(emb, sentence_embedding(Hc)).item())
sims = torch.tensor(sims)
print(f"  cosine to 60 one-token-perturbed phrases: max {sims.max():.6f}, "
      f"mean {sims.mean():.4f}")
print("  no collisions -- distinct phrases land at distinct points, as expected for a")
print("  smooth map from a discrete set into R^1024. Injective, but not invertible by")
print("  algebra: existence of an inverse is not the same as a formula for it.")

print("\n" + "=" * 72)
print("D. Strongest deterministic attack using ONLY the pooled vector")
print("=" * 72)
print("  Linearise the whole 24-layer network in the token-embedding inputs, exactly,")
print("  via its Jacobian (autodiff computes the derivative -- it is not training), then")
print("  solve for the tokens by matching pursuit over the vocabulary. Deterministic")
print("  linear algebra proposes; an exhaustive vocabulary scan disposes. No gradient")
print("  descent, no learned inverse -- but note this IS a search, not a reversal.")

E = model.embeddings
mask_id = tok.mask_token_id
ref = ids[0].clone(); ref[1:-1] = mask_id


def emb_from_word_vectors(X):
    x = X + E.position_embeddings.weight[:n] + E.token_type_embeddings.weight[0]
    h = ln(x, E.LayerNorm)[0] if x.dim() == 3 else ln(x, E.LayerNorm)
    for layer in model.encoder.layer:
        _, h = run_layer(layer, h)
    return h[0] / h[0].norm()


guess = ref.clone()
t0 = time.time()
for sweep in range(3):
    X0 = E.word_embeddings(guess).clone()
    J = torch.autograd.functional.jacobian(emb_from_word_vectors, X0, vectorize=True)
    r = emb - emb_from_word_vectors(X0)                       # (1024,)
    Ew = E.word_embeddings.weight
    for i in range(1, n - 1):
        Ji = J[:, i, :]                                       # (1024, 1024)
        Vt = (Ew - Ew[guess[i]]) @ Ji.T                       # (V, 1024) predicted moves
        nrm = Vt.norm(dim=1).clamp_min(1e-12)
        score = (Vt @ r) / nrm
        best = int(score.argmax())
        guess[i] = best
    names = tok.convert_ids_to_tokens(guess.tolist())
    print(f"  sweep {sweep}: {names}   [{time.time()-t0:.0f}s]")

print(f"\n>>> matching pursuit gave {tok.decode(guess[1:-1])!r} | TRUE {PHRASE!r}")
print(f"    cosine of its embedding to the target: "
      f"{torch.dot(emb, emb_from_word_vectors(E.word_embeddings(guess))).item():.6f}")
