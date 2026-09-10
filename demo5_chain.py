"""DEMO 5 -- full backward chain, given all rows of the final hidden state.

Recipe assembled from the experiments in FINDINGS.md:

  feed-forward sublayer : direct Newton. The preimage is unique in practice, so one
                          solve suffices; verify by replay.
  attention sublayer    : preconditioned Newton finds *a* preimage, but not always the
                          right one -- there are genuinely several. So ENUMERATE roots by
                          Brown-Gearhart deflation (repulsion strength swept, since the
                          right scale depends on how far apart the roots are), then CHOOSE
                          among them with a reachable-set test: a real layer-L state lies
                          mostly inside the subspace that real states occupy, and the
                          impostors measurably do not.
  embedding block       : exact, one matmul.

Only the replay check and the reachable-set statistics are used to choose -- never the
ground truth. Ground truth is printed alongside purely to score the run.
"""
import time
import torch
from fastinv import (load, forward_states, prev_layernorms, ln, on_manifold,
                     invert_ffn, replay_attn, recover_tokens, attn_delta)
from fastinv import _newton_attn

PHRASE = "magic is real"
NREF = 200                      # random phrases used to characterise the reachable set
KPCA = 30
ALPHAS = (3e2, 1e3, 3e3)        # deflation strengths to sweep
torch.manual_seed(0)

tok, model = load()
ids = tok(PHRASE, return_tensors="pt")["input_ids"]
true_toks = tok.convert_ids_to_tokens(ids[0])
A, H = forward_states(model, ids)
n, d = H[0].shape
lnp = prev_layernorms(model)
V = model.config.vocab_size

print(f"phrase {PHRASE!r} -> {true_toks}")
print(f"given: final hidden states {tuple(H[-1].shape)}\n")

# ---- characterise the reachable set at every layer, in ONE forward sweep -------------
t0 = time.time()
bank = [[] for _ in range(25)]
for _ in range(NREF):
    seq = torch.cat([torch.tensor([101]),
                     torch.randint(1996, V, (n - 2,)), torch.tensor([102])])[None]
    _, Hk = forward_states(model, seq)
    for L in range(25):
        bank[L].append(Hk[L])
stats = []
for L in range(25):
    S_ = torch.cat(bank[L], 0)
    mu = S_.mean(0)
    _, _, Vt = torch.linalg.svd(S_ - mu, full_matrices=False)
    B = Vt[:KPCA]
    off = lambda x, mu=mu, B=B: (((x - mu) - ((x - mu) @ B.T) @ B).norm(dim=-1)
                                 / (x - mu).norm(dim=-1)).mean().item()
    stats.append((off, off(S_)))
print(f"reachable-set statistics for 25 layers from {NREF} phrases  [{time.time()-t0:.0f}s]\n")


def enumerate_attention_roots(layer, a, ln_in, ln_prev, off, base_off):
    """Find a preimage; if it looks unlike a real hidden state, deflate and look again."""
    base = ln(a - ln_in.bias, ln_prev)
    cands = []
    h = _newton_attn(layer, a, ln_in, ln_prev, base, iters=60)
    if replay_attn(layer, h, a, ln_in) < 1e-3:
        cands.append(h)
        if off(h) - base_off < 0.10:                 # looks plausible -> stop early
            return cands, "accepted first"
    for alpha in ALPHAS:
        hh = _newton_attn(layer, a, ln_in, ln_prev, base, iters=60,
                          deflate=tuple(cands), defl_alpha=alpha)
        if replay_attn(layer, hh, a, ln_in) < 1e-3 and \
           not any((hh - q).abs().max() < 1e-2 for q in cands):
            cands.append(hh)
    return cands, f"enumerated {len(cands)}"


y = H[-1].clone()
t0 = time.time()
for Li in range(23, -1, -1):
    layer = model.encoder.layer[Li]
    ln_in = layer.attention.output.LayerNorm
    a_rec, rep_f, ok_f = invert_ffn(layer, y, ln_in, restarts=4)
    off, base_off = stats[Li]
    cands, how = enumerate_attention_roots(layer, a_rec, ln_in, lnp[Li], off, base_off)
    if not cands:
        print(f"layer {Li:2d}  FFN replay={rep_f:.1e} | attention: NO ROOT FOUND -- stop")
        break
    pick = min(range(len(cands)), key=lambda j: abs(off(cands[j]) - base_off))
    h_rec = cands[pick]
    errs = [f"{(c-H[Li]).abs().max():.2f}" for c in cands]
    print(f"layer {Li:2d}  FFN repl={rep_f:.0e} err={(a_rec-A[Li]).abs().max():.0e} | "
          f"ATTN {how:<16} cand_errs={errs} picked #{pick} "
          f"err={(h_rec-H[Li]).abs().max():.2e} | [{time.time()-t0:4.0f}s]")
    y = h_rec

print(f"\nreconstructed h0 vs true: max abs err {(y - H[0]).abs().max():.3e}")
res = recover_tokens(y, model)
got = [r[0][0] for r in res]
for pos, r in enumerate(res):
    nm = tok.convert_ids_to_tokens([r[0][0]])[0]
    print(f"  pos {pos}: true {true_toks[pos]:8s} got {nm:12s} "
          f"{'OK' if nm==true_toks[pos] else 'MISS'}  margin x{r[1][1]/max(r[1][0],1e-30):.1f}")
print(f"\n>>> RECOVERED {tok.decode(got, skip_special_tokens=True)!r} | TRUE {PHRASE!r} | "
      f"{'MATCH' if got==ids[0].tolist() else 'MISMATCH'}   ({time.time()-t0:.0f}s)")
