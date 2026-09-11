"""Optimised end-to-end attack: continuous solve + discrete decode.

Two speed decisions matter:
  * use HF's own encoder for forward passes (it takes hidden_states directly, so it works
    for both "from tokens" and "from h0"), and BATCH candidate sequences -- ~5x per sequence
  * give LSMR an adaptive budget: a rough Newton step is fine while the residual is large,
    so early iterations use few Krylov steps and only the endgame pays for accuracy
"""
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, lsmr
from transformers import AutoTokenizer, AutoModel

torch.set_grad_enabled(False)
EPS = 1e-12
MODEL_NAME = "BAAI/bge-large-en-v1.5"


def load(name=MODEL_NAME, dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained(name)
    # eager attention: HF's fused SDPA kernel has no forward-mode AD rule, which the
    # Gauss-Newton solver needs for Jacobian-vector products.
    model = AutoModel.from_pretrained(name, attn_implementation="eager").to(dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def ln(x, mod):
    mu = x.mean(-1, keepdim=True)
    v = ((x - mu) ** 2).mean(-1, keepdim=True)
    return mod.weight * (x - mu) / torch.sqrt(v + EPS) + mod.bias


def embed_ids(model, ids):
    """token ids -> h0, batched. (B,n) -> (B,n,d)"""
    E = model.embeddings
    n = ids.shape[1]
    x = (E.word_embeddings(ids) + E.position_embeddings(torch.arange(n)[None])
         + E.token_type_embeddings(torch.zeros_like(ids)))
    return ln(x, E.LayerNorm)


def encode_h0(model, h0):
    """h0 -> final hidden states, batched, via HF's own encoder. (B,n,d) -> (B,n,d)"""
    return model.encoder(h0).last_hidden_state


def fwd_ids(model, ids):
    """token ids -> final hidden states, batched."""
    return encode_h0(model, embed_ids(model, ids))


# --------------------------------------------------------------- continuous solve
def solve_h0(model, target, x0, iters=60, tol=1e-4, verbose=False, log_every=10, tok=None):
    """Find h0 with Encoder(h0) = target, by damped Gauss-Newton.

    One unknown and one residual measured against the TRUE given data, so unlike peeling
    layer by layer nothing compounds: an error made here is not fed into the next solve.
    """
    n, d = target.shape
    eln = model.embeddings.LayerNorm

    def F(v):
        h0 = v.view(1, n, d)
        z = (h0[0] - eln.bias) / eln.weight
        return torch.cat([(encode_h0(model, h0)[0] - target).reshape(-1),
                          z.mean(1), z.pow(2).mean(1) - 1])

    x = x0.reshape(-1).clone()
    N, M = x.numel(), F(x).numel()
    mu = 1e-3
    for it in range(iters):
        F0 = F(x)
        nrm = F0.norm().item()
        if verbose and it % log_every == 0:
            print(f"      it{it:3d} |F|={nrm:.4e} mu={mu:.1e}")
        if nrm < tol:
            break
        # adaptive Krylov budget: rough steps are fine while the residual is large
        budget = 20 if nrm > 5 else (50 if nrm > 0.5 else 100)

        def jv(v):
            vt = torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float32)
            with torch.enable_grad():
                return torch.func.jvp(F, (x,), (vt,))[1].numpy()

        def jtv(v):
            vt = torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float32)
            with torch.enable_grad():
                _, f = torch.func.vjp(F, x)
                return f(vt)[0].numpy()

        Aop = LinearOperator((M, N), matvec=jv, rmatvec=jtv, dtype=np.float32)
        sol = lsmr(Aop, (-F0).numpy(), damp=mu, maxiter=budget, atol=1e-7, btol=1e-7)[0]
        step = torch.as_tensor(np.ascontiguousarray(sol), dtype=torch.float32)
        lam, ok = 1.0, False
        for _ in range(20):
            if F(x + lam * step).norm().item() < nrm:
                ok = True
                break
            lam *= 0.5
        if not ok:
            mu *= 10
            if mu > 1e4:
                break
            continue
        mu = max(mu * 0.5, 1e-8)
        x = x + lam * step
    return x.view(n, d), F(x).norm().item()


# ------------------------------------------------------------------ discrete decode
def rank_tokens(model, h0, topk=64):
    """Rank the whole vocabulary per position by fit to the embedding-block equation."""
    E = model.embeddings
    out = []
    for pos in range(h0.shape[0]):
        w = (h0[pos] - E.LayerNorm.bias) / E.LayerNorm.weight
        Q, _ = torch.linalg.qr(torch.stack([w, torch.ones_like(w)], 1))
        R = (E.word_embeddings.weight + E.position_embeddings.weight[pos]
             + E.token_type_embeddings.weight[0])
        resid = (R - (R @ Q) @ Q.T).norm(dim=1)
        v, i = torch.topk(-resid, topk)
        out.append(i)
    return out


def decode(model, h0_est, target, K=64, sweeps=6, batch=64, verbose=False, tok=None):
    """Slice the rough h0 to the nearest valid token sequence.

    The continuous inverse is ill-posed, but valid inputs are discrete, so we never need to
    pin h0 down -- only get close enough to round to the right sequence. Rank the vocabulary
    per position, then coordinate-descend, scoring each candidate by a FULL forward pass.
    Exhaustive within the shortlist, and self-verifying: the winner's residual says whether
    it is exactly right.
    """
    n = h0_est.shape[0]
    cand = rank_tokens(model, h0_est, K)
    seq = torch.stack([c[0] for c in cand])[None]
    best = (fwd_ids(model, seq)[0] - target).norm().item()
    for sw in range(sweeps):
        changed = False
        for i in range(n):
            trials = seq.repeat(K, 1)
            trials[:, i] = cand[i]
            scores = []
            for b in range(0, K, batch):
                blk = trials[b:b + batch]
                scores.append((fwd_ids(model, blk) - target).flatten(1).norm(dim=1))
            scores = torch.cat(scores)
            j = int(scores.argmin())
            if scores[j].item() < best - 1e-6:
                best = scores[j].item()
                seq = trials[j:j + 1].clone()
                changed = True
        if verbose:
            nm = tok.convert_ids_to_tokens(seq[0]) if tok else seq[0].tolist()
            print(f"      sweep {sw}: {nm} resid={best:.3e}")
        if not changed:
            break
    return seq, best


def attack(model, tok, target, n, rounds=8, chunk=15, K=64, verbose=True):
    """Full attack: alternate continuous refinement with discrete projection.

    Continuous solve gets h0 into the neighbourhood; decode rounds it to a real token
    sequence. Rounding is a HARD DECISION -- it lands exactly on the manifold of things the
    model can actually be fed -- so it wipes out accumulated drift, and the next continuous
    round starts from a clean point. If a decode ever replays the target exactly we are done
    and we KNOW it, without ever having seen the answer.
    """
    ids0 = torch.full((1, n), tok.mask_token_id)
    ids0[0, 0] = tok.cls_token_id
    ids0[0, -1] = tok.sep_token_id
    h0 = embed_ids(model, ids0)[0]
    best_seq, best_res = ids0, (fwd_ids(model, ids0)[0] - target).norm().item()
    for r in range(rounds):
        h0, fn = solve_h0(model, target, h0, iters=chunk)
        seq, res = decode(model, h0, target, K=K)
        if verbose:
            print(f"    round {r}: |F|={fn:.3e} -> decoded {tok.convert_ids_to_tokens(seq[0])} "
                  f"resid={res:.3e}")
        if res < best_res:
            best_seq, best_res = seq, res
        if best_res < 1e-3:                       # exact replay: verified, stop
            break
        # hard decision: restart the continuous solve from the decoded point if it helps
        cand_h0 = embed_ids(model, best_seq)[0]
        if (encode_h0(model, cand_h0[None])[0] - target).norm().item() < fn:
            h0 = cand_h0
    return best_seq, best_res
