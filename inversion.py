"""Deterministic, closed-form-where-possible inversion of BAAI/bge-large-en-v1.5.

No gradient descent, no learned inverse model. Every step here is either an explicit
algebraic formula or a deterministic root-find (Newton / Newton-Krylov / a self-consistent
field iteration) on a *square* system of equations.

The pipeline being inverted, for a post-LN BERT encoder:

    h0        = LN_e( E[t] + P[pos] + T[0] )                      embedding block
    a_L       = LN_a( h_L + Attn_L(h_L) )                         attention sublayer
    h_{L+1}   = LN_o( a_L + FFN_L(a_L) )                          feed-forward sublayer
    embedding = h_24[CLS] / || h_24[CLS] ||                       pooling + normalisation

The one idea that makes this tractable:

    LayerNorm looks lossy -- it throws away the input's mean and variance, 2 numbers per
    row -- but its OUTPUT is confined to a known (d-2)-dimensional manifold:
    z = (x - beta)/gamma always has mean 0 and mean-square 1. Every tensor we need to
    recover is itself the output of some LayerNorm, so those 2 lost numbers are pinned by
    2 extra equations. That turns each sublayer inversion into a square system with
    (generically) isolated solutions instead of an underdetermined family.
"""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

torch.set_default_dtype(torch.float64)          # float32 is not enough to peel 24 layers
EPS = 1e-12
MODEL_NAME = "BAAI/bge-large-en-v1.5"


def load(name=MODEL_NAME):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).double().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def ln(x, mod):
    mu = x.mean(-1, keepdim=True)
    v = ((x - mu) ** 2).mean(-1, keepdim=True)
    return mod.weight * (x - mu) / torch.sqrt(v + EPS) + mod.bias


def prev_layernorms(model):
    """For each encoder layer, the LayerNorm whose output feeds it."""
    return [model.embeddings.LayerNorm] + [l.output.LayerNorm for l in model.encoder.layer[:-1]]


# --------------------------------------------------------------------------- forward
def embed(model, ids):
    E = model.embeddings
    n = ids.shape[1]
    x = (E.word_embeddings(ids)
         + E.position_embeddings(torch.arange(n)[None])
         + E.token_type_embeddings(torch.zeros_like(ids)))
    return ln(x, E.LayerNorm)


def attn_probs(layer, h, nh=16):
    sa = layer.attention.self
    n, d = h.shape
    dh = d // nh
    shp = lambda t: t.view(n, nh, dh).transpose(0, 1)
    q = shp(h @ sa.query.weight.T + sa.query.bias)
    k = shp(h @ sa.key.weight.T + sa.key.bias)
    return torch.softmax(q @ k.transpose(-1, -2) / np.sqrt(dh), -1)          # (nh, n, n)


def value_path(layer, h, P, nh=16, with_bias=True):
    """W_O concat_heads(P_head V_head) (+ biases). Linear in h when with_bias=False."""
    sa, ao = layer.attention.self, layer.attention.output
    n, d = h.shape
    dh = d // nh
    v = h @ sa.value.weight.T + (sa.value.bias if with_bias else 0)
    ctx = (P @ v.view(n, nh, dh).transpose(0, 1)).transpose(0, 1).reshape(n, d)
    return ctx @ ao.dense.weight.T + (ao.dense.bias if with_bias else 0)


def attn_delta(layer, h):
    return value_path(layer, h, attn_probs(layer, h), with_bias=True)


def ffn_delta(layer, a):
    g = torch.nn.functional.gelu(a @ layer.intermediate.dense.weight.T
                                 + layer.intermediate.dense.bias)
    return g @ layer.output.dense.weight.T + layer.output.dense.bias


def run_layer(layer, h):
    a = ln(h + attn_delta(layer, h), layer.attention.output.LayerNorm)
    return a, ln(a + ffn_delta(layer, a), layer.output.LayerNorm)


def forward_states(model, ids):
    """Return (h0, [a_L], [h_L]) computed from raw weights."""
    h = embed(model, ids)[0]
    H, A = [h], []
    for layer in model.encoder.layer:
        a, h = run_layer(layer, h)
        A.append(a)
        H.append(h)
    return A, H


def sentence_embedding(H):
    cls = H[-1][0]
    return cls / cls.norm()


# ------------------------------------------------------------------- step 1: unnormalise
def undo_l2(u, ln_last):
    """emb = h/||h||, and h is a LayerNorm output. Recover h EXACTLY, in closed form.

    h = s*u for some s>0. z = (h - beta)/gamma must have mean 0, so
        s * mean(u/gamma) = mean(beta/gamma)
    which is one linear equation in one unknown -- no search, no ambiguity. The second
    LayerNorm constraint, mean(z^2) = 1, is then redundant and serves as a check.
    """
    g, b = ln_last.weight, ln_last.bias
    s = (b / g).mean() / (u / g).mean()
    z = s * (u / g) - b / g
    return s * u, s.item(), (z.pow(2).mean() - 1).abs().item()


# ---------------------------------------------------- step 2: invert the FFN sublayer
def _ffn_row_pieces(layer, ln_in, w, d):
    W1, b1 = layer.intermediate.dense.weight, layer.intermediate.dense.bias
    W2, b2 = layer.output.dense.weight, layer.output.dense.bias
    g, b = ln_in.weight, ln_in.bias
    W1f, W2f, I = W1.float(), W2.float(), torch.eye(d)

    def resid(X, t):
        a, s, m = X[:d], X[d], X[d + 1]
        pre = a @ W1.T + b1
        f = torch.nn.functional.gelu(pre) @ W2.T + b2
        z = (a - b) / g
        return torch.cat([a + t * f - s * w - m,
                          torch.stack([z.mean(), z.pow(2).mean() - 1])]), pre, z, f

    def jac(X, t, pre, z):
        dg = (0.5 * (1 + torch.erf(pre / np.sqrt(2)))
              + pre * torch.exp(-pre ** 2 / 2) / np.sqrt(2 * np.pi)).float()
        J = torch.zeros(d + 2, d + 2)
        J[:d, :d] = I + t * ((W2f * dg) @ W1f).double()
        J[:d, d] = -w
        J[:d, d + 1] = -1
        J[d, :d] = (1 / g) / d
        J[d + 1, :d] = 2 * z / g / d
        return J

    return resid, jac


def _newton_at(resid, jac, X, t, iters=25, tol=1e-11):
    for _ in range(iters):
        F, pre, z, f = resid(X, t)
        nrm = F.norm()
        if nrm < tol:
            return X, nrm.item(), True
        step = torch.linalg.solve(jac(X, t, pre, z), -F)
        d = X.shape[0] - 2
        lam = 1.0
        for _ in range(40):
            # s is a LayerNorm standard deviation: s <= 0 solves our constraint equations
            # but NOT the original LN(x + delta(x)) = y, so those roots must be rejected.
            if X[d] + lam * step[d] > 0 and resid(X + lam * step, t)[0].norm() < nrm:
                break
            lam *= 0.5
        X = X + lam * step
    return X, resid(X, t)[0].norm().item(), resid(X, t)[0].norm().item() < 1e-9


def invert_ffn(layer, y, ln_in, max_steps=120, verbose=False):
    """Recover a from y = LN_out(a + FFN(a)), knowing a lies on ln_in's output manifold.

    Per row: unknowns a (d) plus the scale s and shift m LayerNorm discarded; equations
    a + FFN(a) = s*w + m (w = (y-beta)/gamma) plus the 2 manifold constraints. Square,
    and rows are independent -- the feed-forward block touches one position at a time.

    GELU is not injective, so this square system has several isolated roots and plain
    Newton happily lands on a wrong one. We therefore trace the branch: scale the FFN
    contribution by t, start at t=0 (where the system is linear and the root unique), and
    walk t to 1 using a tangent predictor with adaptive step control. Predictor:
    J X' = -dF/dt = -FFN(a). If a step fails to converge, or the solution moves far more
    than the tangent said it should, the step is halved -- that is what stops the trace
    from hopping onto a neighbouring branch.
    """
    ln_out = layer.output.LayerNorm
    n, d = y.shape
    W = (y - ln_out.bias) / ln_out.weight
    a0 = ln(y - ln_out.bias, ln_in)
    out = torch.empty_like(y)
    worst_r, all_ok = 0.0, True
    for i in range(n):
        resid, jac = _ffn_row_pieces(layer, ln_in, W[i], d)
        X = torch.cat([a0[i], torch.ones(1), torch.zeros(1)])
        X, r, ok = _newton_at(resid, jac, X, 0.0)
        t, hstep, nsteps = 0.0, 0.125, 0
        while t < 1.0 - 1e-12 and nsteps < max_steps:
            nsteps += 1
            F, pre, z, f = resid(X, t)
            tang = torch.linalg.solve(jac(X, t, pre, z),
                                      -torch.cat([f, torch.zeros(2)]))
            hstep = min(hstep, 1.0 - t)
            while True:
                Xn, r, ok = _newton_at(resid, jac, X + hstep * tang, t + hstep)
                drift = (Xn - X - hstep * tang).abs().max()
                if ok and drift < 0.25 * hstep * tang.abs().max().clamp_min(1e-3):
                    break
                hstep *= 0.5
                if hstep < 1e-5:
                    all_ok = False
                    break
            if hstep < 1e-5:
                break
            t += hstep
            X = Xn
            hstep = min(hstep * 1.7, 1.0 - t if t < 1.0 else hstep)
        out[i] = X[:d]
        worst_r = max(worst_r, r)
        all_ok = all_ok and ok and nsteps < max_steps
        if verbose:
            print(f"        ffn row {i}: |F|={r:.2e} ok={ok}")
    return out, [worst_r], all_ok and worst_r < 1e-8


# ---------------------------------------------- step 3: invert the attention sublayer
def _frozen_operator(layer, P, n, d, nh=16):
    """Assemble I + T_P densely, where T_P = sum_heads P_head (x) M_head and
    M_head = W_O[:, head] @ W_V[head, :] is fixed for the layer. Cheap to build; the
    resulting (n*d) x (n*d) matrix is then LU-factored once and reused for every
    right-hand side in this SCF iteration."""
    dh = d // nh
    Wv, Wo = layer.attention.self.value.weight, layer.attention.output.dense.weight
    M = torch.stack([Wo[:, h * dh:(h + 1) * dh] @ Wv[h * dh:(h + 1) * dh, :] for h in range(nh)])
    #        (n,n,d,d) = sum_h P[h,i,j] * M[h]
    blocks = torch.einsum('hij,hpq->ipjq', P, M).reshape(n * d, n * d)
    blocks += torch.eye(n * d)
    return torch.linalg.lu_factor(blocks)


def invert_attention(layer, a, ln_in, ln_prev, h_init, iters=80, tol=1e-11,
                     anderson=4, scale=1.0, verbose=False):
    """Recover h from a = ln_in(h + Attn(h)), knowing h lies on ln_prev's manifold.

    Attn is LINEAR in h through the value path once the softmax probabilities P are
    frozen. So we alternate: read P off the current h (16*n*n numbers -- a tiny object),
    then solve the sublayer equation EXACTLY with P held fixed, then recompute P.

    With P frozen, h is an affine function of the 2n LayerNorm unknowns (s, m):
    solve (I+T_P) x = rhs once per basis direction, then impose the 2n manifold
    constraints by Newton on a 2n-dimensional system. Nothing here is a search.
    """
    n, d = a.shape
    w = (a - ln_in.bias) / ln_in.weight
    g, b = ln_prev.weight, ln_prev.bias
    eye2n = torch.eye(2 * n)

    rhs = torch.zeros(2 * n + 1, n, d)                 # ds_i, dm_i, and the bias term
    for i in range(n):
        rhs[i, i] = w[i]
        rhs[n + i, i] = 1.0

    h = h_init.clone()
    sm = torch.cat([torch.ones(n), torch.zeros(n)])
    hist, Xs, Gs = [], [], []
    for it in range(iters):
        P = attn_probs(layer, h)
        rhs[2 * n] = scale * value_path(layer, torch.zeros(n, d), P, with_bias=True)
        LU = _frozen_operator(layer, scale * P, n, d)
        sols = torch.linalg.lu_solve(*LU, rhs.reshape(2 * n + 1, n * d).T).T.view(2 * n + 1, n, d)
        U, V, p = sols[:n], sols[n:2 * n], sols[2 * n]

        def h_of(sm):
            return (sm[:n, None] * U.view(n, -1)).sum(0).view(n, d) \
                 + (sm[n:, None] * V.view(n, -1)).sum(0).view(n, d) - p

        def cons(sm):
            z = (h_of(sm) - b) / g
            return torch.cat([z.mean(1), z.pow(2).mean(1) - 1])

        for _ in range(60):                            # Newton on the tiny (s,m) system
            r = cons(sm)
            rn = r.norm()
            if rn < 1e-13:
                break
            J = torch.stack([(cons(sm + 1e-7 * eye2n[j]) - r) / 1e-7 for j in range(2 * n)], 1)
            step = torch.linalg.lstsq(J, -r.unsqueeze(1)).solution.squeeze(1)
            t = 1.0
            for _ in range(40):
                cand = sm + t * step
                if (cand[:n] > 0).all() and cons(cand).norm() < rn:
                    break
                t *= 0.5
            sm = sm + t * step

        h_new = h_of(sm)
        gk = h_new - h
        delta = gk.abs().max().item()
        hist.append(delta)
        if verbose:
            print(f"        scf{it:2d} |dh|={delta:.3e}")
        if delta < tol:
            return h, hist, True
        if anderson:                                   # Anderson(m) on the SCF map
            Xs.append(h.reshape(-1)); Gs.append(gk.reshape(-1))
            Xs, Gs = Xs[-anderson:], Gs[-anderson:]
            if len(Gs) > 1:
                dG = torch.stack([Gs[j + 1] - Gs[j] for j in range(len(Gs) - 1)], 1)
                dX = torch.stack([Xs[j + 1] - Xs[j] for j in range(len(Xs) - 1)], 1)
                gam = torch.linalg.lstsq(dG, Gs[-1].unsqueeze(1)).solution.squeeze(1)
                cand = (Xs[-1] + Gs[-1] - (dX + dG) @ gam).view(n, d)
                if torch.isfinite(cand).all() and (cand - h).abs().max() < 50 * max(delta, 1e-9):
                    h = cand
                    continue
        h = h_new
    return h, hist, False


def invert_attention_continued(layer, a, ln_in, ln_prev, nt=5, verbose=False):
    """Same continuation trick as the FFN: ramp the attention contribution from 0 to 1,
    tracking the root, so we follow the branch the forward pass actually took."""
    h = ln(a - ln_in.bias, ln_prev)
    ok = False
    for k in range(nt + 1):
        h, hist, ok = invert_attention(layer, a, ln_in, ln_prev, h,
                                       scale=k / nt, verbose=verbose)
    return h, hist, ok


def invert_attention_best(layer, a, ln_in, ln_prev, verbose=False):
    """Run both branch-tracing strategies and keep whichever actually solves the equation.

    Neither plain SCF nor continuation is reliable on its own -- each succeeds on layers
    where the other fails. But a candidate is CHECKABLE without knowing the answer: replay
    it forwards and see whether it reproduces the observed output. So try both and keep
    the one that replays. This is verification, not search: two candidates, one test.
    """
    best, best_rep, best_hist = None, float('inf'), [float('inf')]
    for init, sc in ((ln(a - ln_in.bias, ln_prev), None), (None, 5)):
        if sc is None:
            h, hist, _ = invert_attention(layer, a, ln_in, ln_prev, init)
        else:
            h, hist, _ = invert_attention_continued(layer, a, ln_in, ln_prev, nt=sc)
        rep = (ln(h + attn_delta(layer, h), ln_in) - a).abs().max().item()
        if verbose:
            print(f"        attn {'plain' if sc is None else 'cont '}: replay={rep:.1e}")
        if rep < best_rep:
            best, best_rep, best_hist = h, rep, hist
    return best, best_hist, best_rep < 1e-8


def invert_layer(layer, y, ln_prev, verbose=False):
    """Invert one whole encoder layer: y -> (a, h)."""
    ln_in = layer.attention.output.LayerNorm
    a, h1, ok1 = invert_ffn(layer, y, ln_in, verbose=verbose)
    h, h2, ok2 = invert_attention_best(layer, a, ln_in, ln_prev, verbose=verbose)
    return a, h, (ok1, ok2), (h1, h2)


# ------------------------------------------------ step 4: invert the embedding block
def recover_tokens(h0, model, topk=3):
    """Invert the embedding block EXACTLY, with no search over sequences.

    h0 = LN(E[t] + P[pos] + T[0]) means E[t] + P[pos] + T[0] = s*w + m*1 for the known
    w = (h0 - beta)/gamma, i.e. it must lie in a known 2-dimensional affine plane.
    So score every one of the 30522 vocabulary rows by its distance to that plane: the
    true token sits at ~1e-15, everything else at ~1. One matmul per position.
    """
    E = model.embeddings
    out = []
    for pos in range(h0.shape[0]):
        w = (h0[pos] - E.LayerNorm.bias) / E.LayerNorm.weight
        Q, _ = torch.linalg.qr(torch.stack([w, torch.ones_like(w)], 1))
        R = (E.word_embeddings.weight + E.position_embeddings.weight[pos]
             + E.token_type_embeddings.weight[0])
        resid = (R - (R @ Q) @ Q.T).norm(dim=1)
        v, i = torch.topk(-resid, topk)
        out.append((i.tolist(), (-v).tolist()))
    return out
