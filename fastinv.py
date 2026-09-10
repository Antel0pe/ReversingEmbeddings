"""Fast experimental harness for inverting bge-large-en-v1.5.

Differences from inversion.py, all justified by the goal being *recover the phrase*
rather than *match to 11 decimals*:

  * float32 throughout. The token readout tolerates ~0.1 absolute error in h0 (measured);
    float32 carries ~7 digits, which is ample, and it roughly halves every matmul and
    factorisation.
  * No homotopy continuation. Multistart testing showed the feed-forward preimage is
    unique in practice, so the 120-step branch trace was buying nothing at ~20x the cost.
    Direct Newton from a decent start, ~6 iterations.
  * Multistart + replay as the universal safety net: if a solve does not reproduce the
    observed output, retry from a different point on the LayerNorm manifold.

Everything is verified by REPLAY (push the answer forwards, compare to what we were given),
which needs no ground truth and so is available during a real inversion.
"""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

torch.set_grad_enabled(False)
EPS = 1e-12
MODEL_NAME = "BAAI/bge-large-en-v1.5"


def load(name=MODEL_NAME, dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def ln(x, mod):
    mu = x.mean(-1, keepdim=True)
    v = ((x - mu) ** 2).mean(-1, keepdim=True)
    return mod.weight * (x - mu) / torch.sqrt(v + EPS) + mod.bias


def prev_layernorms(model):
    return [model.embeddings.LayerNorm] + [l.output.LayerNorm for l in model.encoder.layer[:-1]]


def on_manifold(ln_mod, ref):
    """A random point on ln_mod's output manifold, shaped like ref."""
    r = torch.randn_like(ref)
    r = (r - r.mean(-1, keepdim=True)) / r.std(-1, keepdim=True, unbiased=False)
    return ln_mod.weight * r + ln_mod.bias


# ------------------------------------------------------------------ forward
def embed(model, ids):
    E = model.embeddings
    n = ids.shape[1]
    x = (E.word_embeddings(ids) + E.position_embeddings(torch.arange(n)[None])
         + E.token_type_embeddings(torch.zeros_like(ids)))
    return ln(x, E.LayerNorm)


def attn_probs(layer, h, nh=16):
    sa = layer.attention.self
    n, d = h.shape
    dh = d // nh
    shp = lambda t: t.view(n, nh, dh).transpose(0, 1)
    q = shp(h @ sa.query.weight.T + sa.query.bias)
    k = shp(h @ sa.key.weight.T + sa.key.bias)
    return torch.softmax(q @ k.transpose(-1, -2) / np.sqrt(dh), -1)


def value_path(layer, h, P, nh=16, with_bias=True):
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
    h = embed(model, ids)[0]
    H, A = [h], []
    for layer in model.encoder.layer:
        a, h = run_layer(layer, h)
        A.append(a)
        H.append(h)
    return A, H


# ------------------------------------------------------------- replay checks
def replay_ffn(layer, a, y):
    return (ln(a + ffn_delta(layer, a), layer.output.LayerNorm) - y).abs().max().item()


def replay_attn(layer, h, a, ln_in):
    return (ln(h + attn_delta(layer, h), ln_in) - a).abs().max().item()


# ------------------------------------------------- feed-forward sublayer inverse
def _ffn_newton_row(layer, w, ln_in, X, iters=40, tol=1e-5):
    """Newton on one row of  a + FFN(a) = s*w + m,  a on ln_in's manifold."""
    W1, b1 = layer.intermediate.dense.weight, layer.intermediate.dense.bias
    W2, b2 = layer.output.dense.weight, layer.output.dense.bias
    g, b = ln_in.weight, ln_in.bias
    d = w.shape[0]
    I = torch.eye(d)

    def resid(X):
        a, s, m = X[:d], X[d], X[d + 1]
        pre = a @ W1.T + b1
        f = torch.nn.functional.gelu(pre) @ W2.T + b2
        z = (a - b) / g
        return torch.cat([a + f - s * w - m,
                          torch.stack([z.mean(), z.pow(2).mean() - 1])]), pre, z

    for _ in range(iters):
        F, pre, z = resid(X)
        nrm = F.norm()
        if nrm < tol:
            break
        dg = 0.5 * (1 + torch.erf(pre / np.sqrt(2))) \
             + pre * torch.exp(-pre ** 2 / 2) / np.sqrt(2 * np.pi)
        J = torch.zeros(d + 2, d + 2)
        J[:d, :d] = I + (W2 * dg) @ W1
        J[:d, d] = -w
        J[:d, d + 1] = -1
        J[d, :d] = (1 / g) / d
        J[d + 1, :d] = 2 * z / g / d
        step = torch.linalg.solve(J, -F)
        lam = 1.0
        for _ in range(25):
            cand = X + lam * step
            if cand[d] > 0 and resid(cand)[0].norm() < nrm:
                break
            lam *= 0.5
        X = X + lam * step
    return X, resid(X)[0].norm().item()


def invert_ffn(layer, y, ln_in, restarts=6, tol=1e-4, verbose=False):
    """Recover a from y = LN_out(a + FFN(a)). Rows are independent. Multistart + replay."""
    ln_out = layer.output.LayerNorm
    n, d = y.shape
    W = (y - ln_out.bias) / ln_out.weight
    a0 = ln(y - ln_out.bias, ln_in)
    out = torch.empty_like(y)
    tries = []
    for i in range(n):
        best, best_rep = None, float('inf')
        for k in range(restarts):
            start = a0[i] if k == 0 else on_manifold(ln_in, a0[i])
            X = torch.cat([start, torch.ones(1), torch.zeros(1)])
            X, _ = _ffn_newton_row(layer, W[i], ln_in, X)
            a = X[:d]
            rep = (ln(a + ffn_delta(layer, a[None]), ln_out)[0] - y[i]).abs().max().item()
            if rep < best_rep:
                best, best_rep = a, rep
            if best_rep < tol:
                break
        out[i] = best
        tries.append(k + 1)
    rep = replay_ffn(layer, out, y)
    if verbose:
        print(f"      ffn: replay={rep:.2e} starts_used={tries}")
    return out, rep, rep < tol


# --------------------------------------------------- attention sublayer inverse
def _frozen_lu(layer, P, n, d, nh=16):
    dh = d // nh
    Wv, Wo = layer.attention.self.value.weight, layer.attention.output.dense.weight
    M = torch.stack([Wo[:, h * dh:(h + 1) * dh] @ Wv[h * dh:(h + 1) * dh, :] for h in range(nh)])
    Amat = torch.einsum('hij,hpq->ipjq', P, M).reshape(n * d, n * d) + torch.eye(n * d)
    return torch.linalg.lu_factor(Amat)


def _scf(layer, a, ln_in, ln_prev, h_init, iters=60, tol=1e-5, damped=True, log=None):
    """Freeze the attention pattern, solve exactly, recompute.

    The plain iteration h <- h_new is a quasi-Newton step: it uses the exact Jacobian of
    the value path but DROPS the derivative of the softmax with respect to h. Where that
    dropped term is large the iteration can oscillate or walk away, which is exactly what
    happens at layer 23. So instead of taking the full step, take h + lam*(h_new - h) with
    lam backtracked until the REPLAY error actually decreases. Replay is the quantity we
    care about, is computable without ground truth, and gives the iteration a merit
    function it must monotonically reduce -- turning a divergent scheme into a descent one.
    """
    n, d = a.shape
    w = (a - ln_in.bias) / ln_in.weight
    g, b = ln_prev.weight, ln_prev.bias
    eye2n = torch.eye(2 * n)
    rhs = torch.zeros(2 * n + 1, n, d)
    for i in range(n):
        rhs[i, i] = w[i]
        rhs[n + i, i] = 1.0
    h = h_init.clone()
    sm = torch.cat([torch.ones(n), torch.zeros(n)])
    merit = replay_attn(layer, h, a, ln_in)
    for _it in range(iters):
        P = attn_probs(layer, h)
        rhs[2 * n] = value_path(layer, torch.zeros(n, d), P, with_bias=True)
        LU = _frozen_lu(layer, P, n, d)
        sols = torch.linalg.lu_solve(*LU, rhs.reshape(2 * n + 1, n * d).T).T.view(2 * n + 1, n, d)
        U, V, p = sols[:n], sols[n:2 * n], sols[2 * n]

        def h_of(sm):
            return ((sm[:n, None] * U.view(n, -1)).sum(0).view(n, d)
                    + (sm[n:, None] * V.view(n, -1)).sum(0).view(n, d) - p)

        def cons(sm):
            z = (h_of(sm) - b) / g
            return torch.cat([z.mean(1), z.pow(2).mean(1) - 1])

        for _ in range(40):
            r = cons(sm)
            rn = r.norm()
            if rn < 1e-6:
                break
            J = torch.stack([(cons(sm + 1e-4 * eye2n[j]) - r) / 1e-4 for j in range(2 * n)], 1)
            step = torch.linalg.lstsq(J, -r.unsqueeze(1)).solution.squeeze(1)
            t = 1.0
            for _ in range(30):
                cand = sm + t * step
                if (cand[:n] > 0).all() and cons(cand).norm() < rn:
                    break
                t *= 0.5
            sm = sm + t * step

        h_new = h_of(sm)
        if not torch.isfinite(h_new).all():
            break
        if damped:
            lam, ok_step = 1.0, False
            for _ in range(25):
                cand = h + lam * (h_new - h)
                mc = replay_attn(layer, cand, a, ln_in)
                if mc < merit:
                    ok_step = True
                    break
                lam *= 0.5
            if not ok_step:                      # no downhill direction left
                break
            delta = (cand - h).abs().max().item()
            h, merit = cand, mc
        else:
            delta = (h_new - h).abs().max().item()
            h = h_new
            merit = replay_attn(layer, h, a, ln_in)
        if log is not None:
            log.append((merit, delta))
        if merit < tol or delta > 1e6:
            break
    return h


def _newton_attn(layer, a, ln_in, ln_prev, h_init, iters=25, tol=1e-5, log=None,
                 deflate=(), defl_alpha=1.0):
    """Proper Newton on the attention sublayer, preconditioned by the frozen-P solve.

    The SCF drops the softmax's derivative, which is why it oscillates at layer 23. Here
    the full Jacobian is used, applied matrix-free as Jacobian-vector products, and the
    linear system is solved by GMRES preconditioned with (I + T_P)^-1 -- the frozen-P
    operator, which captures the bulk of the Jacobian and is cheap to factor. So the
    preconditioner IS the old SCF step; Newton just supplies the part it was missing.
    """
    from scipy.sparse.linalg import LinearOperator, gmres
    n, d = a.shape
    N = n * d + 2 * n
    w = (a - ln_in.bias) / ln_in.weight
    g, b = ln_prev.weight, ln_prev.bias

    def F_raw(X):
        h = X[:n * d].view(n, d)
        s, m = X[n * d:n * d + n], X[n * d + n:]
        z = (h - b) / g
        return torch.cat([(h + attn_delta(layer, h) - s[:, None] * w - m[:, None]).reshape(-1),
                          z.mean(1), z.pow(2).mean(1) - 1])

    # Brown-Gearhart deflation: multiply the residual by a factor that blows up at each
    # already-known root, so Newton is repelled from them and has to find a different one.
    # Away from those points the factor is ~1, so the remaining roots are untouched.
    defl = [D.reshape(-1) for D in deflate]

    def F(X):
        out = F_raw(X)
        for Dv in defl:
            out = out * (1.0 + defl_alpha / ((X[:n * d] - Dv).pow(2).sum() + 1e-9))
        return out

    X = torch.cat([h_init.reshape(-1), torch.ones(n), torch.zeros(n)])
    for _it in range(iters):
        F0 = F(X)
        nrm = F0.norm().item()
        if log is not None:
            log.append(nrm)
        if nrm < tol:
            break
        h_cur = X[:n * d].view(n, d)
        LU = _frozen_lu(layer, attn_probs(layer, h_cur), n, d)

        def mv(v):
            vt = torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float32)
            with torch.enable_grad():
                jv = torch.func.jvp(F, (X,), (vt,))[1]
            return jv.numpy()

        def pc(v):                                    # apply (I + T_P)^-1 to the h block
            vt = torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float32)
            out = vt.clone()
            out[:n * d] = torch.linalg.lu_solve(*LU, vt[:n * d].unsqueeze(1)).squeeze(1)
            return out.numpy()

        A_ = LinearOperator((N, N), matvec=mv, dtype=np.float32)
        M_ = LinearOperator((N, N), matvec=pc, dtype=np.float32)
        dX, _ = gmres(A_, (-F0).numpy(), M=M_, rtol=1e-4, restart=60, maxiter=4)
        dX = torch.as_tensor(dX)
        lam, stepped = 1.0, False
        for _ in range(30):
            cand = X + lam * dX
            if cand[n * d:n * d + n].gt(0).all() and F(cand).norm().item() < nrm:
                stepped = True
                break
            lam *= 0.5
        if not stepped:
            break
        X = X + lam * dX
    return X[:n * d].view(n, d)


def invert_attention(layer, a, ln_in, ln_prev, restarts=8, tol=1e-4, verbose=False):
    """Recover h from a = LN_in(h + Attn(h)). Multistart on the SCF, keep what replays."""
    base = ln(a - ln_in.bias, ln_prev)
    best, best_rep, used = None, float('inf'), 0
    for k in range(restarts):
        init = base if k == 0 else on_manifold(ln_prev, base)
        used = k + 1
        for solver in (_scf, _newton_attn):           # cheap one first, then the robust one
            h = solver(layer, a, ln_in, ln_prev, init)
            rep = replay_attn(layer, h, a, ln_in)
            if rep < best_rep:
                best, best_rep = h, rep
            if best_rep < tol:
                break
        if best_rep < tol:
            break
    if verbose:
        print(f"      attn: replay={best_rep:.2e} starts_used={used}")
    return best, best_rep, best_rep < tol


# ----------------------------------------------------------- embedding inverse
def undo_l2(u, ln_last):
    g, b = ln_last.weight, ln_last.bias
    s = (b / g).mean() / (u / g).mean()
    return s * u


def recover_tokens(h0, model, topk=3):
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
