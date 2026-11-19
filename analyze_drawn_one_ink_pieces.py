"""Ink pieces: let the template carry a few learned extra-ink patterns.

Run after analyze_drawn_one_warp_scorecard.py has cached its fits:
    OPENBLAS_NUM_THREADS=1 .venv/bin/python analyze_drawn_one_ink_pieces.py <name> <warp count>

A warp only moves ink that the template already has. It cannot grow a hook or
a base bar. So each image becomes

    image = warp( template + c1 * piece1 + ... + cK * pieceK )

where the warp numbers say WHERE the ink goes (as before) and the K piece
numbers say WHAT extra ink the image has. The pieces live in the template's
own frame, so a hook drawn there follows the stroke when it leans or moves.

Pieces are learned from the training images only: each image's leftover after
its warp is carried back into the template frame, and the main patterns of
those carried-back leftovers (principal directions) become the pieces. For
every image, with its warp held fixed, decoding is linear in the template, so
the piece numbers are a least-squares solve.

Compared at equal TOTAL number count against more warp numbers and against PCA,
and, on the drawn 1s, scored on whether the piece numbers carry the hook and bar
knobs the warp numbers missed.
"""

import json
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

import warp_charts as wc
from analyze_drawn_one_warp_scorecard import (MODELS, N_TEST, N_TRAIN, OUT, load,
                                              pca_coords, quad_ridge_r2, where)
import knob_strokes as ks

PIECES = (0, 1, 2, 4, 8, 16)


def carry_back(R, TH, m):
    """Each 28x28 leftover, separately, into the template frame (G x G)."""
    return np.stack([wc.splat(r[None], th[None], m) for r, th in zip(R, TH)])


def piece_images(A, TH, m):
    """(n, K, 784): every piece decoded through every image's warp."""
    out = np.empty((len(TH), len(A), 784))
    for k, a in enumerate(A):
        out[:, k] = wc.decode(a, TH, m).reshape(len(TH), -1)
    return out


def solve(Pk, R):
    """Per-image least-squares piece numbers, tiny ridge for stability."""
    C = np.empty((len(R), Pk.shape[1]))
    for i in range(len(R)):
        B = Pk[i].T
        H = B.T @ B
        C[i] = np.linalg.solve(H + 1e-6 * np.trace(H) / len(H) * np.eye(len(H)),
                               B.T @ R[i])
    return C


def run(name, nw):
    Z = np.load(OUT / f"fit_{name}.npz")
    I, P = load(name)
    tr, te = np.arange(N_TRAIN), np.arange(N_TRAIN, N_TRAIN + N_TEST)
    m = MODELS[nw]
    T, THtr, THte = Z[f"T{nw}"], Z[f"tr{nw}"], Z[f"te{nw}"]
    Dtr, Dte = wc.decode(T, THtr, m), Z[f"dec{nw}"]
    Rtr = (I[tr] - Dtr).reshape(len(tr), -1)
    Rte = (I[te] - Dte).reshape(len(te), -1)

    back = carry_back(I[tr] - Dtr, THtr, m).reshape(len(tr), -1)
    _, s, V = np.linalg.svd(back, full_matrices=False)       # uncentred: no average piece
    A = V[:max(PIECES)].reshape(-1, wc.G, wc.G)
    Ptr, Pte = piece_images(A, THtr, m), piece_images(A, THte, m)

    F = I.reshape(len(I), -1)
    res = {"warp": nw, "rows": {}}
    print(f"\n=== {name}, {nw} warp numbers + K ink pieces ===")
    print(f"  {'pieces':>7}{'total numbers':>15}{'leftover':>10}"
          f"{'PCA same total':>16}")
    for K in PIECES:
        if K == 0:
            left, Cte, Ctr = wc.rel_error(I[te], Dte), None, None
        else:
            Ctr, Cte = solve(Ptr[:, :K], Rtr), solve(Pte[:, :K], Rte)
            rec = Dte.reshape(len(te), -1) + np.einsum("nk,nkp->np", Cte, Pte[:, :K])
            left = wc.rel_error(F[te], rec)
        C, mu, Vp = pca_coords(F, tr, nw + K)
        pca = wc.rel_error(F[te], mu + C[te] @ Vp)
        row = {"total": nw + K, "leftover": left, "pca": pca}
        if K:
            row["where"] = where(I[te], (F[te] - rec).reshape(-1, 28, 28))
        if P is not None and P.size and K:
            X_tr = np.c_[THtr, Ctr]
            X_te = np.c_[THte, Cte]
            row["knob_r2"] = dict(zip(ks.NAMES[:P.shape[1]], map(float, quad_ridge_r2(
                X_tr, P[tr], X_te, P[te]))))
        res["rows"][K] = row
        print(f"  {K:>7}{nw + K:>15}{left * 100:9.2f}%{pca * 100:15.2f}%"
              + ("" if "knob_r2" not in row else "   knobs: " + " ".join(
                  f"{k[:6]} {v:.2f}" for k, v in row["knob_r2"].items())))
    res["piece_energy"] = (s[:max(PIECES)] ** 2 / (s ** 2).sum()).tolist()
    np.save(OUT / f"pieces_{name}_{nw}.npy", A[:8])
    (OUT / f"pieces_{name}_{nw}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]))
