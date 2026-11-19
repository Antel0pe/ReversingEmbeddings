"""The simplest learned map from a few numbers to a 28x28 image of a 1.

Model: a plain autoencoder. Encoder 784 -> 256 -> k numbers; decoder k numbers
-> 256 -> 784 pixels (ReLU hidden layers, sigmoid output, squared-error loss).
The DECODER is the object of interest: a mapping from k low-dimensional
coordinates to a full image. Nothing about strokes, warps or knobs is built in.

Data: real MNIST training 1s (5,742 to train, 1,000 held out), plus a CONTROL
family whose answer is known: generated 1s from `grey_ones.render` with five
knobs drawn uniformly from their default ranges (5,000 train, 1,000 held out).

Parts (python analyze_mnist_one_autoencoder.py [sweep] [meaning] [gaps]):
  sweep    held-out error for k = 1..30 coordinates, two seeds, against PCA
  meaning  what each coordinate of the k=5 (and k=12) model tracks: measured
           ink, centre, height, slant, width and bend; true knobs on the control
  gaps     do NEW coordinates decode to 1s? (a) random draws, (b) the midpoint
           between two real 1s' coordinates vs. the plain pixel average

Outputs in figures/mnist_one_autoencoder/.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

import mnist
import warp_charts as wc
from grey_ones import KNOBS, RANGES, render

torch.set_num_threads(2)
OUT = Path(__file__).resolve().parent / "figures" / "mnist_one_autoencoder"
OUT.mkdir(parents=True, exist_ok=True)
KS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30)
EPOCHS, HIDDEN = 200, 256
SEED1_KS = (3, 5, 8, 12, 20)   # second seed only here, to show seed spread


# ---------------------------------------------------------------- data

def datasets():
    X, y = mnist.load("train")
    I = X[y == 1].reshape(-1, 784).astype(np.float32) / 255.0
    order = np.random.default_rng(0).permutation(len(I))
    I = I[order]
    rng = np.random.default_rng(1)
    knobs = RANGES[:, 0] + rng.random((6000, 5)) * (RANGES[:, 1] - RANGES[:, 0])
    G = render(knobs).reshape(-1, 784).astype(np.float32)
    return {"mnist": (I[:-1000], I[-1000:], None, None),
            "generated": (G[:5000], G[5000:], knobs[:5000], knobs[5000:])}


# ---------------------------------------------------------------- model

class AE(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.enc = torch.nn.Sequential(torch.nn.Linear(784, HIDDEN), torch.nn.ReLU(),
                                       torch.nn.Linear(HIDDEN, k))
        self.dec = torch.nn.Sequential(torch.nn.Linear(k, HIDDEN), torch.nn.ReLU(),
                                       torch.nn.Linear(HIDDEN, 784), torch.nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x))


def train(Xtr, k, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    model = AE(k)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    X = torch.from_numpy(Xtr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for b in torch.randperm(len(X), generator=g).split(128):
            loss = ((model(X[b]) - X[b]) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model.eval()


@torch.no_grad()
def encode(model, X):
    return model.enc(torch.from_numpy(X)).numpy()


@torch.no_grad()
def decode(model, Z):
    return model.dec(torch.from_numpy(Z.astype(np.float32))).numpy()


def errors(A, B):
    """Whole-set relative error (repo convention) and per-image relative errors."""
    per = np.linalg.norm(A - B, axis=1) / np.linalg.norm(A, axis=1)
    return wc.rel_error(A, B), per


def pca_errors(Xtr, Xte, ks):
    mu = Xtr.mean(0)
    _, _, V = np.linalg.svd(Xtr - mu, full_matrices=False)
    out = {}
    for k in ks:
        R = mu + (Xte - mu) @ V[:k].T @ V[:k]
        out[k] = wc.rel_error(Xte, R)
    return out


def model_path(name, k, seed):
    return OUT / f"ae_{name}_k{k}_s{seed}.pt"


def get_model(name, Xtr, k, seed):
    p = model_path(name, k, seed)
    if p.exists():
        m = AE(k); m.load_state_dict(torch.load(p)); return m.eval()
    m = train(Xtr, k, seed)
    torch.save(m.state_dict(), p)
    return m


# ---------------------------------------------------------------- parts

def part_sweep(D, R):
    print("\n=== SWEEP: held-out error vs number of coordinates ===")
    R["sweep"] = {}
    for name, (Xtr, Xte, _, _) in D.items():
        pca = pca_errors(Xtr, Xte, list(range(1, 201)))
        rows = {}
        for k in KS:
            res = []
            for seed in ((0, 1) if k in SEED1_KS else (0,)):
                t = time.time()
                m = get_model(name, Xtr, k, seed)
                tot, per = errors(Xte, decode(m, encode(m, Xte)))
                tr_tot, _ = errors(Xtr, decode(m, encode(m, Xtr)))
                res.append({"heldout": tot, "train": tr_tot,
                            "median_image": float(np.median(per)),
                            "worst_tenth_image": float(np.percentile(per, 90)),
                            "worst_image": float(per.max())})
                match = next((j for j in range(1, 201) if pca[j] <= tot), None)
                print(f"  {name:9s} k={k:2d} seed {seed}: held-out {tot:6.2%} "
                      f"(train {tr_tot:6.2%}); per image median {np.median(per):6.2%}, "
                      f"90th pct {np.percentile(per, 90):6.2%}, worst {per.max():6.2%}; "
                      f"PCA needs {match} for the same  [{time.time() - t:.0f}s]")
            rows[k] = res
        R["sweep"][name] = {"ae": rows, "pca": {k: pca[k] for k in range(1, 201)}}


def measurements(X):
    """Plain measurements of each image, no model: ink, centre, height, slant,
    width (from ink moments, as in analyze_mnist_one_knob_search) and bend
    (curvature of the row-by-row ink centre)."""
    I = X.reshape(-1, 28, 28).astype(np.float64)
    P = wc.moment_chart(I)
    vyy = P[:, 4] ** 2
    ys = np.arange(28) + 0.5
    xs = np.arange(28) + 0.5
    rows = I.sum(2)
    xc = (I * xs).sum(2) / np.maximum(rows, 1e-9)
    bend = np.empty(len(I))
    for n in range(len(I)):
        w = rows[n]
        keep = w > 0.3
        y = (ys[keep] - P[n, 1]) / 10.0
        A = np.stack([np.ones_like(y), y, y ** 2], 1) * np.sqrt(w[keep])[:, None]
        coef = np.linalg.lstsq(A, xc[n, keep] * np.sqrt(w[keep]), rcond=None)[0]
        bend[n] = coef[2]
    return {"ink": I.sum((1, 2)), "centre x": P[:, 0], "centre y": P[:, 1],
            "height": np.sqrt(12 * vyy), "slant": -(P[:, 3] * P[:, 4]) / vyy,
            "width": np.sqrt(12 * P[:, 2] ** 2), "bend": bend}


def readout(Ztr, Zte, ytr, yte, degree):
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-9
    a, b = (Ztr - mu) / sd, (Zte - mu) / sd
    if degree == 2:
        f = PolynomialFeatures(2, include_bias=False); a, b = f.fit_transform(a), f.transform(b)
    p = Ridge(alpha=1e-2).fit(a, ytr).predict(b)
    return float(1 - ((p - yte) ** 2).sum() / ((yte - ytr.mean()) ** 2).sum())


def code_directions(Z):
    """Principal directions of the code cloud, largest spread first."""
    mu = Z.mean(0)
    _, s, V = np.linalg.svd(Z - mu, full_matrices=False)
    return mu, V, (Z - mu) @ V.T


def part_meaning(D, R):
    print("\n=== MEANING: what the coordinates track ===")
    R["meaning"] = {}
    for name, (Xtr, Xte, Ktr, Kte) in D.items():
        Mtr, Mte = measurements(Xtr), measurements(Xte)
        if Ktr is not None:
            for j, kn in enumerate(KNOBS):
                Mtr["TRUE " + kn], Mte["TRUE " + kn] = Ktr[:, j], Kte[:, j]
        R["meaning"][name] = {}
        for k in (5, 12):
            m = get_model(name, Xtr, k, 0)
            Ztr, Zte = encode(m, Xtr), encode(m, Xte)
            res = {"readout": {}, "directions": []}
            print(f"  {name}, k={k}: held-out R^2 of each measurement read from the code")
            for key in Mtr:
                l1 = readout(Ztr, Zte, Mtr[key], Mte[key], 1)
                l2 = readout(Ztr, Zte, Mtr[key], Mte[key], 2)
                res["readout"][key] = {"linear": l1, "quadratic": l2}
                print(f"    {key:16s} linear {l1:6.3f}  quadratic {l2:6.3f}")
            mu, V, S = code_directions(Ztr)
            spread = S.std(0)
            print(f"    code directions (largest spread first), |correlation| with each measurement:")
            for d in range(k):
                c = {key: float(np.corrcoef(S[:, d], Mtr[key])[0, 1]) for key in Mtr}
                top = sorted(c, key=lambda q: -abs(c[q]))[:3]
                res["directions"].append({"spread": float(spread[d]), "corr": c})
                print(f"      dir {d + 1}: spread {spread[d]:6.2f}  " +
                      ", ".join(f"{q} {c[q]:+.2f}" for q in top))
            R["meaning"][name][k] = res


def realism(S, ref, ref_norm_sq):
    d = ref_norm_sq[None, :] + (S ** 2).sum(1)[:, None] - 2 * S @ ref.T
    near = np.sqrt(np.maximum(d.min(1), 0)) / np.linalg.norm(S, axis=1)
    grey = ((S > 0.2) & (S < 0.8)).sum(1) / np.maximum((S > 0.2).sum(1), 1)
    return {"dist_to_nearest_real": float(np.median(near)),
            "grey_share": float(np.median(grey))}


def part_gaps(D, R):
    print("\n=== GAPS: do NEW coordinates decode to 1s? ===")
    rng = np.random.default_rng(7)
    R["gaps"] = {}
    for name, (Xtr, Xte, _, _) in D.items():
        ref = Xtr.astype(np.float64); rn = (ref ** 2).sum(1)
        a, b = rng.integers(0, len(Xte), 600), rng.integers(0, len(Xte), 600)
        keep = a != b; a, b = a[keep], b[keep]
        gap = np.median(np.linalg.norm(Xte[a] - Xte[b], axis=1) /
                        np.linalg.norm(Xte[a], axis=1))
        rows = {"real held-out 1s": realism(Xte.astype(np.float64), ref, rn),
                "pixel average of two real 1s": realism(
                    0.5 * (Xte[a] + Xte[b]).astype(np.float64), ref, rn)}
        for k in (5, 12, 30):
            m = get_model(name, Xtr, k, 0)
            Ztr, Zte = encode(m, Xtr), encode(m, Xte)
            rows[f"k={k} reconstruction of held-out"] = realism(
                decode(m, Zte).astype(np.float64), ref, rn)
            rows[f"k={k} midpoint of two codes"] = realism(
                decode(m, 0.5 * (Zte[a] + Zte[b])).astype(np.float64), ref, rn)
            C = np.cov(Ztr.T) + 1e-9 * np.eye(k)
            draw = rng.multivariate_normal(Ztr.mean(0), C, size=600)
            rows[f"k={k} random new code"] = realism(decode(m, draw).astype(np.float64), ref, rn)
        print(f"  {name}: random pairs are {gap:.1%} of an image apart")
        print(f"    {'source':38s} {'dist to nearest real':>21s} {'grey share':>11s}")
        for lab, v in rows.items():
            print(f"    {lab:38s} {v['dist_to_nearest_real']:21.3f} {v['grey_share']:11.3f}")
        R["gaps"][name] = {"pair_gap": float(gap), "rows": rows}


if __name__ == "__main__":
    args = sys.argv[1:]
    only = [a.split("=")[1] for a in args if a.startswith("only=")]
    parts = [a for a in args if "=" not in a] or ["sweep", "meaning", "gaps"]
    D = datasets()
    if only:
        D = {only[0]: D[only[0]]}
    path = OUT / f"results_{only[0] if only else 'all'}.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    for p in parts:
        {"sweep": part_sweep, "meaning": part_meaning, "gaps": part_gaps}[p](D, R)
        path.write_text(json.dumps(R, indent=1, default=str))
