"""Does reshaping the 1s BEFORE learning cut the numbers needed or the error?

Same plain autoencoder as analyze_mnist_one_autoencoder.py (input -> 256 -> k ->
256 -> input, 200 epochs), trained on transformed images (pre_transforms.py).
Every decoded image is mapped back with the transform's undo and scored against
the ORIGINAL held-out image, so lost detail counts. Numbers set aside by a
transform count toward the total.

  python analyze_mnist_one_pretransform.py pose mnist
  python analyze_mnist_one_pretransform.py pose generated
  python analyze_mnist_one_pretransform.py sdf mnist
  python analyze_mnist_one_pretransform.py blur mnist
"""
import json
import sys
import time

import numpy as np
import torch

import analyze_mnist_one_autoencoder as A
import pre_transforms as T
import warp_charts as wc

torch.set_num_threads(2)
OUT = A.OUT / "pretransform"
OUT.mkdir(parents=True, exist_ok=True)
KS = (1, 2, 3, 5, 8, 12, 20)


class AE(torch.nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.enc = torch.nn.Sequential(torch.nn.Linear(n, 256), torch.nn.ReLU(), torch.nn.Linear(256, k))
        self.dec = torch.nn.Sequential(torch.nn.Linear(k, 256), torch.nn.ReLU(),
                                       torch.nn.Linear(256, n), torch.nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x))


def train(Y, k, seed=0, epochs=A.EPOCHS):
    torch.manual_seed(seed)
    m = AE(Y.shape[1], k)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    X = torch.from_numpy(Y)
    g = torch.Generator().manual_seed(seed)
    # loss weight so a 56x56 pose image counts like a 28x28 one
    w = 784.0 / Y.shape[1]
    for _ in range(epochs):
        for b in torch.randperm(len(X), generator=g).split(128):
            loss = w * ((m(X[b]) - X[b]) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return m.eval()


@torch.no_grad()
def roundtrip(m, Y):
    return m(torch.from_numpy(Y)).numpy()


def main(tname, dname):
    Xtr, Xte, _, _ = A.datasets()[dname]
    if tname == "pose":
        Ytr, _ = T.pose_forward(Xtr); Yte, Pte = T.pose_forward(Xte)
        back, extra = (lambda Y: T.pose_inverse(Y, Pte)), 5
    elif tname == "deslant":
        Ytr, _ = T.deslant_forward(Xtr); Yte, Pte = T.deslant_forward(Xte)
        back, extra = (lambda Y: T.pose_inverse(Y, Pte)), 2
    elif tname == "upsample":   # control: 56x56 resampling, nothing removed
        Ytr, _ = T.upsample_forward(Xtr); Yte, Pte = T.upsample_forward(Xte)
        back, extra = (lambda Y: T.pose_inverse(Y, Pte)), 0
    elif tname == "sdf":
        Ytr, _ = T.sdf_forward(Xtr); Yte, _ = T.sdf_forward(Xte)
        T.SDF_T = T.fit_sdf_t(Xtr[:2000])
        back, extra = T.sdf_inverse, 0
    elif tname == "blur":
        Ytr, _ = T.blur_forward(Xtr); Yte, _ = T.blur_forward(Xte)
        back, extra = (lambda Y: Y), 0
    floor = wc.rel_error(Xte, back(Yte))
    print(f"{tname} on {dname}: going there and back with no network loses {floor:.2%}")
    res = {"floor": floor, "extra_numbers": extra, "k": {}}
    if tname in ("pose", "deslant"):   # zero learned numbers: the average standard-pose 1, re-posed
        avg = np.repeat(Ytr.mean(0, keepdims=True), len(Yte), 0)
        e0 = wc.rel_error(Xte, back(avg))
        res["k"][0] = {"vs_original": e0}
        print(f"  k= 0 (+{extra} set aside = {extra} total): vs original {e0:.2%}")
    for k in (KS if len(sys.argv) < 4 else [int(a) for a in sys.argv[3].split(",")]):
        t = time.time()
        m = train(Ytr, k)
        R = roundtrip(m, Yte)
        e = wc.rel_error(Xte, back(R)); et = wc.rel_error(Yte, R)
        per = np.linalg.norm(Xte - back(R), axis=1) / np.linalg.norm(Xte, axis=1)
        res["k"][k] = {"vs_original": e, "vs_transformed": et,
                       "median_image": float(np.median(per)),
                       "worst_tenth_image": float(np.percentile(per, 90))}
        print(f"  k={k:2d} (+{extra} set aside = {k + extra:2d} total): vs original {e:.2%}  "
              f"(vs its own transformed target {et:.2%}); 90th pct image {np.percentile(per, 90):.2%}"
              f"  [{time.time() - t:.0f}s]", flush=True)
        (OUT / f"{tname}_{dname}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
