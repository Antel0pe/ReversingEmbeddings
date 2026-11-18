"""MNIST 1s with the picture hidden: same geometry, no visual handle.

    from scrambled_ones import load
    X = load()                 # (6742, 784) float32, train 1s, fully hidden
    X = load("test")           # (1135, 784) the test-set 1s, same scramble
    X = load(level="shuffle")  # pixels only re-ordered (values still ink 0..1)

What is done to every image, identically:
  1. ink scaled to 0..1 (uint8 / 255), flattened to 784 numbers;
  2. the image ORDER is shuffled, so neighbouring rows mean nothing;
  3. level="shuffle": the 784 pixel positions are re-ordered by one fixed
     random permutation;
     level="rotate" (default): additionally multiplied by one fixed random
     784x784 rotation. Every coordinate becomes a mix of all pixels, so values
     go negative, nothing is zero, and no single number means "this pixel".

Both levels keep EXACTLY: every distance between two images, every angle,
every nearest neighbour, the PCA spectrum, intrinsic dimension, persistent
homology, Isomap/LLE. So any distance-based result is identical to running on
the real pixels.

How much they hide:
  "shuffle" hides the picture from your eyes but not from an algorithm: a pixel's
  most-correlated partner is a touching grid neighbour 98.7% of the time, so the
  28x28 layout can be rebuilt. It also keeps ink values, sparsity (most entries
  0) and total ink as a plain sum.
  "rotate" hides all of that. Rebuilding the grid would first need the rotation
  undone (e.g. searching for the rotation that makes the data non-negative and
  sparse again) -- possible in principle, nothing simple does it.

The key (`key()`, `unscramble()`, `original_index()`) exists for GRADING only:
checking a method's answer against the real images or the real MNIST index.
Do not feed it to a method you are testing.

Speed: the first call builds a cache in data/scrambled_ones/ (gitignored, ~50 MB
for all four arrays, a few seconds); later calls memory-map it in milliseconds. Delete the folder or
pass rebuild=True to regenerate -- the same seed always gives the same scramble.

Run `python scrambled_ones.py` to build the caches and print self-checks.
"""

from pathlib import Path

import numpy as np

import mnist

SEED = 20261117
CACHE = Path(__file__).resolve().parent / "data" / "scrambled_ones"
SPLITS = ("train", "test")
LEVELS = ("rotate", "shuffle")


def _rng(tag):
    return np.random.default_rng([SEED, sum(map(ord, tag))])


def key():
    """Return dict(perm, rotation): x_scrambled = x[perm] @ rotation.

    perm[j] is the original pixel (row*28 + col) that lands in slot j.
    rotation is orthogonal (rotation @ rotation.T == identity).
    """
    perm = _rng("perm").permutation(784)
    q, r = np.linalg.qr(_rng("rotation").standard_normal((784, 784)))
    rotation = q * np.sign(np.diag(r))   # sign fix -> uniformly random rotation
    return {"perm": perm, "rotation": rotation}


def _real_ones(split):
    images, labels = mnist.load(split)
    idx = np.flatnonzero(labels == 1)
    order = _rng("order-" + split).permutation(len(idx))
    idx = idx[order]
    return images[idx].reshape(len(idx), 784).astype(np.float64) / 255.0, idx


def _build(split, level):
    x, idx = _real_ones(split)
    k = key()
    x = x[:, k["perm"]]
    if level == "rotate":
        x = x @ k["rotation"]
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / f"{split}_{level}.npy", x.astype(np.float32))
    np.save(CACHE / f"{split}_index.npy", idx)


def load(split="train", level="rotate", rebuild=False):
    """Scrambled MNIST 1s as an (n, 784) float32 array (read-only memory map).

    Call np.array(X) if you need a writable copy.
    """
    if split not in SPLITS or level not in LEVELS:
        raise ValueError(f"split in {SPLITS}, level in {LEVELS}")
    path = CACHE / f"{split}_{level}.npy"
    if rebuild or not path.exists():
        _build(split, level)
    return np.load(path, mmap_mode="r")


def original_index(split="train"):
    """Row i of load(split) is MNIST image original_index(split)[i]. Grading only."""
    path = CACHE / f"{split}_index.npy"
    if not path.exists():
        _build(split, "shuffle")
    return np.load(path)


def unscramble(x, level="rotate"):
    """Map scrambled rows back to ordinary (n, 28, 28) images. Grading only."""
    k = key()
    x = np.asarray(x, dtype=np.float64).reshape(-1, 784)
    if level == "rotate":
        x = x @ k["rotation"].T
    out = np.empty_like(x)
    out[:, k["perm"]] = x
    return out.reshape(-1, 28, 28)


if __name__ == "__main__":
    import time

    from scipy.spatial.distance import pdist

    for split in SPLITS:
        for level in LEVELS:
            load(split, level, rebuild=True)
    t = time.perf_counter()
    X = load()
    print(f"load() from cache: {1000 * (time.perf_counter() - t):.1f} ms, "
          f"shape {X.shape}, {X.dtype}")

    real, _ = _real_ones("train")
    for level in LEVELS:
        S = np.asarray(load(level=level), dtype=np.float64)
        d0, d1 = pdist(real[:1500]), pdist(S[:1500])
        back = np.abs(unscramble(S, level).reshape(-1, 784) - real).max()
        print(f"{level:8s} worst relative distance change {np.max(np.abs(d1 - d0) / d0):.1e}"
              f" | min value {S.min():+.3f} | share of exact zeros {(S == 0).mean():.1%}"
              f" | unscramble error {back:.1e}")
    print("original pixels: share of exact zeros", f"{(real == 0).mean():.1%}")
    print("cache:", ", ".join(f"{p.name} {p.stat().st_size / 1e6:.1f} MB"
                              for p in sorted(CACHE.iterdir())))
