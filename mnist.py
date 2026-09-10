"""Dependency-free MNIST fetch + load.

torchvision has no wheel matching this env's torch (2.14+cpu), so we pull the
four idx.gz files straight from the S3 mirror and parse the idx format by hand.
Downloads are skipped if the file is already on disk.
"""

import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path

import numpy as np

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"

# filename -> md5 of the gzipped file
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}

DEFAULT_ROOT = Path(__file__).resolve().parent / "data" / "mnist"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(root=DEFAULT_ROOT, verbose=True):
    """Fetch any of the four MNIST archives that aren't already present."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, md5 in FILES.items():
        dest = root / name
        if dest.exists() and _md5(dest) == md5:
            if verbose:
                print(f"skip     {name} (already downloaded)")
            continue
        if verbose:
            print(f"download {name} ...")
        urllib.request.urlretrieve(MIRROR + name, dest)
        got = _md5(dest)
        if got != md5:
            dest.unlink()
            raise RuntimeError(f"{name}: md5 mismatch (got {got}, want {md5})")
        if verbose:
            print(f"         {name} ok ({dest.stat().st_size / 1e6:.1f} MB)")
    return root


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        zero, dtype, ndim = struct.unpack(">HBB", f.read(4))
        if zero != 0 or dtype != 0x08:
            raise ValueError(f"{path.name}: not a uint8 idx file")
        shape = struct.unpack(f">{ndim}I", f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load(split="train", root=DEFAULT_ROOT):
    """Return (images uint8 [N,28,28], labels uint8 [N]) for 'train' or 'test'."""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    root = Path(root)
    prefix = "train" if split == "train" else "t10k"
    images = _read_idx(root / f"{prefix}-images-idx3-ubyte.gz")
    labels = _read_idx(root / f"{prefix}-labels-idx1-ubyte.gz")
    return images, labels


if __name__ == "__main__":
    download()
    for split in ("train", "test"):
        x, y = load(split)
        print(split, x.shape, y.shape, x.dtype, "labels", sorted(set(y.tolist())))
