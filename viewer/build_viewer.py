"""Build the self-contained 'Walking the Ones' page.

Packs the giant component of the bridged 1-graph into one HTML file: every node's
28x28 image goes into a sprite-sheet PNG, base64'd inline alongside a real/synthetic
flag array. The page decodes the sprite to a Uint8Array on load and recomputes
distance from the current position to all 12k nodes on every move (~10M ops, ~30ms),
which is what lets the current position be an arbitrary 784-vector rather than a node.

    python viewer/build_viewer.py            # writes viewer/walking-the-ones.html

Needs synth3.npy (the synthesised bridge nodes); regenerate with morph.bridge_path2
over the MST bridges if it is missing -- see OnesManifold.ipynb Part VI.
"""

import base64
import os
import sys

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mnist  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def build_nodes(synth_path):
    """Giant component of the THETA-graph over real + synthesised 1s."""
    X, y = mnist.load("train")
    X = X.reshape(len(X), -1).astype(np.float32) / 255.0
    ones = X[y == 1]
    n = len(ones)
    sq = (ones ** 2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * ones @ ones.T, 0))
    np.fill_diagonal(D, 0)
    theta = float((D + np.eye(n, dtype=np.float32) * 1e9).min(1).mean())

    S = np.load(synth_path) if os.path.exists(synth_path) else np.zeros((0, 784), np.float32)
    P = np.vstack([ones, S]).astype(np.float32)
    N = len(P)
    psq = (P ** 2).sum(1)

    rows, cols = [], []
    for s in range(0, N, 1500):
        ch = P[s:s + 1500]
        d = np.sqrt(np.maximum((ch ** 2).sum(1)[:, None] + psq[None, :] - 2 * ch @ P.T, 0))
        r, c = np.where(d <= theta)
        k = (r + s) != c
        rows.append(r[k] + s)
        cols.append(c[k])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    _, lab = connected_components(
        coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N)), directed=False)
    keep = np.where(lab == np.argmax(np.bincount(lab)))[0]
    return P[keep], (keep >= n).astype(np.uint8), theta


def sprite(P):
    """All node images tiled into one square grayscale PNG. MNIST compresses ~6x."""
    M = len(P)
    G = int(np.ceil(np.sqrt(M)))
    sheet = np.zeros((G * 28, G * 28), np.uint8)
    for i in range(M):
        r, c = divmod(i, G)
        sheet[r*28:(r+1)*28, c*28:(c+1)*28] = np.clip(P[i].reshape(28, 28) * 255, 0, 255)
    path = os.path.join(HERE, "_sprite.png")
    Image.fromarray(sheet, mode="L").save(path, optimize=True)
    return path, G


def main():
    P, flags, theta = build_nodes(os.path.join(HERE, "synth3.npy"))
    print(f"theta={theta:.4f}  giant component {len(P)} nodes "
          f"({int((flags == 0).sum())} real, {int(flags.sum())} synthetic)")
    png, G = sprite(P)
    print(f"sprite {G}x{G} tiles -> {os.path.getsize(png)/1e6:.2f} MB")

    html = open(os.path.join(HERE, "template.html")).read()
    html = (html.replace("__M__", str(len(P))).replace("__G__", str(G))
                .replace("__THETA__", f"{theta:.6f}")
                .replace("__FLAGS__", base64.b64encode(flags.tobytes()).decode())
                .replace("__SPRITE__", base64.b64encode(open(png, "rb").read()).decode()))
    out = os.path.join(HERE, "walking-the-ones.html")
    open(out, "w").write(html)
    os.remove(png)
    print(f"wrote {out}  ({os.path.getsize(out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
