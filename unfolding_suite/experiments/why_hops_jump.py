"""Side view of the S-curl: where hops jump between layers, and what the flatness check saw.

The S-curl is one curve swept sideways, so looking along the sweep shows every layer.
Run from the repo root: .venv/bin/python -m unfolding_suite.experiments.why_hops_jump
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.spatial import cKDTree

from unfolding_suite.experiments.adaptive_hops import K_MAX, flat_neighbour_counts
from unfolding_suite.sheets import SHEETS, make_instance

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures" / "why_hops_jump"
OUT.mkdir(parents=True, exist_ok=True)
FAR = 0.3
INK, MUTED, RED = "#0b0b0b", "#52514e", "#d03b3b"

I = make_instance(SHEETS["s_curl"], 65, 0)
X, U = I.xyz, I.uv
curvature = np.abs(7 * np.sin(np.pi * (U[:, 0] + 1)))
dist, near = cKDTree(X).query(X, k=K_MAX + 1)
spacing = float(np.median(dist[:, 1]))

# How close is the nearest other layer to each dot?
d_all, idx_all = cKDTree(X).query(X, k=400)
other = np.linalg.norm(U[idx_all] - U[:, None], axis=2) > FAR
gap = np.where(other, d_all, np.inf).min(1)


def links_for(counts):
    src = np.repeat(np.arange(len(X)), counts)
    dst = np.concatenate([near[i, 1:c + 1] for i, c in enumerate(counts)])
    return src, dst, np.linalg.norm(U[src] - U[dst], axis=1) > FAR, np.linalg.norm(X[src] - X[dst], axis=1)


def out_of_plane(k):
    patch = X[near[:, :k + 1]]
    patch = patch - patch.mean(1, keepdims=True)
    s = np.linalg.svd(patch, compute_uv=False)
    return s[:, 2] / np.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2)


rows, plots = {}, {}
for label, counts in [("Walking, 6 neighbours", np.full(len(X), 6)), ("Walking, 10 neighbours", np.full(len(X), 10)),
                      *[(f"Flatness hops, tolerance {t}", flat_neighbour_counts(X, t)[1]) for t in (0.05, 0.10, 0.20)]]:
    src, dst, jump, length = links_for(counts)
    r = {"links": int(len(jump)), "links_jumping": int(jump.sum()),
         "neighbours_median": float(np.median(counts)), "neighbours_max": int(counts.max()),
         "longest_link": float(length.max()), "median_link": float(np.median(length))}
    if jump.any():
        sources = np.unique(src[jump])
        r.update({"jumping_dots": len(sources), "neighbours_at_jumping_dots_median": float(np.median(counts[sources])),
                  "jump_link_length_median": float(np.median(length[jump])),
                  "curvature_at_jumping_dots_median": float(np.median(curvature[sources])),
                  "gap_to_other_layer_at_jumping_dots_median": float(np.median(gap[sources]))})
    rows[label] = r
    plots[label] = (counts, src, dst, jump)
    print(label, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}), flush=True)

# Did the flatness score notice a second layer entering the patch?
tests = {}
for k in (10, 20, 30, 40):
    has_other = (np.linalg.norm(U[near[:, 1:k + 1]] - U[:, None], axis=2) > FAR).any(1)
    score = out_of_plane(k)
    tests[k] = {"patches_touching_another_layer": int(has_other.sum()),
                "score_with_other_layer_median": float(np.median(score[has_other])) if has_other.any() else None,
                "score_one_layer_median": float(np.median(score[~has_other])),
                "score_one_layer_p95": float(np.percentile(score[~has_other], 95))}
    print(k, tests[k], flush=True)

# At the moment the flatness check let a jumping link in, how many intruders were in the patch?
counts = plots["Flatness hops, tolerance 0.1"][0]
src, dst, jump = plots["Flatness hops, tolerance 0.1"][1:]
intruders, sizes, scores = [], [], []
for i in np.unique(src[jump]):
    k = counts[i]
    intruders.append(int((np.linalg.norm(U[near[i, 1:k + 1]] - U[i], axis=1) > FAR).sum()))
    sizes.append(int(k))
    patch = X[near[i, :k + 1]] - X[near[i, :k + 1]].mean(0)
    sv = np.linalg.svd(patch, compute_uv=False)
    scores.append(float(sv[2] / np.sqrt(sv[0] ** 2 + sv[1] ** 2)))
print("tolerance 0.1, dots with a jumping link: patch size median", int(np.median(sizes)), " other-layer dots in it: median",
      float(np.median(intruders)), "max", max(intruders), " flatness score median", round(float(np.median(scores)), 3))

# How flat is the sheet where the curve bends hardest, compared with where it is straight?
straight = curvature < np.percentile(curvature, 25)
bent = curvature > np.percentile(curvature, 75)
print("score at k=20: straightest quarter", round(float(np.median(out_of_plane(20)[straight])), 4),
      " most curved quarter", round(float(np.median(out_of_plane(20)[bent])), 4))
print("dot spacing", round(spacing, 4), " closest layers", round(float(gap.min()), 4))
(OUT / "results.json").write_text(json.dumps({"rows": rows, "flatness_score_by_patch_size": tests,
                                              "dot_spacing": spacing, "closest_layers": float(gap.min())}, indent=1))

# ---- figure
mid = np.abs(U[:, 1]) < 0.12          # a thin slice along the sweep, so links show clearly
names = ["Walking, 6 neighbours", "Walking, 10 neighbours", "Flatness hops, tolerance 0.1"]
fig = plt.figure(figsize=(16, 7.4), facecolor="white")
fig.text(0.03, 0.945, "The S-curl seen from the side: where hops jump between layers", fontsize=20, fontweight="bold", color=INK)
fig.text(0.03, 0.895, "Looking along the sheet, so each dot sits on one curve. Red lines: links joining dots that are far apart on "
         "the paper but close in 3D.", fontsize=12.5, color=MUTED)
for j, name in enumerate(names):
    counts, src, dst, jump = plots[name]
    ax = fig.add_axes([0.025 + 0.325 * j, 0.20, 0.30, 0.64])
    if j == 2:
        sc = ax.scatter(X[:, 0], X[:, 2], c=counts, s=7, cmap="Blues", vmin=0, vmax=K_MAX, linewidths=0, zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 2], c="#b9b8b1", s=7, linewidths=0, zorder=2)
    for a, b in zip(src[jump], dst[jump]):
        ax.plot(X[[a, b], 0], X[[a, b], 2], color=RED, lw=1.0, alpha=0.8, zorder=3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#dddcd6")
    ax.set_title(name, fontsize=14, loc="left", color=INK)
    ax.text(0, -0.03, f"{int(jump.sum())} links jump, of {len(jump):,}", transform=ax.transAxes, va="top", fontsize=12.5,
            color=RED if jump.any() else "#008300", fontweight="bold")
    ax.text(0, -0.095, f"Longest hop {rows[name]['longest_link']:.3f}. Closest two layers {gap.min():.3f} apart.",
            transform=ax.transAxes, va="top", fontsize=12, color=MUTED)
cb = fig.add_axes([0.675, 0.115, 0.25, 0.018])
fig.colorbar(sc, cax=cb, orientation="horizontal")
cb.set_xlabel("Right panel: neighbours per dot (dark = more, longer hops)", fontsize=10.5, color=INK)
fig.savefig(OUT / "side_view.png", dpi=125, facecolor="white")
