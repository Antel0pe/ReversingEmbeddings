"""Show a 2-knob sheet of generated 1s as a curvilinear grid.

Figure 1  the curvy grid, beside the square knob grid it came from
Figure 2  why it curves: a knob's direction is an image with moving, local support
Figure 3  what survives the choice of projection

Window: height x lean, with cx=14, cy=14, width=3.2 held at root-1 values.
Height 14-20 extends grey_ones.RANGES (19.0-20.5); lean +-12 sits inside it.
No stroke in the window touches the canvas border.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from grey_ones import render

OUT = "figures/generated_one_curvy_grid"
HR, LR = (14.0, 20.0), (-12.0, 12.0)
CX, CY, W = 14.0, 14.0, 3.2
EPS_H, EPS_L = 1e-3, 1e-2
C_H, C_L = "#1f6feb", "#d1491c"
BOX = dict(boxstyle="round,pad=0.35", fc="white", ec="#999999", alpha=0.93)


def flat(hs, ls):
    return render([[CX, CY, h, W, l] for h, l in zip(hs, ls)], dtype=np.float64).reshape(len(hs), -1)


def img(h, l):
    return render([[CX, CY, h, W, l]], dtype=np.float64)[0]


def secants(h, l):
    cols = []
    for k, e in zip((2, 4), (EPS_H, EPS_L)):
        a = [CX, CY, h, W, l]
        b = [CX, CY, h, W, l]
        a[k] -= e
        b[k] += e
        cols.append((render([b], dtype=np.float64)[0] - render([a], dtype=np.float64)[0]).ravel() / (2 * e))
    return np.stack(cols, 1)


def turning(xy):
    d = np.diff(xy, axis=0)
    n = np.linalg.norm(d, axis=1)
    d, n = d[n > 1e-12], n[n > 1e-12]
    if len(d) < 2:
        return 0.0
    u = d / n[:, None]
    return float(np.degrees(np.arccos(np.clip((u[:-1] * u[1:]).sum(1), -1, 1))).sum())


NLINE, NSAMP = 7, 121
hs_line, ls_line = np.linspace(*HR, NLINE), np.linspace(*LR, NLINE)
h_dense, l_dense = np.linspace(*HR, NSAMP), np.linspace(*LR, NSAMP)
const_lean = [(h_dense, np.full(NSAMP, l)) for l in ls_line]
const_h = [(np.full(NSAMP, h), l_dense) for h in hs_line]

# best flat map: PCA plane of the whole sheet
gh, gl = np.meshgrid(np.linspace(*HR, 25), np.linspace(*LR, 25))
SHEET = flat(gh.ravel(), gl.ravel())
MU = SHEET.mean(0)
_, SV, VT = np.linalg.svd(SHEET - MU, full_matrices=False)
VAR2 = 100 * (SV[:2] ** 2).sum() / (SV ** 2).sum()
VAR3 = 100 * (SV[:3] ** 2).sum() / (SV ** 2).sum()


def pmap(hs, ls, k=2):
    return (flat(hs, ls) - MU) @ VT[:k].T


# ============================================================ FIGURE 1
fig = plt.figure(figsize=(14.0, 15.0))
fig.patch.set_facecolor("white")
fig.text(0.5, 0.982, "A 2-knob sheet of generated 1s, laid out flat",
         ha="center", va="top", fontsize=19, fontweight="bold")
fig.text(0.5, 0.948,
         "height x lean, with cx=14, cy=14, width=3.2 fixed.   Blue = lines of constant lean;   orange = lines of constant height.",
         ha="center", va="top", fontsize=11.5, color="#333")

ax = fig.add_axes([0.065, 0.575, 0.375, 0.300])
NT, GAP = 5, 3
ht, lt = np.linspace(*HR, NT), np.linspace(*LR, NT)
mos = np.full((NT * 28 + (NT - 1) * GAP, NT * 28 + (NT - 1) * GAP), np.nan)
for i, l in enumerate(lt):
    for j, h in enumerate(ht):
        mos[i * 31:i * 31 + 28, j * 31:j * 31 + 28] = img(h, l)
cm = plt.get_cmap("gray_r").copy()
cm.set_bad("#c9c9c9")
ax.imshow(mos, cmap=cm, vmin=0, vmax=1, interpolation="nearest")
ax.set_xticks([j * 31 + 14 for j in range(NT)])
ax.set_xticklabels([f"{h:.1f}" for h in ht], fontsize=10)
ax.set_yticks([i * 31 + 14 for i in range(NT)])
ax.set_yticklabels([f"{l:+.0f}" for l in lt], fontsize=10)
ax.set_xlabel("height  (pixels)", fontsize=11)
ax.set_ylabel("lean  (degrees)", fontsize=11)
ax.set_title("1   The object: 25 of the 1s on this sheet\nblack = full ink (1.0),  white = no ink (0.0)", fontsize=12.5, pad=9)

ax = fig.add_axes([0.575, 0.575, 0.375, 0.300])
for hh, ll in const_lean:
    ax.plot(hh, ll, color=C_H, lw=1.5)
for hh, ll in const_h:
    ax.plot(hh, ll, color=C_L, lw=1.5)
ax.set_xlabel("height  (pixels)", fontsize=11)
ax.set_ylabel("lean  (degrees)", fontsize=11)
ax.set_title("2   BASELINE - the knobs you dial\nstraight, parallel, evenly spaced, by construction\ntotal turning of every line:  0.0 deg", fontsize=12.5, pad=9)
for s in ax.spines.values():
    s.set_linewidth(2.6)
    s.set_color("#222")

ax = fig.add_axes([0.075, 0.135, 0.370, 0.325])
turns = []
for hh, ll in const_lean:
    xy = pmap(hh, ll)
    ax.plot(xy[:, 0], xy[:, 1], color=C_H, lw=1.6)
    turns.append(turning(xy))
for hh, ll in const_h:
    xy = pmap(hh, ll)
    ax.plot(xy[:, 0], xy[:, 1], color=C_L, lw=1.6)
    turns.append(turning(xy))
for h in hs_line:
    for l in ls_line:
        xy = pmap([h], [l])
        ax.plot(xy[0, 0], xy[0, 1], "o", ms=3.4, color="#222", zorder=5)
ax.set_xlabel("pixel distance, 1st axis of the best flat map", fontsize=10.5)
ax.set_ylabel("pixel distance, 2nd axis", fontsize=10.5)
ax.set_title(f"3   THE SAME GRID, in pixel space\nthe flattest honest view of the sheet  (holds {VAR2:.0f}% of it)\n"
             f"total turning per line:  median {np.median(turns):.0f} deg,  max {max(turns):.0f} deg", fontsize=12.5, pad=9)
ax.set_aspect("equal")

ax = fig.add_axes([0.525, 0.085, 0.45, 0.400], projection="3d")
for hh, ll in const_lean:
    xyz = pmap(hh, ll, 3)
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=C_H, lw=1.3)
for hh, ll in const_h:
    xyz = pmap(hh, ll, 3)
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=C_L, lw=1.3)
ax.set_xlabel("axis 1", fontsize=9.5, labelpad=1)
ax.set_ylabel("axis 2", fontsize=9.5, labelpad=1)
ax.set_zlabel("axis 3", fontsize=9.5, labelpad=1)
ax.tick_params(labelsize=8)
ax.view_init(elev=24, azim=-58)
fig.text(0.75, 0.513, f"4   The same sheet with a third axis added\n"
         f"a curved surface, not a bent plane  (holds {VAR3:.0f}%)",
         ha="center", va="top", fontsize=12.5, linespacing=1.4)

fig.text(0.5, 0.012,
         "Panel 3 is a PROJECTION: the 2-d plane through the 784-dimensional sheet that spreads it out most, so it distorts as little as any flat view can.\n"
         "Dialling a knob moves you along a line of its colour. In panel 2 those lines are a perfect square grid; in panel 3 the SAME lines bend, fan out,\n"
         "and crowd together - so a step of '+1 height' is a different move in pixels at every point on the sheet. No stroke in this window is clipped.",
         ha="center", va="bottom", fontsize=10.2, color="#333", linespacing=1.6)
fig.savefig(f"{OUT}/curvy_grid.png", dpi=125, facecolor="white")
plt.close(fig)
print("figure 1 done")


# ============================================================ FIGURE 2
fig = plt.figure(figsize=(14.0, 10.2))
fig.patch.set_facecolor("white")
fig.text(0.5, 0.980, "Why the grid curves: a knob's direction is an image, and it moves",
         ha="center", va="top", fontsize=19, fontweight="bold")
fig.text(0.5, 0.947,
         "'The height direction' is the picture of what one extra unit of height does to the ink. It lives only where the stroke's two ends currently are.",
         ha="center", va="top", fontsize=11.5, color="#333")

HSHOW = [15.0, 17.0, 19.0]
dirs = [secants(h, 0.0)[:, 0].reshape(28, 28) for h in HSHOW]
M = max(np.abs(d).max() for d in dirs)
for i, (h, d) in enumerate(zip(HSHOW, dirs)):
    ax = fig.add_axes([0.045 + 0.180 * i, 0.545, 0.170, 0.285])
    ax.imshow(d, cmap="bwr_r", vmin=-M, vmax=M, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"height = {h:.0f}", fontsize=12.5, pad=7)
    ax.text(0.5, -0.09, "one more unit of height", transform=ax.transAxes,
            ha="center", va="top", fontsize=10, color="#333")
cax = fig.add_axes([0.592, 0.545, 0.013, 0.285])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-M, M), cmap="bwr_r"), cax=cax)
cb.set_label("blue = ink gained,   red = ink lost", fontsize=9.5, labelpad=2)
cb.ax.tick_params(labelsize=8.5)

ax = fig.add_axes([0.680, 0.545, 0.275, 0.285])
cos = np.array([[float(dirs[i].ravel() @ dirs[j].ravel() /
                       (np.linalg.norm(dirs[i]) * np.linalg.norm(dirs[j])))
                 for j in range(3)] for i in range(3)])
im = ax.imshow(cos, cmap="RdYlBu_r", vmin=-1, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cos[i, j]:.3f}", ha="center", va="center", fontsize=13,
                color="white" if abs(cos[i, j]) > 0.55 else "#111", fontweight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels([f"{h:.0f}" for h in HSHOW], fontsize=10.5)
ax.set_yticks(range(3)); ax.set_yticklabels([f"{h:.0f}" for h in HSHOW], fontsize=10.5)
ax.set_xlabel("height", fontsize=10.5); ax.set_ylabel("height", fontsize=10.5)
ax.set_title("cosine between those three pictures\n0.000 = they share no pixel at all", fontsize=12.5, pad=7)

ax = fig.add_axes([0.075, 0.145, 0.375, 0.265])
dh = np.linspace(0, 4, 161)
d0 = secants(17.0, 0.0)[:, 0]; d0 /= np.linalg.norm(d0)
ch = []
for x in dh:
    v = secants(17.0 + x, 0.0)[:, 0]
    ch.append(float(d0 @ v / np.linalg.norm(v)))
e0 = secants(17.0, 0.0)[:, 1]; e0 /= np.linalg.norm(e0)
dl = np.linspace(0, 30, 121)
cl = []
for x in dl:
    v = secants(17.0, 0.0 + x)[:, 1]
    cl.append(float(e0 @ v / np.linalg.norm(v)))
ax.plot(dh / 2.0, ch, color=C_H, lw=2.4, label="height direction  (x = how far each end moved)")
ax.plot(dl / 30.0 * 2.0, cl, color=C_L, lw=2.4, ls="--", label="lean direction  (x rescaled: 0-30 deg over 0-2)")
ax.axhline(0, color="#888", lw=1.0)
ax.axvline(0.5, color="#222", lw=1.3, ls=":")
ax.text(0.56, 0.42, "the end crosses into\nthe next pixel row", fontsize=9.6, color="#222")
ax.set_xlim(0, 2); ax.set_ylim(-0.25, 1.05)
ax.set_xlabel("how far the stroke's end has moved  (pixels)", fontsize=10.5)
ax.set_ylabel("cosine with the direction at height 17", fontsize=10.5)
ax.legend(fontsize=9.3, loc="upper right", framealpha=0.95)
ax.set_title("A   How long a direction stays itself\nheight: a cliff at half a pixel.   lean: a gentle fade.",
             fontsize=12.5, pad=7)

ax = fig.add_axes([0.575, 0.145, 0.375, 0.265])
Q = np.linalg.qr(secants(17.0, 0.0))[0]
base = img(17.0, 0.0).ravel()
hs = np.linspace(14, 20, 121)
true = [float(np.linalg.norm(img(h, 0.0).ravel() - base)) for h in hs]
seen = [float(np.linalg.norm((img(h, 0.0).ravel() - base) @ Q)) for h in hs]
ax.plot(hs, true, color="#222", lw=2.4, label="true distance in pixel space")
ax.plot(hs, seen, color=C_H, lw=2.4, ls="--", label="distance the anchor's flat plane sees")
ax.axvline(17, color="#888", lw=1.1, ls=":")
ax.text(17.08, max(true) * 0.93, "anchor", fontsize=10, color="#555")
ax.set_xlabel("height  (pixels)", fontsize=10.5)
ax.set_ylabel("distance from the anchor 1", fontsize=10.5)
ax.legend(fontsize=9.6, loc="upper center", framealpha=0.95)
ax.set_title("B   Why you cannot just call one 1 'the straight one'\nits flat plane saturates: past ~1 px it stops seeing the change",
             fontsize=12.5, pad=7)

fig.text(0.5, 0.012,
         "Left: each 28x28 picture is a DIRECTION, not a 1 - it shows where ink arrives and leaves when height alone goes up, on a shared colour scale.\n"
         "They do not share a single pixel, so their cosines are exactly 0.000: the same knob is a completely different move in pixels at 15, 17 and 19.\n"
         "That is what makes the grid in Figure 1 bend, and it is why anchoring to a single 1 (panel B) cannot describe the whole sheet.",
         ha="center", va="bottom", fontsize=10.2, color="#333", linespacing=1.6)
fig.savefig(f"{OUT}/why_it_curves.png", dpi=125, facecolor="white")
plt.close(fig)
print("figure 2 done")


# ============================================================ FIGURE 3
fig = plt.figure(figsize=(13.5, 6.6))
fig.patch.set_facecolor("white")
fig.text(0.5, 0.968, "What survives the choice of picture",
         ha="center", va="top", fontsize=19, fontweight="bold")
fig.text(0.5, 0.922,
         "Every flat drawing of the sheet involved a choice. These two readings do not: both come only from measured pixel distances.",
         ha="center", va="top", fontsize=11.5, color="#333")

ax = fig.add_axes([0.060, 0.150, 0.370, 0.640])
NN = 5
us = np.linspace(0, 1, NN)
th = np.linspace(0, 2 * np.pi, 181)
RS = []
for u in us:
    for v in us:
        h = HR[0] + u * (HR[1] - HR[0])
        l = LR[0] + v * (LR[1] - LR[0])
        J = secants(h, l) * np.array([HR[1] - HR[0], LR[1] - LR[0]])
        g = J.T @ J
        d = np.stack([np.cos(th), np.sin(th)])
        RS.append((u, v, np.sqrt(np.einsum("it,ij,jt->t", d, g, d))))
RMAX = max(r.max() for _, _, r in RS)
SC = 0.105 / RMAX
for u, v, r in RS:
    ax.plot(u + SC * r * np.cos(th), v + SC * r * np.sin(th), color="#6a2c8f", lw=1.7)
    ax.plot([u], [v], ".", color="#222", ms=3)
for u in us:
    ax.plot([u, u], [0, 1], color="#cccccc", lw=0.8, zorder=0)
    ax.plot([0, 1], [u, u], color="#cccccc", lw=0.8, zorder=0)
rmin = min(r.min() for _, _, r in RS)
ax.set_xlim(-0.14, 1.14); ax.set_ylim(-0.14, 1.14)
ax.set_xticks(us); ax.set_xticklabels([f"{HR[0] + u * (HR[1] - HR[0]):.1f}" for u in us], fontsize=10)
ax.set_yticks(us); ax.set_yticklabels([f"{LR[0] + u * (LR[1] - LR[0]):+.0f}" for u in us], fontsize=10)
ax.set_xlabel("height  (pixels)", fontsize=11)
ax.set_ylabel("lean  (degrees)", fontsize=11)
ax.set_aspect("equal")
ax.set_title("1   The knob grid, with the truth drawn on top\n"
             "each ring = how far one equal knob-step actually moves you,\n"
             "in every direction, at that 1", fontsize=12.3, pad=9)
ax.text(0.02, 0.02, f"biggest step / smallest step across the sheet:  {RMAX / rmin:.1f}x",
        transform=ax.transAxes, fontsize=10.2, va="bottom", bbox=BOX)

ax = fig.add_axes([0.560, 0.150, 0.390, 0.640])
D2 = np.linalg.norm(SHEET[:, None, :] - SHEET[None, :, :], axis=2) ** 2
n = len(SHEET)
Jc = np.eye(n) - np.ones((n, n)) / n
w = np.clip(np.linalg.eigvalsh(-0.5 * Jc @ D2 @ Jc)[::-1], 0, None)
share = 100 * w[:7] / w.sum()
cols = ["#888888", "#888888"] + ["#c0392b"] * 5
ax.bar(range(1, 8), share, color=cols, edgecolor="#333", lw=0.8)
for i, v in enumerate(share):
    ax.text(i + 1, v + 1.1, f"{v:.1f}%", ha="center", fontsize=10.2,
            fontweight="bold" if i == 2 else "normal")
ax.axhline(0, color="#333", lw=1.0)
ax.set_xlabel("flat-map axis", fontsize=11)
ax.set_ylabel("share of the sheet's spread", fontsize=11)
ax.set_ylim(0, max(share) * 1.22)
ax.set_title("2   Can the sheet be drawn flat without lying?\n"
             "grey = the two axes any flat map gets.  red = what is left over.\n"
             "on truly flat paper every red bar would be 0.0%", fontsize=12.3, pad=9)
ax.text(0.97, 0.80, f"3rd axis = {share[2]:.1f}% of the spread\n"
                    f"-> NO flat map of this sheet\n    can be faithful",
        transform=ax.transAxes, ha="right", va="top", fontsize=10.6, bbox=BOX)

fig.text(0.5, 0.012,
         "Left: the rings are an honest local ruler - a circle of equal knob-steps, drawn at the size and shape it really has in pixels. They change size and tilt across the\n"
         "sheet, which IS the bending, with no projection chosen. Right: these numbers come from the matrix of distances between the 1s, which no choice of view can alter.",
         ha="center", va="bottom", fontsize=10.2, color="#333", linespacing=1.6)
fig.savefig(f"{OUT}/what_survives.png", dpi=125, facecolor="white")
plt.close(fig)
print("figure 3 done")
